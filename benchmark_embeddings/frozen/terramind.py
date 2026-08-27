#!/usr/bin/env python3
"""Canonical TerraMind extraction for the county-yield benchmark.

Two Sentinel-2 L2A spectral contracts are supported from the same ten-band
source cohort:

``s2_6_prithvi``
    B02, B03, B04, B8A, B11, and B12, supplied through TerraMind's official
    band-selection interface so the corresponding pretrained patch weights
    are retained without artificial channels.

``s2_10_zero_pad``
    All ten available benchmark bands placed in TerraMind's official 12-band
    S2L2A order, with unavailable B01 and B09 explicitly set to zero.

Source patches are first harmonized to the benchmark's 256x256 footprint and
then center-cropped to TerraMind's native 224x224 input.  No interpolation is
used.  TerraMind has no CLS token: the representation is the mean of all 196
final-layer spatial patch tokens.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ..data.io import (
    RAW_S2_12_BANDS,
    S2_10_BANDS,
    UNDERSIZE_POLICIES,
    band_names,
    metadata as npz_metadata,
    normalise_band_name,
    normalise_county,
    normalise_undersize_policy,
    reindex_bands,
    safe_float,
    screen_undersized_patches,
)
from .schema import validate_embeddings, write_embeddings


OFFICIAL_REPOSITORY = "https://github.com/IBM/terramind"
OFFICIAL_TERRATORCH_REPOSITORY = "https://github.com/torchgeo/terratorch"
OFFICIAL_MODEL_ORGANIZATION = "ibm-esa-geospatial"
TERRAMIND_MODALITY = "S2L2A"

EXPERIMENT_S2_6 = "s2_6_prithvi"
EXPERIMENT_S2_10_ZERO_PAD = "s2_10_zero_pad"
EXPERIMENTS = (EXPERIMENT_S2_6, EXPERIMENT_S2_10_ZERO_PAD)

SOURCE_BANDS_S2_6 = ("B02", "B03", "B04", "B8A", "B11", "B12")
TERRAMIND_BANDS_S2_6 = (
    "BLUE",
    "GREEN",
    "RED",
    "NIR_NARROW",
    "SWIR_1",
    "SWIR_2",
)
TERRAMIND_BANDS_S2_12 = (
    "COASTAL_AEROSOL",
    "BLUE",
    "GREEN",
    "RED",
    "RED_EDGE_1",
    "RED_EDGE_2",
    "RED_EDGE_3",
    "NIR_BROAD",
    "NIR_NARROW",
    "WATER_VAPOR",
    "SWIR_1",
    "SWIR_2",
)
ZERO_PADDED_SOURCE_BANDS = ("B01", "B09")

MODEL_DIMS = {
    "terramind_v1_tiny": 192,
    "terramind_v1_small": 384,
    "terramind_v1_base": 768,
    "terramind_v1_large": 1024,
}
MODEL_DEPTHS = {
    "terramind_v1_tiny": 12,
    "terramind_v1_small": 12,
    "terramind_v1_base": 12,
    "terramind_v1_large": 24,
}

DEFAULT_MODEL = "terramind_v1_base"
DEFAULT_SOURCE_SIZE = (256, 256)
DEFAULT_MODEL_SIZE = (224, 224)
DEFAULT_TOKEN_COUNT = 196  # (224 / patch 16)^2, TerraMind has no CLS token


def terramind_token_count(model_size, patch_size: int = 16) -> int:
    """Spatial token count for a given model input; TerraMind has no CLS token.

    Derived rather than pinned to 196: the backbone accepts other sizes
    (verified 224 -> 196 tokens, 256 -> 256 tokens), so a hardcoded expectation
    would reject a valid non-224 configuration.
    """
    height, width = int(model_size[0]), int(model_size[1])
    return (height // patch_size) * (width // patch_size)

_COUNTY_YEAR = re.compile(
    r"county[_-](?P<county>\d+)(?:[_-]year)?[_-](?P<year>\d{4})",
    re.IGNORECASE,
)
_TIMESTEP = re.compile(
    r"(?:^|_)(?:t|interval|timestep)[_-]?(?P<t>\d+)(?:$|_)",
    re.IGNORECASE,
)
_SPATIAL_ID = re.compile(
    r"(?:^|_)x[_-]?(?P<x>-?\d+).*?(?:^|_)y[_-]?(?P<y>-?\d+)",
    re.IGNORECASE,
)


def _pair(value: int | Sequence[int], *, label: str) -> tuple[int, int]:
    if isinstance(value, (int, np.integer)):
        result = (int(value), int(value))
    else:
        result = tuple(int(part) for part in value)
        if len(result) != 2:
            raise ValueError(f"{label} must be an integer or [height, width]")
    if min(result) <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _first(mapping: dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None and not (isinstance(value, str) and value == ""):
            return value
    return default


def _source_fields(path: Path) -> tuple[str, int, str, int]:
    county_year = _COUNTY_YEAR.search(path.stem)
    timestep = _TIMESTEP.search(path.stem + "_")
    spatial = _SPATIAL_ID.search(path.stem)
    if county_year is None or timestep is None:
        raise ValueError(
            f"cannot infer county/year/timestep from {path.name}; retain those "
            "fields in every source filename"
        )
    county = normalise_county(county_year.group("county"))
    year = int(county_year.group("year"))
    raw_timestep = int(timestep.group("t"))
    if spatial is not None:
        patch_id = f"x{spatial.group('x')}_y{spatial.group('y')}"
    else:
        patch_id = re.sub(
            r"(?:_stack_COG)?_(?:t|interval|timestep)[_-]?\d+(?:_.*)?$",
            "",
            path.stem,
            flags=re.IGNORECASE,
        )
    if not patch_id:
        raise ValueError(f"cannot derive a stable spatial patch_id from {path.name}")
    return county, year, patch_id, raw_timestep


def _resolve_timestep_base(values: Sequence[int], expected: int, requested: str | int) -> int:
    if str(requested) in {"0", "1"}:
        return int(requested)
    unique = set(int(value) for value in values)
    if 0 in unique:
        return 0
    if 1 in unique and expected in unique:
        return 1
    raise ValueError(
        "timestep base is ambiguous; pass timestep_base=0 or timestep_base=1 "
        f"(observed values: {sorted(unique)})"
    )


def _pixel_array(z: Any) -> np.ndarray:
    key = next(
        (
            name
            for name in ("pixels", "patch", "cube", "data", "s2", "image", "array")
            if name in z.files
        ),
        None,
    )
    if key is None:
        raise KeyError(f"no Sentinel-2 pixel array found; keys={z.files}")
    array = np.asarray(z[key], dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"expected one patch [C,H,W] or [H,W,C], got {array.shape}")
    if array.shape[0] not in (10, 12) and array.shape[-1] in (10, 12):
        array = np.moveaxis(array, -1, 0)
    if array.shape[0] not in (10, 12):
        raise ValueError(
            "TerraMind experiments must start from the shared ten/12-band source "
            f"cohort, got {array.shape[0]} channels"
        )
    return array


def _center_crop(
    image: np.ndarray,
    *,
    target_size: tuple[int, int],
    path: Path,
    stage: str,
) -> np.ndarray:
    height, width = int(image.shape[-2]), int(image.shape[-1])
    target_height, target_width = target_size
    if height < target_height or width < target_width:
        raise ValueError(
            f"{path}: source patch is {height}x{width}, below {stage} expectation "
            f"{target_height}x{target_width}; padding and interpolation are disabled"
        )
    top = (height - target_height) // 2
    left = (width - target_width) // 2
    return image[..., top : top + target_height, left : left + target_width]


def _input_bands(
    z: Any,
    meta: dict[str, Any],
    *,
    channels: int,
    require_names: bool,
) -> tuple[str, ...]:
    detected = band_names(z)
    if detected is None:
        value = _first(meta, ("band_names", "bands", "band_order"))
        if value is not None:
            detected = tuple(normalise_band_name(item) for item in value)
    if detected is None:
        if require_names:
            raise ValueError(
                "Sentinel-2 NPZ has no band_names; pass "
                "--assume-canonical-band-order only after verifying the source order"
            )
        detected = S2_10_BANDS if channels == 10 else RAW_S2_12_BANDS
    if len(detected) != channels:
        raise ValueError(f"band metadata has {len(detected)} names for {channels} channels")
    normalized = tuple(normalise_band_name(value) for value in detected)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"band metadata contains duplicate names: {normalized}")
    return normalized


def _to_reflectance(pixels: np.ndarray, *, source_units: str, path: Path) -> np.ndarray:
    pixels = np.asarray(pixels, dtype=np.float32)
    finite = pixels[np.isfinite(pixels)]
    max_abs = float(np.max(np.abs(finite))) if finite.size else 0.0
    if source_units == "auto":
        source_units = "reflectance" if max_abs <= 2.0 else "dn"
    if source_units == "dn":
        pixels = pixels / 10000.0
    elif source_units == "reflectance":
        if max_abs > 2.0:
            raise ValueError(
                f"{path.name}: values look like digital numbers but source_units=reflectance"
            )
    else:
        raise ValueError("source_units must be 'auto', 'dn', or 'reflectance'")
    # This intentionally follows the historical direct-extraction script.
    return np.clip(pixels, 0.0, 1.0)


def prepare_terramind_bands(
    pixels: np.ndarray,
    input_bands: Sequence[str],
    experiment: str,
) -> np.ndarray:
    """Select six official bands or construct the explicit 10+2-zero input."""
    experiment = str(experiment).strip().lower()
    if experiment == EXPERIMENT_S2_6:
        return reindex_bands(pixels[None, ...], input_bands, SOURCE_BANDS_S2_6)[0]
    if experiment != EXPERIMENT_S2_10_ZERO_PAD:
        raise ValueError(f"unknown TerraMind experiment {experiment!r}")

    observed = reindex_bands(pixels[None, ...], input_bands, S2_10_BANDS)[0]
    by_band = {band: observed[index] for index, band in enumerate(S2_10_BANDS)}
    output = []
    for band in RAW_S2_12_BANDS:
        if band in by_band:
            output.append(by_band[band])
        else:
            output.append(np.zeros_like(observed[0], dtype=np.float32))
    return np.stack(output, axis=0)


def terramind_model_kwargs(experiment: str) -> dict[str, Any]:
    """Return the official TerraTorch constructor contract for an experiment."""
    experiment = str(experiment).strip().lower()
    kwargs: dict[str, Any] = {
        "modalities": [TERRAMIND_MODALITY],
        "merge_method": "mean",
    }
    if experiment == EXPERIMENT_S2_6:
        kwargs["bands"] = {TERRAMIND_MODALITY: list(TERRAMIND_BANDS_S2_6)}
    elif experiment != EXPERIMENT_S2_10_ZERO_PAD:
        raise ValueError(f"unknown TerraMind experiment {experiment!r}")
    return kwargs


@dataclass(frozen=True)
class TerraMindPatchIndex:
    path: Path
    source_id: str
    patch_id: str
    county_id: str
    year: int
    raw_timestep: int
    timestep: int


class TerraMindPatchDataset(Dataset):
    """Read one shared-cohort Sentinel-2 patch-timestep for TerraMind."""

    def __init__(
        self,
        npz_dir: str | Path,
        *,
        experiment: str,
        source_size: int | Sequence[int] = DEFAULT_SOURCE_SIZE,
        model_size: int | Sequence[int] = DEFAULT_MODEL_SIZE,
        expected_timesteps: int = 7,
        timestep_base: str | int = "auto",
        expected_input_count: int | None = None,
        require_band_names: bool = True,
        source_units: str = "auto",
        nonfinite_policy: str = "error",
        undersize_policy: str = "error",
        max_files: int | None = None,
    ):
        self.root = Path(npz_dir)
        if not self.root.exists():
            raise FileNotFoundError(f"Sentinel-2 NPZ directory does not exist: {self.root}")
        self.experiment = str(experiment).strip().lower()
        if self.experiment not in EXPERIMENTS:
            raise ValueError(f"experiment must be one of {EXPERIMENTS}")
        self.source_size = _pair(source_size, label="source_size")
        self.model_size = _pair(model_size, label="model_size")
        if self.model_size[0] > self.source_size[0] or self.model_size[1] > self.source_size[1]:
            raise ValueError("model_size cannot exceed the harmonized source_size")
        self.expected_timesteps = int(expected_timesteps)
        self.require_band_names = bool(require_band_names)
        self.source_units = str(source_units).strip().lower()
        if self.source_units not in {"auto", "dn", "reflectance"}:
            raise ValueError("source_units must be 'auto', 'dn', or 'reflectance'")
        self.nonfinite_policy = str(nonfinite_policy).strip().lower()
        if self.nonfinite_policy not in {"error", "zero"}:
            raise ValueError("nonfinite_policy must be 'error' or 'zero'")
        self.undersize_policy = normalise_undersize_policy(undersize_policy)

        all_paths = sorted(self.root.rglob("*.npz"))
        if not all_paths:
            raise FileNotFoundError(f"no NPZ files found below {self.root}")
        self.source_cohort_count = len(all_paths)
        if expected_input_count is not None and len(all_paths) != int(expected_input_count):
            raise ValueError(
                f"TerraMind input cohort has {len(all_paths):,} files, expected "
                f"{int(expected_input_count):,}"
            )
        parsed = [_source_fields(path) for path in all_paths]
        base = _resolve_timestep_base(
            [fields[3] for fields in parsed], self.expected_timesteps, timestep_base
        )
        indices = []
        excluded_paths = []
        for path, (county, year, patch_id, raw_timestep) in zip(all_paths, parsed):
            timestep = raw_timestep - base
            if not 0 <= timestep < self.expected_timesteps:
                excluded_paths.append(path)
                continue
            indices.append(
                TerraMindPatchIndex(
                    path=path,
                    source_id=path.stem,
                    patch_id=patch_id,
                    county_id=county,
                    year=year,
                    raw_timestep=raw_timestep,
                    timestep=timestep,
                )
            )
        keys = [
            (item.county_id, item.year, item.patch_id, item.timestep)
            for item in indices
        ]
        if len(keys) != len(set(keys)):
            raise ValueError(
                "source files collapse to duplicate county/year/patch/timestep keys; "
                "spatial patch identity has not been preserved"
            )
        # Screen per spatial patch, not per file: dropping one timestep would
        # leave a ragged sequence the county aggregation assumes cannot occur.
        grouped_paths: dict[tuple[str, int, str], list[Path]] = {}
        for item in indices:
            grouped_paths.setdefault(
                (item.county_id, item.year, item.patch_id), []
            ).append(item.path)
        _, undersized_keys, undersized_files = screen_undersized_patches(
            grouped_paths, target_size=self.source_size, policy=self.undersize_policy
        )
        dropped = set(undersized_keys)
        if dropped:
            indices = [
                item
                for item in indices
                if (item.county_id, item.year, item.patch_id) not in dropped
            ]
        self._undersized_patches_excluded = len(dropped)
        self._undersized_files_excluded = int(undersized_files)

        self.indices = indices if max_files is None else indices[: int(max_files)]
        if not self.indices:
            raise ValueError("max_files selected no TerraMind inputs")
        self.timestep_base = base
        self._out_of_schedule_paths = tuple(excluded_paths)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.indices[index]
        with np.load(item.path, allow_pickle=True) as z:
            meta = npz_metadata(z)
            pixels = _pixel_array(z)
            input_bands = _input_bands(
                z,
                meta,
                channels=int(pixels.shape[0]),
                require_names=self.require_band_names,
            )

        pixels = _center_crop(
            pixels,
            target_size=self.source_size,
            path=item.path,
            stage="benchmark source-size",
        )
        pixels = _center_crop(
            pixels,
            target_size=self.model_size,
            path=item.path,
            stage="TerraMind model-size",
        )
        valid_fraction = float(np.isfinite(pixels).all(axis=0).mean())
        if valid_fraction < 1.0 and self.nonfinite_policy == "error":
            raise ValueError(
                f"{item.path.name}: non-finite pixels remain after compositing "
                f"(valid spatial fraction={valid_fraction:.6f})"
            )
        pixels = np.nan_to_num(pixels, nan=0.0, posinf=0.0, neginf=0.0)
        pixels = _to_reflectance(pixels, source_units=self.source_units, path=item.path)
        pixels = prepare_terramind_bands(pixels, input_bands, self.experiment)

        meta_county = normalise_county(
            _first(meta, ("county_fips", "county", "fips", "GEOID"), item.county_id)
        )
        meta_year = int(safe_float(_first(meta, ("year",), item.year), item.year))
        if meta_county != item.county_id or meta_year != item.year:
            raise ValueError(
                f"{item.path.name}: filename county/year {item.county_id}/{item.year} "
                f"disagrees with metadata {meta_county}/{meta_year}"
            )

        return {
            "pixels": torch.from_numpy(np.ascontiguousarray(pixels, dtype=np.float32)),
            "county_id": item.county_id,
            "year": item.year,
            "patch_id": item.patch_id,
            "timestep": item.timestep,
            "source_id": item.source_id,
            "source_file": str(item.path),
            "valid_fraction": valid_fraction,
        }

    def describe(self) -> dict[str, Any]:
        frame = pd.DataFrame(
            {
                "county_id": [item.county_id for item in self.indices],
                "year": [item.year for item in self.indices],
                "patch_id": [item.patch_id for item in self.indices],
                "timestep": [item.timestep for item in self.indices],
            }
        )
        patch_counts = (
            frame[["county_id", "year", "patch_id"]]
            .drop_duplicates()
            .groupby(["county_id", "year"])
            .size()
        )
        coverage = frame.groupby(["county_id", "year", "patch_id"])["timestep"].nunique()
        complete = coverage.eq(self.expected_timesteps)
        if self.experiment == EXPERIMENT_S2_6:
            model_bands = list(TERRAMIND_BANDS_S2_6)
            source_bands = list(SOURCE_BANDS_S2_6)
            padded = []
        else:
            model_bands = list(TERRAMIND_BANDS_S2_12)
            source_bands = list(S2_10_BANDS)
            padded = list(ZERO_PADDED_SOURCE_BANDS)
        return {
            "source_cohort_files": int(self.source_cohort_count),
            "schedule_input_files": int(
                self.source_cohort_count - len(self._out_of_schedule_paths)
            ),
            "selected_input_files": int(len(frame)),
            "out_of_schedule_files_excluded": int(len(self._out_of_schedule_paths)),
            "out_of_schedule_timestep_policy": "audit_then_exclude",
            "county_years": int(frame[["county_id", "year"]].drop_duplicates().shape[0]),
            "spatial_patches": int(
                frame[["county_id", "year", "patch_id"]].drop_duplicates().shape[0]
            ),
            "complete_spatial_patches": int(complete.sum()),
            "incomplete_spatial_patches": int((~complete).sum()),
            "patch_count_min": int(patch_counts.min()),
            "patch_count_median": float(patch_counts.median()),
            "patch_count_max": int(patch_counts.max()),
            "timesteps": sorted(int(value) for value in frame["timestep"].unique()),
            "timestep_base": int(self.timestep_base),
            "experiment": self.experiment,
            "source_size": list(self.source_size),
            "model_size": list(self.model_size),
            "oversize_policy": (
                f"center_crop_to_{self.source_size[0]}"
                f"_then_center_crop_to_{self.model_size[0]}"
            ),
            "undersize_policy": self.undersize_policy,
            "undersized_spatial_patches_excluded": int(
                self._undersized_patches_excluded
            ),
            "undersized_files_excluded": int(self._undersized_files_excluded),
            "interpolation": "none",
            "source_units": self.source_units,
            "source_units_policy": (
                "detect_each_file_then_convert_to_reflectance"
                if self.source_units == "auto"
                else "declared_uniform"
            ),
            "normalization": (
                "detect_units_then_convert_to_reflectance_then_clip_0_1"
                if self.source_units == "auto"
                else (
                    "divide_dn_by_10000_then_clip_0_1"
                    if self.source_units == "dn"
                    else "clip_reflectance_0_1"
                )
            ),
            "nonfinite_policy": self.nonfinite_policy,
            "observed_source_bands": source_bands,
            "model_band_names": model_bands,
            "zero_padded_source_bands": padded,
            "model_input_channels": 6 if self.experiment == EXPERIMENT_S2_6 else 12,
        }


def terramind_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty TerraMind batch")
    output: dict[str, Any] = {
        "pixels": torch.stack([sample["pixels"] for sample in batch])
    }
    for name in batch[0]:
        if name != "pixels":
            output[name] = [sample[name] for sample in batch]
    return output


def pool_terramind_tokens(
    output: Any,
    *,
    expected_token_count: int | None = DEFAULT_TOKEN_COUNT,
    expected_embedding_dim: int | None = None,
) -> torch.Tensor:
    """Mean-pool all final-layer spatial tokens; TerraMind has no CLS token."""
    if isinstance(output, (list, tuple)):
        if not output:
            raise ValueError("TerraMind returned no encoder layers")
        tokens = output[-1]
    else:
        tokens = output
    if not isinstance(tokens, torch.Tensor):
        raise TypeError(f"TerraMind final output must be a tensor, got {type(tokens)}")
    if tokens.ndim != 3:
        raise ValueError(
            f"TerraMind final layer must have shape [B,L,D], got {tuple(tokens.shape)}"
        )
    if expected_token_count is not None and tokens.shape[1] != int(expected_token_count):
        raise ValueError(
            f"TerraMind produced {tokens.shape[1]} tokens, expected "
            f"{int(expected_token_count)}; the count is (size/16)^2, so check "
            f"the configured model input size"
        )
    if expected_embedding_dim is not None and tokens.shape[2] != int(expected_embedding_dim):
        raise ValueError(
            f"TerraMind embedding dimension is {tokens.shape[2]}, expected "
            f"{int(expected_embedding_dim)}"
        )
    return tokens.mean(dim=1)


def extract_terramind_embeddings(
    dataset: TerraMindPatchDataset,
    model: Callable[[dict[str, torch.Tensor]], Any],
    *,
    model_name: str,
    device: torch.device | str,
    batch_size: int = 8,
    num_workers: int = 0,
    expected_token_count: int | None = DEFAULT_TOKEN_COUNT,
    expected_embedding_dim: int | None = None,
) -> pd.DataFrame:
    """Extract one final-layer spatial-mean vector for every source row."""
    if model_name not in MODEL_DIMS and expected_embedding_dim is None:
        raise ValueError(
            f"unknown TerraMind model {model_name!r}; pass expected_embedding_dim explicitly"
        )
    expected_embedding_dim = (
        MODEL_DIMS.get(model_name)
        if expected_embedding_dim is None
        else int(expected_embedding_dim)
    )
    device = torch.device(device)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=terramind_collate,
        pin_memory=device.type == "cuda",
    )
    backbone = f"{model_name}_{dataset.experiment}"
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            output = model({TERRAMIND_MODALITY: batch["pixels"].to(device)})
            vectors = pool_terramind_tokens(
                output,
                expected_token_count=expected_token_count,
                expected_embedding_dim=expected_embedding_dim,
            ).detach().cpu().to(torch.float32).numpy()
            for row_index, vector in enumerate(vectors):
                rows.append(
                    {
                        "county_id": batch["county_id"][row_index],
                        "year": batch["year"][row_index],
                        "patch_id": batch["patch_id"][row_index],
                        "timestep": batch["timestep"][row_index],
                        "backbone": backbone,
                        "embedding": vector.tolist(),
                        "source_id": batch["source_id"][row_index],
                        "source_file": batch["source_file"][row_index],
                        "valid_fraction": batch["valid_fraction"][row_index],
                        "representation_scope": "timestep",
                        "spectral_experiment": dataset.experiment,
                        "encoder_layer": "final",
                        "token_pool": "mean_all_spatial_tokens",
                    }
                )
    if len(rows) != len(dataset):
        raise RuntimeError(
            f"TerraMind extraction emitted {len(rows):,} rows for "
            f"{len(dataset):,} source files"
        )
    return validate_embeddings(pd.DataFrame(rows))


def _device(value: str) -> torch.device:
    value = str(value).lower()
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_terramind(
    *,
    model_name: str,
    experiment: str,
    device: torch.device,
    checkpoint_path: str | Path | None = None,
) -> torch.nn.Module:
    """Build the official TerraTorch backbone and validate its input adapter."""
    if model_name not in MODEL_DIMS:
        raise ValueError(f"model_name must be one of {tuple(MODEL_DIMS)}")
    try:
        from terratorch.registry import BACKBONE_REGISTRY
    except ImportError as exc:
        raise ImportError(
            "TerraMind extraction requires an installed official TerraTorch package"
        ) from exc

    kwargs = terramind_model_kwargs(experiment)
    if checkpoint_path is None:
        kwargs["pretrained"] = True
    else:
        checkpoint = Path(checkpoint_path).resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"TerraMind checkpoint does not exist: {checkpoint}")
        kwargs.update(pretrained=False, ckpt_path=str(checkpoint))
    model = BACKBONE_REGISTRY.build(model_name, **kwargs).to(device).eval()

    try:
        embedding = model.encoder_embeddings[model.mod_name_mapping[TERRAMIND_MODALITY]]
        patch_height, patch_width = embedding.patch_size
        channels = int(embedding.proj.in_features // (patch_height * patch_width))
    except (AttributeError, KeyError, TypeError) as exc:
        raise TypeError("loaded TerraMind model has no inspectable S2L2A patch adapter") from exc
    expected_channels = 6 if experiment == EXPERIMENT_S2_6 else 12
    if channels != expected_channels:
        raise ValueError(
            f"TerraMind S2L2A adapter has {channels} channels, expected {expected_channels}"
        )
    return model


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-dir", required=True)
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument("--model", choices=tuple(MODEL_DIMS), default=DEFAULT_MODEL)
    parser.add_argument("--checkpoint", help="Optional local official TerraMind checkpoint")
    parser.add_argument("--output", help="Canonical output Parquet")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--expected-input-count", type=int)
    parser.add_argument("--expected-timesteps", type=int, default=7)
    parser.add_argument("--timestep-base", choices=("auto", "0", "1"), default="auto")
    parser.add_argument(
        "--source-units",
        choices=("auto", "dn", "reflectance"),
        default="auto",
    )
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--assume-canonical-band-order", action="store_true")
    parser.add_argument("--nonfinite-policy", choices=("error", "zero"), default="error")
    parser.add_argument(
        "--source-size",
        type=int,
        default=DEFAULT_SOURCE_SIZE[0],
        help=(
            "Harmonized center-crop footprint in pixels, applied before the native 224x224 model crop. Default 256 preserves the historical contract; 224 makes the harmonized footprint equal the model input, so all encoders share one footprint and fewer edge patches are rejected."
        ),
    )
    parser.add_argument(
        "--undersize-policy",
        choices=UNDERSIZE_POLICIES,
        default="error",
        help=(
            "'error' (default) rejects patches below the 256x256 source footprint. "
            "'skip' drops those spatial patches and records the counts; it is a "
            "recorded deviation that changes per-county patch counts."
        ),
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)

    dataset = TerraMindPatchDataset(
        args.npz_dir,
        experiment=args.experiment,
        source_size=(args.source_size, args.source_size),
        model_size=DEFAULT_MODEL_SIZE,
        expected_timesteps=args.expected_timesteps,
        timestep_base=args.timestep_base,
        expected_input_count=args.expected_input_count,
        require_band_names=not args.assume_canonical_band_order,
        source_units=args.source_units,
        nonfinite_policy=args.nonfinite_policy,
        undersize_policy=args.undersize_policy,
        max_files=args.max_files,
    )
    description = dataset.describe()
    print(json.dumps(description, indent=2))
    if args.preflight_only:
        for index in sorted({0, len(dataset) // 2, len(dataset) - 1}):
            _ = dataset[index]
        return 0
    if not args.output:
        parser.error("--output is required for extraction")

    device = _device(args.device)
    model = load_terramind(
        model_name=args.model,
        experiment=args.experiment,
        device=device,
        checkpoint_path=args.checkpoint,
    )
    frame = extract_terramind_embeddings(
        dataset,
        model,
        model_name=args.model,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        # Derived from the configured model input, not pinned to 196.
        expected_token_count=terramind_token_count(dataset.model_size),
    )
    output = write_embeddings(frame, args.output)
    try:
        terratorch_version = importlib.metadata.version("terratorch")
    except importlib.metadata.PackageNotFoundError:
        terratorch_version = "unknown"
    checkpoint = (
        str(Path(args.checkpoint).resolve())
        if args.checkpoint
        else f"{OFFICIAL_MODEL_ORGANIZATION}/TerraMind-1.0-{args.model.rsplit('_', 1)[-1]}"
    )
    provenance = {
        "schema_version": 1,
        "backbone": frame["backbone"].iloc[0],
        "model_name": args.model,
        "model_depth": MODEL_DEPTHS[args.model],
        "official_repository": OFFICIAL_REPOSITORY,
        "official_terratorch_repository": OFFICIAL_TERRATORCH_REPOSITORY,
        "terratorch_version": terratorch_version,
        "checkpoint": checkpoint,
        "input_root": str(Path(args.npz_dir).resolve()),
        "output": str(output.resolve()),
        "output_rows": int(len(frame)),
        "embedding_dim": int(len(frame["embedding"].iloc[0])),
        "encoder_layer": "final",
        "token_pool": "mean_all_196_spatial_tokens",
        "device": str(device),
        "dataset": description,
    }
    sidecar = output.with_suffix(output.suffix + ".provenance.json")
    sidecar.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"output": str(output), "provenance": str(sidecar)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
