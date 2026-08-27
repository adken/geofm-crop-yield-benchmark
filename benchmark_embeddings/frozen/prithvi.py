#!/usr/bin/env python3
"""Canonical Prithvi-EO-2.0-300M-TL extraction for the county benchmark.

The source cohort is the same collection of per-patch, per-timestep
Sentinel-2 composites used by the other image encoders. Complete seven-step
spatial-patch schedules are audited without changing county patch counts, but
each timestep is encoded independently. Each source image is center-cropped
to the benchmark's 256x256 footprint and then to Prithvi's native 224x224
input; smaller images are rejected and no interpolation or padding is used.

Prithvi's six semantic HLS bands (Blue, Green, Red, Narrow NIR, SWIR1, SWIR2)
map to Sentinel-2 B02, B03, B04, B8A, B11, and B12.  The official V2 digital-
number mean and standard deviation are applied once. Every model sample is one
image with shape [B,6,1,224,224], one (year, day-of-year) pair, and one
(latitude, longitude) pair. The representation spatially mean-pools the 196
final-layer non-CLS tokens and emits one row per patch-timestep. Presto is the
only benchmark encoder that receives the complete seven-step time series.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
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


OFFICIAL_REPOSITORY = "https://github.com/NASA-IMPACT/Prithvi-EO-2.0"
OFFICIAL_MODEL_CARD = (
    "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL"
)
OFFICIAL_CHECKPOINT = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL"

PRITHVI_BACKBONE = "prithvi_eo_v2_300_tl"
PRITHVI_SOURCE_BANDS = ("B02", "B03", "B04", "B8A", "B11", "B12")
PRITHVI_MODEL_BANDS = (
    "BLUE",
    "GREEN",
    "RED",
    "NIR_NARROW",
    "SWIR_1",
    "SWIR_2",
)
PRITHVI_V2_MEAN = np.asarray(
    (1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0), dtype=np.float32
)
PRITHVI_V2_STD = np.asarray(
    (2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0), dtype=np.float32
)

POOL_PER_TIMESTEP_SPATIAL_MEAN = "per_timestep_spatial_mean"
POOLING_CHOICES = (POOL_PER_TIMESTEP_SPATIAL_MEAN,)

DEFAULT_SOURCE_SIZE = (256, 256)
DEFAULT_MODEL_SIZE = (224, 224)
DEFAULT_TIMESTEPS = 7
INFERENCE_TIMESTEPS = 1
PRETRAINING_TIMESTEPS = 4
DEFAULT_EMBEDDING_DIM = 1024
DEFAULT_PATCH_SIZE = (1, 16, 16)

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
_DATE = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})")


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


def _parse_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, np.datetime64):
        return datetime.fromisoformat(str(value)[:10]).date()
    text = str(value).strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        try:
            return datetime.strptime(text[:10], "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError(f"invalid acquisition date {value!r}") from exc


def _acquisition_date(path: Path, meta: dict[str, Any]) -> date:
    metadata_value = _first(meta, ("datetime", "timestamp", "date", "acquisition_date"))
    filename_match = _DATE.search(path.stem)
    metadata_date = _parse_date(metadata_value) if metadata_value is not None else None
    filename_date = _parse_date(filename_match.group("date")) if filename_match else None
    if metadata_date is not None and filename_date is not None and metadata_date != filename_date:
        raise ValueError(
            f"{path.name}: metadata date {metadata_date} disagrees with filename {filename_date}"
        )
    result = metadata_date or filename_date
    if result is None:
        raise ValueError(
            f"{path.name}: Prithvi-300M-TL needs an acquisition date for year/day-of-year"
        )
    return result


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
            "Prithvi must start from the shared ten/12-band source cohort, "
            f"got {array.shape[0]} channels"
        )
    return array


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
    return tuple(normalise_band_name(value) for value in detected)


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


def _latlon(z: Any, meta: dict[str, Any]) -> tuple[float, float] | None:
    root = {
        key: (z[key].item() if getattr(z[key], "shape", None) == () else z[key])
        for key in ("raw_lat", "raw_lon", "latitude", "longitude", "lat", "lon")
        if key in z.files
    }
    latitude = _first(root, ("raw_lat", "latitude", "lat"))
    longitude = _first(root, ("raw_lon", "longitude", "lon"))
    latitude = _first(meta, ("latitude", "latitude_deg", "lat"), latitude)
    longitude = _first(meta, ("longitude", "longitude_deg", "lon", "lng"), longitude)
    lat = safe_float(latitude)
    lon = safe_float(longitude)
    if not np.isfinite(lat) or not np.isfinite(lon):
        return None
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        raise ValueError(f"invalid latitude/longitude: {lat}, {lon}")
    return float(lat), float(lon)


def expected_token_count(
    timesteps: int = INFERENCE_TIMESTEPS,
    model_size: int | Sequence[int] = DEFAULT_MODEL_SIZE,
    patch_size: Sequence[int] = DEFAULT_PATCH_SIZE,
) -> int:
    """Return CLS plus all temporal-spatial tokens for the configured input."""
    height, width = _pair(model_size, label="model_size")
    temporal_patch, patch_height, patch_width = tuple(int(value) for value in patch_size)
    if timesteps % temporal_patch or height % patch_height or width % patch_width:
        raise ValueError("timesteps/model_size must be divisible by the 3-D patch size")
    return 1 + (timesteps // temporal_patch) * (height // patch_height) * (width // patch_width)


def normalize_prithvi(
    pixels: np.ndarray,
    *,
    source_units: str,
    path: Path,
) -> np.ndarray:
    """Apply the official V2 per-band DN statistics exactly once."""
    source_units = str(source_units).strip().lower()
    finite = pixels[np.isfinite(pixels)]
    max_abs = float(np.max(np.abs(finite))) if finite.size else 0.0
    if source_units == "auto":
        source_units = "reflectance" if max_abs <= 2.0 else "dn"
    if source_units == "dn":
        if 0.0 < max_abs <= 2.0:
            raise ValueError(
                f"{path.name}: values look like reflectance, but source_units='dn'; "
                "use --source-units reflectance"
            )
        dn = pixels
    elif source_units == "reflectance":
        if max_abs > 4.0:
            raise ValueError(
                f"{path.name}: values look like digital numbers, but "
                "source_units='reflectance'; use --source-units dn"
            )
        dn = pixels * 10000.0
    else:
        raise ValueError("source_units must be 'auto', 'dn', or 'reflectance'")
    return (dn - PRITHVI_V2_MEAN[:, None, None]) / PRITHVI_V2_STD[:, None, None]


@dataclass(frozen=True)
class PrithviPatchSequence:
    county_id: str
    year: int
    patch_id: str
    paths: tuple[Path, ...]


class PrithviPatchDataset(Dataset):
    """Audit complete schedules and load their frames for independent inference."""

    def __init__(
        self,
        npz_dir: str | Path,
        *,
        source_size: int | Sequence[int] = DEFAULT_SOURCE_SIZE,
        model_size: int | Sequence[int] = DEFAULT_MODEL_SIZE,
        expected_timesteps: int = DEFAULT_TIMESTEPS,
        timestep_base: str | int = "auto",
        expected_input_count: int | None = None,
        require_band_names: bool = True,
        source_units: str = "auto",
        nonfinite_policy: str = "error",
        undersize_policy: str = "error",
        max_sequences: int | None = None,
    ):
        self.root = Path(npz_dir)
        if not self.root.exists():
            raise FileNotFoundError(f"Sentinel-2 NPZ directory does not exist: {self.root}")
        self.source_size = _pair(source_size, label="source_size")
        self.model_size = _pair(model_size, label="model_size")
        if self.model_size[0] > self.source_size[0] or self.model_size[1] > self.source_size[1]:
            raise ValueError("model_size cannot exceed the harmonized source_size")
        self.expected_timesteps = int(expected_timesteps)
        if self.expected_timesteps <= 0:
            raise ValueError("expected_timesteps must be positive")
        self.require_band_names = bool(require_band_names)
        self.source_units = str(source_units).strip().lower()
        if self.source_units not in {"auto", "dn", "reflectance"}:
            raise ValueError("source_units must be 'auto', 'dn', or 'reflectance'")
        self.nonfinite_policy = str(nonfinite_policy).strip().lower()
        if self.nonfinite_policy not in {"error", "zero"}:
            raise ValueError("nonfinite_policy must be 'error' or 'zero'")
        self.undersize_policy = normalise_undersize_policy(undersize_policy)

        paths = sorted(self.root.rglob("*.npz"))
        if not paths:
            raise FileNotFoundError(f"no NPZ files found below {self.root}")
        self.source_cohort_count = len(paths)
        if expected_input_count is not None and len(paths) != int(expected_input_count):
            raise ValueError(
                f"Prithvi Sentinel-2 input cohort has {len(paths):,} files, expected "
                f"{int(expected_input_count):,}"
            )
        parsed = [_source_fields(path) for path in paths]
        base = _resolve_timestep_base(
            [fields[3] for fields in parsed], self.expected_timesteps, timestep_base
        )
        grouped: dict[tuple[str, int, str], dict[int, Path]] = {}
        excluded_paths: list[Path] = []
        raw_keys = []
        for path, (county, year, patch_id, raw_timestep) in zip(paths, parsed):
            timestep = raw_timestep - base
            if not 0 <= timestep < self.expected_timesteps:
                excluded_paths.append(path)
                continue
            key = (county, year, patch_id, timestep)
            raw_keys.append(key)
            grouped.setdefault((county, year, patch_id), {})[timestep] = path
        if len(raw_keys) != len(set(raw_keys)):
            raise ValueError(
                "Sentinel-2 files collapse to duplicate county/year/patch/timestep keys"
            )

        schedule = set(range(self.expected_timesteps))
        complete_keys = sorted(
            key for key, by_time in grouped.items() if set(by_time) == schedule
        )
        if not complete_keys:
            raise ValueError("no spatial patches have the complete benchmark timestep schedule")

        # Keep schedule-completeness and undersize as separate exclusion reasons.
        self._schedule_complete_patches = len(complete_keys)
        complete_keys, undersized_keys, undersized_files = screen_undersized_patches(
            {key: grouped[key] for key in complete_keys},
            target_size=self.source_size,
            policy=self.undersize_policy,
        )
        self._undersized_patches_excluded = len(undersized_keys)
        self._undersized_files_excluded = int(undersized_files)

        selected_keys = complete_keys
        if max_sequences is not None:
            selected_keys = selected_keys[: int(max_sequences)]
        if not selected_keys:
            raise ValueError("max_sequences selected no Prithvi inputs")

        self.indices = [
            PrithviPatchSequence(
                county_id=county,
                year=year,
                patch_id=patch_id,
                paths=tuple(
                    grouped[(county, year, patch_id)][timestep]
                    for timestep in range(self.expected_timesteps)
                ),
            )
            for county, year, patch_id in selected_keys
        ]
        self.timestep_base = base
        self._out_of_schedule_paths = tuple(excluded_paths)
        self._all_groups = grouped
        self._all_complete_keys = complete_keys

    def __len__(self) -> int:
        return len(self.indices)

    @staticmethod
    def _validate_identity(
        path: Path,
        meta: dict[str, Any],
        *,
        county: str,
        year: int,
    ) -> None:
        meta_county = _first(meta, ("county_fips", "county", "fips", "GEOID"))
        meta_year = _first(meta, ("year",))
        if meta_county is not None and normalise_county(meta_county) != county:
            raise ValueError(f"{path.name}: metadata county disagrees with filename")
        if meta_year is not None and int(safe_float(meta_year)) != year:
            raise ValueError(f"{path.name}: metadata year disagrees with filename")

    def _read_frame(
        self,
        path: Path,
        *,
        county: str,
        year: int,
    ) -> tuple[np.ndarray, date, tuple[float, float], float]:
        with np.load(path, allow_pickle=True) as z:
            meta = npz_metadata(z)
            pixels = _pixel_array(z)
            names = _input_bands(
                z,
                meta,
                channels=int(pixels.shape[0]),
                require_names=self.require_band_names,
            )
            acquisition = _acquisition_date(path, meta)
            location = _latlon(z, meta)
        self._validate_identity(path, meta, county=county, year=year)

        pixels = reindex_bands(pixels[None], names, PRITHVI_SOURCE_BANDS)[0]
        pixels = _center_crop(
            pixels,
            target_size=self.source_size,
            path=path,
            stage="benchmark source-size",
        )
        pixels = _center_crop(
            pixels,
            target_size=self.model_size,
            path=path,
            stage="Prithvi model-size",
        )
        valid_fraction = float(np.isfinite(pixels).all(axis=0).mean())
        if valid_fraction < 1.0 and self.nonfinite_policy == "error":
            raise ValueError(
                f"{path.name}: non-finite pixels remain after compositing "
                f"(valid spatial fraction={valid_fraction:.6f})"
            )
        pixels = np.nan_to_num(pixels, nan=0.0, posinf=0.0, neginf=0.0)
        pixels = normalize_prithvi(pixels, source_units=self.source_units, path=path)
        if not np.isfinite(pixels).all():
            raise ValueError(f"{path.name}: Prithvi normalization produced non-finite values")
        if location is None:
            raise ValueError(
                f"{path.name}: Prithvi-300M-TL requires latitude and longitude metadata"
            )
        return pixels.astype(np.float32), acquisition, location, valid_fraction

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.indices[index]
        frames, dates, locations, valid_fractions = [], [], [], []
        for path in item.paths:
            pixels, acquisition, location, valid_fraction = self._read_frame(
                path, county=item.county_id, year=item.year
            )
            frames.append(pixels)
            dates.append(acquisition)
            locations.append(location)
            valid_fractions.append(valid_fraction)

        if dates != sorted(dates) or len(set(dates)) != self.expected_timesteps:
            raise ValueError(
                f"{item.county_id}/{item.year}/{item.patch_id}: acquisition dates must be "
                "distinct and chronological in timestep order"
            )
        location_array = np.asarray(locations, dtype=np.float64)
        if not np.allclose(location_array, location_array[0], atol=1e-5, rtol=0.0):
            raise ValueError(
                f"{item.county_id}/{item.year}/{item.patch_id}: latitude/longitude "
                "changes across timesteps"
            )
        temporal = np.asarray(
            [(value.year, value.timetuple().tm_yday) for value in dates],
            dtype=np.float32,
        )
        # Keep the source schedule together for auditing/collation. The extractor
        # turns this into B*T independent [C,1,H,W] model samples.
        pixels = np.moveaxis(np.stack(frames), 0, 1)
        return {
            "pixels": torch.from_numpy(np.ascontiguousarray(pixels, dtype=np.float32)),
            "temporal_coords": torch.from_numpy(temporal),
            # TerraTorch's LocationEncoder documents [latitude, longitude].
            "location_coords": torch.from_numpy(location_array[0].astype(np.float32)),
            "county_id": item.county_id,
            "year": item.year,
            "patch_id": item.patch_id,
            "dates": [value.isoformat() for value in dates],
            "day_of_year": [int(value.timetuple().tm_yday) for value in dates],
            "source_files": [str(path) for path in item.paths],
            "valid_fraction_min": float(min(valid_fractions)),
        }

    def describe(self) -> dict[str, Any]:
        selected = pd.DataFrame(
            [
                {
                    "county_id": item.county_id,
                    "year": item.year,
                    "patch_id": item.patch_id,
                }
                for item in self.indices
            ]
        )
        patch_counts = selected.groupby(["county_id", "year"]).size()
        incomplete = len(self._all_groups) - self._schedule_complete_patches
        return {
            "sentinel2_input_files": int(self.source_cohort_count),
            "sentinel2_schedule_files": int(
                self.source_cohort_count - len(self._out_of_schedule_paths)
            ),
            "out_of_schedule_files_excluded": int(len(self._out_of_schedule_paths)),
            "out_of_schedule_timestep_policy": "audit_then_exclude",
            "source_spatial_patches": int(len(self._all_groups)),
            "complete_spatial_patches": int(len(self._all_complete_keys)),
            "incomplete_spatial_patches_excluded": int(incomplete),
            "complete_source_sequences": int(len(self.indices)),
            "output_timestep_rows": int(len(self.indices) * self.expected_timesteps),
            "county_years": int(selected[["county_id", "year"]].drop_duplicates().shape[0]),
            "patch_count_min": int(patch_counts.min()),
            "patch_count_median": float(patch_counts.median()),
            "patch_count_max": int(patch_counts.max()),
            "expected_timesteps": int(self.expected_timesteps),
            "model_inference_timesteps": INFERENCE_TIMESTEPS,
            "pretraining_timesteps": PRETRAINING_TIMESTEPS,
            "timestep_base": int(self.timestep_base),
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
                "detect_each_file_then_convert_to_dn"
                if self.source_units == "auto"
                else "declared_uniform"
            ),
            "source_bands": list(PRITHVI_SOURCE_BANDS),
            "model_band_names": list(PRITHVI_MODEL_BANDS),
            "zero_padded_bands": [],
            "normalization": "official_prithvi_v2_dn_mean_std",
            "normalization_mean": PRITHVI_V2_MEAN.tolist(),
            "normalization_std": PRITHVI_V2_STD.tolist(),
            "temporal_coordinates": "per_timestep_year_and_one_based_day_of_year",
            "location_coordinates": "latitude_longitude",
            "nonfinite_policy": self.nonfinite_policy,
        }


def prithvi_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty Prithvi batch")
    tensor_names = ("pixels", "temporal_coords", "location_coords")
    output: dict[str, Any] = {
        name: torch.stack([sample[name] for sample in batch]) for name in tensor_names
    }
    for name in batch[0]:
        if name not in tensor_names:
            output[name] = [sample[name] for sample in batch]
    return output


def pool_prithvi_tokens(
    output: Any,
    *,
    pooling: str = POOL_PER_TIMESTEP_SPATIAL_MEAN,
    expected_tokens: int | None = None,
    expected_embedding_dim: int | None = DEFAULT_EMBEDDING_DIM,
) -> torch.Tensor:
    """Spatially pool one independently encoded timestep per model sample."""
    pooling = str(pooling).strip().lower()
    if pooling not in POOLING_CHOICES:
        raise ValueError(f"pooling must be one of {POOLING_CHOICES}")
    if isinstance(output, (list, tuple)):
        if not output:
            raise ValueError("Prithvi returned no encoder layers")
        tokens = output[-1]
    else:
        tokens = output
    if not isinstance(tokens, torch.Tensor):
        raise TypeError(f"Prithvi final output must be a tensor, got {type(tokens)}")
    if tokens.ndim != 3:
        raise ValueError(
            f"Prithvi final layer must have shape [B,L,D], got {tuple(tokens.shape)}"
        )
    if expected_tokens is not None and tokens.shape[1] != int(expected_tokens):
        raise ValueError(
            f"Prithvi produced {tokens.shape[1]} tokens, expected {int(expected_tokens)}; "
            "check num_frames, image size, and patch size"
        )
    if expected_embedding_dim is not None and tokens.shape[2] != int(expected_embedding_dim):
        raise ValueError(
            f"Prithvi embedding dimension is {tokens.shape[2]}, expected "
            f"{int(expected_embedding_dim)}"
        )
    if not torch.isfinite(tokens).all():
        raise ValueError("Prithvi produced non-finite encoder tokens")
    if tokens.shape[1] <= 1:
        raise ValueError("Prithvi spatial mean needs at least one non-CLS token")
    return tokens[:, 1:].mean(dim=1)


def extract_prithvi_embeddings(
    dataset: PrithviPatchDataset,
    model: Callable[..., Any],
    *,
    device: torch.device | str,
    batch_size: int = 1,
    num_workers: int = 0,
    pooling: str = POOL_PER_TIMESTEP_SPATIAL_MEAN,
    expected_tokens: int | None = None,
    expected_embedding_dim: int = DEFAULT_EMBEDDING_DIM,
) -> pd.DataFrame:
    """Encode every source timestep independently and emit one row per timestep."""
    device = torch.device(device)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=prithvi_collate,
        pin_memory=device.type == "cuda",
    )
    if expected_tokens is None:
        expected_tokens = expected_token_count(
            timesteps=INFERENCE_TIMESTEPS,
            model_size=dataset.model_size,
        )
    tokens_per_timestep = int(expected_tokens) - 1
    pooling = str(pooling).strip().lower()
    backbone = f"{PRITHVI_BACKBONE}_{pooling}"
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            pixels = batch["pixels"]
            temporal = batch["temporal_coords"]
            locations = batch["location_coords"]
            batch_count, channels, timesteps, height, width = pixels.shape
            if timesteps != dataset.expected_timesteps:
                raise ValueError(
                    f"Prithvi batch has {timesteps} source timesteps, expected "
                    f"{dataset.expected_timesteps}"
                )
            # Each element of this model batch contains exactly one frame. Batching
            # B*T independent samples does not expose temporal neighbours to the
            # encoder, while avoiding seven separate forward calls.
            frame_pixels = pixels.permute(0, 2, 1, 3, 4).reshape(
                batch_count * timesteps, channels, INFERENCE_TIMESTEPS, height, width
            )
            frame_temporal = temporal.reshape(
                batch_count * timesteps, INFERENCE_TIMESTEPS, 2
            )
            frame_locations = locations.repeat_interleave(timesteps, dim=0)
            output = model(
                frame_pixels.to(device),
                frame_temporal.to(device),
                frame_locations.to(device),
            )
            vectors = pool_prithvi_tokens(
                output,
                pooling=pooling,
                expected_tokens=expected_tokens,
                expected_embedding_dim=expected_embedding_dim,
            ).reshape(batch_count, timesteps, expected_embedding_dim)
            vectors = vectors.detach().cpu().to(torch.float32).numpy()
            for row_index, item_vectors in enumerate(vectors):
                for timestep, vector in enumerate(item_vectors):
                    rows.append({
                        "county_id": batch["county_id"][row_index],
                        "year": batch["year"][row_index],
                        "patch_id": batch["patch_id"][row_index],
                        "timestep": timestep,
                        "backbone": backbone,
                        "embedding": vector.tolist(),
                        "experiment_family": "main_benchmark",
                        "fusion_stage": "none",
                        "representation_scope": "timestep",
                        "temporal_ingestion": "single_timestep_independent",
                        "input_modalities": "Sentinel-2",
                        "sequence_timesteps": dataset.expected_timesteps,
                        "inference_timesteps": INFERENCE_TIMESTEPS,
                        "pretraining_timesteps": PRETRAINING_TIMESTEPS,
                        "date": batch["dates"][row_index][timestep],
                        "day_of_year": batch["day_of_year"][row_index][timestep],
                        "location_coords_latlon": (
                            batch["location_coords"][row_index].tolist()
                        ),
                        "source_file": batch["source_files"][row_index][timestep],
                        "valid_fraction_min": batch["valid_fraction_min"][row_index],
                        "encoder_layer": "final",
                        "token_pool": (
                            f"mean_{tokens_per_timestep}_non_cls_spatial_tokens_per_timestep"
                        ),
                    })
    expected_rows = len(dataset) * dataset.expected_timesteps
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Prithvi extraction emitted {len(rows):,} rows, expected {expected_rows:,} "
            f"for {len(dataset):,} sequences"
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


def load_prithvi(
    *,
    device: torch.device,
    num_frames: int = INFERENCE_TIMESTEPS,
    checkpoint_path: str | Path | None = None,
) -> torch.nn.Module:
    """Build the official 300M-TL encoder for one-frame independent inference."""
    if int(num_frames) != INFERENCE_TIMESTEPS:
        raise ValueError(
            "canonical Prithvi extraction requires num_frames=1; complete temporal "
            "sequences are ingested only by Presto"
        )
    try:
        from terratorch.registry import BACKBONE_REGISTRY
    except ImportError as exc:
        raise ImportError(
            "Prithvi extraction requires an installed official TerraTorch package"
        ) from exc

    kwargs: dict[str, Any] = {
        "pretrained": True,
        "bands": list(PRITHVI_MODEL_BANDS),
        "num_frames": int(num_frames),
        # Return only the final layer from the registry wrapper.
        "out_indices": [23],
    }
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path).resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"Prithvi checkpoint does not exist: {checkpoint}")
        kwargs["ckpt_path"] = str(checkpoint)
    model = BACKBONE_REGISTRY.build(PRITHVI_BACKBONE, **kwargs).to(device).eval()

    observed_patch = tuple(int(value) for value in model.patch_embed.patch_size)
    checks = {
        "embedding dimension": (int(model.embed_dim), DEFAULT_EMBEDDING_DIM),
        "input channels": (int(model.in_chans), len(PRITHVI_MODEL_BANDS)),
        "number of frames": (int(model.num_frames), int(num_frames)),
        "encoder depth": (len(model.blocks), 24),
    }
    for label, (observed, expected) in checks.items():
        if observed != expected:
            raise ValueError(f"Prithvi {label} is {observed}, expected {expected}")
    if observed_patch != DEFAULT_PATCH_SIZE:
        raise ValueError(
            f"Prithvi patch size is {observed_patch}, expected {DEFAULT_PATCH_SIZE}"
        )
    if not model.temporal_encoding or not model.location_encoding:
        raise ValueError("loaded Prithvi model is not the temporal/location 300M-TL variant")
    if getattr(model, "out_indices", None) != [23]:
        raise ValueError("Prithvi registry did not retain final-layer-only out_indices=[23]")
    return model


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-dir", required=True)
    parser.add_argument("--checkpoint", help="Optional local official 300M-TL checkpoint")
    parser.add_argument("--output", help="Canonical output Parquet")
    parser.add_argument(
        "--pooling", choices=POOLING_CHOICES, default=POOL_PER_TIMESTEP_SPATIAL_MEAN
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--expected-input-count", type=int)
    parser.add_argument("--expected-timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--timestep-base", choices=("auto", "0", "1"), default="auto")
    parser.add_argument(
        "--source-units",
        choices=("auto", "dn", "reflectance"),
        default="auto",
    )
    parser.add_argument("--max-sequences", type=int)
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

    dataset = PrithviPatchDataset(
        args.npz_dir,
        source_size=(args.source_size, args.source_size),
        model_size=DEFAULT_MODEL_SIZE,
        expected_timesteps=args.expected_timesteps,
        timestep_base=args.timestep_base,
        expected_input_count=args.expected_input_count,
        require_band_names=not args.assume_canonical_band_order,
        source_units=args.source_units,
        nonfinite_policy=args.nonfinite_policy,
        undersize_policy=args.undersize_policy,
        max_sequences=args.max_sequences,
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
    model = load_prithvi(
        device=device,
        num_frames=INFERENCE_TIMESTEPS,
        checkpoint_path=args.checkpoint,
    )
    frame = extract_prithvi_embeddings(
        dataset,
        model,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pooling=args.pooling,
    )
    output = write_embeddings(frame, args.output)
    try:
        terratorch_version = importlib.metadata.version("terratorch")
    except importlib.metadata.PackageNotFoundError:
        terratorch_version = "unknown"
    checkpoint = (
        str(Path(args.checkpoint).resolve()) if args.checkpoint else OFFICIAL_CHECKPOINT
    )
    provenance = {
        "schema_version": 1,
        "backbone": frame["backbone"].iloc[0],
        "model_name": PRITHVI_BACKBONE,
        "model_parameters": "300M",
        "model_depth": 24,
        "official_repository": OFFICIAL_REPOSITORY,
        "official_model_card": OFFICIAL_MODEL_CARD,
        "terratorch_version": terratorch_version,
        "checkpoint": checkpoint,
        "input_root": str(Path(args.npz_dir).resolve()),
        "output": str(output.resolve()),
        "output_rows": int(len(frame)),
        "embedding_dim": DEFAULT_EMBEDDING_DIM,
        "encoder_layer": "final",
        "token_pool": "mean_196_non_cls_spatial_tokens_per_timestep",
        "representation_scope": "timestep",
        "temporal_ingestion": "single_timestep_independent",
        "source_sequence_timesteps": int(args.expected_timesteps),
        "inference_frames": INFERENCE_TIMESTEPS,
        "pretraining_frames": PRETRAINING_TIMESTEPS,
        "coordinate_order": "latitude_longitude",
        "device": str(device),
        "dataset": description,
    }
    sidecar = output.with_suffix(output.suffix + ".provenance.json")
    sidecar.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"output": str(output), "provenance": str(sidecar)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
