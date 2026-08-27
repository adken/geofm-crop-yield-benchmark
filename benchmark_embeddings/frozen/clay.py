#!/usr/bin/env python3
"""Clay v1.5 extraction for the canonical county-yield benchmark.

The implementation follows the official Clay v1.5 input contract while
preserving the benchmark's spatial patch identity.  In particular, it uses
the sensor statistics and wavelengths from Clay's ``metadata.yaml``, applies
the official cyclic time/location encodings, center-crops oversized inputs to
256x256, and emits exactly one 1024-dimensional vector per source
patch-timestep.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from ..data.io import (
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


CLAY_PLATFORM = "sentinel-2-l2a"
CLAY_BACKBONE = "clay_v1_5"
OFFICIAL_REPOSITORY = "https://github.com/Clay-foundation/model"
DEFAULT_SPATIAL_SIZE = (256, 256)
DEFAULT_EMBEDDING_DIM = 1024
CLAY_PATCH_SIZE = 8  # Clay v1.5 encoder patch size
DEFAULT_TOKEN_COUNT = 1025  # CLS + (256 / patch_size=8)^2 spatial tokens


def clay_token_count(
    spatial_size: tuple[int, int], patch_size: int = CLAY_PATCH_SIZE
) -> int:
    """CLS + spatial tokens for a given input footprint.

    Clay v1.5 is not fixed at 256x256. Its encoder uses a ``DynamicEmbedding``
    with ``img_size=None`` and ``patch_size=8``, so any multiple of 8 is valid;
    verified by forward pass at 256 (1025 tokens), 224 (785), and 192 (577).
    The token count must therefore follow the configured footprint rather than
    being pinned to 256.
    """
    height, width = int(spatial_size[0]), int(spatial_size[1])
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"Clay input {height}x{width} is not a multiple of patch_size={patch_size}"
        )
    return 1 + (height // patch_size) * (width // patch_size)

_CLAY_TO_S2 = {
    "blue": "B02",
    "green": "B03",
    "red": "B04",
    "rededge1": "B05",
    "rededge2": "B06",
    "rededge3": "B07",
    "nir": "B08",
    "nir08": "B8A",
    "swir16": "B11",
    "swir22": "B12",
}
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


def _first(mapping: dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None and value != "":
            return value
    return default


def _pair(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, (int, np.integer)):
        result = (int(value), int(value))
    else:
        result = tuple(int(part) for part in value)
        if len(result) != 2:
            raise ValueError("target_size must be an integer or [height, width]")
    if min(result) <= 0:
        raise ValueError("target_size must be positive")
    return result


def _to_reflectance_units(values: np.ndarray) -> np.ndarray:
    """Convert Clay's Sentinel-2 DN statistics to [0,1] reflectance units."""
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size and float(np.max(np.abs(finite))) > 2.0:
        values = values / 10000.0
    return values


def _reflectance_array(z: Any) -> np.ndarray:
    """Read [C,H,W] reflectance without clipping valid values above one.

    Clay's official transform standardizes the source values directly; unlike
    the historical benchmark loader, it does not clamp reflectance to [0,1].
    Raw Sentinel-2 digital numbers are converted to reflectance units only so
    they can be paired with equivalently converted Clay statistics.
    """
    key = next(
        (
            name
            for name in ("pixels", "patch", "cube", "data", "s2", "image", "array")
            if name in z.files
        ),
        None,
    )
    if key is None:
        raise KeyError(f"no pixel array found; keys={z.files}")
    array = np.asarray(z[key], dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"expected one Clay patch [C,H,W] or [H,W,C], got {array.shape}")
    if array.shape[0] not in (10, 12) and array.shape[-1] in (10, 12):
        array = np.moveaxis(array, -1, 0)
    finite = array[np.isfinite(array)]
    if finite.size and float(np.max(np.abs(finite))) > 2.0:
        array = array / 10000.0
    return array


@dataclass(frozen=True)
class ClaySensorMetadata:
    """The official Clay metadata aligned to the benchmark's ten S2 bands."""

    bands: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    wavelengths_um: tuple[float, ...]
    gsd: float

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ClaySensorMetadata":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Clay metadata file does not exist: {path}")
        document = yaml.safe_load(path.read_text())
        block = document.get(CLAY_PLATFORM) if isinstance(document, dict) else None
        if not isinstance(block, dict):
            raise ValueError(f"{path} has no {CLAY_PLATFORM!r} metadata block")
        clay_order = tuple(str(name) for name in block.get("band_order", ()))
        mapped_order = tuple(_CLAY_TO_S2.get(name, normalise_band_name(name)) for name in clay_order)
        expected = tuple(S2_10_BANDS)
        if mapped_order != expected:
            raise ValueError(
                "Clay Sentinel-2 band order differs from the benchmark contract: "
                f"mapped={mapped_order}, expected={expected}"
            )
        bands = block.get("bands", {})
        try:
            mean = np.asarray([bands["mean"][name] for name in clay_order], dtype=np.float32)
            std = np.asarray([bands["std"][name] for name in clay_order], dtype=np.float32)
            waves = np.asarray(
                [bands["wavelength"][name] for name in clay_order], dtype=np.float32
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{path} has incomplete Clay band statistics") from exc
        mean = _to_reflectance_units(mean)
        std = _to_reflectance_units(std)
        # The v1.5 direct ``module.model.encoder`` path consumes the values in
        # the checkpoint metadata verbatim (micrometres). Newer convenience
        # APIs may convert to nm internally, but this extractor targets the
        # direct v1.5 encoder bundled with the official checkpoint code.
        waves = np.where(waves > 100.0, waves / 1000.0, waves)
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
            raise ValueError("Clay mean/std values must be finite and std must be positive")
        if not np.isfinite(waves).all():
            raise ValueError("Clay wavelengths must be finite")
        return cls(
            bands=expected,
            mean=tuple(float(value) for value in mean),
            std=tuple(float(value) for value in std),
            wavelengths_um=tuple(float(value) for value in waves),
            gsd=float(block.get("gsd", 10.0)),
        )


@dataclass(frozen=True)
class ClayPatchIndex:
    path: Path
    source_id: str
    patch_id: str
    county_id: str
    year: int
    raw_timestep: int
    timestep: int


def encode_clay_time(value: str | date | datetime) -> tuple[torch.Tensor, int, str]:
    """Apply Clay's official week/hour sine-cosine encoding."""
    if isinstance(value, datetime):
        timestamp = value
    elif isinstance(value, date):
        timestamp = datetime.combine(value, datetime.min.time())
    else:
        text = str(value).strip().replace("Z", "+00:00")
        try:
            timestamp = datetime.fromisoformat(text)
        except ValueError:
            try:
                timestamp = datetime.strptime(text[:10], "%Y-%m-%d")
            except ValueError as exc:
                raise ValueError(f"invalid Clay acquisition date/time: {value!r}") from exc
    week_angle = timestamp.isocalendar().week * 2.0 * math.pi / 52.0
    hour = timestamp.hour + timestamp.minute / 60.0 + timestamp.second / 3600.0
    hour_angle = hour * 2.0 * math.pi / 24.0
    encoded = torch.tensor(
        [
            math.sin(week_angle),
            math.cos(week_angle),
            math.sin(hour_angle),
            math.cos(hour_angle),
        ],
        dtype=torch.float32,
    )
    return encoded, int(timestamp.timetuple().tm_yday), timestamp.isoformat()


def encode_clay_location(latitude: Any, longitude: Any) -> torch.Tensor:
    """Apply Clay's official latitude/longitude sine-cosine encoding."""
    lat = safe_float(latitude)
    lon = safe_float(longitude)
    if not np.isfinite(lat) or not np.isfinite(lon):
        raise ValueError(f"non-finite latitude/longitude: {latitude!r}, {longitude!r}")
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        raise ValueError(f"invalid latitude/longitude: {lat}, {lon}")
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    return torch.tensor(
        [math.sin(lat_rad), math.cos(lat_rad), math.sin(lon_rad), math.cos(lon_rad)],
        dtype=torch.float32,
    )


def _source_fields(path: Path) -> tuple[str, int, str, int]:
    county_year = _COUNTY_YEAR.search(path.stem)
    timestep = _TIMESTEP.search(path.stem + "_")
    spatial = _SPATIAL_ID.search(path.stem)
    if county_year is None or timestep is None:
        raise ValueError(
            f"cannot infer county/year/timestep from {path.name}; retain those fields "
            "in the source filename or NPZ metadata"
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


def _resolve_timestep_base(values: Sequence[int], expected_timesteps: int, requested: str | int) -> int:
    if str(requested) in {"0", "1"}:
        return int(requested)
    unique = set(int(value) for value in values)
    if 0 in unique:
        return 0
    if 1 in unique and expected_timesteps in unique:
        return 1
    raise ValueError(
        "timestep base is ambiguous; pass timestep_base=0 or timestep_base=1 "
        f"(observed values: {sorted(unique)})"
    )


class ClayPatchDataset(Dataset):
    """Read one Sentinel-2 patch-timestep per NPZ for Clay v1.5 extraction."""

    def __init__(
        self,
        npz_dir: str | Path,
        *,
        metadata_path: str | Path,
        target_size: int | Sequence[int] = DEFAULT_SPATIAL_SIZE,
        expected_timesteps: int = 7,
        timestep_base: str | int = "auto",
        expected_input_count: int | None = None,
        require_band_names: bool = True,
        allow_missing_spatiotemporal_metadata: bool = False,
        nonfinite_policy: str = "error",
        undersize_policy: str = "error",
        max_files: int | None = None,
    ):
        self.root = Path(npz_dir)
        if not self.root.exists():
            raise FileNotFoundError(f"Sentinel-2 NPZ directory does not exist: {self.root}")
        self.sensor = ClaySensorMetadata.from_yaml(metadata_path)
        self.target_size = _pair(target_size)
        self.expected_timesteps = int(expected_timesteps)
        self.require_band_names = bool(require_band_names)
        self.allow_missing_spatiotemporal_metadata = bool(
            allow_missing_spatiotemporal_metadata
        )
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
                f"Clay input cohort has {len(all_paths):,} files, expected "
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
                ClayPatchIndex(
                    path=path,
                    source_id=path.stem,
                    patch_id=patch_id,
                    county_id=county,
                    year=year,
                    raw_timestep=raw_timestep,
                    timestep=timestep,
                )
            )
        key = [
            (item.county_id, item.year, item.patch_id, item.timestep)
            for item in indices
        ]
        if len(key) != len(set(key)):
            raise ValueError(
                "source files collapse to duplicate county/year/patch/timestep keys; "
                "spatial patch identity has not been preserved"
            )
        # Undersize is screened per spatial patch, not per file: dropping a single
        # timestep would leave a ragged sequence that the county aggregation and
        # complete-patch accounting both assume cannot happen.
        grouped_paths: dict[tuple[str, int, str], list[Path]] = {}
        for item in indices:
            grouped_paths.setdefault(
                (item.county_id, item.year, item.patch_id), []
            ).append(item.path)
        _, undersized_keys, undersized_files = screen_undersized_patches(
            grouped_paths, target_size=self.target_size, policy=self.undersize_policy
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
            raise ValueError("max_files selected no Clay inputs")
        self._out_of_schedule_paths = tuple(excluded_paths)
        self.timestep_base = base

    def __len__(self) -> int:
        return len(self.indices)

    def _center_crop(self, image: np.ndarray, *, path: Path) -> np.ndarray:
        height, width = int(image.shape[-2]), int(image.shape[-1])
        target_height, target_width = self.target_size
        if height < target_height or width < target_width:
            raise ValueError(
                f"{path}: source patch is {height}x{width}, below Clay benchmark "
                f"expectation {target_height}x{target_width}; padding is disabled"
            )
        top = (height - target_height) // 2
        left = (width - target_width) // 2
        return image[..., top : top + target_height, left : left + target_width]

    def _input_bands(self, z: Any, meta: dict[str, Any], channels: int) -> tuple[str, ...]:
        detected = band_names(z)
        if detected is None and "band_names" in meta:
            detected = tuple(normalise_band_name(value) for value in meta["band_names"])
        if detected is None:
            if self.require_band_names:
                raise ValueError(
                    "NPZ has no band_names; pass --assume-canonical-band-order only "
                    "after verifying B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12"
                )
            if channels != len(S2_10_BANDS):
                raise ValueError(f"cannot assume ten-band order for {channels} channels")
            detected = tuple(S2_10_BANDS)
        return tuple(normalise_band_name(value) for value in detected)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.indices[index]
        with np.load(item.path, allow_pickle=True) as z:
            meta = npz_metadata(z)
            pixels = _reflectance_array(z)
            input_bands = self._input_bands(z, meta, int(pixels.shape[0]))
        pixels = reindex_bands(
            pixels[None, ...], input_bands, self.sensor.bands
        )[0]
        pixels = self._center_crop(pixels, path=item.path)
        valid_fraction = float(np.isfinite(pixels).all(axis=0).mean())
        if valid_fraction < 1.0 and self.nonfinite_policy == "error":
            raise ValueError(
                f"{item.path.name}: non-finite pixels remain after compositing "
                f"(valid spatial fraction={valid_fraction:.6f})"
            )
        pixels = np.nan_to_num(pixels, nan=0.0, posinf=1.0, neginf=0.0)
        mean = np.asarray(self.sensor.mean, dtype=np.float32)[:, None, None]
        std = np.asarray(self.sensor.std, dtype=np.float32)[:, None, None]
        pixels = (pixels - mean) / std

        meta_county = normalise_county(
            _first(meta, ("county_fips", "county", "fips", "GEOID"), item.county_id)
        )
        meta_year = int(safe_float(_first(meta, ("year",), item.year), item.year))
        if meta_county != item.county_id or meta_year != item.year:
            raise ValueError(
                f"{item.path.name}: filename county/year {item.county_id}/{item.year} "
                f"disagrees with metadata {meta_county}/{meta_year}"
            )

        timestamp_value = _first(meta, ("datetime", "timestamp", "date", "acquisition_date"))
        latitude = _first(meta, ("latitude", "latitude_deg", "lat"))
        longitude = _first(meta, ("longitude", "longitude_deg", "lon", "lng"))
        if timestamp_value is None or latitude is None or longitude is None:
            if not self.allow_missing_spatiotemporal_metadata:
                raise ValueError(
                    f"{item.path.name}: Clay needs date/time and latitude/longitude metadata"
                )
            time = torch.zeros(4, dtype=torch.float32)
            latlon = torch.zeros(4, dtype=torch.float32)
            day_of_year = 0
            timestamp = ""
            latitude_value = float("nan")
            longitude_value = float("nan")
        else:
            time, day_of_year, timestamp = encode_clay_time(timestamp_value)
            latlon = encode_clay_location(latitude, longitude)
            latitude_value = float(latitude)
            longitude_value = float(longitude)

        return {
            "pixels": torch.from_numpy(np.ascontiguousarray(pixels, dtype=np.float32)),
            "time": time,
            "latlon": latlon,
            "waves": torch.tensor(self.sensor.wavelengths_um, dtype=torch.float32),
            "gsd": torch.tensor(self.sensor.gsd, dtype=torch.float32),
            "platform": CLAY_PLATFORM,
            "county_id": item.county_id,
            "year": item.year,
            "patch_id": item.patch_id,
            "timestep": item.timestep,
            "source_id": item.source_id,
            "source_file": str(item.path),
            "timestamp": timestamp,
            "day_of_year": day_of_year,
            "latitude_deg": latitude_value,
            "longitude_deg": longitude_value,
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
            "target_size": list(self.target_size),
            "oversize_policy": "center_crop",
            "undersize_policy": self.undersize_policy,
            "undersized_spatial_patches_excluded": int(
                self._undersized_patches_excluded
            ),
            "undersized_files_excluded": int(self._undersized_files_excluded),
            "nonfinite_policy": self.nonfinite_policy,
            "bands": list(self.sensor.bands),
            "normalization": "official_clay_sensor_mean_std",
            "wavelength_units": "micrometres",
        }


def clay_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("cannot collate an empty Clay batch")
    tensor_fields = ("pixels", "time", "latlon")
    output: dict[str, Any] = {
        name: torch.stack([sample[name] for sample in batch]) for name in tensor_fields
    }
    output["waves"] = batch[0]["waves"]
    output["gsd"] = batch[0]["gsd"]
    for name in batch[0]:
        if name not in {*tensor_fields, "waves", "gsd"}:
            output[name] = [sample[name] for sample in batch]
    return output


def pool_clay_tokens(
    tokens: torch.Tensor,
    pooling: str,
    *,
    expected_token_count: int | None = DEFAULT_TOKEN_COUNT,
    expected_embedding_dim: int | None = DEFAULT_EMBEDDING_DIM,
) -> torch.Tensor:
    """Turn Clay's ``[B, CLS+spatial, D]`` output into one explicit vector."""
    if tokens.ndim != 3:
        raise ValueError(f"Clay encoder must return [B,L,D] tokens, got {tuple(tokens.shape)}")
    if expected_token_count is not None and tokens.shape[1] != int(expected_token_count):
        raise ValueError(
            f"Clay produced {tokens.shape[1]} tokens, expected {expected_token_count}; "
            f"the count is 1+(size/{CLAY_PATCH_SIZE})^2, so check --spatial-size "
            f"and that the checkpoint uses patch_size={CLAY_PATCH_SIZE}"
        )
    if expected_embedding_dim is not None and tokens.shape[2] != int(expected_embedding_dim):
        raise ValueError(
            f"Clay embedding dimension is {tokens.shape[2]}, expected {expected_embedding_dim}"
        )
    pooling = str(pooling).lower()
    if pooling == "cls":
        return tokens[:, 0, :]
    if pooling == "spatial_mean":
        if tokens.shape[1] <= 1:
            raise ValueError("spatial_mean requires at least one non-CLS token")
        return tokens[:, 1:, :].mean(dim=1)
    raise ValueError("pooling must be 'cls' or 'spatial_mean'")


def extract_clay_embeddings(
    dataset: ClayPatchDataset,
    encoder: Callable[[dict[str, torch.Tensor]], Any],
    *,
    device: torch.device | str,
    pooling: str = "cls",
    batch_size: int = 8,
    num_workers: int = 0,
    expected_token_count: int | None = DEFAULT_TOKEN_COUNT,
    expected_embedding_dim: int | None = DEFAULT_EMBEDDING_DIM,
) -> pd.DataFrame:
    """Extract canonical rows without dropping or collapsing source patches."""
    device = torch.device(device)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=clay_collate,
        pin_memory=device.type == "cuda",
    )
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            model_input = {
                name: batch[name].to(device)
                for name in ("pixels", "time", "latlon", "waves", "gsd")
            }
            encoded = encoder(model_input)
            tokens = encoded[0] if isinstance(encoded, tuple) else encoded
            vectors = pool_clay_tokens(
                tokens,
                pooling,
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
                        "backbone": f"{CLAY_BACKBONE}_{pooling}",
                        "embedding": vector.tolist(),
                        "source_id": batch["source_id"][row_index],
                        "source_file": batch["source_file"][row_index],
                        "timestamp": batch["timestamp"][row_index],
                        "day_of_year": batch["day_of_year"][row_index],
                        "latitude_deg": batch["latitude_deg"][row_index],
                        "longitude_deg": batch["longitude_deg"][row_index],
                        "valid_fraction": batch["valid_fraction"][row_index],
                        "token_pool": pooling,
                        # Contract fields the LOYO and temporal-ablation
                        # validators require. Clay emits one row per patch and
                        # per composite, so the scope is a single timestep.
                        "representation_scope": "timestep",
                        "experiment_family": "main_benchmark",
                        "fusion_stage": "none",
                        "input_modalities": "Sentinel-2",
                        "temporal_ingestion": "single_timestep_independent",
                    }
                )
    if len(rows) != len(dataset):
        raise RuntimeError(
            f"Clay extraction emitted {len(rows):,} rows for {len(dataset):,} source files"
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


def load_clay_encoder(
    *,
    clay_repo: str | Path,
    checkpoint_path: str | Path,
    metadata_path: str | Path,
    device: torch.device,
) -> tuple[torch.nn.Module, Callable[[dict[str, torch.Tensor]], Any]]:
    """Load either the current ``claymodel`` package or the v1.5 ``src`` tree."""
    clay_repo = Path(clay_repo).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Clay checkpoint does not exist: {checkpoint_path}")
    if not clay_repo.exists():
        raise FileNotFoundError(f"Clay repository does not exist: {clay_repo}")
    sys.path.insert(0, str(clay_repo))
    errors = []
    module_class = None
    for import_name in ("claymodel.module", "src.module"):
        try:
            module_class = importlib.import_module(import_name).ClayMAEModule
            break
        except (ImportError, AttributeError) as exc:
            errors.append(f"{import_name}: {exc}")
    if module_class is None:
        raise ImportError("cannot import ClayMAEModule; " + " | ".join(errors))
    module = module_class.load_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        model_size="large",
        metadata_path=str(Path(metadata_path).resolve()),
        dolls=[16, 32, 64, 128, 256, 768, 1024],
        doll_weights=[1, 1, 1, 1, 1, 1, 1],
        mask_ratio=0.0,
        shuffle=False,
        map_location="cpu",
    )
    module = module.to(device).eval()
    if not hasattr(module, "model") or not hasattr(module.model, "encoder"):
        raise TypeError("loaded Clay checkpoint has no module.model.encoder")
    return module, module.model.encoder


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-dir", required=True)
    parser.add_argument("--metadata", required=True, help="Official Clay configs/metadata.yaml")
    parser.add_argument("--checkpoint", help="Official Clay v1.5 checkpoint")
    parser.add_argument("--clay-repo", help="Official Clay checkout or installed source root")
    parser.add_argument("--output", help="Canonical output Parquet")
    parser.add_argument("--pooling", choices=("cls", "spatial_mean"), default="cls")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--expected-input-count", type=int)
    parser.add_argument("--expected-timesteps", type=int, default=7)
    parser.add_argument("--timestep-base", choices=("auto", "0", "1"), default="auto")
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--assume-canonical-band-order", action="store_true")
    parser.add_argument("--allow-missing-spatiotemporal-metadata", action="store_true")
    parser.add_argument("--nonfinite-policy", choices=("error", "zero"), default="error")
    parser.add_argument(
        "--spatial-size",
        type=int,
        default=DEFAULT_SPATIAL_SIZE[0],
        help=(
            "Harmonized center-crop footprint in pixels; must be a multiple of 8. "
            "Default 256 preserves the historical contract. 224 matches Prithvi's "
            "and TerraMind's native input, so all encoders share one footprint "
            "and fewer edge patches are rejected."
        ),
    )
    parser.add_argument(
        "--undersize-policy",
        choices=UNDERSIZE_POLICIES,
        default="error",
        help=(
            "'error' (default) rejects patches below the target footprint. "
            "'skip' drops those spatial patches and records the counts; it is a "
            "recorded deviation that changes per-county patch counts."
        ),
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)

    dataset = ClayPatchDataset(
        args.npz_dir,
        metadata_path=args.metadata,
        target_size=(args.spatial_size, args.spatial_size),
        expected_timesteps=args.expected_timesteps,
        timestep_base=args.timestep_base,
        expected_input_count=args.expected_input_count,
        require_band_names=not args.assume_canonical_band_order,
        allow_missing_spatiotemporal_metadata=args.allow_missing_spatiotemporal_metadata,
        nonfinite_policy=args.nonfinite_policy,
        undersize_policy=args.undersize_policy,
        max_files=args.max_files,
    )
    description = dataset.describe()
    print(json.dumps(description, indent=2))
    if args.preflight_only:
        # Exercise metadata, bands, normalization, and spatial shape on a few files.
        for index in sorted({0, len(dataset) // 2, len(dataset) - 1}):
            _ = dataset[index]
        return 0
    if not args.checkpoint or not args.clay_repo or not args.output:
        parser.error("--checkpoint, --clay-repo, and --output are required for extraction")

    device = _device(args.device)
    module, encoder = load_clay_encoder(
        clay_repo=args.clay_repo,
        checkpoint_path=args.checkpoint,
        metadata_path=args.metadata,
        device=device,
    )
    frame = extract_clay_embeddings(
        dataset,
        encoder,
        device=device,
        pooling=args.pooling,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        # Derived, not the 256-pinned constant: Clay's token count is
        # 1 + (size/8)^2, so it must track --spatial-size.
        expected_token_count=clay_token_count(dataset.target_size),
    )
    output = write_embeddings(frame, args.output)
    provenance = {
        "schema_version": 1,
        "backbone": frame["backbone"].iloc[0],
        "official_repository": OFFICIAL_REPOSITORY,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "metadata": str(Path(args.metadata).resolve()),
        "input_root": str(Path(args.npz_dir).resolve()),
        "output": str(output.resolve()),
        "output_rows": int(len(frame)),
        "embedding_dim": int(len(frame["embedding"].iloc[0])),
        "encoder_layer": "final_encoder_tokens",
        "token_pool": args.pooling,
        "device": str(device),
        "dataset": description,
    }
    sidecar = output.with_suffix(output.suffix + ".provenance.json")
    sidecar.write_text(json.dumps(provenance, indent=2) + "\n")
    # Keep the module alive until writes complete; this also makes the intended
    # ownership of the bound encoder explicit for static analysis.
    _ = module
    print(f"Wrote {len(frame):,} canonical Clay rows to {output}")
    print(f"Wrote provenance to {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
