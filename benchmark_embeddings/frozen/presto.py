#!/usr/bin/env python3
"""Canonical Presto extraction for the county-yield benchmark.

Presto is a temporal pixel model rather than a spatial image encoder.  Each
Sentinel-2 patch is center-cropped to the benchmark's 256x256 source footprint
and encoded as complete seven-composite sequences.  Two spatial reductions are
available.  ``--spatial-mode mean`` reduces each composite to one band vector,
which is what the published run did; ``--spatial-mode sample`` draws real pixel
time series, encodes each, and averages the embeddings, which respects that
Presto was pretrained on single pixels and never on a spatial average.  Raw
Sentinel-2 values and optional
ERA5-Land variables are passed through Presto's official
``construct_single_presto_input`` helper, which owns normalization, band
masking, and NDVI calculation.  The paper's Daymet experiment is county-level
late fusion and is handled by the shared probe, not by this extractor.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import subprocess
import sys
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


OFFICIAL_REPOSITORY = "https://github.com/nasaharvest/presto"
PRESTO_BACKBONE_S2 = "presto_s2"
PRESTO_BACKBONE_S2_ERA5 = "presto_s2_era5"
PRESTO_S2_BANDS = ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12")
PRESTO_ERA5_BANDS = ("temperature_2m", "total_precipitation")
DEFAULT_ERA5_SOURCE_BANDS = PRESTO_ERA5_BANDS
DEFAULT_SPATIAL_SIZE = (256, 256)
DEFAULT_EMBEDDING_DIM = 128
DEFAULT_MODEL_BANDS = 17
DEFAULT_PIXEL_SAMPLES = 64
SPATIAL_MODES = ("mean", "sample")
NONFINITE_POLICIES = ("error", "zero", "mask")

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

_ERA5_BAND_ALIASES = {
    "temperature_2m": "temperature_2m",
    "2m_temperature": "temperature_2m",
    "t2m": "temperature_2m",
    "temp_mean_k": "temperature_2m",
    "era5_temp_mean_k": "temperature_2m",
    # The Earth Engine export writes 'tmean_K' / 'prcp_m' into the NPZ
    # metadata. Aliased rather than renamed at the source so existing archives
    # stay readable; the values themselves are already Kelvin and metres, which
    # the plausibility bounds in _read_era5 confirm.
    "tmean_k": "temperature_2m",
    "era5_tmean_k": "temperature_2m",
    "total_precipitation": "total_precipitation",
    "tp": "total_precipitation",
    "precip_sum_m": "total_precipitation",
    "era5_precip_sum_m": "total_precipitation",
    "prcp_m": "total_precipitation",
    "era5_prcp_m": "total_precipitation",
}


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


def _first(mapping: dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None and value != "":
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
        raise ValueError(f"{path.name}: Presto needs an acquisition date for zero-based months")
    return result


def _pixel_array(z: Any, *, expected_channels: tuple[int, ...], label: str) -> np.ndarray:
    key = next(
        (
            name
            for name in ("pixels", "patch", "cube", "data", "s2", "image", "array")
            if name in z.files
        ),
        None,
    )
    if key is None:
        raise KeyError(f"no {label} pixel array found; keys={z.files}")
    array = np.asarray(z[key], dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"expected {label} [C,H,W] or [H,W,C], got {array.shape}")
    if array.shape[0] not in expected_channels and array.shape[-1] in expected_channels:
        array = np.moveaxis(array, -1, 0)
    if array.shape[0] not in expected_channels:
        raise ValueError(
            f"expected {label} channels in {expected_channels}, got {array.shape[0]}"
        )
    return array


def _center_crop(
    image: np.ndarray,
    *,
    target_size: tuple[int, int],
    path: Path,
    label: str,
) -> np.ndarray:
    height, width = int(image.shape[-2]), int(image.shape[-1])
    target_height, target_width = target_size
    if height < target_height or width < target_width:
        raise ValueError(
            f"{path}: {label} patch is {height}x{width}, below benchmark expectation "
            f"{target_height}x{target_width}; padding is disabled"
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
                "Sentinel-2 NPZ has no band_names; pass --assume-canonical-band-order "
                "only after verifying the ten/12-band source order"
            )
        detected = S2_10_BANDS if channels == 10 else RAW_S2_12_BANDS
    if len(detected) != channels:
        raise ValueError(f"band metadata has {len(detected)} names for {channels} channels")
    return tuple(normalise_band_name(value) for value in detected)


def _canonical_era5_band_name(value: Any) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    if token not in _ERA5_BAND_ALIASES:
        raise ValueError(
            f"unsupported ERA5-Land band name {value!r}; expected aliases for "
            "temperature_2m and total_precipitation"
        )
    return _ERA5_BAND_ALIASES[token]


def _era5_input_bands(
    z: Any,
    meta: dict[str, Any],
    *,
    fallback: Sequence[str],
) -> tuple[str, str]:
    """Resolve source channels, preferring the per-file NPZ declaration."""
    detected = None
    if "band_names" in z.files:
        try:
            detected = tuple(z["band_names"].tolist())
        except Exception as exc:
            raise ValueError("cannot read ERA5-Land band_names from NPZ") from exc
    if detected is None:
        value = _first(meta, ("band_names", "bands", "band_order"))
        if value is not None:
            detected = tuple(value)
    source = tuple(fallback) if detected is None else detected
    canonical = tuple(_canonical_era5_band_name(value) for value in source)
    if len(canonical) != 2 or set(canonical) != set(PRESTO_ERA5_BANDS):
        raise ValueError(
            "ERA5-Land band metadata must identify temperature_2m and "
            f"total_precipitation once; got {source}"
        )
    return canonical


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


def _patch_rng(county_id: str, year: int, patch_id: str, salt: int) -> np.random.Generator:
    """Deterministic per-patch generator, independent of processing order.

    A single generator shared across the cohort makes each patch's draw depend
    on every draw before it, so a resumed or reordered run samples different
    pixels.  Seeding from the patch identity keeps the sample stable across
    restarts, worker counts, and cohort changes -- which matters here because
    extraction runs for hours on a filesystem that has already eaten one job.
    """
    material = f"{county_id}-{year}-{patch_id}-{int(salt)}".encode()
    digest = hashlib.sha256(material).digest()[:8]
    return np.random.default_rng(int.from_bytes(digest, "big"))


def _sample_pixel_series(
    cube: np.ndarray,
    valid: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
    label: str,
) -> tuple[np.ndarray, bool]:
    """Draw ``samples`` pixel time series from the valid positions of ``cube``.

    ``cube`` is [T,C,H,W] and ``valid`` is [H,W].  Returns ``[K,T,C]`` plus a
    flag recording whether replacement was needed.  Presto is a pixel model, so
    each drawn location is a legitimate input sequence; the spatial mean is not,
    because it is a spectrum belonging to no real pixel.
    """
    positions = np.argwhere(valid)
    if positions.size == 0:
        raise ValueError(f"{label}: no valid pixels remain to sample")
    replaced = len(positions) < int(samples)
    chosen = rng.choice(len(positions), size=int(samples), replace=replaced)
    rows = positions[chosen, 0]
    cols = positions[chosen, 1]
    # [T,C,K] -> [K,T,C]
    return np.ascontiguousarray(cube[:, :, rows, cols].transpose(2, 0, 1)), replaced


@dataclass(frozen=True)
class PrestoPatchSequence:
    county_id: str
    year: int
    patch_id: str
    s2_paths: tuple[Path, ...]
    era5_paths: tuple[Path, ...] | None


class PrestoPatchDataset(Dataset):
    """Read complete spatial-patch sequences without changing patch counts."""

    def __init__(
        self,
        s2_dir: str | Path,
        *,
        era5_dir: str | Path | None = None,
        target_size: int | Sequence[int] = DEFAULT_SPATIAL_SIZE,
        expected_timesteps: int = 7,
        timestep_base: str | int = "auto",
        expected_input_count: int | None = None,
        expected_era5_input_count: int | None = None,
        require_band_names: bool = True,
        s2_units: str = "auto",
        era5_source_bands: Sequence[str] = DEFAULT_ERA5_SOURCE_BANDS,
        allow_missing_latlon: bool = False,
        nonfinite_policy: str = "error",
        undersize_policy: str = "error",
        spatial_mode: str = "mean",
        pixel_samples: int = DEFAULT_PIXEL_SAMPLES,
        sample_seed: int = 0,
        min_valid_fraction: float = 0.0,
        max_sequences: int | None = None,
    ):
        self.s2_root = Path(s2_dir)
        if not self.s2_root.exists():
            raise FileNotFoundError(f"Sentinel-2 NPZ directory does not exist: {self.s2_root}")
        self.era5_root = Path(era5_dir) if era5_dir is not None else None
        if self.era5_root is not None and not self.era5_root.exists():
            raise FileNotFoundError(
                f"ERA5-Land NPZ directory does not exist: {self.era5_root}"
            )
        self.target_size = _pair(target_size)
        self.expected_timesteps = int(expected_timesteps)
        self.require_band_names = bool(require_band_names)
        self.s2_units = str(s2_units).strip().lower()
        if self.s2_units not in {"auto", "dn", "reflectance"}:
            raise ValueError("s2_units must be 'auto', 'dn', or 'reflectance'")
        self.allow_missing_latlon = bool(allow_missing_latlon)
        self.nonfinite_policy = str(nonfinite_policy).strip().lower()
        if self.nonfinite_policy not in NONFINITE_POLICIES:
            raise ValueError(f"nonfinite_policy must be one of {NONFINITE_POLICIES}")
        self.undersize_policy = normalise_undersize_policy(undersize_policy)
        self.spatial_mode = str(spatial_mode).strip().lower()
        if self.spatial_mode not in SPATIAL_MODES:
            raise ValueError(f"spatial_mode must be one of {SPATIAL_MODES}")
        self.pixel_samples = int(pixel_samples)
        if self.spatial_mode == "sample" and self.pixel_samples < 1:
            raise ValueError("pixel_samples must be >= 1 in spatial_mode='sample'")
        self.sample_seed = int(sample_seed)
        self.min_valid_fraction = float(min_valid_fraction)
        if not 0.0 <= self.min_valid_fraction <= 1.0:
            raise ValueError("min_valid_fraction must lie in [0, 1]")
        if self.min_valid_fraction > 0.0 and self.nonfinite_policy != "mask":
            raise ValueError(
                "min_valid_fraction only has meaning with nonfinite_policy='mask'; "
                "under 'error' any invalid pixel already halts, and under 'zero' "
                "invalid pixels are folded into the mean rather than excluded"
            )
        self._replacement_sampled_patches = 0
        source_bands = tuple(str(value).strip().lower() for value in era5_source_bands)
        if len(source_bands) != 2 or set(source_bands) != set(PRESTO_ERA5_BANDS):
            raise ValueError(
                "era5_source_bands must contain temperature_2m and "
                "total_precipitation once"
            )
        self.era5_source_bands = source_bands

        s2_paths = sorted(self.s2_root.rglob("*.npz"))
        if not s2_paths:
            raise FileNotFoundError(f"no NPZ files found below {self.s2_root}")
        if expected_input_count is not None and len(s2_paths) != int(expected_input_count):
            raise ValueError(
                f"Presto Sentinel-2 input cohort has {len(s2_paths):,} files, expected "
                f"{int(expected_input_count):,}"
            )
        parsed = [_source_fields(path) for path in s2_paths]
        base = _resolve_timestep_base(
            [fields[3] for fields in parsed], self.expected_timesteps, timestep_base
        )
        grouped: dict[tuple[str, int, str], dict[int, Path]] = {}
        excluded_s2_paths: list[Path] = []
        raw_keys = []
        for path, (county, year, patch_id, raw_timestep) in zip(s2_paths, parsed):
            timestep = raw_timestep - base
            if not 0 <= timestep < self.expected_timesteps:
                excluded_s2_paths.append(path)
                continue
            key = (county, year, patch_id, timestep)
            raw_keys.append(key)
            grouped.setdefault((county, year, patch_id), {})[timestep] = path
        if len(raw_keys) != len(set(raw_keys)):
            raise ValueError(
                "Sentinel-2 files collapse to duplicate county/year/patch/timestep keys"
            )

        expected_schedule = set(range(self.expected_timesteps))
        complete_keys = sorted(
            key for key, by_time in grouped.items() if set(by_time) == expected_schedule
        )
        if not complete_keys:
            raise ValueError("no spatial patches have the complete benchmark timestep schedule")

        # Keep the schedule-completeness count separate from the undersize
        # count, so the contract does not conflate the two exclusion reasons.
        self._schedule_complete_patches = len(complete_keys)

        # Screen undersized spatial patches here, at index time, so the excluded
        # counts land in the data contract and extraction cannot die halfway
        # through a multi-hour run on a 4.4% minority of the cohort.
        complete_keys, undersized_keys, undersized_files = screen_undersized_patches(
            {key: grouped[key] for key in complete_keys},
            target_size=self.target_size,
            policy=self.undersize_policy,
        )
        self._undersized_patches_excluded = len(undersized_keys)
        self._undersized_files_excluded = int(undersized_files)

        selected_keys = complete_keys
        if max_sequences is not None:
            selected_keys = selected_keys[: int(max_sequences)]

        era5_lookup: dict[tuple[str, int, str, int], Path] = {}
        era5_count = 0
        excluded_era5_paths: list[Path] = []
        if self.era5_root is not None:
            era5_paths = sorted(self.era5_root.rglob("*.npz"))
            era5_count = len(era5_paths)
            if expected_era5_input_count is not None and era5_count != int(
                expected_era5_input_count
            ):
                raise ValueError(
                    f"Presto ERA5-Land input cohort has {era5_count:,} files, "
                    f"expected {int(expected_era5_input_count):,}"
                )
            for path in era5_paths:
                county, year, patch_id, raw_timestep = _source_fields(path)
                timestep = raw_timestep - base
                if not 0 <= timestep < self.expected_timesteps:
                    excluded_era5_paths.append(path)
                    continue
                key = (county, year, patch_id, timestep)
                if key in era5_lookup:
                    raise ValueError(f"duplicate ERA5-Land key {key}")
                era5_lookup[key] = path

        sequences = []
        for county, year, patch_id in selected_keys:
            by_time = grouped[(county, year, patch_id)]
            matched_era5 = None
            if self.era5_root is not None:
                missing = [
                    timestep
                    for timestep in range(self.expected_timesteps)
                    if (county, year, patch_id, timestep) not in era5_lookup
                ]
                if missing:
                    raise ValueError(
                        f"ERA5-Land is missing timesteps {missing} for "
                        f"{county}/{year}/{patch_id}"
                    )
                matched_era5 = tuple(
                    era5_lookup[(county, year, patch_id, timestep)]
                    for timestep in range(self.expected_timesteps)
                )
            sequences.append(
                PrestoPatchSequence(
                    county_id=county,
                    year=year,
                    patch_id=patch_id,
                    s2_paths=tuple(by_time[t] for t in range(self.expected_timesteps)),
                    era5_paths=matched_era5,
                )
            )
        self.indices = sequences
        self.timestep_base = base
        self._input_files = len(s2_paths)
        self._era5_input_files = era5_count
        self._excluded_s2_paths = tuple(excluded_s2_paths)
        self._excluded_era5_paths = tuple(excluded_era5_paths)
        self._all_groups = grouped
        self._all_complete_keys = complete_keys

    @property
    def include_era5(self) -> bool:
        return self.era5_root is not None

    @property
    def backbone(self) -> str:
        return PRESTO_BACKBONE_S2_ERA5 if self.include_era5 else PRESTO_BACKBONE_S2

    def __len__(self) -> int:
        return len(self.indices)

    def _validate_source_identity(
        self, path: Path, meta: dict[str, Any], *, county: str, year: int
    ) -> None:
        meta_county = _first(meta, ("county_fips", "county", "fips", "GEOID"))
        meta_year = _first(meta, ("year",))
        if meta_county is not None and normalise_county(meta_county) != county:
            raise ValueError(f"{path.name}: metadata county disagrees with filename")
        if meta_year is not None and int(safe_float(meta_year)) != year:
            raise ValueError(f"{path.name}: metadata year disagrees with filename")

    def _handle_nonfinite(
        self, values: np.ndarray, *, path: Path, label: str
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Return ``(values, valid_mask, valid_fraction)`` for one [C,H,W] array.

        ``zero`` reproduces the historical behaviour: non-finite pixels become
        zeros and then enter the spatial mean, which biases every band toward
        zero in proportion to cloud cover.  ``mask`` instead excludes them from
        the statistics entirely, which is what a cloud-masked pixel warrants --
        it carries no observation, not an observation of zero.
        """
        finite = np.isfinite(values).all(axis=0)
        valid_fraction = float(finite.mean())
        if valid_fraction < 1.0 and self.nonfinite_policy == "error":
            raise ValueError(
                f"{path.name}: non-finite {label} pixels remain after compositing "
                f"(valid spatial fraction={valid_fraction:.6f})"
            )
        if self.nonfinite_policy == "zero":
            return (
                np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0),
                np.ones(values.shape[-2:], dtype=bool),
                valid_fraction,
            )
        return values, finite, valid_fraction

    def _read_s2(
        self, path: Path, *, county: str, year: int
    ) -> tuple[
        np.ndarray, np.ndarray, date, tuple[float, float] | None, float, tuple[int, int]
    ]:
        with np.load(path, allow_pickle=True) as z:
            meta = npz_metadata(z)
            pixels = _pixel_array(z, expected_channels=(10, 12), label="Sentinel-2")
            names = _input_bands(
                z,
                meta,
                channels=int(pixels.shape[0]),
                require_names=self.require_band_names,
            )
            acquisition = _acquisition_date(path, meta)
            location = _latlon(z, meta)
        self._validate_source_identity(path, meta, county=county, year=year)
        pixels = reindex_bands(pixels[None], names, S2_10_BANDS)[0]
        source_shape = (int(pixels.shape[-2]), int(pixels.shape[-1]))
        pixels = _center_crop(
            pixels, target_size=self.target_size, path=path, label="Sentinel-2"
        )
        pixels, valid_mask, valid_fraction = self._handle_nonfinite(
            pixels, path=path, label="Sentinel-2"
        )
        finite = pixels[np.isfinite(pixels)]
        max_abs = float(np.max(np.abs(finite))) if finite.size else 0.0
        source_units = self.s2_units
        if source_units == "auto":
            source_units = "reflectance" if max_abs <= 2.0 else "dn"
        if source_units == "dn" and 0.0 < max_abs <= 2.0:
            raise ValueError(
                f"{path.name}: values look like reflectance, but s2_units='dn'; "
                "use --s2-units reflectance"
            )
        if source_units == "reflectance" and max_abs > 4.0:
            raise ValueError(
                f"{path.name}: values look like digital numbers, but "
                "s2_units='reflectance'; use --s2-units dn"
            )
        # Convert on the cube rather than after reduction: the spatial mean and
        # any sampled pixel must reach construct_single_presto_input in the same
        # digital-number units the helper's normalization assumes.
        if source_units == "reflectance":
            pixels = pixels.astype(np.float32) * 10000.0
        return (
            pixels.astype(np.float32),
            valid_mask,
            acquisition,
            location,
            valid_fraction,
            source_shape,
        )

    def _read_era5(
        self, path: Path, *, expected_source_shape: tuple[int, int]
    ) -> tuple[np.ndarray, date, float]:
        with np.load(path, allow_pickle=True) as z:
            meta = npz_metadata(z)
            pixels = _pixel_array(z, expected_channels=(2,), label="ERA5-Land")
            source_bands = _era5_input_bands(
                z,
                meta,
                fallback=self.era5_source_bands,
            )
            acquisition = _acquisition_date(path, meta)
        source_shape = (int(pixels.shape[-2]), int(pixels.shape[-1]))
        if source_shape != expected_source_shape:
            raise ValueError(
                f"{path.name}: ERA5-Land grid {source_shape[0]}x{source_shape[1]} does not "
                f"match its Sentinel-2 grid {expected_source_shape[0]}x"
                f"{expected_source_shape[1]}; geospatial resampling must happen upstream"
            )
        pixels = _center_crop(
            pixels, target_size=self.target_size, path=path, label="ERA5-Land"
        )
        pixels, era5_valid, valid_fraction = self._handle_nonfinite(
            pixels, path=path, label="ERA5-Land"
        )
        source_index = {
            name: index for index, name in enumerate(source_bands)
        }
        pixels = pixels[[source_index[name] for name in PRESTO_ERA5_BANDS]]
        # ERA5-Land and Daymet are coarse fields upsampled onto the S2 grid, so
        # they carry no meaningful within-patch spatial structure to sample.
        # They stay a patch statistic and are broadcast across sampled pixels.
        if era5_valid.any():
            spatial_mean = (
                pixels[:, era5_valid].mean(axis=-1, dtype=np.float64).astype(np.float32)
            )
        else:
            raise ValueError(f"{path.name}: no valid ERA5-Land pixels remain")
        temperature, precipitation = map(float, spatial_mean)
        if not 150.0 <= temperature <= 350.0:
            raise ValueError(
                f"{path.name}: temperature_2m={temperature:.3f} is not plausible Kelvin"
            )
        if precipitation < 0.0 or precipitation > 20.0:
            raise ValueError(
                f"{path.name}: total_precipitation={precipitation:.3f} is not plausible metres"
            )
        return spatial_mean, acquisition, valid_fraction

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.indices[index]
        s2_values = []
        s2_masks = []
        dates = []
        locations = []
        valid_fractions = []
        source_shapes = []
        for path in item.s2_paths:
            (
                values,
                valid_mask,
                acquisition,
                location,
                valid_fraction,
                source_shape,
            ) = self._read_s2(path, county=item.county_id, year=item.year)
            s2_values.append(values)
            s2_masks.append(valid_mask)
            dates.append(acquisition)
            if location is not None:
                locations.append(location)
            valid_fractions.append(valid_fraction)
            source_shapes.append(source_shape)
        if dates != sorted(dates) or len(set(dates)) != self.expected_timesteps:
            raise ValueError(
                f"{item.county_id}/{item.year}/{item.patch_id}: acquisition dates must be "
                "distinct and chronological in timestep order"
            )
        if locations:
            location_array = np.asarray(locations, dtype=np.float64)
            if not np.allclose(location_array, location_array[0], atol=1e-5, rtol=0.0):
                raise ValueError(
                    f"{item.county_id}/{item.year}/{item.patch_id}: latitude/longitude "
                    "changes across timesteps"
                )
            location = tuple(float(value) for value in location_array[0])
        elif self.allow_missing_latlon:
            location = (0.0, 0.0)
        else:
            raise ValueError(
                f"{item.county_id}/{item.year}/{item.patch_id}: Presto needs latitude/longitude"
            )

        era5_values = np.empty((self.expected_timesteps, 0), dtype=np.float32)
        if item.era5_paths is not None:
            era5 = []
            for timestep, path in enumerate(item.era5_paths):
                values, acquisition, valid_fraction = self._read_era5(
                    path, expected_source_shape=source_shapes[timestep]
                )
                if acquisition != dates[timestep]:
                    raise ValueError(
                        f"{path.name}: ERA5-Land date {acquisition} disagrees with "
                        f"Sentinel-2 date {dates[timestep]}"
                    )
                era5.append(values)
                valid_fractions.append(valid_fraction)
            era5_values = np.stack(era5).astype(np.float32)

        # [T,C,H,W] with a single [H,W] validity mask shared by the whole
        # sequence: a pixel is usable only where every timestep observed it.
        cube = np.stack(s2_values).astype(np.float32)
        valid = np.logical_and.reduce(np.stack(s2_masks))
        if self.nonfinite_policy == "mask":
            # All-zero locations are cloud-mask fill, not dark ground. Encoding
            # them would feed Presto a fabricated observation of zero.
            valid = valid & (cube != 0).any(axis=(0, 1))
        sequence_valid_fraction = float(valid.mean())
        if sequence_valid_fraction < self.min_valid_fraction:
            raise ValueError(
                f"{item.county_id}/{item.year}/{item.patch_id}: valid pixel fraction "
                f"{sequence_valid_fraction:.6f} is below --min-valid-fraction "
                f"{self.min_valid_fraction:.6f}"
            )
        if not valid.any():
            raise ValueError(
                f"{item.county_id}/{item.year}/{item.patch_id}: no pixel is valid at "
                "every timestep"
            )

        replaced = False
        if self.spatial_mode == "sample":
            rng = _patch_rng(
                item.county_id, item.year, item.patch_id, self.sample_seed
            )
            s2_series, replaced = _sample_pixel_series(
                cube,
                valid,
                samples=self.pixel_samples,
                rng=rng,
                label=f"{item.county_id}/{item.year}/{item.patch_id}",
            )
        else:
            # [T,C,N_valid] -> [T,C], then a leading axis of 1 so both modes
            # share one downstream shape contract.
            s2_series = (
                cube[:, :, valid].mean(axis=-1, dtype=np.float64).astype(np.float32)
            )[None]

        return {
            "s2_dn": torch.from_numpy(np.ascontiguousarray(s2_series)),
            "era5": torch.from_numpy(era5_values),
            "sequence_valid_fraction": sequence_valid_fraction,
            "replacement_sampled": bool(replaced),
            "months": torch.tensor([value.month - 1 for value in dates], dtype=torch.long),
            "latlon": torch.tensor(location, dtype=torch.float32),
            "county_id": item.county_id,
            "year": item.year,
            "patch_id": item.patch_id,
            "dates": [value.isoformat() for value in dates],
            "s2_source_files": [str(path) for path in item.s2_paths],
            "era5_source_files": (
                [str(path) for path in item.era5_paths]
                if item.era5_paths is not None
                else []
            ),
            "valid_fraction_min": float(min(valid_fractions)),
        }

    def describe(self) -> dict[str, Any]:
        incomplete = len(self._all_groups) - self._schedule_complete_patches
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
        return {
            "sentinel2_input_files": int(self._input_files),
            "era5_input_files": int(self._era5_input_files),
            "sentinel2_schedule_files": int(
                self._input_files - len(self._excluded_s2_paths)
            ),
            "era5_schedule_files": int(
                self._era5_input_files - len(self._excluded_era5_paths)
            ),
            "sentinel2_out_of_schedule_files_excluded": int(
                len(self._excluded_s2_paths)
            ),
            "era5_out_of_schedule_files_excluded": int(
                len(self._excluded_era5_paths)
            ),
            "source_spatial_patches": int(len(self._all_groups)),
            "complete_spatial_patches": int(len(self._all_complete_keys)),
            "incomplete_spatial_patches_excluded": int(incomplete),
            "output_sequence_rows": int(len(self.indices)),
            "county_years": int(selected[["county_id", "year"]].drop_duplicates().shape[0]),
            "patch_count_min": int(patch_counts.min()),
            "patch_count_median": float(patch_counts.median()),
            "patch_count_max": int(patch_counts.max()),
            "expected_timesteps": int(self.expected_timesteps),
            "normalized_timestep_schedule": list(range(self.expected_timesteps)),
            "out_of_schedule_timestep_policy": "audit_then_exclude",
            "timestep_base": int(self.timestep_base),
            "target_size": list(self.target_size),
            "oversize_policy": (
                "center_crop_before_pixel_sampling"
                if self.spatial_mode == "sample"
                else "center_crop_before_spatial_mean"
            ),
            "spatial_mode": self.spatial_mode,
            "pixel_samples": (
                int(self.pixel_samples) if self.spatial_mode == "sample" else None
            ),
            "pixel_sample_seed": (
                int(self.sample_seed) if self.spatial_mode == "sample" else None
            ),
            "pixel_sample_seed_source": (
                "sha256(county-year-patch-seed)"
                if self.spatial_mode == "sample"
                else None
            ),
            "nonfinite_policy": self.nonfinite_policy,
            "min_valid_fraction": float(self.min_valid_fraction),
            "undersize_policy": self.undersize_policy,
            "undersized_spatial_patches_excluded": int(
                self._undersized_patches_excluded
            ),
            "undersized_files_excluded": int(self._undersized_files_excluded),
            "s2_units_at_source": self.s2_units,
            "s2_units_policy": (
                "detect_each_file_then_convert_reflectance_to_dn"
                if self.s2_units == "auto"
                else "declared_uniform"
            ),
            "presto_input_units": "raw_dn",
            "normalization_owner": "presto.construct_single_presto_input",
            "ndvi_owner": "presto.construct_single_presto_input",
            "era5_source_bands": (
                list(self.era5_source_bands) if self.include_era5 else []
            ),
            "era5_source_band_resolution": (
                "npz_metadata_then_cli_fallback" if self.include_era5 else None
            ),
            "era5_model_bands": list(PRESTO_ERA5_BANDS) if self.include_era5 else [],
            "era5_units": (
                {"temperature_2m": "K", "total_precipitation": "m"}
                if self.include_era5
                else {}
            ),
        }


def _collate(samples: Sequence[dict[str, Any]]) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    tensors = {
        key: torch.stack([sample[key] for sample in samples])
        for key in ("s2_dn", "era5", "months", "latlon")
    }
    metadata = [
        {key: value for key, value in sample.items() if key not in tensors}
        for sample in samples
    ]
    return tensors, metadata


def build_presto_batch(
    tensors: dict[str, torch.Tensor],
    construct_input: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    include_era5: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use the official helper once per input sequence, without pre-normalizing.

    ``s2_dn`` is [B,K,T,C]: K is 1 in spatial_mode='mean' and the sampled pixel
    count otherwise.  The batch is flattened to B*K sequences here and folded
    back to B embeddings after the encoder, so the helper still sees exactly one
    complete temporal sequence per call.
    """
    batch, samples = int(tensors["s2_dn"].shape[0]), int(tensors["s2_dn"].shape[1])
    flat_s2 = tensors["s2_dn"].reshape(batch * samples, *tensors["s2_dn"].shape[2:])
    xs, masks, dynamic_world = [], [], []
    for row in range(batch * samples):
        kwargs: dict[str, Any] = {
            "s2": flat_s2[row],
            "s2_bands": list(PRESTO_S2_BANDS),
        }
        if include_era5:
            kwargs.update(
                {
                    "era5": tensors["era5"][row // samples],
                    "era5_bands": list(PRESTO_ERA5_BANDS),
                }
            )
        x, mask, dw = construct_input(**kwargs)
        if x.ndim != 2 or x.shape[1] != DEFAULT_MODEL_BANDS:
            raise ValueError(
                f"official Presto helper returned x={tuple(x.shape)}, expected [T,17]"
            )
        if mask.shape != x.shape or dw.shape != x.shape[:1]:
            raise ValueError(
                f"official Presto helper returned incompatible shapes "
                f"x={tuple(x.shape)}, mask={tuple(mask.shape)}, dw={tuple(dw.shape)}"
            )
        if not torch.isfinite(x).all():
            raise ValueError("official Presto preprocessing produced non-finite values")
        xs.append(x.float())
        masks.append(mask.float())
        dynamic_world.append(dw.long())
    return torch.stack(xs), torch.stack(masks), torch.stack(dynamic_world)


def extract_presto_embeddings(
    dataset: PrestoPatchDataset,
    model: torch.nn.Module,
    construct_input: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device | str,
    batch_size: int = 256,
    num_workers: int = 0,
) -> pd.DataFrame:
    """Emit one official globally pooled 128-D vector per complete patch sequence."""
    device = torch.device(device)
    model = model.to(device).eval()
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=_collate,
    )
    rows = []
    with torch.inference_mode():
        for tensors, metadata in loader:
            x, mask, dw = build_presto_batch(
                tensors, construct_input, include_era5=dataset.include_era5
            )
            samples = int(tensors["s2_dn"].shape[1])
            encoded = model.encoder(
                x=x.to(device),
                dynamic_world=dw.to(device),
                mask=mask.to(device),
                latlons=tensors["latlon"].repeat_interleave(samples, dim=0).to(device),
                month=tensors["months"].repeat_interleave(samples, dim=0).to(device),
                eval_task=True,
            )
            if encoded.ndim != 2 or encoded.shape[1] != DEFAULT_EMBEDDING_DIM:
                raise ValueError(
                    f"Presto encoder returned {tuple(encoded.shape)}, expected [B,128]"
                )
            if not torch.isfinite(encoded).all():
                raise ValueError("Presto encoder produced non-finite embeddings")
            # Pool the K pixel embeddings per patch. Presto is nonlinear, so this
            # is not the same as encoding a spatial mean, and it is the ordering
            # the model was pretrained for: encode real pixels, then average.
            encoded = encoded.reshape(-1, samples, DEFAULT_EMBEDDING_DIM).mean(dim=1)
            encoded_np = encoded.detach().cpu().numpy().astype(np.float32)
            for vector, item, months in zip(encoded_np, metadata, tensors["months"]):
                rows.append(
                    {
                        "county_id": item["county_id"],
                        "year": item["year"],
                        "patch_id": item["patch_id"],
                        # Presto has already globally pooled the whole temporal sequence.
                        "timestep": 0,
                        "backbone": dataset.backbone,
                        "embedding": vector.tolist(),
                        "experiment_family": (
                            "auxiliary_climate_fusion"
                            if dataset.include_era5
                            else "main_benchmark"
                        ),
                        "fusion_stage": (
                            "presto_encoder_input" if dataset.include_era5 else "none"
                        ),
                        "representation_scope": "sequence",
                        "input_modalities": (
                            "Sentinel-2,ERA5-Land"
                            if dataset.include_era5
                            else "Sentinel-2"
                        ),
                        "sequence_timesteps": dataset.expected_timesteps,
                        "months_zero_based": months.tolist(),
                        "dates": item["dates"],
                        "s2_source_files": item["s2_source_files"],
                        "era5_source_files": item["era5_source_files"],
                        "valid_fraction_min": item["valid_fraction_min"],
                        "sequence_valid_fraction": item["sequence_valid_fraction"],
                        "replacement_sampled": item["replacement_sampled"],
                        "spatial_reduction": (
                            (
                                f"pixel_sample_k{dataset.pixel_samples}"
                                f"_embedding_mean_after_center_crop_"
                                f"{dataset.target_size[0]}x{dataset.target_size[1]}"
                            )
                            if dataset.spatial_mode == "sample"
                            # Unchanged wording when no masking is in force, so
                            # the published configuration keeps its contract.
                            else (
                                f"valid_pixel_mean_after_center_crop_"
                                f"{dataset.target_size[0]}x{dataset.target_size[1]}"
                                if dataset.nonfinite_policy == "mask"
                                else f"mean_after_center_crop_"
                                f"{dataset.target_size[0]}x{dataset.target_size[1]}"
                            )
                        ),
                        "temporal_reduction": "official_global_token_mean",
                    }
                )
    if len(rows) != len(dataset):
        raise RuntimeError(
            f"Presto extraction emitted {len(rows):,} rows for {len(dataset):,} sequences"
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


def load_presto(
    *, presto_repo: str | Path | None, device: torch.device
) -> tuple[Any, torch.nn.Module, str]:
    """Load the official package and its bundled pretrained checkpoint."""
    repo = Path(presto_repo).resolve() if presto_repo is not None else None
    if repo is not None:
        if not repo.exists():
            raise FileNotFoundError(f"official Presto repository does not exist: {repo}")
        sys.path.insert(0, str(repo))
    try:
        module = importlib.import_module("presto")
    except ImportError as exc:
        raise ImportError(
            "cannot import the official Presto package; install its repository with "
            "`pip install -e .` in the extraction environment"
        ) from exc
    if not hasattr(module, "Presto") or not hasattr(module, "construct_single_presto_input"):
        raise TypeError("imported `presto` is not the nasaharvest/presto package")
    model = module.Presto.load_pretrained().to(device).eval()
    try:
        utils = importlib.import_module("presto.utils")
        checkpoint = str(Path(utils.default_model_path).resolve())
    except (ImportError, AttributeError):
        checkpoint = "official bundled default_model.pt"
    return module, model, checkpoint


def _git_revision(path: str | Path | None) -> str | None:
    if path is None:
        return None
    result = subprocess.run(
        ["git", "-C", str(Path(path).resolve()), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s2-dir", required=True)
    parser.add_argument(
        "--era5-dir",
        help="Co-located ERA5-Land NPZs; Daymet late fusion belongs in the probe",
    )
    parser.add_argument("--presto-repo", help="Official nasaharvest/presto checkout")
    parser.add_argument("--output", help="Canonical output Parquet")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--expected-input-count", type=int)
    parser.add_argument("--expected-era5-input-count", type=int)
    parser.add_argument("--expected-timesteps", type=int, default=7)
    parser.add_argument("--timestep-base", choices=("auto", "0", "1"), default="auto")
    parser.add_argument(
        "--s2-units",
        choices=("auto", "dn", "reflectance"),
        default="auto",
        help="Source units; auto supports audited collections containing both encodings",
    )
    parser.add_argument(
        "--era5-source-bands",
        nargs=2,
        default=DEFAULT_ERA5_SOURCE_BANDS,
        metavar=("BAND_1", "BAND_2"),
        help="Channel order in ERA5 NPZs; values are reordered to Presto's official order",
    )
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--assume-canonical-band-order", action="store_true")
    parser.add_argument("--allow-missing-latlon", action="store_true")
    parser.add_argument(
        "--nonfinite-policy",
        choices=NONFINITE_POLICIES,
        default="error",
        help=(
            "'error' (default) halts on any non-finite pixel. 'zero' reproduces "
            "the published run by substituting zeros, which biases the spatial "
            "mean toward zero in proportion to cloud cover. 'mask' excludes "
            "non-finite and all-zero fill pixels from the statistics instead."
        ),
    )
    parser.add_argument(
        "--spatial-mode",
        choices=SPATIAL_MODES,
        default="mean",
        help=(
            "'mean' (default) reduces each composite to one band vector over the "
            "valid pixels, matching the published run. 'sample' draws "
            "--pixel-samples real pixel time series, encodes each, and averages "
            "the embeddings -- Presto is a pixel model, so a spatial mean is a "
            "spectrum belonging to no pixel it was pretrained on."
        ),
    )
    parser.add_argument(
        "--pixel-samples",
        type=int,
        default=DEFAULT_PIXEL_SAMPLES,
        help=(
            "Pixels drawn per patch in --spatial-mode sample. Multiplies the "
            "encoder batch by this factor, so lower --batch-size to match."
        ),
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help=(
            "Salt for the per-patch sampling seed. Each patch seeds from "
            "sha256(county-year-patch-seed), so the draw survives restarts and "
            "reordering; vary this to measure sampling sensitivity."
        ),
    )
    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=0.0,
        help=(
            "Reject a patch whose share of pixels valid at every timestep falls "
            "below this. Requires --nonfinite-policy mask. Default 0 disables it."
        ),
    )
    parser.add_argument(
        "--spatial-size",
        type=int,
        default=DEFAULT_SPATIAL_SIZE[0],
        help=(
            "Harmonized center-crop footprint in pixels before the per-composite "
            "spatial mean. Presto imposes no spatial requirement of its own, so "
            "this exists purely to match the other encoders' footprint."
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

    dataset = PrestoPatchDataset(
        args.s2_dir,
        era5_dir=args.era5_dir,
        target_size=(args.spatial_size, args.spatial_size),
        expected_timesteps=args.expected_timesteps,
        timestep_base=args.timestep_base,
        expected_input_count=args.expected_input_count,
        expected_era5_input_count=args.expected_era5_input_count,
        require_band_names=not args.assume_canonical_band_order,
        s2_units=args.s2_units,
        era5_source_bands=args.era5_source_bands,
        allow_missing_latlon=args.allow_missing_latlon,
        nonfinite_policy=args.nonfinite_policy,
        undersize_policy=args.undersize_policy,
        spatial_mode=args.spatial_mode,
        pixel_samples=args.pixel_samples,
        sample_seed=args.sample_seed,
        min_valid_fraction=args.min_valid_fraction,
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
    presto_module, model, checkpoint = load_presto(
        presto_repo=args.presto_repo, device=device
    )
    frame = extract_presto_embeddings(
        dataset,
        model,
        presto_module.construct_single_presto_input,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    output = write_embeddings(frame, args.output)
    provenance = {
        "schema_version": 1,
        "backbone": dataset.backbone,
        "experiment_family": (
            "auxiliary_climate_fusion" if dataset.include_era5 else "main_benchmark"
        ),
        "fusion_stage": (
            "presto_encoder_input" if dataset.include_era5 else "none"
        ),
        "official_repository": OFFICIAL_REPOSITORY,
        "official_revision": _git_revision(args.presto_repo),
        "checkpoint": checkpoint,
        "input_root": str(Path(args.s2_dir).resolve()),
        "era5_root": str(Path(args.era5_dir).resolve()) if args.era5_dir else None,
        "output": str(output.resolve()),
        "output_rows": int(len(frame)),
        "embedding_dim": DEFAULT_EMBEDDING_DIM,
        "encoder_layer": "final_encoder_global_pool",
        "representation_scope": "complete_patch_sequence",
        "device": str(device),
        "dataset": description,
    }
    sidecar = output.with_suffix(output.suffix + ".provenance.json")
    sidecar.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Wrote {len(frame):,} canonical Presto sequence rows to {output}")
    print(f"Wrote provenance to {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
