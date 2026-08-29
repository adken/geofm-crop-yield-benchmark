"""Small, dependency-free readers for the two county NPZ layouts.

This module intentionally owns the county-file parsing contract so the
benchmark package remains standalone.
"""

from __future__ import annotations

import logging
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

UNDERSIZE_POLICIES = ("error", "skip")

S2_10_BANDS = (
    "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B11", "B12",
)
RAW_S2_12_BANDS = (
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B09", "B11", "B12",
)

COUNTY_FILE_RE = re.compile(
    r"county_(?P<county>\d+)_year_(?P<year>\d{4}).*?interval_(?P<interval>\d+)",
    re.IGNORECASE,
)
PATCH_ID_RE = re.compile(
    r"(?:patch|tile|chip)[_-]?(?P<patch>[A-Za-z0-9]+)|"
    r"(?:row|r)[_-]?(?P<row>\d+).*?(?:col|c)[_-]?(?P<col>\d+)|"
    r"x[_-]?(?P<x>\d+).*?y[_-]?(?P<y>\d+)",
    re.IGNORECASE,
)


def normalise_undersize_policy(value: str) -> str:
    """Validate the undersized-patch policy.

    ``error`` is the documented benchmark contract: a source patch smaller than
    the target footprint in either dimension is rejected, because padding it
    would fabricate pixels the encoder would then treat as observations.

    ``skip`` drops such spatial patches before extraction and records how many
    were removed.  It is an explicitly recorded deviation, not a default: it
    reduces per-county patch counts and therefore changes the complete-patch
    identity hashes that the parity audit compares.
    """
    policy = str(value).strip().lower()
    if policy not in UNDERSIZE_POLICIES:
        raise ValueError(f"undersize_policy must be one of {UNDERSIZE_POLICIES}")
    return policy


def npz_spatial_shape(path: Path, key: str = "pixels") -> tuple[int, int]:
    """Return the trailing (height, width) of an NPZ member without inflating it.

    Reads only the NPY header inside the zip member, so screening the full
    77,813-file cohort costs seconds rather than decompressing gigabytes.
    """
    member = f"{key}.npy"
    with zipfile.ZipFile(path) as archive:
        if member not in archive.namelist():
            raise KeyError(f"{path} has no {key!r} array")
        with archive.open(member) as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, _, _ = np.lib.format.read_array_header_1_0(stream)
            elif version in {(2, 0), (3, 0)}:
                shape, _, _ = np.lib.format.read_array_header_2_0(stream)
            else:  # pragma: no cover - NumPy currently emits only these versions
                raise ValueError(f"unsupported NPY header version {version} in {path}")
    if len(shape) < 2:
        raise ValueError(f"{path}: {key!r} has shape {tuple(shape)}, expected >=2 dims")
    return int(shape[-2]), int(shape[-1])


def screen_undersized_patches(
    grouped: Mapping[Any, Mapping[Any, Path] | Iterable[Path]],
    *,
    target_size: tuple[int, int],
    policy: str = "error",
    pixel_key: str = "pixels",
) -> tuple[list[Any], list[Any], int]:
    """Split spatial-patch groups into kept and undersized-dropped.

    A group is dropped when *any* of its timesteps is below ``target_size``,
    keeping each retained sequence complete rather than ragged. Returns
    ``(kept_keys, dropped_keys, dropped_file_count)``.

    With ``policy='error'`` nothing is dropped and nothing is read; the existing
    per-file check raises later, preserving current behaviour exactly.
    """
    policy = normalise_undersize_policy(policy)
    keys = sorted(grouped)
    if policy == "error":
        return keys, [], 0

    target_height, target_width = int(target_size[0]), int(target_size[1])
    kept: list[Any] = []
    dropped: list[Any] = []
    dropped_files = 0
    for key in keys:
        entry = grouped[key]
        paths = list(entry.values()) if isinstance(entry, Mapping) else list(entry)
        undersized = 0
        for path in paths:
            try:
                height, width = npz_spatial_shape(Path(path), pixel_key)
            except (KeyError, ValueError, OSError, zipfile.BadZipFile):
                # Unreadable headers are not an undersize decision; leave them
                # to the per-file reader so the original error still surfaces.
                continue
            if height < target_height or width < target_width:
                undersized += 1
        if undersized:
            dropped.append(key)
            dropped_files += undersized
        else:
            kept.append(key)
    if not kept:
        raise ValueError(
            "undersize_policy='skip' removed every spatial patch; the target size "
            f"{target_height}x{target_width} exceeds the whole source cohort"
        )
    return kept, dropped, dropped_files


def normalise_band_name(value: str) -> str:
    band = str(value).strip().upper()
    if re.fullmatch(r"B[0-9]", band):
        return f"B0{band[1:]}"
    return band


def reindex_bands(
    cube: np.ndarray,
    input_bands: Sequence[str],
    target_bands: Sequence[str],
) -> np.ndarray:
    """Reorder ``[...,C,H,W]`` data without silently filling real S2 bands."""
    normalized = {normalise_band_name(b): i for i, b in enumerate(input_bands)}
    targets = [normalise_band_name(b) for b in target_bands]
    missing = [band for band in targets if band not in normalized]
    if missing:
        raise KeyError(f"input is missing bands {missing}; has {tuple(input_bands)}")
    return cube[..., [normalized[band] for band in targets], :, :]


def npz_value(z: Any, key: str, default: Any) -> Any:
    if key not in z.files:
        return default
    value = z[key]
    return value.item() if getattr(value, "shape", None) == () else value


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if isinstance(value, str) and value.strip().upper() in {
            "", "NA", "N/A", "NONE", "NULL", "NAN", "UNKNOWN",
        }:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def normalise_county(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value)).zfill(5)
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(5) if text.isdigit() else text


def guess_input_bands(channels: int) -> tuple[str, ...]:
    if channels == 10:
        return S2_10_BANDS
    if channels == 12:
        return RAW_S2_12_BANDS
    raise ValueError(
        f"unknown channel count {channels}; provide data.input_bands explicitly"
    )


def band_names(z: Any) -> tuple[str, ...] | None:
    if "band_names" not in z.files:
        return None
    try:
        return tuple(normalise_band_name(v) for v in z["band_names"].tolist())
    except Exception:
        return None


def metadata(z: Any) -> dict[str, Any]:
    if "metadata" not in z.files:
        return {}
    value = z["metadata"]
    if getattr(value, "shape", None) == ():
        value = value.item()
    return value if isinstance(value, dict) else {}


def pixel_array(z: Any) -> np.ndarray:
    key = next(
        (name for name in ("pixels", "patch", "cube", "data", "s2", "image", "array") if name in z.files),
        None,
    )
    if key is None:
        raise KeyError(f"no pixel array found; keys={z.files}")
    array = z[key].astype(np.float32)
    if array.ndim != 3:
        raise ValueError(f"expected one patch [C,H,W] or [H,W,C], got {array.shape}")
    if array.shape[0] not in (6, 10, 12) and array.shape[-1] in (6, 10, 12):
        array = np.moveaxis(array, -1, 0)
    finite = array[np.isfinite(array)]
    if finite.size and float(finite.max()) > 2.0:
        array = array / 10000.0
    return np.clip(array, 0.0, 1.0)


def yield_bu_per_acre(group: dict[str, Any]) -> float:
    """Return an unconverted USDA corn-yield label in bushels per acre."""
    return safe_float(group.get("yield_bu_per_acre", group.get("yield", np.nan)))


def load_yield_lookup(path: Path) -> dict[tuple[str, int], float]:
    if not path.exists():
        raise FileNotFoundError(f"yield CSV not found: {path}")
    frame = pd.read_csv(path)
    county_col = _pick_column(frame, ("county", "county_fips", "fips", "GEOID", "geoid"))
    year_col = _pick_column(frame, ("year", "Year"))
    yield_col = _pick_column(frame, ("yield_bu_per_acre", "yield", "Yield", "Value", "value"))
    if county_col is None or year_col is None or yield_col is None:
        raise ValueError(f"{path} must contain county/year/yield columns")
    lookup: dict[tuple[str, int], float] = {}
    for _, row in frame.iterrows():
        county = normalise_county(row[county_col])
        value = safe_float(row[yield_col])
        if county and np.isfinite(value):
            lookup[(county, int(row[year_col]))] = value
    return lookup


def iter_patch_file_groups(
    files: Sequence[Path],
    max_counties: int | None,
    yield_lookup: dict[tuple[str, int], float],
    *,
    expected_timesteps: int,
    require_complete_schedule: bool,
    fast_filename_index: bool = False,
    yield_lookup_is_authoritative: bool = False,
) -> list[dict[str, Any]]:
    """Group per-interval files into deterministic county/patch sequences."""
    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    for path in files:
        try:
            with np.load(path, allow_pickle=True) as z:
                meta = metadata(z)
            match = COUNTY_FILE_RE.search(path.stem)
            county = normalise_county(_first(meta, ("county_fips", "county", "fips", "GEOID"), ""))
            year = int(safe_float(_first(meta, ("year",), 0), 0))
            interval = int(safe_float(_first(meta, ("interval_index", "interval", "timestep"), -1), -1))
            if match:
                county = county or normalise_county(match.group("county"))
                year = year or int(match.group("year"))
                interval = interval if interval >= 0 else int(match.group("interval"))
            if not county or year <= 0 or interval < 0:
                continue
            key = (county, year)
            if key not in grouped:
                if max_counties is not None and len(grouped) >= int(max_counties):
                    break
                # A supplied label table is authoritative. Embedded targets are
                # retained only as a synthetic/backwards-compatible fallback.
                embedded_yield = _first(
                    meta, ("yield_bu_per_acre", "yield", "yield_value"), np.nan
                )
                yield_bu = safe_float(
                    yield_lookup.get(
                        key,
                        np.nan if yield_lookup_is_authoritative else embedded_yield,
                    )
                )
                if not np.isfinite(yield_bu):
                    continue
                grouped[key] = {
                    "county_fips": county,
                    "year": year,
                    "crop": str(meta.get("crop", "corn")),
                    "yield_bu_per_acre": yield_bu,
                    "patch_files": defaultdict(dict),
                }
            patch_id = _patch_id(path, meta)
            grouped[key]["patch_files"][patch_id][interval] = path
        except Exception as exc:
            LOG.warning("Skipping unreadable NPZ %s: %s", path, exc)

    output: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group = grouped[key]
        present = sorted(
            {interval for by_interval in group["patch_files"].values() for interval in by_interval}
        )
        schedule = _expected_schedule(present, expected_timesteps)
        missing = sorted(set(schedule).difference(present))
        if missing and require_complete_schedule:
            continue
        entries = [
            (patch_id, dict(sorted(by_interval.items())))
            for patch_id, by_interval in sorted(group["patch_files"].items())
            if set(schedule).issubset(by_interval) or not require_complete_schedule
        ]
        if entries:
            item = dict(group)
            item.pop("patch_files")
            item["patch_entries"] = entries
            item["intervals"] = schedule
            output.append(item)
    return output


def _pick_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    lower = {column.lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def _first(mapping: dict[str, Any], keys: Sequence[str], default: Any) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return default


def _patch_id(path: Path, meta: dict[str, Any]) -> str:
    for key in ("patch_id", "patch_idx", "patch_index", "tile_id", "chip_id"):
        if key in meta and meta[key] not in (None, ""):
            return str(meta[key])
    match = PATCH_ID_RE.search(path.stem)
    if match:
        if match.group("patch"):
            return match.group("patch")
        if match.group("row") and match.group("col"):
            return f"r{match.group('row')}_c{match.group('col')}"
        if match.group("x") and match.group("y"):
            return f"x{match.group('x')}_y{match.group('y')}"
    return re.sub(r"interval[_-]?\d+", "interval", path.stem, flags=re.IGNORECASE)


def _expected_schedule(present: Sequence[int], timesteps: int) -> list[int]:
    if present and 0 not in present and min(present) >= 1 and max(present) <= timesteps:
        return list(range(1, timesteps + 1))
    return list(range(timesteps))
