"""Canonical county-year Daymet features for late fusion.

The manuscript's Presto+Daymet experiment does not send Daymet through
Presto's ERA5 input slot.  It concatenates 35 county-year covariates (five
variables across seven compositing intervals) after county pooling.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .data.io import normalise_county


DAYMET_VARIABLES = ("dayl", "prcp", "srad", "tmax", "tmin")
_FEATURE = re.compile(r"^(dayl|prcp|srad|tmax|tmin)_(\d+)$", re.IGNORECASE)


def _read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Daymet table does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(source)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    raise ValueError(f"unsupported Daymet table format: {source.suffix}")


def _pick(frame: pd.DataFrame, choices: Sequence[str]) -> str | None:
    lower = {str(column).strip().lower(): str(column) for column in frame.columns}
    return next((lower[value.lower()] for value in choices if value.lower() in lower), None)


def _county_name(values: pd.Series) -> pd.Series:
    return (
        values.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+county$", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
    )


def _fips_lookup(path: str | Path) -> pd.DataFrame:
    frame = _read_table(path)
    name_col = _pick(frame, ("county_name", "county", "name"))
    state_col = _pick(frame, ("statefp", "state_fips", "state ansi"))
    fips_col = _pick(frame, ("county_id", "fips", "geoid", "county_fips"))
    ansi_col = _pick(frame, ("county ansi", "county_ansi"))
    if name_col is None or state_col is None or (fips_col is None and ansi_col is None):
        raise ValueError(
            "Daymet FIPS map needs county name, statefp, and either county FIPS "
            "or County ANSI"
        )
    lookup = pd.DataFrame(
        {
            "_county_name": _county_name(frame[name_col]),
            "_statefp": pd.to_numeric(frame[state_col], errors="raise").astype(int),
        }
    )
    if fips_col is not None:
        lookup["county_id"] = frame[fips_col].map(normalise_county)
    else:
        state = pd.to_numeric(frame[state_col], errors="raise").astype(int).astype(str).str.zfill(2)
        ansi = pd.to_numeric(frame[ansi_col], errors="raise").astype(int).astype(str).str.zfill(3)
        lookup["county_id"] = state + ansi
    lookup = lookup.drop_duplicates()
    if lookup.duplicated(["_county_name", "_statefp"], keep=False).any():
        raise ValueError("Daymet FIPS map has ambiguous county/state rows")
    return lookup


def load_daymet_features(
    path: str | Path,
    *,
    fips_map: str | Path | None = None,
    expected_timesteps: int = 7,
) -> pd.DataFrame:
    """Load the manuscript's ordered 35-D county-year Daymet vector."""
    frame = _read_table(path)
    year_col = _pick(frame, ("year",))
    if year_col is None:
        raise ValueError("Daymet table needs a year column")

    fips_col = _pick(frame, ("county_id", "fips", "geoid", "county_fips"))
    unmapped_rows = 0
    if fips_col is not None:
        county_ids = frame[fips_col].map(normalise_county)
    else:
        name_col = _pick(frame, ("county_name", "county"))
        state_col = _pick(frame, ("statefp", "state_fips", "state ansi"))
        if name_col is None or state_col is None or fips_map is None:
            raise ValueError(
                "Daymet rows without county FIPS require county name, statefp, "
                "and --daymet-fips-map"
            )
        keys = pd.DataFrame(
            {
                "_county_name": _county_name(frame[name_col]),
                "_statefp": pd.to_numeric(frame[state_col], errors="raise").astype(int),
            }
        )
        mapped = keys.merge(
            _fips_lookup(fips_map),
            on=["_county_name", "_statefp"],
            how="left",
            validate="many_to_one",
        )
        keep = mapped["county_id"].notna().to_numpy()
        unmapped_rows = int((~keep).sum())
        if not keep.any():
            raise ValueError("Daymet FIPS map does not match any Daymet rows")
        frame = frame.loc[keep].reset_index(drop=True)
        county_ids = mapped.loc[keep, "county_id"].reset_index(drop=True)

    by_variable: dict[str, list[tuple[int, str]]] = {name: [] for name in DAYMET_VARIABLES}
    for column in frame.columns:
        match = _FEATURE.fullmatch(str(column).strip())
        if match:
            by_variable[match.group(1).lower()].append((int(match.group(2)), str(column)))
    schedules = []
    feature_names = []
    for variable in DAYMET_VARIABLES:
        entries = sorted(by_variable[variable])
        if len(entries) != int(expected_timesteps):
            raise ValueError(
                f"Daymet variable {variable} has {len(entries)} intervals; "
                f"expected {expected_timesteps}"
            )
        schedules.append(tuple(index for index, _ in entries))
        feature_names.extend(column for _, column in entries)
    if len(set(schedules)) != 1:
        raise ValueError(f"Daymet variables use inconsistent interval schedules: {schedules}")

    matrix = frame[feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    if not np.isfinite(matrix).all():
        bad = int((~np.isfinite(matrix)).any(axis=1).sum())
        raise ValueError(f"Daymet table has {bad} rows with missing/non-finite features")
    output = pd.DataFrame(
        {
            "county_id": county_ids.map(normalise_county),
            "year": pd.to_numeric(frame[year_col], errors="raise").astype(int),
            "daymet_features": list(matrix),
        }
    )
    if output["county_id"].eq("").any():
        raise ValueError("Daymet table contains an empty county FIPS")
    if output.duplicated(["county_id", "year"]).any():
        raise ValueError("Daymet table contains duplicate county-year rows")
    output.attrs["feature_names"] = feature_names
    output.attrs["interval_schedule"] = list(schedules[0])
    output.attrs["unmapped_rows_excluded"] = unmapped_rows
    return output


def fuse_daymet_features(
    county_features: pd.DataFrame,
    daymet: pd.DataFrame,
) -> pd.DataFrame:
    """Concatenate Daymet after county pooling, matching the manuscript."""
    if county_features.duplicated(["county_id", "year"]).any():
        raise ValueError("county features contain duplicate county-year rows")
    merged = county_features.merge(
        daymet[["county_id", "year", "daymet_features"]],
        on=["county_id", "year"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("Daymet has no county-years in common with the embedding cohort")
    merged["features"] = [
        np.concatenate(
            [np.asarray(embedding, dtype=np.float32), np.asarray(climate, dtype=np.float32)]
        )
        for embedding, climate in zip(merged["features"], merged["daymet_features"])
    ]
    return merged.drop(columns="daymet_features")
