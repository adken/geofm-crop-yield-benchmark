"""Canonical 21-D Sentinel-2 vegetation-index county-year baseline."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .data.io import normalise_county
from .daymet import _county_name, _fips_lookup, _pick, _read_table


S2_INDEX_VARIABLES = ("evi", "lai", "fpar")
_FEATURE = re.compile(r"^(evi|lai|fpar)_(\d+)$", re.IGNORECASE)


def load_s2_index_features(
    path: str | Path,
    *,
    fips_map: str | Path | None = None,
    expected_timesteps: int = 7,
) -> pd.DataFrame:
    """Load EVI, LAI, and fPAR across the seven benchmark intervals."""
    frame = _read_table(path)
    year_col = _pick(frame, ("year",))
    if year_col is None:
        raise ValueError("Sentinel-2 index table needs a year column")
    fips_col = _pick(frame, ("county_id", "fips", "geoid", "county_fips"))
    unmapped_rows = 0
    if fips_col is not None:
        county_ids = frame[fips_col].map(normalise_county)
    else:
        name_col = _pick(frame, ("county_name", "county"))
        state_col = _pick(frame, ("statefp", "state_fips", "state ansi"))
        if name_col is None or state_col is None or fips_map is None:
            raise ValueError(
                "Sentinel-2 index rows without county FIPS require county name, "
                "statefp, and --s2-indices-fips-map"
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
            raise ValueError("Sentinel-2 index FIPS map does not match any source row")
        frame = frame.loc[keep].reset_index(drop=True)
        county_ids = mapped.loc[keep, "county_id"].reset_index(drop=True)

    by_variable: dict[str, list[tuple[int, str]]] = {
        name: [] for name in S2_INDEX_VARIABLES
    }
    for column in frame.columns:
        match = _FEATURE.fullmatch(str(column).strip())
        if match:
            by_variable[match.group(1).lower()].append((int(match.group(2)), str(column)))
    schedules = []
    feature_names = []
    for variable in S2_INDEX_VARIABLES:
        entries = sorted(by_variable[variable])
        if len(entries) != int(expected_timesteps):
            raise ValueError(
                f"Sentinel-2 index {variable} has {len(entries)} intervals; "
                f"expected {expected_timesteps}"
            )
        schedules.append(tuple(index for index, _ in entries))
        feature_names.extend(column for _, column in entries)
    if len(set(schedules)) != 1:
        raise ValueError(
            f"Sentinel-2 indices use inconsistent interval schedules: {schedules}"
        )
    matrix = frame[feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    complete = np.isfinite(matrix).all(axis=1)
    incomplete_rows = int((~complete).sum())
    if not complete.any():
        raise ValueError("Sentinel-2 index table has no complete seven-interval rows")
    frame = frame.loc[complete].reset_index(drop=True)
    county_ids = county_ids.loc[complete].reset_index(drop=True)
    matrix = matrix[complete]
    output = pd.DataFrame(
        {
            "county_id": county_ids.map(normalise_county),
            "year": pd.to_numeric(frame[year_col], errors="raise").astype(int),
            "features": list(matrix),
            "n_patches": 0,
            "representation_scope": "county_year",
        }
    )
    if output.duplicated(["county_id", "year"]).any():
        raise ValueError("Sentinel-2 index table contains duplicate county-year rows")
    output.attrs["feature_names"] = feature_names
    output.attrs["interval_schedule"] = list(schedules[0])
    output.attrs["unmapped_rows_excluded"] = unmapped_rows
    output.attrs["incomplete_rows_excluded"] = incomplete_rows
    return output
