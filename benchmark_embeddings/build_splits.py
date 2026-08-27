#!/usr/bin/env python3
"""Build the canonical five-fold county-grouped benchmark manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .data.io import normalise_county
from .daymet import load_daymet_features
from .frozen import read_embeddings
from .frozen.presto import PrestoPatchDataset
from .s2_indices import load_s2_index_features


DEFAULT_YEARS = (2019, 2020, 2021, 2022)


def _pick(frame: pd.DataFrame, names: Sequence[str]) -> str | None:
    lower = {str(column).strip().lower(): str(column) for column in frame.columns}
    return next((lower[name.lower()] for name in names if name.lower() in lower), None)


def _label_keys(path: str | Path, years: set[int]) -> set[str]:
    frame = pd.read_csv(path)
    county = _pick(frame, ("county_id", "county", "county_fips", "fips", "geoid"))
    year = _pick(frame, ("year",))
    target = _pick(frame, ("yield_bu_per_acre", "yield", "value", "observed_yield"))
    if county is None or year is None or target is None:
        raise ValueError("yield labels need county, year, and yield columns")
    normalized = pd.DataFrame(
        {
            "county_id": frame[county].map(normalise_county),
            "year": pd.to_numeric(frame[year], errors="raise").astype(int),
            "yield": pd.to_numeric(frame[target], errors="coerce"),
        }
    ).dropna()
    normalized = normalized.loc[normalized["year"].isin(years)]
    if normalized.duplicated(["county_id", "year"]).any():
        raise ValueError("yield labels contain duplicate county-year rows")
    return set(normalized["county_id"] + "-" + normalized["year"].astype(str))


def _alphaearth_keys(path: str | Path, years: set[int]) -> set[str]:
    frame = read_embeddings(path)
    if sorted(frame["backbone"].astype(str).unique()) != ["alphaearth"]:
        raise ValueError("canonical AlphaEarth input must use backbone='alphaearth'")
    selected = frame.loc[frame["year"].astype(int).isin(years)]
    if selected.duplicated(["county_id", "year"]).any():
        raise ValueError("canonical AlphaEarth input contains duplicate county-years")
    return set(selected["county_id"] + "-" + selected["year"].astype(str))


def _source_patch_keys(
    s2_dir: str | Path,
    *,
    years: set[int],
    expected_input_count: int | None,
) -> tuple[set[str], dict[str, tuple[str, ...]], dict[str, Any]]:
    dataset = PrestoPatchDataset(
        s2_dir,
        expected_input_count=expected_input_count,
        expected_timesteps=7,
        timestep_base="auto",
        require_band_names=False,
        s2_units="auto",
    )
    patch_ids: dict[str, list[str]] = {}
    for item in dataset.indices:
        if item.year not in years:
            continue
        key = f"{item.county_id}-{item.year}"
        patch_ids.setdefault(key, []).append(item.patch_id)
    frozen = {key: tuple(sorted(values)) for key, values in patch_ids.items()}
    return set(frozen), frozen, dataset.describe()


def _hash_lines(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def build_common_split_manifest(
    *,
    s2_dir: str | Path,
    s2_daymet_merged: str | Path,
    s2_fips_map: str | Path | None,
    alphaearth_path: str | Path,
    labels_path: str | Path,
    output_path: str | Path,
    years: Sequence[int] = DEFAULT_YEARS,
    n_splits: int = 5,
    validation_fold_offset: int = 1,
    expected_input_count: int | None = 77813,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    years = tuple(sorted(int(value) for value in years))
    if not years or len(set(years)) != len(years):
        raise ValueError("years must be unique and non-empty")
    if int(n_splits) < 3:
        raise ValueError("at least three folds are required for train/val/test roles")
    if int(validation_fold_offset) % int(n_splits) == 0:
        raise ValueError("validation fold offset cannot select the outer test fold")
    year_set = set(years)

    patch_keys, patch_ids, patch_contract = _source_patch_keys(
        s2_dir,
        years=year_set,
        expected_input_count=expected_input_count,
    )
    s2 = load_s2_index_features(
        s2_daymet_merged,
        fips_map=s2_fips_map,
        expected_timesteps=7,
    )
    s2 = s2.loc[s2["year"].isin(year_set)]
    s2_keys = set(s2["county_id"] + "-" + s2["year"].astype(str))
    daymet = load_daymet_features(
        s2_daymet_merged,
        fips_map=s2_fips_map,
        expected_timesteps=7,
    )
    daymet = daymet.loc[daymet["year"].isin(year_set)]
    daymet_keys = set(daymet["county_id"] + "-" + daymet["year"].astype(str))
    alphaearth_keys = _alphaearth_keys(alphaearth_path, year_set)
    label_keys = _label_keys(labels_path, year_set)
    source_sets = {
        "complete_patch_sequences": patch_keys,
        "sentinel2_indices": s2_keys,
        "daymet": daymet_keys,
        "alphaearth": alphaearth_keys,
        "yield_labels": label_keys,
    }
    common = set.intersection(*source_sets.values())
    if not common:
        raise ValueError("the benchmark sources have no common county-year cohort")
    keys = sorted(common)
    cohort = pd.DataFrame(
        {
            "fips_year": keys,
            "county_id": [key.split("-", 1)[0] for key in keys],
            "year": [int(key.split("-", 1)[1]) for key in keys],
        }
    )
    if cohort["county_id"].nunique() < int(n_splits):
        raise ValueError("common cohort has fewer counties than requested folds")

    outer = GroupKFold(n_splits=int(n_splits))
    fold_by_county: dict[str, int] = {}
    dummy = np.zeros(len(cohort), dtype=np.float32)
    for fold, (_, test_indices) in enumerate(
        outer.split(dummy, groups=cohort["county_id"].to_numpy())
    ):
        for county in cohort.iloc[test_indices]["county_id"].unique():
            if county in fold_by_county:
                raise RuntimeError(f"county {county} was assigned to multiple outer folds")
            fold_by_county[county] = int(fold)
    if set(fold_by_county) != set(cohort["county_id"]):
        raise RuntimeError("not every common-cohort county received an outer fold")

    rows = []
    partition_counts: dict[str, dict[str, int]] = {}
    for fold in range(int(n_splits)):
        validation_fold = (fold + int(validation_fold_offset)) % int(n_splits)
        counts = {"train": 0, "val": 0, "test": 0}
        years_by_role = {"train": set(), "val": set(), "test": set()}
        for item in cohort.itertuples(index=False):
            county_fold = fold_by_county[item.county_id]
            role = (
                "test"
                if county_fold == fold
                else ("val" if county_fold == validation_fold else "train")
            )
            counts[role] += 1
            years_by_role[role].add(int(item.year))
            rows.append(
                {
                    "fips_year": item.fips_year,
                    "county_id": item.county_id,
                    "year": int(item.year),
                    "fold": fold,
                    "split": role,
                    "county_outer_fold": county_fold,
                    "validation_fold": validation_fold,
                }
            )
        for role, observed_years in years_by_role.items():
            if observed_years != year_set:
                raise ValueError(
                    f"fold {fold} {role} years are {sorted(observed_years)}, "
                    f"expected {list(years)}"
                )
        partition_counts[str(fold)] = counts
    manifest = pd.DataFrame(rows).sort_values(["fold", "fips_year"]).reset_index(drop=True)
    if manifest.duplicated(["fold", "fips_year"]).any():
        raise RuntimeError("generated split manifest contains duplicate fold/key rows")

    patch_hash_lines = [
        f"{key}\x1f{'\x1f'.join(patch_ids[key])}" for key in keys
    ]
    contract = {
        "schema_version": 1,
        "split_method": "sklearn.model_selection.GroupKFold",
        "grouping": "county_all_years_together",
        "shuffle": False,
        "n_splits": int(n_splits),
        "validation_fold_offset": int(validation_fold_offset),
        "years": list(years),
        "county_years": len(keys),
        "counties": int(cohort["county_id"].nunique()),
        "cohort_sha256": _hash_lines(keys),
        "complete_patch_identity_sha256": _hash_lines(patch_hash_lines),
        "source_counts": {name: len(values) for name, values in source_sets.items()},
        "source_rows_excluded": {
            name: len(values - common) for name, values in source_sets.items()
        },
        "partition_counts": partition_counts,
        "source_patch_contract": patch_contract,
        "inputs": {
            "sentinel2_npz": str(Path(s2_dir).resolve()),
            "s2_daymet_merged": str(Path(s2_daymet_merged).resolve()),
            "s2_fips_map": str(Path(s2_fips_map).resolve()) if s2_fips_map else None,
            "alphaearth": str(Path(alphaearth_path).resolve()),
            "yield_labels": str(Path(labels_path).resolve()),
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    output.with_suffix(output.suffix + ".contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )
    return manifest, contract


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s2-dir", required=True)
    parser.add_argument("--s2-daymet-merged", required=True)
    parser.add_argument("--s2-fips-map")
    parser.add_argument("--alphaearth", required=True, help="Canonical AlphaEarth Parquet")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--validation-fold-offset", type=int, default=1)
    parser.add_argument("--expected-input-count", type=int, default=77813)
    args = parser.parse_args(argv)
    _, contract = build_common_split_manifest(
        s2_dir=args.s2_dir,
        s2_daymet_merged=args.s2_daymet_merged,
        s2_fips_map=args.s2_fips_map,
        alphaearth_path=args.alphaearth,
        labels_path=args.labels,
        output_path=args.output,
        years=args.years,
        n_splits=args.folds,
        validation_fold_offset=args.validation_fold_offset,
        expected_input_count=args.expected_input_count,
    )
    print(json.dumps(contract, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
