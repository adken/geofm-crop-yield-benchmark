"""Leakage-safe train/validation/test partition loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class FoldPartitions:
    train: list[str]
    val: list[str]
    test: list[str]
    outer_fold: int
    validation_fold: int | None


def years_from_keys(keys: Iterable[str]) -> list[int]:
    """Return the sorted years encoded in canonical ``county-year`` keys."""
    values = pd.Series([str(key) for key in keys], dtype="string")
    if values.empty:
        raise ValueError("at least one county-year key is required")
    extracted = values.str.extract(r"^.+-(?P<year>\d{4})$")
    if extracted.isna().any(axis=None):
        bad = values[extracted["year"].isna()].tolist()
        raise ValueError(f"malformed county-year keys: {bad[:5]}")
    return sorted(extracted["year"].astype(int).unique().tolist())


def validate_all_years_in_partitions(
    parts: FoldPartitions,
    *,
    expected_years: Iterable[int],
) -> dict[str, list[int]]:
    """Require train, validation, and test to each contain every cohort year."""
    expected = sorted({int(year) for year in expected_years})
    if not expected:
        raise ValueError("expected_years must not be empty")
    observed = {
        "train": years_from_keys(parts.train),
        "validation": years_from_keys(parts.val),
        "test": years_from_keys(parts.test),
    }
    drift = {
        name: years
        for name, years in observed.items()
        if years != expected
    }
    if drift:
        raise ValueError(
            f"outer fold {parts.outer_fold} does not contain every cohort year "
            f"{expected} in each partition: {drift}"
        )
    return observed


def load_fold_partitions(
    path: str | Path,
    *,
    fold: int,
    id_column: str = "fips_year",
    validation_fold: int | None = None,
    validation_fold_offset: int = 1,
) -> FoldPartitions:
    """Load a split manifest while keeping the outer test fold untouched."""
    path = Path(path)
    frame = pd.read_csv(path)
    required = {id_column, "fold", "split"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    frame = frame.copy()
    frame[id_column] = frame[id_column].astype(str).str.replace("_", "-", regex=False)
    frame["split"] = frame["split"].astype(str).str.lower()
    frame.loc[frame["split"] == "validation", "split"] = "val"
    unexpected_splits = sorted(set(frame["split"]) - {"train", "val", "test"})
    if unexpected_splits:
        raise ValueError(f"{path} contains unsupported split labels {unexpected_splits}")
    if frame.duplicated([id_column, "fold"]).any():
        raise ValueError(f"{path} contains duplicate {id_column}/fold rows")
    if id_column == "fips_year":
        extracted = frame[id_column].str.extract(r"^(?P<county>.+)-(?P<year>\d{4})$")
        if extracted.isna().any(axis=None):
            raise ValueError(f"{path} has malformed fips_year values")
        frame["_county_group"] = extracted["county"]
        for manifest_fold, rows in frame.groupby("fold"):
            county_roles = rows.groupby("_county_group")["split"].nunique()
            leaked = sorted(county_roles[county_roles != 1].index.tolist())
            if leaked:
                raise ValueError(
                    f"{path} fold {manifest_fold} leaks counties across train/test "
                    "or train/val/test: "
                    f"{leaked[:5]}"
                )
        test_rows = frame[frame["split"] == "test"]
        assignments = test_rows.groupby("_county_group")["fold"].nunique()
        if (assignments != 1).any():
            bad = assignments[assignments != 1].index.tolist()
            raise ValueError(
                f"{path} assigns different years of a county to different test folds: "
                f"{bad[:5]}"
            )
    folds = sorted(int(value) for value in frame["fold"].unique())
    if fold not in folds:
        raise ValueError(f"outer fold {fold} is absent from {path}")
    outer = frame[frame["fold"] == fold]
    test = set(outer.loc[outer["split"] == "test", id_column])
    outer_train = set(outer.loc[outer["split"] == "train", id_column])
    explicit_val = set(outer.loc[outer["split"].isin({"val", "validation"}), id_column])
    selected_validation_fold: int | None = None
    if explicit_val:
        val = explicit_val
    else:
        if validation_fold is None:
            outer_position = folds.index(fold)
            selected_validation_fold = folds[
                (outer_position + int(validation_fold_offset)) % len(folds)
            ]
        else:
            selected_validation_fold = int(validation_fold)
        if selected_validation_fold == fold:
            raise ValueError("validation fold must differ from the outer test fold")
        validation_rows = frame[
            (frame["fold"] == selected_validation_fold) & (frame["split"] == "test")
        ]
        val = outer_train.intersection(validation_rows[id_column])
    train = outer_train.difference(val)
    if (train & val) or (train & test) or (val & test):
        raise ValueError(f"non-disjoint partitions in {path}")
    if not train or not val or not test:
        raise ValueError(
            f"empty partition from {path}: train={len(train)}, val={len(val)}, test={len(test)}"
        )
    return FoldPartitions(
        train=sorted(train),
        val=sorted(val),
        test=sorted(test),
        outer_fold=int(fold),
        validation_fold=selected_validation_fold,
    )
