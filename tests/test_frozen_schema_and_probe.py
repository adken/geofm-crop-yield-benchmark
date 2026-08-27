from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmark_embeddings.data import load_fold_partitions
from benchmark_embeddings.frozen import validate_embeddings
from benchmark_embeddings.probe import (
    AUXILIARY_CLIMATE_FUSION,
    MAIN_BENCHMARK,
    evaluate_fold,
    pool_county_embeddings,
    resolve_experiment_contract,
)


def _embedding_rows() -> pd.DataFrame:
    rows = []
    for county_index, county in enumerate(("17001", "17003", "17005", "17007", "17009", "17011")):
        for patch in range(1 + county_index % 2):
            for timestep in range(7):
                rows.append(
                    {
                        "county_id": county,
                        "year": 2020,
                        "patch_id": f"{county}-p{patch}",
                        "timestep": timestep,
                        "backbone": "test_backbone",
                        "embedding": [float(county_index), float(timestep), float(patch)],
                    }
                )
    return validate_embeddings(pd.DataFrame(rows))


def test_schema_accepts_variable_patch_counts_and_rejects_duplicate_keys() -> None:
    frame = _embedding_rows()
    counts = frame.groupby("county_id")["patch_id"].nunique().tolist()
    assert sorted(set(counts)) == [1, 2]

    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate embedding keys"):
        validate_embeddings(duplicate)


def test_county_pooling_is_invariant_to_row_order() -> None:
    frame = _embedding_rows()
    labels = pd.DataFrame(
        {
            "county_id": frame["county_id"].unique(),
            "year": 2020,
            "yield": np.arange(frame["county_id"].nunique(), dtype=float) + 140.0,
        }
    )
    first = pool_county_embeddings(frame, labels, spatial_pool="mean").sort_values("key")
    second = pool_county_embeddings(
        frame.sample(frac=1.0, random_state=7), labels, spatial_pool="mean"
    ).sort_values("key")
    np.testing.assert_allclose(
        np.stack(first["features"]),
        np.stack(second["features"]),
    )


def test_county_pooling_uses_only_complete_patch_sequences() -> None:
    frame = _embedding_rows()
    incomplete_patch = pd.DataFrame(
        [
            {
                "county_id": "17001",
                "year": 2020,
                "patch_id": "incomplete",
                "timestep": timestep,
                "backbone": "test_backbone",
                "embedding": [1000.0, 1000.0, 1000.0],
            }
            for timestep in range(6)
        ]
    )
    frame = validate_embeddings(pd.concat([frame, incomplete_patch], ignore_index=True))
    labels = pd.DataFrame(
        {
            "county_id": frame["county_id"].unique(),
            "year": 2020,
            "yield": 150.0,
        }
    )

    pooled = pool_county_embeddings(frame, labels, spatial_pool="mean").set_index("county_id")

    assert pooled.loc["17001", "n_patches"] == 1
    np.testing.assert_allclose(
        pooled.loc["17001", "features"],
        np.array([0.0, 3.0, 0.0], dtype=np.float32),
    )


def test_mean_std_pooling_is_spatial_then_temporal() -> None:
    rows = []
    for patch, offset in (("a", 0.0), ("b", 2.0)):
        for timestep in range(7):
            rows.append(
                {
                    "county_id": "17001",
                    "year": 2020,
                    "patch_id": patch,
                    "timestep": timestep,
                    "backbone": "test_backbone",
                    "embedding": [offset + timestep],
                }
            )
    frame = validate_embeddings(pd.DataFrame(rows))
    labels = pd.DataFrame(
        {"county_id": ["17001"], "year": [2020], "yield": [150.0]}
    )

    pooled = pool_county_embeddings(frame, labels)

    # Per timestep: patch mean is t+1 and population std is 1. Temporal mean
    # across t=0..6 therefore yields [4, 1].
    np.testing.assert_allclose(pooled.loc[0, "features"], [4.0, 1.0])


def test_sequence_scope_uses_one_presto_row_per_patch_without_requiring_seven_rows() -> None:
    frame = validate_embeddings(
        pd.DataFrame(
            [
                {
                    "county_id": "17001",
                    "year": 2020,
                    "patch_id": patch,
                    "timestep": 0,
                    "backbone": "presto_s2",
                    "embedding": [value],
                    "representation_scope": "sequence",
                }
                for patch, value in (("a", 2.0), ("b", 4.0))
            ]
        )
    )
    labels = pd.DataFrame(
        {"county_id": ["17001"], "year": [2020], "yield": [150.0]}
    )

    pooled = pool_county_embeddings(frame, labels)

    np.testing.assert_allclose(pooled.loc[0, "features"], [3.0, 1.0])
    assert pooled.loc[0, "n_patches"] == 2
    assert pooled.loc[0, "representation_scope"] == "sequence"


def test_fold_loader_and_probe_keep_test_out_of_alpha_selection(tmp_path: Path) -> None:
    frame = _embedding_rows()
    counties = frame["county_id"].unique().tolist()
    labels = pd.DataFrame(
        {
            "county_id": counties,
            "year": 2020,
            "yield": np.linspace(140.0, 180.0, len(counties)),
        }
    )
    data = pool_county_embeddings(frame, labels, spatial_pool="mean")
    keys = data["key"].tolist()
    manifest = pd.DataFrame(
        [
            *({"fips_year": key, "fold": 0, "split": "train"} for key in keys[:4]),
            *({"fips_year": key, "fold": 0, "split": "val"} for key in keys[4:5]),
            *({"fips_year": key, "fold": 0, "split": "test"} for key in keys[5:]),
        ]
    )
    split_path = tmp_path / "split.csv"
    manifest.to_csv(split_path, index=False)

    partitions = load_fold_partitions(split_path, fold=0)
    assert set(partitions.train).isdisjoint(partitions.test)
    result, predictions = evaluate_fold(
        data,
        split_path=split_path,
        fold=0,
        alphas=(0.1, 1.0),
    )

    assert result["selection"]["source"] == "validation"
    assert result["test"]["county_n"] == 1
    assert predictions["county_id"].tolist() == [counties[-1]]


def test_fold_loader_rejects_cross_year_county_leakage(tmp_path: Path) -> None:
    manifest = pd.DataFrame(
        [
            {"fips_year": "17001-2019", "fold": 0, "split": "test"},
            {"fips_year": "17001-2020", "fold": 0, "split": "train"},
            {"fips_year": "17003-2019", "fold": 0, "split": "train"},
        ]
    )
    split_path = tmp_path / "leaky.csv"
    manifest.to_csv(split_path, index=False)

    with pytest.raises(ValueError, match="leaks counties across train/test"):
        load_fold_partitions(split_path, fold=0)


def test_main_and_auxiliary_fusion_experiment_families_stay_separate() -> None:
    assert resolve_experiment_contract("presto_s2", daymet=False)["family"] == MAIN_BENCHMARK
    era5 = resolve_experiment_contract("presto_s2_era5", daymet=False)
    assert era5 == {
        "family": AUXILIARY_CLIMATE_FUSION,
        "climate_source": "ERA5-Land",
        "fusion_stage": "presto_encoder_input",
    }
    daymet = resolve_experiment_contract("clay_v1_5_cls", daymet=True)
    assert daymet["family"] == AUXILIARY_CLIMATE_FUSION
    assert daymet["fusion_stage"] == "county_year_late"

    with pytest.raises(ValueError, match=r"do not stack Daymet"):
        resolve_experiment_contract("presto_s2_era5", daymet=True)
