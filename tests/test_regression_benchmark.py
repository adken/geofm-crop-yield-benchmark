from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import benchmark_embeddings.regression_benchmark as benchmark_module
from benchmark_embeddings.data import (
    FoldPartitions,
    load_fold_partitions,
    validate_all_years_in_partitions,
)
from benchmark_embeddings.frozen import write_embeddings
from benchmark_embeddings.loyo import EXPECTED_BACKBONES, REPRESENTATIONS
from benchmark_embeddings.regression_benchmark import (
    CLIMATE_REPRESENTATIONS,
    FAMILY_CLIMATE,
    FAMILY_MAIN,
    load_all_fold_partitions,
    run_regression_benchmark,
)


COUNTIES = ("17001", "17003", "17005", "17007", "17009", "17011")
YEAR = 2020


def _embedding_rows(name: str, dimension: int, *, era5: bool = False) -> pd.DataFrame:
    backbone = "presto_s2_era5" if era5 else EXPECTED_BACKBONES[name]
    rows = []
    for county_index, county in enumerate(COUNTIES):
        base = float(county_index)
        if name in {"presto", "alphaearth"}:
            rows.append(
                {
                    "county_id": county,
                    "year": YEAR,
                    "patch_id": (
                        "shared_patch" if name == "presto" else f"alpha-{county}"
                    ),
                    "timestep": 0,
                    "backbone": backbone,
                    "embedding": [base + 0.01 * index for index in range(dimension)],
                    "representation_scope": "sequence",
                    "experiment_family": (
                        "auxiliary_climate_fusion" if era5 else "main_benchmark"
                    ),
                    "input_modalities": (
                        "Sentinel-2,ERA5-Land"
                        if era5
                        else (
                            "precomputed multimodal"
                            if name == "alphaearth"
                            else "Sentinel-2"
                        )
                    ),
                }
            )
        else:
            for timestep in range(7):
                rows.append(
                    {
                        "county_id": county,
                        "year": YEAR,
                        "patch_id": "shared_patch",
                        "timestep": timestep,
                        "backbone": backbone,
                        "embedding": [
                            base + 0.1 * timestep + 0.01 * index
                            for index in range(dimension)
                        ],
                        "representation_scope": "timestep",
                        "experiment_family": "main_benchmark",
                        "input_modalities": "Sentinel-2",
                    }
                )
    frame = pd.DataFrame(rows)
    if name == "prithvi":
        frame["temporal_ingestion"] = "single_timestep_independent"
    return frame


def _write_inputs(tmp_path: Path):
    paths = {}
    for index, name in enumerate(REPRESENTATIONS[:-1]):
        path = tmp_path / f"{name}.parquet"
        write_embeddings(_embedding_rows(name, index + 2), path)
        paths[name] = path
    era5_path = tmp_path / "presto_era5.parquet"
    write_embeddings(_embedding_rows("presto", 5, era5=True), era5_path)
    labels_path = tmp_path / "labels.csv"
    pd.DataFrame(
        {
            "county_id": COUNTIES,
            "year": YEAR,
            "yield": np.linspace(140.0, 190.0, len(COUNTIES)),
        }
    ).to_csv(labels_path, index=False)
    indices_path = tmp_path / "indices.csv"
    index_rows = []
    daymet_rows = []
    for county_index, county in enumerate(COUNTIES):
        index_row = {"county_id": county, "year": YEAR}
        for variable_index, variable in enumerate(("evi", "lai", "fpar")):
            for timestep in range(7):
                index_row[f"{variable}_{timestep}"] = (
                    county_index + variable_index + 0.1 * timestep
                )
        index_rows.append(index_row)
        daymet_row = {"county_id": county, "year": YEAR}
        for variable_index, variable in enumerate(
            ("dayl", "prcp", "srad", "tmax", "tmin")
        ):
            for timestep in range(7):
                daymet_row[f"{variable}_{timestep}"] = (
                    county_index + variable_index + 0.1 * timestep
                )
        daymet_rows.append(daymet_row)
    pd.DataFrame(index_rows).to_csv(indices_path, index=False)
    daymet_path = tmp_path / "daymet.csv"
    pd.DataFrame(daymet_rows).to_csv(daymet_path, index=False)
    split_path = tmp_path / "folds.csv"
    rows = []
    for fold, test_counties, val_county in (
        (0, COUNTIES[:3], COUNTIES[3]),
        (1, COUNTIES[3:], COUNTIES[0]),
    ):
        for county in COUNTIES:
            split = (
                "test"
                if county in test_counties
                else ("val" if county == val_county else "train")
            )
            rows.append(
                {"fips_year": f"{county}-{YEAR}", "fold": fold, "split": split}
            )
    pd.DataFrame(rows).to_csv(split_path, index=False)
    return paths, era5_path, labels_path, indices_path, daymet_path, split_path


def test_all_fold_loader_enforces_exact_outer_test_coverage(tmp_path: Path) -> None:
    _, _, _, _, _, split_path = _write_inputs(tmp_path)
    keys = [f"{county}-{YEAR}" for county in COUNTIES]

    parts = load_all_fold_partitions(
        split_path,
        expected_folds=(0, 1),
        cohort_keys=keys,
    )

    assert set(parts) == {0, 1}
    assert set(parts[0].test).isdisjoint(parts[1].test)
    assert set(parts[0].test) | set(parts[1].test) == set(keys)

    with pytest.raises(ValueError, match="split folds"):
        load_all_fold_partitions(
            split_path,
            expected_folds=(0,),
            cohort_keys=keys,
        )


def test_split_guards_require_county_roles_and_all_years() -> None:
    parts = FoldPartitions(
        train=["17001-2019", "17001-2020"],
        val=["17003-2019"],
        test=["17005-2019", "17005-2020"],
        outer_fold=0,
        validation_fold=None,
    )
    with pytest.raises(ValueError, match="does not contain every cohort year"):
        validate_all_years_in_partitions(parts, expected_years=(2019, 2020))


def test_split_loader_rejects_county_split_between_train_and_validation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "leaky.csv"
    pd.DataFrame(
        [
            {"fips_year": "17001-2019", "fold": 0, "split": "train"},
            {"fips_year": "17001-2020", "fold": 0, "split": "val"},
            {"fips_year": "17003-2019", "fold": 0, "split": "test"},
        ]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="train/val/test"):
        load_fold_partitions(path, fold=0)


def test_main_family_runs_identical_regressors_on_identical_folds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, _, labels, indices, _, split = _write_inputs(tmp_path)
    original_registry = benchmark_module.regressor_registry

    def fast_registry(*, n_jobs: int, **kwargs):
        # Forward anything the real registry gains (ebm_interactions, ...) so
        # this stub does not have to track its signature.
        registry = original_registry(n_jobs=n_jobs, **kwargs)
        registry["random_forest"]["n_estimators"] = 5
        return registry

    monkeypatch.setattr(benchmark_module, "regressor_registry", fast_registry)
    output = run_regression_benchmark(
        family=FAMILY_MAIN,
        embedding_paths=paths,
        sentinel2_indices_path=indices,
        labels_path=labels,
        split_path=split,
        out_dir=tmp_path / "main_results",
        folds=(0, 1),
        regressors=("ridge", "random_forest"),
        seeds=(0, 1),
        ridge_alphas=(0.1, 1.0),
        n_jobs=1,
    )

    expected_runs = len(REPRESENTATIONS) * 2 * (1 + 2)
    assert len(output["results"]) == expected_runs
    assert len(output["summary"]) == len(REPRESENTATIONS) * 2
    assert set(output["summary"].loc[output["summary"]["regressor"] == "ridge", "seeds_per_fold"]) == {1}
    assert set(output["summary"].loc[output["summary"]["regressor"] == "random_forest", "seeds_per_fold"]) == {2}
    assert set(output["summary"]["folds"]) == {2}
    assert set(output["summary"]["test_n_total"]) == {6}
    assert np.allclose(
        output["summary"]["rmse_bu_per_acre_mean"],
        output["summary"]["rmse_mean"],
    )
    assert output["data_contract"]["target_and_metric_units"]["canonical_yield"] == (
        "bushels_per_acre"
    )
    assert output["data_contract"]["split"]["grouping"] == "county_all_years_together"
    assert output["data_contract"]["estimator_family"] == "classical_ml"
    assert output["data_contract"]["workflow_role"] == (
        "main_encoder_embedding_comparison"
    )
    assert output["data_contract"]["feature_aggregation"]["operation_order"] == (
        "complete_patches_then_spatial_pool_per_timestep_then_temporal_pool"
    )
    assert output["data_contract"]["feature_aggregation"]["temporal_pool"] == "mean"
    assert output["data_contract"]["split"]["cohort_years"] == [YEAR]
    for filename in (
        "data_contract.json",
        "results_by_fold_and_seed.csv",
        "results_by_fold.csv",
        "summary_across_folds.csv",
        "predictions.csv",
    ):
        assert (tmp_path / "main_results" / filename).is_file()


def test_climate_family_is_separate_but_has_the_same_regressor_and_fold_contract(
    tmp_path: Path,
) -> None:
    paths, era5, labels, indices, daymet, split = _write_inputs(tmp_path)
    daymet_frame = pd.read_csv(daymet)
    extra = daymet_frame.iloc[[0]].copy()
    extra["county_id"] = "99999"
    pd.concat([daymet_frame, extra], ignore_index=True).to_csv(daymet, index=False)
    climate_paths = {name: paths[name] for name in ("clay", "prithvi", "terramind", "presto")}
    output = run_regression_benchmark(
        family=FAMILY_CLIMATE,
        embedding_paths=climate_paths,
        presto_era5_path=era5,
        sentinel2_indices_path=indices,
        daymet_path=daymet,
        labels_path=labels,
        split_path=split,
        out_dir=tmp_path / "climate_results",
        folds=(0, 1),
        regressors=("ridge",),
        seeds=(0, 1),
        ridge_alphas=(0.1, 1.0),
        n_jobs=1,
    )

    assert set(output["summary"]["representation"]) == set(CLIMATE_REPRESENTATIONS)
    assert set(output["summary"]["regressor"]) == {"ridge"}
    assert set(output["summary"]["folds"]) == {2}
    era5_contract = output["data_contract"]["representations"]["presto_era5"]
    assert era5_contract["fusion_stage"] == "presto_encoder_input"
    assert era5_contract["added_regressor_feature_count"] == 0
    daymet_contract = output["data_contract"]["representations"]["clay_daymet"]
    assert daymet_contract["added_regressor_feature_count"] == 35
    assert daymet_contract["daymet_rows_outside_benchmark_excluded"] == 1


def test_main_family_rejects_representation_cohort_drift(tmp_path: Path) -> None:
    paths, _, labels, indices, _, split = _write_inputs(tmp_path)
    alpha = pd.read_parquet(paths["alphaearth"]).iloc[:-1]
    alpha.to_parquet(paths["alphaearth"], index=False)

    with pytest.raises(ValueError, match="cohort mismatch for alphaearth"):
        run_regression_benchmark(
            family=FAMILY_MAIN,
            embedding_paths=paths,
            sentinel2_indices_path=indices,
            labels_path=labels,
            split_path=split,
            out_dir=tmp_path / "unused",
            folds=(0, 1),
            regressors=("ridge",),
            seeds=(0,),
            ridge_alphas=(1.0,),
            n_jobs=1,
            preflight_only=True,
        )
