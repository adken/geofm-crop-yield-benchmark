from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmark_embeddings.frozen import adapt_alphaearth_csv, write_embeddings
from benchmark_embeddings.loyo import (
    DEFAULT_YEARS,
    EXPECTED_BACKBONES,
    REPRESENTATIONS,
    run_main_benchmark_loyo,
    validate_unfused_main_embedding,
)


COUNTIES = ("17001", "17003", "17005")


def _embedding_rows(name: str, dimension: int) -> pd.DataFrame:
    rows = []
    backbone = EXPECTED_BACKBONES[name]
    for year_index, year in enumerate(DEFAULT_YEARS):
        for county_index, county in enumerate(COUNTIES):
            base = float(year_index + county_index)
            if name in {"presto", "alphaearth"}:
                rows.append(
                    {
                        "county_id": county,
                        "year": year,
                        "patch_id": (
                            "shared_patch" if name == "presto" else f"alphaearth-{county}-{year}"
                        ),
                        "timestep": 0,
                        "backbone": backbone,
                        "embedding": [base + 0.01 * index for index in range(dimension)],
                        "representation_scope": "sequence",
                        "experiment_family": "main_benchmark",
                        "input_modalities": (
                            "Sentinel-2" if name == "presto" else "precomputed multimodal"
                        ),
                    }
                )
            else:
                for timestep in range(7):
                    rows.append(
                        {
                            "county_id": county,
                            "year": year,
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


def _write_loyo_inputs(tmp_path: Path) -> tuple[dict[str, Path], Path, Path]:
    paths = {}
    for index, name in enumerate(REPRESENTATIONS[:-1]):
        path = tmp_path / f"{name}.parquet"
        write_embeddings(_embedding_rows(name, dimension=index + 2), path)
        paths[name] = path
    labels_path = tmp_path / "labels.csv"
    labels = []
    index_rows = []
    for year_index, year in enumerate(DEFAULT_YEARS):
        for county_index, county in enumerate(COUNTIES):
            value = 140.0 + 5.0 * year_index + 2.0 * county_index
            labels.append(
                {"county_id": county, "year": year, "yield": value}
            )
            row = {"county_id": county, "year": year}
            for variable_index, variable in enumerate(("evi", "lai", "fpar")):
                for timestep in range(7):
                    row[f"{variable}_{timestep}"] = (
                        value + variable_index + 0.1 * timestep
                    )
            index_rows.append(row)
    pd.DataFrame(labels).to_csv(labels_path, index=False)
    indices_path = tmp_path / "s2_indices.csv"
    pd.DataFrame(index_rows).to_csv(indices_path, index=False)
    return paths, labels_path, indices_path


def test_alphaearth_adapter_marks_annual_rows_as_sequence_scope(tmp_path: Path) -> None:
    source = tmp_path / "alpha.csv"
    pd.DataFrame(
        {
            "county_id": ["17001"],
            "year": [2020],
            "mean_A00": [1.0],
            "mean_A01": [2.0],
        }
    ).to_csv(source, index=False)

    adapted = adapt_alphaearth_csv(source)

    assert adapted.loc[0, "representation_scope"] == "sequence"


def test_alphaearth_adapter_prefers_geoid_over_county_name(tmp_path: Path) -> None:
    source = tmp_path / "alpha_with_names.csv"
    pd.DataFrame(
        {
            "GEOID": [17001, 19001],
            "county": ["ADAMS", "ADAIR"],
            "year": [2020, 2020],
            "mean_A00": [1.0, 2.0],
        }
    ).to_csv(source, index=False)

    adapted = adapt_alphaearth_csv(source)

    assert adapted["county_id"].tolist() == ["17001", "19001"]


def test_climate_free_guard_rejects_era5_and_daymet_inputs() -> None:
    era5 = _embedding_rows("presto", 2)
    era5["backbone"] = "presto_s2_era5"
    with pytest.raises(ValueError, match="requires backbone 'presto_s2'"):
        validate_unfused_main_embedding("presto", era5)

    daymet = _embedding_rows("presto", 2)
    daymet["input_modalities"] = "Sentinel-2,Daymet"
    with pytest.raises(ValueError, match="added climate inputs"):
        validate_unfused_main_embedding("presto", daymet)


def test_prithvi_guard_rejects_joint_sequence_inference() -> None:
    joint = _embedding_rows("prithvi", 2)
    joint["temporal_ingestion"] = "joint_sequence_per_frame_readout"
    with pytest.raises(ValueError, match="single_timestep_independent"):
        validate_unfused_main_embedding("prithvi", joint)


def test_main_benchmark_loyo_is_matched_and_climate_free(tmp_path: Path) -> None:
    paths, labels_path, indices_path = _write_loyo_inputs(tmp_path)
    out_dir = tmp_path / "loyo"

    output = run_main_benchmark_loyo(
        paths,
        sentinel2_indices_path=indices_path,
        labels_path=labels_path,
        out_dir=out_dir,
        years=DEFAULT_YEARS,
        seeds=(0,),
        n_estimators=5,
        n_jobs=1,
    )

    expected_runs = len(REPRESENTATIONS) * len(DEFAULT_YEARS)
    assert len(output["results"]) == expected_runs
    assert len(output["by_year"]) == expected_runs
    assert len(output["summary"]) == len(REPRESENTATIONS)
    assert set(output["summary"]["held_out_years"]) == {4}
    assert set(output["summary"]["county_years_tested"]) == {12}
    assert len(output["predictions"]) == expected_runs * len(COUNTIES)
    assert output["data_contract"]["fusion_contract"] == {
        "added_climate_features": [],
        "daymet_late_fusion": False,
        "presto_era5_input_fusion": False,
        "alphaearth_note": "precomputed_multimodal_reference_not_added_fusion",
    }
    assert output["data_contract"]["workflow_role"] == "temporal_generalization"
    assert output["data_contract"]["feature_aggregation"]["operation_order"] == (
        "complete_patches_then_spatial_pool_per_timestep_then_temporal_pool"
    )
    assert output["data_contract"]["feature_aggregation"]["temporal_pool"] == "mean"
    assert output["data_contract"]["evaluation"]["regressor_key"] == "random_forest"
    assert output["data_contract"]["evaluation"]["model_selection"] == (
        "none_fixed_protocol"
    )
    assert set(output["results"]["regressor"]) == {"random_forest"}
    assert set(output["predictions"]["regressor"]) == {"random_forest"}
    assert len(output["data_contract"]["matched_key_sha256"]) == 64
    assert len(
        output["data_contract"]["matched_complete_patch_identity_sha256"]
    ) == 64
    for filename in (
        "data_contract.json",
        "results_by_year_and_seed.csv",
        "results_by_year.csv",
        "summary_across_years.csv",
        "predictions.csv",
    ):
        assert (out_dir / filename).is_file()


def test_loyo_rejects_county_year_cohort_drift(tmp_path: Path) -> None:
    paths, labels_path, indices_path = _write_loyo_inputs(tmp_path)
    alpha = pd.read_parquet(paths["alphaearth"])
    alpha = alpha.iloc[:-1].copy()
    alpha.to_parquet(paths["alphaearth"], index=False)

    with pytest.raises(ValueError, match="cohort mismatch for alphaearth"):
        run_main_benchmark_loyo(
            paths,
            sentinel2_indices_path=indices_path,
            labels_path=labels_path,
            out_dir=tmp_path / "unused",
            years=DEFAULT_YEARS,
            seeds=(0,),
            n_estimators=5,
            n_jobs=1,
            preflight_only=True,
        )


def test_loyo_contract_records_the_pooling_that_was_used(tmp_path: Path) -> None:
    """The contract must follow --temporal-pool, not a hard-coded default.

    Regression test: the contract was built by calling the aggregation helper
    with no argument, so a joint-pooled run wrote a contract claiming two-stage
    pooling. The numbers and the provenance disagreed, which is precisely what
    these contracts exist to prevent.
    """
    paths, labels_path, indices_path = _write_loyo_inputs(tmp_path)

    for pool, expected_order in (
        ("mean", "complete_patches_then_spatial_pool_per_timestep_then_temporal_pool"),
        ("joint", "complete_patches_then_joint_pool_over_patch_timestep_rows"),
    ):
        output = run_main_benchmark_loyo(
            paths,
            sentinel2_indices_path=indices_path,
            labels_path=labels_path,
            out_dir=tmp_path / f"loyo_{pool}",
            years=DEFAULT_YEARS,
            seeds=(0,),
            n_estimators=5,
            n_jobs=1,
            temporal_pool=pool,
        )
        aggregation = output["data_contract"]["feature_aggregation"]
        assert aggregation["temporal_pool"] == pool
        assert aggregation["operation_order"] == expected_order
        for name, contract in output["data_contract"]["representations"].items():
            # The Sentinel-2 index baseline arrives preaggregated to county-year
            # and never enters the pooling code, so it records neither field.
            if contract.get("temporal_pool") == "preaggregated_21d":
                assert "temporal_pool_requested" not in contract, name
                continue
            assert contract["temporal_pool_requested"] == pool, name
            # Sequence-scoped representations pool an axis of length one, so the
            # applied pool is an identity regardless of what was requested.
            if contract["temporal_pool"] != "preencoded_global_pool":
                assert contract["temporal_pool"] == pool, name
