from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from benchmark_embeddings.frozen import write_embeddings
from benchmark_embeddings.loyo import EXPECTED_BACKBONES
from benchmark_embeddings.probe import AGGREGATION_OPERATION_ORDER
from benchmark_embeddings.temporal_ablation import (
    ENCODERS,
    STRATEGIES,
    SequenceStandardizer,
    TemporalAblationRegressor,
    run_temporal_ablation,
    validate_matched_cohorts,
)
from benchmark_embeddings.temporal_ablation_aggregate import aggregate_temporal_folds


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_temporal_readouts_have_a_shared_output_contract_and_gradients(
    strategy: str,
) -> None:
    model = TemporalAblationRegressor(
        strategy=strategy,
        timesteps=7,
        feature_dim=6,
        readout_dim=8,
        conv_channels=5,
        mlp_hidden=4,
        dropout=0.0,
    )
    sequence = torch.randn(3, 7, 6)

    prediction = model(sequence)
    prediction.square().mean().backward()

    assert prediction.shape == (3,)
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_sequence_standardization_is_fit_on_training_counties_only() -> None:
    train = np.arange(3 * 7 * 2, dtype=np.float32).reshape(3, 7, 2)
    held_out = np.full((2, 7, 2), 1_000_000.0, dtype=np.float32)

    scaler = SequenceStandardizer.fit(train)
    transformed = scaler.transform(np.concatenate([train, held_out], axis=0))

    np.testing.assert_allclose(scaler.mean, train.mean(axis=0), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(transformed[:3].mean(axis=0), 0.0, atol=1e-6)
    assert transformed[3:].min() > 1_000.0


def _cohort_frame(*, drop_last: bool = False, changed_count: bool = False) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "key": ["17001-2020", "17003-2020"],
            "n_patches": [1, 2],
            "complete_patch_ids": [("a",), ("b", "c")],
            "yield_bu_per_acre": [150.0, 160.0],
        }
    )
    if drop_last:
        return frame.iloc[:-1].copy()
    if changed_count:
        frame.loc[1, "n_patches"] = 3
    return frame


def test_matched_cohort_guard_rejects_key_or_patch_count_drift() -> None:
    exact = {name: _cohort_frame() for name in ENCODERS}
    assert validate_matched_cohorts(exact) == ["17001-2020", "17003-2020"]

    missing = dict(exact)
    missing["prithvi"] = _cohort_frame(drop_last=True)
    with pytest.raises(ValueError, match="cohort mismatch"):
        validate_matched_cohorts(missing)

    changed = dict(exact)
    changed["terramind"] = _cohort_frame(changed_count=True)
    with pytest.raises(ValueError, match="patch-count mismatch"):
        validate_matched_cohorts(changed)

    changed_identity = dict(exact)
    changed_identity["terramind"] = _cohort_frame()
    changed_identity["terramind"].at[1, "complete_patch_ids"] = ("b", "d")
    with pytest.raises(ValueError, match="patch-identity mismatch"):
        validate_matched_cohorts(changed_identity)


def _embedding_table(name: str, dimension: int) -> pd.DataFrame:
    rows = []
    counties = ("17001", "17003", "17005", "17007", "17009", "17011")
    for county_index, county in enumerate(counties):
        patch_count = 1 + county_index % 2
        for patch_index in range(patch_count):
            for timestep in range(7):
                base = county_index + 0.2 * timestep + 0.1 * patch_index
                rows.append(
                    {
                        "county_id": county,
                        "year": 2020,
                        "patch_id": f"{county}_p{patch_index}",
                        "timestep": timestep,
                        "backbone": EXPECTED_BACKBONES[name],
                        "embedding": [base + 0.01 * index for index in range(dimension)],
                        "representation_scope": "timestep",
                    }
                )
    frame = pd.DataFrame(rows)
    if name == "prithvi":
        frame["temporal_ingestion"] = "single_timestep_independent"
    return frame


def test_end_to_end_ablation_writes_all_matched_runs(tmp_path: Path) -> None:
    embedding_paths = {}
    for name, dimension in zip(ENCODERS, (2, 3, 4), strict=True):
        path = tmp_path / f"{name}.parquet"
        write_embeddings(_embedding_table(name, dimension), path)
        embedding_paths[name] = path

    counties = ("17001", "17003", "17005", "17007", "17009", "17011")
    labels_path = tmp_path / "labels.csv"
    pd.DataFrame(
        {
            "county_id": counties,
            "year": 2020,
            "yield": np.linspace(140.0, 190.0, len(counties)),
        }
    ).to_csv(labels_path, index=False)
    split_path = tmp_path / "split.csv"
    pd.DataFrame(
        [
            *(
                {"fips_year": f"{county}-2020", "fold": 0, "split": "train"}
                for county in counties[:4]
            ),
            {"fips_year": f"{counties[4]}-2020", "fold": 0, "split": "val"},
            {"fips_year": f"{counties[5]}-2020", "fold": 0, "split": "test"},
        ]
    ).to_csv(split_path, index=False)
    out_dir = tmp_path / "results"

    output = run_temporal_ablation(
        embedding_paths,
        labels_path=labels_path,
        split_path=split_path,
        fold=0,
        out_dir=out_dir,
        seeds=(0,),
        readout_dim=4,
        conv_channels=3,
        mlp_hidden=3,
        dropout=0.0,
        batch_size=4,
        max_epochs=2,
        patience=2,
        device="cpu",
    )

    assert len(output["results"]) == len(ENCODERS) * len(STRATEGIES)
    assert len(output["summary"]) == len(ENCODERS) * len(STRATEGIES)
    assert len(output["predictions"]) == len(ENCODERS) * len(STRATEGIES)
    assert set(output["predictions"]["key"]) == {"17011-2020"}
    assert output["data_contract"]["matched_county_years"] == len(counties)
    assert output["data_contract"]["workflow_role"] == "temporal_pooling_ablation"
    assert output["data_contract"]["prediction_head"] == "mlp"
    assert output["data_contract"]["feature_aggregation"]["operation_order"] == (
        "complete_patches_then_spatial_pool_per_timestep_then_temporal_pool"
    )
    assert set(
        output["data_contract"]["feature_aggregation"]["temporal_pool_strategies"]
    ) == {"mean", "concat", "conv1d"}
    assert output["data_contract"]["protocol"]["prediction_head_scope"] == (
        "temporal_pooling_ablation_only"
    )
    assert set(output["results"]["prediction_head"]) == {"mlp"}
    assert len(output["data_contract"]["matched_key_sha256"]) == 64
    for filename in (
        "data_contract.json",
        "results_by_seed.csv",
        "predictions.csv",
        "summary.csv",
    ):
        assert (out_dir / filename).is_file()
    assert len(list((out_dir / "checkpoints").glob("*.pt"))) == 9


def test_ablation_rejects_county_years_not_in_the_fold(tmp_path: Path) -> None:
    # The exact encoder cohort is not silently reduced to the split manifest.
    embedding_paths = {}
    for name in ENCODERS:
        path = tmp_path / f"{name}.parquet"
        write_embeddings(_embedding_table(name, 2), path)
        embedding_paths[name] = path
    labels_path = tmp_path / "labels.csv"
    counties = ("17001", "17003", "17005", "17007", "17009", "17011")
    pd.DataFrame(
        {"county_id": counties, "year": 2020, "yield": np.arange(140.0, 146.0)}
    ).to_csv(labels_path, index=False)
    split_path = tmp_path / "split.csv"
    pd.DataFrame(
        [
            {"fips_year": "17001-2020", "fold": 0, "split": "train"},
            {"fips_year": "17003-2020", "fold": 0, "split": "val"},
            {"fips_year": "17005-2020", "fold": 0, "split": "test"},
        ]
    ).to_csv(split_path, index=False)

    with pytest.raises(ValueError, match="absent from fold"):
        run_temporal_ablation(
            embedding_paths,
            labels_path=labels_path,
            split_path=split_path,
            fold=0,
            out_dir=tmp_path / "unused",
            preflight_only=True,
        )


def _write_fake_fold(path: Path, fold: int, *, cohort_hash: str = "cohort") -> None:
    path.mkdir()
    protocol = {"seeds": [0, 1], "strategies": list(STRATEGIES)}
    contract = {
        "experiment": "temporal_readout_ablation",
        "encoders": {
            name: {
                "backbone": f"{name}_test",
                "embedding_dim": index + 2,
                "county_timestep_feature_dim": 2 * (index + 2),
                "spatial_pool": "mean_std",
                "spatial_std_ddof": 0,
            }
            for index, name in enumerate(ENCODERS)
        },
        "matched_county_years": 2,
        "matched_key_sha256": cohort_hash,
        "matched_complete_patch_identity_sha256": "patches",
        "spatial_pool": "mean_std",
        "spatial_operation_order": AGGREGATION_OPERATION_ORDER,
        "expected_timesteps": 7,
        "protocol": protocol,
        "split": {"fold": fold},
    }
    (path / "data_contract.json").write_text(json.dumps(contract))
    results = []
    predictions = []
    test_key = f"{17001 + 2 * fold:05d}-2020"
    for encoder_index, encoder in enumerate(ENCODERS):
        for strategy_index, strategy in enumerate(STRATEGIES):
            for seed in protocol["seeds"]:
                score = 0.2 + 0.01 * encoder_index + 0.02 * strategy_index + 0.001 * seed
                results.append(
                    {
                        "encoder": encoder,
                        "backbone": f"{encoder}_test",
                        "strategy": strategy,
                        "fold": fold,
                        "seed": seed,
                        "test_r2": score,
                        "test_rmse": 1.0 - score,
                        "test_mae": 0.5 - score / 2,
                        "test_n": 1,
                        "parameter_count": 100 + 10 * encoder_index + strategy_index,
                    }
                )
                predictions.append(
                    {
                        "key": test_key,
                        "observed_yield": 10.0 + fold,
                        "predicted_yield": 9.5 + fold,
                        "encoder": encoder,
                        "strategy": strategy,
                        "seed": seed,
                        "split_or_fold": f"fold_{fold}",
                    }
                )
    pd.DataFrame(results).to_csv(path / "results_by_seed.csv", index=False)
    pd.DataFrame(predictions).to_csv(path / "predictions.csv", index=False)


def test_fold_aggregation_validates_protocol_and_reports_outer_fold_variation(
    tmp_path: Path,
) -> None:
    folds = [tmp_path / "fold_0", tmp_path / "fold_1"]
    _write_fake_fold(folds[0], 0)
    _write_fake_fold(folds[1], 1)

    output = aggregate_temporal_folds(
        folds,
        out_dir=tmp_path / "aggregate",
        expected_folds=(0, 1),
    )

    assert len(output["summary"]) == len(ENCODERS) * len(STRATEGIES)
    assert set(output["summary"]["folds"]) == {2}
    assert set(output["summary"]["seeds_per_fold"]) == {2}
    assert set(output["summary"]["test_n_total"]) == {2}
    assert output["contract"]["summary_unit"].startswith("outer_fold_mean")
    assert (tmp_path / "aggregate" / "summary_across_folds.csv").is_file()


def test_fold_aggregation_rejects_cohort_drift(tmp_path: Path) -> None:
    folds = [tmp_path / "fold_0", tmp_path / "fold_1"]
    _write_fake_fold(folds[0], 0)
    _write_fake_fold(folds[1], 1, cohort_hash="different")

    with pytest.raises(ValueError, match="protocol or cohort drift"):
        aggregate_temporal_folds(
            folds,
            out_dir=tmp_path / "aggregate",
            expected_folds=(0, 1),
        )
