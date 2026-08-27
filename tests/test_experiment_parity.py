from __future__ import annotations

from copy import deepcopy

import pytest

from benchmark_embeddings.experiment_parity import audit_experiment_contracts
from benchmark_embeddings.loyo import REPRESENTATIONS
from benchmark_embeddings.probe import (
    AGGREGATION_OPERATION_ORDER,
    SPATIAL_POOL_MEAN_STD,
    two_stage_mean_aggregation_contract,
)
from benchmark_embeddings.regression_benchmark import CLIMATE_REPRESENTATIONS


FOLDS = (0, 1)
REGRESSORS = ("ridge", "random_forest")


def _partition(fold: int) -> dict:
    if fold == 0:
        return {
            "validation_fold": 1,
            "train_keys": ["c-2"],
            "validation_keys": ["b-2"],
            "test_keys": ["a-2"],
        }
    return {
        "validation_fold": 0,
        "train_keys": ["a-2"],
        "validation_keys": ["b-2"],
        "test_keys": ["c-2"],
    }


def _tabular_contract(family: str) -> dict:
    names = REPRESENTATIONS if family == "main" else CLIMATE_REPRESENTATIONS
    return {
        "experiment": "matched_tabular_regression_benchmark",
        "workflow_role": (
            "main_encoder_embedding_comparison"
            if family == "main"
            else "auxiliary_climate_fusion_comparison"
        ),
        "estimator_family": "classical_ml",
        "family": family,
        "representations": {name: {} for name in names},
        "matched_county_years": 3,
        "matched_key_sha256": "key-hash",
        "matched_complete_patch_identity_sha256": "patch-hash",
        "labels": "/same/labels.csv",
        "split": {
            "outer_folds": list(FOLDS),
            "grouping": "county_all_years_together",
            "cohort_years": [2],
            "year_policy": "all_cohort_years_in_each_train_validation_test_partition",
            "partitions": {str(fold): _partition(fold) for fold in FOLDS},
        },
        "regressor_registry": {name: {"same": True} for name in REGRESSORS},
        "stochastic_seeds": [0, 1, 2],
        "ridge_seed_policy": "deterministic_once_seed_0",
        "feature_aggregation": two_stage_mean_aggregation_contract(),
        "aggregation": {"within_fold": "mean", "across_folds": "mean_std"},
        "target_and_metric_units": {"canonical_yield": "bushels_per_acre"},
    }


def _temporal_contract(fold: int) -> dict:
    partition = _partition(fold)
    return {
        "experiment": "temporal_readout_ablation",
        "workflow_role": "temporal_pooling_ablation",
        "estimator_family": "neural_ablation",
        "prediction_head": "mlp",
        "protocol": {"prediction_head": "mlp"},
        "feature_aggregation": {
            "operation_order": AGGREGATION_OPERATION_ORDER,
            "spatial_pool": SPATIAL_POOL_MEAN_STD,
            "temporal_pool_strategies": ["mean", "concat", "conv1d"],
        },
        "matched_county_years": 3,
        "matched_key_sha256": "key-hash",
        "matched_complete_patch_identity_sha256": "patch-hash",
        "labels": "/same/labels.csv",
        "split": {
            "fold": fold,
            "validation_fold": partition["validation_fold"],
            "train_keys": partition["train_keys"],
            "val_keys": partition["validation_keys"],
            "test_keys": partition["test_keys"],
        },
    }


def _loyo_contract() -> dict:
    return {
        "experiment": "main_benchmark_climate_free_loyo",
        "workflow_role": "temporal_generalization",
        "estimator_family": "classical_ml",
        "matched_county_years": 3,
        "matched_key_sha256": "key-hash",
        "matched_complete_patch_identity_sha256": "patch-hash",
        "labels": "/same/labels.csv",
        "feature_aggregation": two_stage_mean_aggregation_contract(),
        "fusion_contract": {
            "added_climate_features": [],
            "daymet_late_fusion": False,
            "presto_era5_input_fusion": False,
        },
        "evaluation": {
            "regressor_key": "random_forest",
            "regressor": "RandomForestRegressor",
            "model_selection": "none_fixed_protocol",
        },
    }


def _supervised_contract() -> dict:
    return {
        "experiment": "matched_supervised_sentinel2_cv",
        "workflow_role": "supervised_sentinel2_benchmark",
        "estimator_family": "supervised_deep_learning",
        "models": [
            "supervised_s2_3d_convlstm",
            "supervised_s2_gru",
            "supervised_s2_lstm",
        ],
        "outer_folds": list(FOLDS),
        "grouping": "county_all_years_together",
        "year_policy": "all_cohort_years_in_each_train_validation_test_partition",
        "matched_county_years": 3,
        "matched_key_sha256": "key-hash",
        "partitions": {str(fold): _partition(fold) for fold in FOLDS},
        "raw_data_contract": {"yield_labels": "/same/labels.csv"},
    }


def test_audit_verifies_cross_family_temporal_and_loyo_parity() -> None:
    report = audit_experiment_contracts(
        _tabular_contract("main"),
        _tabular_contract("climate_fusion"),
        temporal_contracts=[_temporal_contract(0), _temporal_contract(1)],
        loyo_contract=_loyo_contract(),
        supervised_contract=_supervised_contract(),
        expected_folds=FOLDS,
        required_regressors=REGRESSORS,
    )

    assert report["status"] == "pass"
    assert report["main_climate_same_partitions"] is True
    assert report["temporal_folds_verified"] == [0, 1]
    assert report["loyo_climate_free_verified"] is True
    assert report["supervised_county_fold_parity_verified"] is True


def test_audit_rejects_fold_or_cohort_drift() -> None:
    climate = _tabular_contract("climate_fusion")
    climate["split"]["partitions"]["1"]["test_keys"] = ["wrong-2"]
    with pytest.raises(ValueError, match="partitions"):
        audit_experiment_contracts(
            _tabular_contract("main"),
            climate,
            expected_folds=FOLDS,
            required_regressors=REGRESSORS,
        )

    temporal = _temporal_contract(0)
    temporal["matched_key_sha256"] = "wrong"
    with pytest.raises(ValueError, match="matched_key_sha256"):
        audit_experiment_contracts(
            _tabular_contract("main"),
            _tabular_contract("climate_fusion"),
            temporal_contracts=[temporal, _temporal_contract(1)],
            expected_folds=FOLDS,
            required_regressors=REGRESSORS,
        )


def test_audit_rejects_regressor_or_climate_fusion_drift() -> None:
    climate = deepcopy(_tabular_contract("climate_fusion"))
    climate["regressor_registry"]["random_forest"]["same"] = False
    with pytest.raises(ValueError, match="regressor_registry"):
        audit_experiment_contracts(
            _tabular_contract("main"),
            climate,
            expected_folds=FOLDS,
            required_regressors=REGRESSORS,
        )

    loyo = _loyo_contract()
    loyo["fusion_contract"]["daymet_late_fusion"] = True
    with pytest.raises(ValueError, match="climate fusion"):
        audit_experiment_contracts(
            _tabular_contract("main"),
            _tabular_contract("climate_fusion"),
            loyo_contract=loyo,
            expected_folds=FOLDS,
            required_regressors=REGRESSORS,
        )
