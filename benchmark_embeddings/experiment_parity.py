#!/usr/bin/env python3
"""Audit cohort, fold, and regressor parity across benchmark experiment families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .loyo import REPRESENTATIONS as MAIN_REPRESENTATIONS
from .regression_benchmark import (
    CLIMATE_REPRESENTATIONS,
    DEFAULT_FOLDS,
    FAMILY_CLIMATE,
    FAMILY_MAIN,
    REGRESSORS,
)
from .probe import (
    AGGREGATION_OPERATION_ORDER,
    SPATIAL_POOL_MEAN_STD,
    two_stage_mean_aggregation_contract,
)
from .supervised_aggregate import MODELS as SUPERVISED_MODELS


def _read_contract(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"contract must be a JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _require_equal(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    field: str,
    *,
    comparison: str,
) -> None:
    if left.get(field) != right.get(field):
        raise ValueError(f"{comparison} mismatch for {field}")


def _validate_tabular_contract(
    contract: Mapping[str, Any],
    *,
    family: str,
    representations: Sequence[str],
    expected_folds: Sequence[int],
    required_regressors: Sequence[str],
) -> None:
    if contract.get("experiment") != "matched_tabular_regression_benchmark":
        raise ValueError(f"{family} is not a matched tabular benchmark contract")
    if contract.get("family") != family:
        raise ValueError(f"expected {family} contract, got {contract.get('family')!r}")
    expected_role = (
        "main_encoder_embedding_comparison"
        if family == FAMILY_MAIN
        else "auxiliary_climate_fusion_comparison"
    )
    if contract.get("workflow_role") != expected_role:
        raise ValueError(f"{family} has the wrong workflow role")
    if contract.get("estimator_family") != "classical_ml":
        raise ValueError(f"{family} must use only the classical-ML regressor family")
    observed_representations = set(contract.get("representations", {}))
    if observed_representations != set(representations):
        raise ValueError(
            f"{family} representations are {sorted(observed_representations)}, "
            f"expected {sorted(representations)}"
        )
    split = contract.get("split", {})
    folds = sorted(int(value) for value in split.get("outer_folds", []))
    expected = sorted(int(value) for value in expected_folds)
    if folds != expected:
        raise ValueError(f"{family} outer folds are {folds}, expected {expected}")
    if split.get("grouping") != "county_all_years_together":
        raise ValueError(f"{family} does not group all years of each county together")
    if split.get("year_policy") != (
        "all_cohort_years_in_each_train_validation_test_partition"
    ):
        raise ValueError(f"{family} does not use all cohort years in every partition")
    if set(split.get("partitions", {})) != {str(value) for value in expected}:
        raise ValueError(f"{family} does not record every required fold partition")
    observed_regressors = set(contract.get("regressor_registry", {}))
    if observed_regressors != set(required_regressors):
        raise ValueError(
            f"{family} regressors are {sorted(observed_regressors)}, "
            f"expected {sorted(required_regressors)}"
        )
    if contract.get("target_and_metric_units", {}).get("canonical_yield") != (
        "bushels_per_acre"
    ):
        raise ValueError(f"{family} does not record canonical yield units")
    _require_declared_aggregation(contract, family)


# LOYO has been run with Random Forest and with Ridge. The study reports one
# head, but the audit's job is to confirm a single fixed head was used, not to
# dictate which -- pinning one made the tool reject correct artefacts after the
# head changed.
_LOYO_REGRESSORS = frozenset({"random_forest", "ridge"})


def _require_declared_aggregation(contract, family: str) -> None:
    """Check the aggregation contract is well formed and internally consistent.

    Previously this compared against a single hard-coded two-stage contract, so
    every joint-pooled run failed the audit even though joint pooling is what
    the main benchmark, LOYO and LOSO now share. What matters for parity is not
    which pooling was chosen but that the contract states it and that the
    analyses being compared agree; the caller enforces agreement.
    """
    aggregation = contract.get("feature_aggregation")
    if not isinstance(aggregation, dict):
        raise ValueError(f"{family} does not record a feature_aggregation contract")
    expected = two_stage_mean_aggregation_contract(
        aggregation.get("temporal_pool", "mean")
    )
    if aggregation != expected:
        differing = sorted(
            key for key in set(aggregation) | set(expected)
            if aggregation.get(key) != expected.get(key)
        )
        raise ValueError(
            f"{family} feature_aggregation is not a recognised pooling "
            f"contract; inconsistent fields: {differing}"
        )


def _validate_supervised_contract(
    contract: Mapping[str, Any],
    *,
    expected_folds: Sequence[int],
) -> None:
    if contract.get("experiment") != "matched_supervised_sentinel2_cv":
        raise ValueError("unexpected supervised benchmark contract type")
    if contract.get("workflow_role") != "supervised_sentinel2_benchmark":
        raise ValueError("supervised contract has the wrong workflow role")
    if contract.get("estimator_family") != "supervised_deep_learning":
        raise ValueError("supervised contract has the wrong estimator family")
    if set(contract.get("models", [])) != set(SUPERVISED_MODELS):
        raise ValueError("supervised contract does not contain the three matched models")
    if sorted(int(value) for value in contract.get("outer_folds", [])) != sorted(
        int(value) for value in expected_folds
    ):
        raise ValueError("supervised outer folds do not match the main benchmark")
    if contract.get("grouping") != "county_all_years_together":
        raise ValueError("supervised contract is not county grouped")
    if contract.get("year_policy") != (
        "all_cohort_years_in_each_train_validation_test_partition"
    ):
        raise ValueError("supervised contract does not retain all years per partition")


def audit_experiment_contracts(
    main_contract: Mapping[str, Any],
    climate_contract: Mapping[str, Any],
    *,
    temporal_contracts: Sequence[Mapping[str, Any]] = (),
    loyo_contract: Mapping[str, Any] | None = None,
    supervised_contract: Mapping[str, Any] | None = None,
    expected_folds: Sequence[int] = DEFAULT_FOLDS,
    required_regressors: Sequence[str] = REGRESSORS,
) -> dict[str, Any]:
    """Fail closed unless all supplied contracts satisfy the shared-data protocol."""
    expected_folds = tuple(int(value) for value in expected_folds)
    required_regressors = tuple(str(value) for value in required_regressors)
    _validate_tabular_contract(
        main_contract,
        family=FAMILY_MAIN,
        representations=MAIN_REPRESENTATIONS,
        expected_folds=expected_folds,
        required_regressors=required_regressors,
    )
    _validate_tabular_contract(
        climate_contract,
        family=FAMILY_CLIMATE,
        representations=CLIMATE_REPRESENTATIONS,
        expected_folds=expected_folds,
        required_regressors=required_regressors,
    )
    for field in (
        "matched_county_years",
        "matched_key_sha256",
        "matched_complete_patch_identity_sha256",
        "labels",
        "regressor_registry",
        "stochastic_seeds",
        "ridge_seed_policy",
        "aggregation",
        "target_and_metric_units",
    ):
        _require_equal(main_contract, climate_contract, field, comparison="main/climate")
    _require_equal(
        main_contract["split"],
        climate_contract["split"],
        "outer_folds",
        comparison="main/climate split",
    )
    _require_equal(
        main_contract["split"],
        climate_contract["split"],
        "grouping",
        comparison="main/climate split",
    )
    _require_equal(
        main_contract["split"],
        climate_contract["split"],
        "partitions",
        comparison="main/climate split",
    )

    temporal_folds: list[int] = []
    if temporal_contracts:
        if len(temporal_contracts) != len(expected_folds):
            raise ValueError(
                f"temporal audit requires one contract per fold ({len(expected_folds)})"
            )
        main_partitions = main_contract["split"]["partitions"]
        for contract in temporal_contracts:
            if contract.get("experiment") != "temporal_readout_ablation":
                raise ValueError("unexpected temporal-ablation contract type")
            if (
                contract.get("workflow_role") != "temporal_pooling_ablation"
                or contract.get("estimator_family") != "neural_ablation"
                or contract.get("prediction_head") != "mlp"
                or contract.get("protocol", {}).get("prediction_head") != "mlp"
            ):
                raise ValueError("temporal ablation must use the shared MLP head")
            feature_aggregation = contract.get("feature_aggregation", {})
            if (
                feature_aggregation.get("operation_order")
                != AGGREGATION_OPERATION_ORDER
                or feature_aggregation.get("spatial_pool") != SPATIAL_POOL_MEAN_STD
                or set(feature_aggregation.get("temporal_pool_strategies", []))
                != {"mean", "concat", "conv1d"}
            ):
                raise ValueError(
                    "temporal ablation must spatially pool per timestep before "
                    "mean, concat, and conv1d readouts"
                )
            fold = int(contract.get("split", {}).get("fold"))
            if fold not in expected_folds or fold in temporal_folds:
                raise ValueError(f"invalid or duplicate temporal fold {fold}")
            temporal_folds.append(fold)
            for field in (
                "matched_county_years",
                "matched_key_sha256",
                "matched_complete_patch_identity_sha256",
                "labels",
            ):
                _require_equal(
                    main_contract,
                    contract,
                    field,
                    comparison=f"main/temporal fold {fold}",
                )
            temporal_split = contract["split"]
            main_split = main_partitions[str(fold)]
            if temporal_split.get("validation_fold") != main_split.get("validation_fold"):
                raise ValueError(f"temporal fold {fold} validation-fold mismatch")
            for temporal_field, main_field in (
                ("train_keys", "train_keys"),
                ("val_keys", "validation_keys"),
                ("test_keys", "test_keys"),
            ):
                if temporal_split.get(temporal_field) != main_split.get(main_field):
                    raise ValueError(
                        f"temporal fold {fold} mismatch for {temporal_field}"
                    )
        if sorted(temporal_folds) != sorted(expected_folds):
            raise ValueError("temporal contracts do not cover every main outer fold")

    if loyo_contract is not None:
        if loyo_contract.get("experiment") != "main_benchmark_climate_free_loyo":
            raise ValueError("unexpected LOYO contract type")
        evaluation = loyo_contract.get("evaluation", {})
        if (
            loyo_contract.get("workflow_role") != "temporal_generalization"
            or loyo_contract.get("estimator_family") != "classical_ml"
            or evaluation.get("regressor_key") not in _LOYO_REGRESSORS
        ):
            raise ValueError(
                "LOYO must use one fixed regression head from "
                f"{sorted(_LOYO_REGRESSORS)}; found "
                f"{evaluation.get('regressor_key')!r}"
            )
        _require_declared_aggregation(loyo_contract, "LOYO")
        if loyo_contract.get("feature_aggregation") != main_contract.get(
            "feature_aggregation"
        ):
            raise ValueError(
                "LOYO and the main benchmark must pool county features "
                "identically; comparing them across different pooling makes the "
                "difference between them uninterpretable"
            )
        for field in (
            "matched_county_years",
            "matched_key_sha256",
            "matched_complete_patch_identity_sha256",
            "labels",
        ):
            _require_equal(main_contract, loyo_contract, field, comparison="main/LOYO")
        fusion = loyo_contract.get("fusion_contract", {})
        if (
            fusion.get("added_climate_features") != []
            or fusion.get("daymet_late_fusion") is not False
            or fusion.get("presto_era5_input_fusion") is not False
        ):
            raise ValueError("LOYO contract contains added climate fusion")

    supervised_verified = False
    if supervised_contract is not None:
        _validate_supervised_contract(
            supervised_contract, expected_folds=expected_folds
        )
        for field in ("matched_county_years", "matched_key_sha256"):
            _require_equal(
                main_contract,
                supervised_contract,
                field,
                comparison="main/supervised",
            )
        main_partitions = main_contract["split"].get("partitions", {})
        supervised_partitions = supervised_contract.get("partitions", {})
        if set(main_partitions) != set(supervised_partitions):
            raise ValueError("main/supervised mismatch for partition folds")
        for fold in main_partitions:
            expected_partition = {
                field: main_partitions[fold].get(field)
                for field in (
                    "validation_fold",
                    "train_keys",
                    "validation_keys",
                    "test_keys",
                )
            }
            if expected_partition != supervised_partitions[fold]:
                raise ValueError(f"main/supervised mismatch for fold {fold}")
        supervised_labels = supervised_contract.get("raw_data_contract", {}).get(
            "yield_labels"
        )
        if supervised_labels not in {None, "embedded_fallback", main_contract.get("labels")}:
            raise ValueError("main/supervised mismatch for yield labels")
        supervised_verified = True

    return {
        "status": "pass",
        "matched_county_years": main_contract["matched_county_years"],
        "matched_key_sha256": main_contract["matched_key_sha256"],
        "matched_complete_patch_identity_sha256": main_contract[
            "matched_complete_patch_identity_sha256"
        ],
        "outer_folds": sorted(expected_folds),
        "regressors": list(required_regressors),
        "stochastic_seeds": main_contract["stochastic_seeds"],
        "main_climate_same_partitions": True,
        "temporal_folds_verified": sorted(temporal_folds),
        "loyo_climate_free_verified": loyo_contract is not None,
        "supervised_county_fold_parity_verified": supervised_verified,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-contract", required=True)
    parser.add_argument("--climate-contract", required=True)
    parser.add_argument("--temporal-contracts", nargs="*", default=())
    parser.add_argument("--loyo-contract")
    parser.add_argument("--supervised-contract")
    parser.add_argument("--expected-folds", nargs="+", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--required-regressors", nargs="+", choices=REGRESSORS, default=REGRESSORS)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = audit_experiment_contracts(
        _read_contract(args.main_contract),
        _read_contract(args.climate_contract),
        temporal_contracts=[_read_contract(path) for path in args.temporal_contracts],
        loyo_contract=(
            _read_contract(args.loyo_contract) if args.loyo_contract else None
        ),
        supervised_contract=(
            _read_contract(args.supervised_contract)
            if args.supervised_contract
            else None
        ),
        expected_folds=args.expected_folds,
        required_regressors=args.required_regressors,
    )
    _write_json(Path(args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
