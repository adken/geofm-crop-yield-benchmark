#!/usr/bin/env python3
"""Validate and aggregate matched temporal-ablation fold outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


RESULT_COLUMNS = {
    "encoder",
    "backbone",
    "strategy",
    "fold",
    "seed",
    "test_r2",
    "test_rmse",
    "test_mae",
    "test_n",
    "parameter_count",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _contract_signature(contract: dict[str, Any]) -> dict[str, Any]:
    encoders = contract.get("encoders", {})
    encoder_signature = {
        name: {
            "backbone": value.get("backbone"),
            "embedding_dim": value.get("embedding_dim"),
            "county_timestep_feature_dim": value.get("county_timestep_feature_dim"),
            "spatial_pool": value.get("spatial_pool"),
            "spatial_std_ddof": value.get("spatial_std_ddof"),
        }
        for name, value in sorted(encoders.items())
    }
    return {
        "experiment": contract.get("experiment"),
        "workflow_role": contract.get("workflow_role"),
        "estimator_family": contract.get("estimator_family"),
        "prediction_head": contract.get("prediction_head"),
        "encoders": encoder_signature,
        "matched_county_years": contract.get("matched_county_years"),
        "matched_key_sha256": contract.get("matched_key_sha256"),
        "matched_complete_patch_identity_sha256": contract.get(
            "matched_complete_patch_identity_sha256"
        ),
        "spatial_pool": contract.get("spatial_pool"),
        "spatial_operation_order": contract.get("spatial_operation_order"),
        "expected_timesteps": contract.get("expected_timesteps"),
        "protocol": contract.get("protocol"),
    }


def aggregate_temporal_folds(
    fold_dirs: Sequence[str | Path],
    *,
    out_dir: str | Path,
    expected_folds: Sequence[int] = (0, 1, 2, 3, 4),
) -> dict[str, Any]:
    paths = tuple(Path(value) for value in fold_dirs)
    expected_folds = tuple(int(value) for value in expected_folds)
    if not paths or not expected_folds or len(set(expected_folds)) != len(expected_folds):
        raise ValueError("fold directories and unique expected folds are required")
    contracts = []
    result_tables = []
    prediction_tables = []
    observed_folds = []
    for path in paths:
        contract = _read_json(path / "data_contract.json")
        results = pd.read_csv(path / "results_by_seed.csv")
        predictions = pd.read_csv(path / "predictions.csv")
        missing = RESULT_COLUMNS.difference(results.columns)
        if missing:
            raise ValueError(f"{path} result table is missing {sorted(missing)}")
        fold = int(contract.get("split", {}).get("fold"))
        if set(pd.to_numeric(results["fold"], errors="raise").astype(int)) != {fold}:
            raise ValueError(f"{path} mixes result folds or disagrees with its contract")
        if set(predictions["split_or_fold"].astype(str)) != {f"fold_{fold}"}:
            raise ValueError(f"{path} prediction fold labels disagree with its contract")
        contracts.append(contract)
        result_tables.append(results)
        prediction_tables.append(predictions)
        observed_folds.append(fold)

    if sorted(observed_folds) != sorted(expected_folds):
        raise ValueError(
            f"fold coverage is {sorted(observed_folds)}, expected {sorted(expected_folds)}"
        )
    if len(set(observed_folds)) != len(observed_folds):
        raise ValueError("fold directories contain duplicate outer folds")
    reference_signature = _contract_signature(contracts[0])
    for path, contract in zip(paths[1:], contracts[1:], strict=True):
        if _contract_signature(contract) != reference_signature:
            raise ValueError(f"temporal protocol or cohort drift detected in {path}")

    results = pd.concat(result_tables, ignore_index=True)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    duplicate_run = results.duplicated(
        ["encoder", "backbone", "strategy", "fold", "seed"], keep=False
    )
    if duplicate_run.any():
        raise ValueError("duplicate encoder/strategy/fold/seed result rows")
    expected_seeds = sorted(int(value) for value in reference_signature["protocol"]["seeds"])
    expected_strategies = sorted(reference_signature["protocol"]["strategies"])
    for (encoder, strategy, fold), group in results.groupby(
        ["encoder", "strategy", "fold"], sort=True
    ):
        if sorted(group["seed"].astype(int)) != expected_seeds:
            raise ValueError(
                f"seed coverage drift for {encoder}/{strategy}/fold {fold}"
            )
        if group["parameter_count"].nunique() != 1 or group["test_n"].nunique() != 1:
            raise ValueError(
                f"model size or test count drifts across seeds for "
                f"{encoder}/{strategy}/fold {fold}"
            )
    combinations = set(zip(results["encoder"], results["strategy"], strict=True))
    expected_combinations = {
        (encoder, strategy)
        for encoder in reference_signature["encoders"]
        for strategy in expected_strategies
    }
    if combinations != expected_combinations:
        raise ValueError("encoder/strategy result coverage is incomplete")
    prediction_key = ["encoder", "strategy", "seed", "key"]
    if predictions.duplicated(prediction_key).any():
        raise ValueError("a county-year appears in multiple outer test folds")
    prediction_counts = (
        predictions.groupby(["encoder", "strategy", "seed", "split_or_fold"])
        .size()
        .rename("prediction_n")
        .reset_index()
    )
    results_with_labels = results.copy()
    results_with_labels["split_or_fold"] = "fold_" + results_with_labels["fold"].astype(str)
    count_check = results_with_labels.merge(
        prediction_counts,
        on=["encoder", "strategy", "seed", "split_or_fold"],
        how="left",
        validate="one_to_one",
    )
    if count_check["prediction_n"].isna().any() or not np.array_equal(
        count_check["test_n"].to_numpy(dtype=int),
        count_check["prediction_n"].to_numpy(dtype=int),
    ):
        raise ValueError("prediction row counts disagree with per-run test_n")
    observed_parity = predictions.groupby("key")["observed_yield"].nunique(dropna=False)
    if (observed_parity != 1).any():
        raise ValueError("observed yields differ across matched predictions")

    fold_results = (
        results.groupby(["encoder", "backbone", "strategy", "fold"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            test_n=("test_n", "first"),
            r2=("test_r2", "mean"),
            rmse=("test_rmse", "mean"),
            mae=("test_mae", "mean"),
            seed_r2_std=("test_r2", lambda values: values.std(ddof=0)),
            seed_rmse_std=("test_rmse", lambda values: values.std(ddof=0)),
            seed_mae_std=("test_mae", lambda values: values.std(ddof=0)),
            parameter_count=("parameter_count", "first"),
        )
        .sort_values(["encoder", "strategy", "fold"])
        .reset_index(drop=True)
    )
    parameter_parity = fold_results.groupby(
        ["encoder", "backbone", "strategy"]
    )["parameter_count"].nunique()
    if (parameter_parity != 1).any():
        raise ValueError("model parameter counts drift across folds")
    summary = (
        fold_results.groupby(["encoder", "backbone", "strategy"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            seeds_per_fold=("seeds", "first"),
            test_n_total=("test_n", "sum"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", lambda values: values.std(ddof=0)),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", lambda values: values.std(ddof=0)),
            mae_mean=("mae", "mean"),
            mae_std=("mae", lambda values: values.std(ddof=0)),
            parameter_count=("parameter_count", "first"),
        )
        .sort_values(["encoder", "strategy"])
        .reset_index(drop=True)
    )
    if not np.isfinite(summary[["rmse_mean", "rmse_std", "mae_mean", "mae_std"]]).all(
        axis=None
    ):
        raise ValueError("aggregate temporal metrics contain non-finite values")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "results_by_fold_and_seed.csv", index=False)
    fold_results.to_csv(out_dir / "results_by_fold.csv", index=False)
    summary.to_csv(out_dir / "summary_across_folds.csv", index=False)
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    aggregation_contract = {
        "schema_version": 1,
        "experiment": "temporal_readout_ablation_aggregate",
        "folds": sorted(observed_folds),
        "fold_directories": [str(path.resolve()) for path in paths],
        "validated_signature": reference_signature,
        "summary_unit": "outer_fold_mean_after_averaging_seeds_within_fold",
        "standard_deviation_ddof": 0,
    }
    (out_dir / "aggregation_contract.json").write_text(
        json.dumps(aggregation_contract, indent=2, sort_keys=True) + "\n"
    )
    return {
        "results": results,
        "fold_results": fold_results,
        "summary": summary,
        "predictions": predictions,
        "contract": aggregation_contract,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-dirs", nargs="+", required=True)
    parser.add_argument("--expected-folds", nargs="+", type=int, default=(0, 1, 2, 3, 4))
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    output = aggregate_temporal_folds(
        args.fold_dirs,
        out_dir=args.out_dir,
        expected_folds=args.expected_folds,
    )
    print(output["summary"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
