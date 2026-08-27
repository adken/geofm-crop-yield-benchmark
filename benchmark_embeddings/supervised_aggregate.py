#!/usr/bin/env python3
"""Validate and aggregate matched supervised 3D-ConvLSTM/GRU/LSTM runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data import years_from_keys


MODELS = (
    "supervised_s2_3d_convlstm",
    "supervised_s2_gru",
    "supervised_s2_lstm",
)
DEFAULT_FOLDS = (0, 1, 2, 3, 4)
DEFAULT_SEEDS = (0, 1, 2)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _protocol_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    config = deepcopy(config)
    config.pop("out_dir", None)
    config.pop("seed", None)
    config.setdefault("model", {}).pop("name", None)
    config.setdefault("split", {}).pop("fold", None)
    return config


def _cohort_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    ignored = {"split", "partition_counts", "partition_keys"}
    return {key: value for key, value in contract.items() if key not in ignored}


def _metric_values(frame: pd.DataFrame) -> dict[str, float | int | None]:
    observed = frame["observed_yield"].to_numpy(dtype=np.float64)
    predicted = frame["predicted_yield"].to_numpy(dtype=np.float64)
    return {
        "county_r2": float(r2_score(observed, predicted)) if len(frame) > 1 else None,
        "county_rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "county_mae": float(mean_absolute_error(observed, predicted)),
        "county_n": int(len(frame)),
    }


def aggregate_supervised_runs(
    run_dirs: Sequence[str | Path],
    *,
    out_dir: str | Path,
    expected_models: Sequence[str] = MODELS,
    expected_folds: Sequence[int] = DEFAULT_FOLDS,
    expected_seeds: Sequence[int] = DEFAULT_SEEDS,
) -> dict[str, Any]:
    expected_models = tuple(str(value) for value in expected_models)
    expected_folds = tuple(int(value) for value in expected_folds)
    expected_seeds = tuple(int(value) for value in expected_seeds)
    expected_combinations = {
        (model, fold, seed)
        for model in expected_models
        for fold in expected_folds
        for seed in expected_seeds
    }
    if len(run_dirs) != len(expected_combinations):
        raise ValueError(
            f"expected {len(expected_combinations)} supervised run directories, "
            f"got {len(run_dirs)}"
        )

    rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    seen: set[tuple[str, int, int]] = set()
    reference_protocol: dict[str, Any] | None = None
    reference_cohort: dict[str, Any] | None = None
    partition_by_fold: dict[int, dict[str, Any]] = {}

    for value in run_dirs:
        run_dir = Path(value)
        result = _read_json(run_dir / "result.json")
        contract = _read_json(run_dir / "data_contract.json")
        protocol = _protocol_config(run_dir / "config_used.yaml")
        model = str(result.get("experiment", {}).get("id"))
        split = result.get("split", {})
        if split.get("mode") != "primary":
            raise ValueError(f"supervised CV aggregation rejects non-primary run {run_dir}")
        fold = int(split.get("fold"))
        seed = int(result.get("seed"))
        combination = (model, fold, seed)
        if combination not in expected_combinations:
            raise ValueError(f"unexpected supervised run {combination}")
        if combination in seen:
            raise ValueError(f"duplicate supervised run {combination}")
        seen.add(combination)

        if reference_protocol is None:
            reference_protocol = protocol
        elif protocol != reference_protocol:
            raise ValueError(
                f"training/configuration drift for {combination}; only model, fold, "
                "seed, and output directory may differ"
            )
        cohort = _cohort_contract(contract)
        if reference_cohort is None:
            reference_cohort = cohort
        elif cohort != reference_cohort:
            raise ValueError(f"raw-data cohort contract drift for {combination}")

        contract_split = contract.get("split", {})
        if contract_split != split:
            raise ValueError(f"result/data-contract split mismatch for {combination}")
        partition = contract.get("partition_keys", {})
        counts = contract.get("partition_counts", {})
        for name in ("train", "val", "test"):
            if len(partition.get(name, [])) != int(counts.get(name, -1)):
                raise ValueError(f"partition count mismatch for {combination}/{name}")
        if fold not in partition_by_fold:
            partition_by_fold[fold] = {
                "validation_fold": split.get("validation_fold"),
                "train_keys": partition["train"],
                "validation_keys": partition["val"],
                "test_keys": partition["test"],
            }
        elif partition_by_fold[fold] != {
            "validation_fold": split.get("validation_fold"),
            "train_keys": partition["train"],
            "validation_keys": partition["val"],
            "test_keys": partition["test"],
        }:
            raise ValueError(f"partition drift across models/seeds for fold {fold}")

        prediction = pd.read_csv(run_dir / "predictions.csv", dtype={"county_id": str})
        required = {"county_id", "year", "observed_yield", "predicted_yield"}
        if not required.issubset(prediction.columns):
            raise ValueError(f"missing prediction columns in {run_dir}")
        prediction["county_id"] = prediction["county_id"].astype(str).str.zfill(5)
        prediction["year"] = pd.to_numeric(prediction["year"], errors="raise").astype(int)
        prediction["key"] = prediction["county_id"] + "-" + prediction["year"].astype(str)
        if prediction["key"].duplicated().any():
            raise ValueError(f"duplicate county-year predictions in {run_dir}")
        if sorted(prediction["key"]) != sorted(partition["test"]):
            raise ValueError(f"prediction/test-partition mismatch for {combination}")
        calculated = _metric_values(prediction)
        reported = result.get("test", {})
        for metric in ("county_r2", "county_rmse", "county_mae"):
            if calculated[metric] is None and reported.get(metric) is None:
                continue
            if not np.isclose(
                float(calculated[metric]), float(reported.get(metric)), rtol=1e-7, atol=1e-9
            ):
                raise ValueError(f"reported {metric} disagrees with predictions for {combination}")
        if int(reported.get("county_n", -1)) != calculated["county_n"]:
            raise ValueError(f"reported county_n disagrees for {combination}")

        rows.append(
            {
                "model": model,
                "fold": fold,
                "validation_fold": split.get("validation_fold"),
                "seed": seed,
                "test_n": calculated["county_n"],
                "test_r2": calculated["county_r2"],
                "test_rmse": calculated["county_rmse"],
                "test_mae": calculated["county_mae"],
                "best_epoch": result.get("selection", {}).get("epoch"),
                "validation_rmse": result.get("selection", {}).get("value"),
            }
        )
        prediction["model"] = model
        prediction["fold"] = fold
        prediction["seed"] = seed
        predictions.append(prediction)

    missing = sorted(expected_combinations - seen)
    if missing:
        raise ValueError(f"missing supervised runs: {missing[:5]}")
    if set(partition_by_fold) != set(expected_folds):
        raise ValueError("supervised runs do not cover every expected fold")

    result_frame = pd.DataFrame(rows).sort_values(["model", "fold", "seed"])
    prediction_frame = pd.concat(predictions, ignore_index=True)
    observed_counts = prediction_frame.groupby("key")["observed_yield"].nunique(dropna=False)
    if (observed_counts != 1).any():
        raise ValueError("observed target drift across supervised models/folds/seeds")
    for (model, seed), group in prediction_frame.groupby(["model", "seed"]):
        if group["key"].duplicated().any():
            raise ValueError(f"county-year appears in multiple test folds for {model}/seed {seed}")

    by_fold = (
        result_frame.groupby(["model", "fold"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            test_n=("test_n", "first"),
            r2=("test_r2", "mean"),
            r2_seed_std=("test_r2", lambda values: values.std(ddof=0)),
            rmse=("test_rmse", "mean"),
            rmse_seed_std=("test_rmse", lambda values: values.std(ddof=0)),
            mae=("test_mae", "mean"),
            mae_seed_std=("test_mae", lambda values: values.std(ddof=0)),
        )
    )
    summary = (
        by_fold.groupby("model", as_index=False)
        .agg(
            folds=("fold", "nunique"),
            seeds_per_fold=("seeds", "first"),
            county_years_tested=("test_n", "sum"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", lambda values: values.std(ddof=0)),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", lambda values: values.std(ddof=0)),
            mae_mean=("mae", "mean"),
            mae_std=("mae", lambda values: values.std(ddof=0)),
        )
    )
    all_keys = sorted(set().union(*(set(value["test_keys"]) for value in partition_by_fold.values())))
    raw_county_years = int((reference_cohort or {}).get("num_county_years", -1))
    if raw_county_years != len(all_keys):
        raise ValueError(
            "supervised outer test folds do not cover the complete raw county-year "
            f"cohort: tested={len(all_keys)}, raw={raw_county_years}"
        )
    cohort_years = years_from_keys(all_keys)
    partition_years: dict[str, dict[str, list[int]]] = {}
    for fold, partition in sorted(partition_by_fold.items()):
        observed = {
            "train": years_from_keys(partition["train_keys"]),
            "validation": years_from_keys(partition["validation_keys"]),
            "test": years_from_keys(partition["test_keys"]),
        }
        if any(years != cohort_years for years in observed.values()):
            raise ValueError(
                f"supervised fold {fold} does not contain all cohort years "
                f"{cohort_years} in every partition: {observed}"
            )
        partition_years[str(fold)] = observed
    data_contract = {
        "schema_version": 1,
        "experiment": "matched_supervised_sentinel2_cv",
        "workflow_role": "supervised_sentinel2_benchmark",
        "estimator_family": "supervised_deep_learning",
        "models": list(expected_models),
        "outer_folds": sorted(expected_folds),
        "seeds": list(expected_seeds),
        "grouping": "county_all_years_together",
        "cohort_years": cohort_years,
        "year_policy": "all_cohort_years_in_each_train_validation_test_partition",
        "partition_years": partition_years,
        "matched_county_years": len(all_keys),
        "matched_key_sha256": hashlib.sha256("\n".join(all_keys).encode()).hexdigest(),
        "partitions": {str(key): value for key, value in sorted(partition_by_fold.items())},
        "shared_protocol": reference_protocol,
        "raw_data_contract": reference_cohort,
        "aggregation": {
            "unit": "county_year",
            "within_fold": "mean_across_seeds",
            "across_folds": "mean_and_population_standard_deviation",
        },
        "target_units": "bushels_per_acre",
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "data_contract.json", data_contract)
    result_frame.to_csv(out_dir / "results_by_fold_and_seed.csv", index=False)
    by_fold.to_csv(out_dir / "results_by_fold.csv", index=False)
    summary.to_csv(out_dir / "summary_across_folds.csv", index=False)
    prediction_frame.to_csv(out_dir / "predictions.csv", index=False)
    return {
        "data_contract": data_contract,
        "results": result_frame,
        "by_fold": by_fold,
        "summary": summary,
        "predictions": prediction_frame,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--folds", nargs="+", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    args = parser.parse_args(argv)
    output = aggregate_supervised_runs(
        args.run_dirs,
        out_dir=args.out_dir,
        expected_models=args.models,
        expected_folds=args.folds,
        expected_seeds=args.seeds,
    )
    print(output["summary"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
