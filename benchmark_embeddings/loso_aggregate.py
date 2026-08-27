#!/usr/bin/env python3
"""Validate and aggregate matched all-encoder leave-one-state-out runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DEFAULT_STATES = (17, 18, 19, 20, 21, 26, 27, 29, 31, 38, 39, 46, 55)
STATE_NAMES = {
    17: "Illinois",
    18: "Indiana",
    19: "Iowa",
    20: "Kansas",
    21: "Kentucky",
    26: "Michigan",
    27: "Minnesota",
    29: "Missouri",
    31: "Nebraska",
    38: "North Dakota",
    39: "Ohio",
    46: "South Dakota",
    55: "Wisconsin",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _metrics(frame: pd.DataFrame) -> dict[str, float | int | None]:
    observed = frame["observed_yield"].to_numpy(dtype=np.float64)
    predicted = frame["predicted_yield"].to_numpy(dtype=np.float64)
    return {
        "r2": float(r2_score(observed, predicted)) if len(frame) > 1 else None,
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "n": int(len(frame)),
    }


def _state_cluster_bootstrap(
    frame: pd.DataFrame,
    *,
    states: Sequence[int],
    counts: np.ndarray,
) -> dict[str, float]:
    """Return bootstrap SDs for pooled metrics after resampling state blocks."""

    sufficient_statistics = []
    for state in states:
        state_frame = frame.loc[frame["held_out_state"] == state]
        if state_frame.empty:
            raise ValueError(f"missing predictions for bootstrap state {state:02d}")
        observed = state_frame["observed_yield"].to_numpy(dtype=np.float64)
        predicted = state_frame["predicted_yield"].to_numpy(dtype=np.float64)
        residual = observed - predicted
        sufficient_statistics.append(
            (
                len(state_frame),
                observed.sum(),
                np.square(observed).sum(),
                np.square(residual).sum(),
                np.abs(residual).sum(),
            )
        )
    statistics = np.asarray(sufficient_statistics, dtype=np.float64)
    sample_n = counts @ statistics[:, 0]
    sample_y = counts @ statistics[:, 1]
    sample_y2 = counts @ statistics[:, 2]
    sample_sse = counts @ statistics[:, 3]
    sample_sae = counts @ statistics[:, 4]
    sample_sst = sample_y2 - np.square(sample_y) / sample_n
    if (sample_sst <= 0).any():
        raise ValueError("state-cluster bootstrap produced a zero-variance target sample")
    values = {
        "pooled_r2_bootstrap_std": np.std(1.0 - sample_sse / sample_sst, ddof=1),
        "pooled_rmse_bootstrap_std": np.std(np.sqrt(sample_sse / sample_n), ddof=1),
        "pooled_mae_bootstrap_std": np.std(sample_sae / sample_n, ddof=1),
    }
    return {name: float(value) for name, value in values.items()}


def _discover_representations(input_dir: Path) -> tuple[str, ...]:
    representations = []
    for path in sorted(input_dir.iterdir()):
        if not path.is_dir():
            continue
        if any(child.is_dir() and child.name.isdigit() for child in path.iterdir()):
            representations.append(path.name)
    if not representations:
        raise ValueError(f"no representation/state run directories found under {input_dir}")
    return tuple(representations)


def _state_directories(path: Path) -> dict[int, Path]:
    return {
        int(child.name): child
        for child in path.iterdir()
        if child.is_dir() and child.name.isdigit()
    }


def _common_signature(result: Mapping[str, Any]) -> dict[str, Any]:
    experiment = result.get("experiment", {})
    return {
        "schema_version": result.get("schema_version"),
        "experiment_family": experiment.get("family"),
        "split_path": result.get("split", {}).get("path"),
        "target_and_metric_units": result.get("target_and_metric_units"),
    }


def _representation_signature(result: Mapping[str, Any]) -> dict[str, Any]:
    experiment = result.get("experiment", {})
    cohort = result.get("cohort", {})
    return {
        "id": experiment.get("id"),
        "representation_type": experiment.get("representation_type"),
        "input_modalities": experiment.get("input_modalities"),
        "aggregation": experiment.get("aggregation"),
        "source_county_years_before_daymet": cohort.get(
            "county_years_before_daymet"
        ),
        "source_county_years_after_daymet": cohort.get("county_years_after_daymet"),
        "feature_dim_before_daymet": cohort.get("feature_dim_before_daymet"),
        "feature_dim_after_daymet": cohort.get("feature_dim_after_daymet"),
    }


def _resolve_split_path(value: Any, *, input_dir: Path) -> Path:
    if not value:
        raise ValueError("LOSO result does not identify its split manifest")
    path = Path(str(value)).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, input_dir / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError(f"LOSO split manifest does not exist: {value}")


def _split_test_keys(
    path: Path, *, expected_states: Sequence[int]
) -> dict[int, list[str]]:
    frame = pd.read_csv(path, dtype={"county_id": str})
    required = {"county_id", "year", "fold", "split"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing split columns {sorted(missing)}")
    frame["county_id"] = frame["county_id"].astype(str).str.zfill(5)
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(int)
    frame["split"] = frame["split"].astype(str).str.lower()
    frame["key"] = frame["county_id"] + "-" + frame["year"].astype(str)
    output: dict[int, list[str]] = {}
    for state in expected_states:
        test = frame.loc[(frame["fold"] == state) & (frame["split"] == "test")]
        if test.empty:
            raise ValueError(f"split manifest has no test rows for state {state:02d}")
        if test["key"].duplicated().any():
            raise ValueError(f"duplicate state-test keys in split manifest for {state:02d}")
        if not test["county_id"].str[:2].eq(f"{state:02d}").all():
            raise ValueError(f"split manifest test rows cross state {state:02d}")
        output[state] = sorted(test["key"].tolist())
    union = [key for state in expected_states for key in output[state]]
    if len(union) != len(set(union)):
        raise ValueError("a county-year is assigned to multiple LOSO outer-test states")
    return output


def _load_predictions(path: Path, *, state: int) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"county_id": str})
    required = {"county_id", "year", "observed_yield", "predicted_yield"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing prediction columns {sorted(missing)}")
    frame["county_id"] = frame["county_id"].astype(str).str.zfill(5)
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    frame["key"] = frame["county_id"] + "-" + frame["year"].astype(str)
    if frame["key"].duplicated().any():
        raise ValueError(f"duplicate county-year predictions in {path}")
    if not frame["county_id"].str[:2].eq(f"{state:02d}").all():
        raise ValueError(f"{path} contains counties outside held-out state {state:02d}")
    if "split_or_fold" in frame.columns:
        labels = set(frame["split_or_fold"].astype(str))
        if labels != {f"fold_{state}"}:
            raise ValueError(f"prediction fold labels disagree with state {state:02d} in {path}")
    unit_columns = {
        "observed_yield_bu_per_acre",
        "predicted_yield_bu_per_acre",
    }
    if unit_columns.issubset(frame.columns):
        for generic, explicit in (
            ("observed_yield", "observed_yield_bu_per_acre"),
            ("predicted_yield", "predicted_yield_bu_per_acre"),
        ):
            if not np.allclose(
                frame[generic].to_numpy(dtype=np.float64),
                frame[explicit].to_numpy(dtype=np.float64),
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError(f"generic and bu/acre columns disagree in {path}")
    values = frame[["observed_yield", "predicted_yield"]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite prediction values in {path}")
    return frame


def _check_reported_metrics(
    calculated: Mapping[str, float | int | None],
    reported: Mapping[str, Any],
    *,
    run_dir: Path,
) -> None:
    mapping = {
        "r2": "county_r2",
        "rmse": "county_rmse",
        "mae": "county_mae",
    }
    for calculated_name, reported_name in mapping.items():
        actual = calculated[calculated_name]
        expected = reported.get(reported_name)
        if actual is None and expected is None:
            continue
        if actual is None or expected is None or not np.isclose(
            float(actual), float(expected), rtol=1e-5, atol=1e-5
        ):
            raise ValueError(
                f"reported {reported_name} disagrees with predictions in {run_dir}"
            )
    if int(reported.get("county_n", -1)) != int(calculated["n"]):
        raise ValueError(f"reported county_n disagrees with predictions in {run_dir}")


def aggregate_loso_runs(
    input_dir: str | Path,
    *,
    out_dir: str | Path,
    expected_states: Sequence[int] = DEFAULT_STATES,
    expected_representations: Sequence[str] | None = None,
    bootstrap_repetitions: int = 10_000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Aggregate a representation/state LOSO directory after parity validation."""

    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"LOSO input directory does not exist: {input_dir}")
    expected_states = tuple(int(value) for value in expected_states)
    if not expected_states or len(set(expected_states)) != len(expected_states):
        raise ValueError("unique expected states are required")
    if bootstrap_repetitions < 2:
        raise ValueError("at least two state-cluster bootstrap repetitions are required")
    discovered = _discover_representations(input_dir)
    if expected_representations is None:
        representations = discovered
    else:
        representations = tuple(str(value) for value in expected_representations)
        missing = sorted(set(representations).difference(discovered))
        if missing:
            raise ValueError(f"missing representation directories: {missing}")
        if len(set(representations)) != len(representations):
            raise ValueError("expected representations must be unique")

    expected_state_set = set(expected_states)
    rows: list[dict[str, Any]] = []
    prediction_tables: list[pd.DataFrame] = []
    representation_contracts: dict[str, dict[str, Any]] = {}
    common_signature: dict[str, Any] | None = None
    split_path: Path | None = None
    expected_test_keys: dict[int, list[str]] | None = None
    reference_keys: list[str] | None = None
    reference_observed: pd.Series | None = None

    for representation in representations:
        representation_dir = input_dir / representation
        state_dirs = _state_directories(representation_dir)
        if set(state_dirs) != expected_state_set:
            raise ValueError(
                f"state coverage for {representation} is {sorted(state_dirs)}, "
                f"expected {sorted(expected_state_set)}"
            )
        representation_signature: dict[str, Any] | None = None
        representation_predictions: list[pd.DataFrame] = []
        for state in expected_states:
            run_dir = state_dirs[state]
            result_path = run_dir / "result.json"
            prediction_path = run_dir / "predictions.csv"
            if not result_path.is_file() or not prediction_path.is_file():
                raise ValueError(f"incomplete LOSO run directory: {run_dir}")
            result = _read_json(result_path)
            split = result.get("split", {})
            if int(split.get("fold", -1)) != state:
                raise ValueError(f"result fold disagrees with directory state in {run_dir}")
            experiment = result.get("experiment", {})
            climate = experiment.get("climate_fusion", {})
            if climate != {"source": "none", "stage": "none"} or result.get("daymet") is not None:
                raise ValueError(f"LOSO aggregation rejects climate-fused run {run_dir}")
            selection = result.get("selection", {})
            if selection.get("source") != "validation" or selection.get("metric") != "county_rmse":
                raise ValueError(f"unexpected model-selection protocol in {run_dir}")

            observed_common = _common_signature(result)
            if common_signature is None:
                common_signature = observed_common
                split_path = _resolve_split_path(
                    observed_common.get("split_path"), input_dir=input_dir
                )
                expected_test_keys = _split_test_keys(
                    split_path, expected_states=expected_states
                )
            elif observed_common != common_signature:
                raise ValueError(f"cohort, split, or target-unit drift detected in {run_dir}")
            observed_representation = _representation_signature(result)
            if representation_signature is None:
                representation_signature = observed_representation
            elif observed_representation != representation_signature:
                raise ValueError(f"representation protocol drift detected in {run_dir}")

            predictions = _load_predictions(prediction_path, state=state)
            assert expected_test_keys is not None
            if sorted(predictions["key"].tolist()) != expected_test_keys[state]:
                raise ValueError(
                    f"predictions do not match split-manifest test keys in {run_dir}"
                )
            calculated = _metrics(predictions)
            _check_reported_metrics(calculated, result.get("test", {}), run_dir=run_dir)
            rows.append(
                {
                    "representation": representation,
                    "state_fips": state,
                    "state_name": STATE_NAMES.get(state, f"FIPS {state:02d}"),
                    "experiment_id": experiment.get("id"),
                    "representation_type": experiment.get("representation_type"),
                    "feature_dim": result.get("cohort", {}).get(
                        "feature_dim_before_daymet"
                    ),
                    "test_n": calculated["n"],
                    "test_r2": calculated["r2"],
                    "test_rmse": calculated["rmse"],
                    "test_mae": calculated["mae"],
                    "selected_alpha": selection.get("alpha"),
                    "validation_rmse": selection.get("value"),
                    "split_path": split.get("path"),
                }
            )
            predictions.insert(0, "state_name", STATE_NAMES.get(state, f"FIPS {state:02d}"))
            predictions.insert(0, "held_out_state", state)
            predictions.insert(0, "representation", representation)
            representation_predictions.append(predictions)

        assert representation_signature is not None
        pooled = pd.concat(representation_predictions, ignore_index=True)
        if pooled["key"].duplicated().any():
            raise ValueError(
                f"county-year appears in multiple held-out states for {representation}"
            )
        assert expected_test_keys is not None
        cohort_count = sum(len(expected_test_keys[state]) for state in expected_states)
        if cohort_count != len(pooled):
            raise ValueError(
                f"complete LOSO predictions for {representation} contain {len(pooled)} "
                f"county-years, but the split manifest defines {cohort_count}"
            )
        ordered = pooled.set_index("key").sort_index()
        if reference_keys is None:
            reference_keys = ordered.index.tolist()
            reference_observed = ordered["observed_yield"].astype(float)
        else:
            if ordered.index.tolist() != reference_keys:
                raise ValueError(f"county-year test-key drift for {representation}")
            assert reference_observed is not None
            if not np.allclose(
                ordered["observed_yield"].to_numpy(dtype=np.float64),
                reference_observed.to_numpy(dtype=np.float64),
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError(f"observed target drift for {representation}")
        representation_contracts[representation] = representation_signature
        prediction_tables.append(pooled)

    results = (
        pd.DataFrame(rows)
        .sort_values(["representation", "state_fips"])
        .reset_index(drop=True)
    )
    predictions = (
        pd.concat(prediction_tables, ignore_index=True)
        .sort_values(["representation", "held_out_state", "county_id", "year"])
        .reset_index(drop=True)
    )
    summary = (
        results.groupby(
            ["representation", "experiment_id", "representation_type", "feature_dim"],
            as_index=False,
        )
        .agg(
            states=("state_fips", "nunique"),
            county_years_tested=("test_n", "sum"),
            r2_mean=("test_r2", "mean"),
            r2_std=("test_r2", lambda values: values.std(ddof=0)),
            rmse_mean=("test_rmse", "mean"),
            rmse_std=("test_rmse", lambda values: values.std(ddof=0)),
            mae_mean=("test_mae", "mean"),
            mae_std=("test_mae", lambda values: values.std(ddof=0)),
        )
    )
    rng = np.random.default_rng(int(bootstrap_seed))
    bootstrap_counts = rng.multinomial(
        len(expected_states),
        np.full(len(expected_states), 1.0 / len(expected_states)),
        size=int(bootstrap_repetitions),
    )
    pooled_rows = []
    for representation, group in predictions.groupby("representation", sort=True):
        values = _metrics(group)
        uncertainty = _state_cluster_bootstrap(
            group, states=expected_states, counts=bootstrap_counts
        )
        pooled_rows.append(
            {
                "representation": representation,
                "pooled_r2": values["r2"],
                "pooled_rmse": values["rmse"],
                "pooled_mae": values["mae"],
                **uncertainty,
            }
        )
    summary = (
        summary.merge(pd.DataFrame(pooled_rows), on="representation", validate="one_to_one")
        .sort_values("representation")
        .reset_index(drop=True)
    )
    numeric_summary = summary[
        [
            "r2_mean",
            "r2_std",
            "rmse_mean",
            "rmse_std",
            "mae_mean",
            "mae_std",
            "pooled_r2",
            "pooled_r2_bootstrap_std",
            "pooled_rmse",
            "pooled_rmse_bootstrap_std",
            "pooled_mae",
            "pooled_mae_bootstrap_std",
        ]
    ]
    if not np.isfinite(numeric_summary.to_numpy(dtype=np.float64)).all():
        raise ValueError("aggregate LOSO metrics contain non-finite values")

    assert reference_keys is not None
    years = sorted(predictions["year"].astype(int).unique().tolist())
    contract = {
        "schema_version": 1,
        "experiment": "main_benchmark_loso_all_encoders_aggregate",
        "workflow_role": "spatial_generalization_ablation",
        "estimator_family": "ridge_regression",
        "input_directory": str(input_dir.resolve()),
        "representations": representation_contracts,
        "held_out_states": {
            str(state): STATE_NAMES.get(state, f"FIPS {state:02d}")
            for state in expected_states
        },
        "state_count": len(expected_states),
        "matched_county_years": len(reference_keys),
        "matched_counties": int(predictions["county_id"].nunique()),
        "cohort_years": years,
        "matched_key_sha256": hashlib.sha256(
            "\n".join(reference_keys).encode()
        ).hexdigest(),
        "split_path": str(split_path),
        "climate_fusion": {"source": "none", "stage": "none"},
        "target_units": "bushels_per_acre",
        "model_selection": {
            "source": "validation_state",
            "metric": "county_rmse",
            "hyperparameter": "ridge_alpha",
        },
        "aggregation": {
            "per_state": "metrics recomputed from held-out county-year predictions",
            "macro_summary": "unweighted mean and population standard deviation across held-out states",
            "pooled_summary": "metrics recomputed after concatenating all outer-test predictions",
            "standard_deviation_ddof": 0,
            "pooled_uncertainty": {
                "method": "held-out-state cluster bootstrap with replacement",
                "unit": "state",
                "repetitions": int(bootstrap_repetitions),
                "seed": int(bootstrap_seed),
                "reported_statistic": "bootstrap standard deviation of the pooled metric",
                "standard_deviation_ddof": 1,
                "paired_resamples_across_representations": True,
            },
        },
        "validated_common_run_signature": common_signature,
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "results_by_state.csv", index=False)
    summary.to_csv(out_dir / "summary_across_states.csv", index=False)
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    _write_json(out_dir / "aggregation_contract.json", contract)
    return {
        "results": results,
        "summary": summary,
        "predictions": predictions,
        "contract": contract,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--representations", nargs="+")
    parser.add_argument(
        "--expected-states", nargs="+", type=int, default=DEFAULT_STATES
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args(argv)
    output = aggregate_loso_runs(
        args.input_dir,
        out_dir=args.out_dir,
        expected_states=args.expected_states,
        expected_representations=args.representations,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(output["summary"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
