from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from benchmark_embeddings.loso_aggregate import aggregate_loso_runs


REPRESENTATIONS = ("encoder_a", "encoder_b")
STATES = (17, 18)
STATE_COUNTIES = {17: ("17001", "17003"), 18: ("18001", "18003")}


def _write_split(root: Path) -> Path:
    path = root / "loso.csv"
    if path.is_file():
        return path
    rows = []
    for fold in STATES:
        for state in STATES:
            for county_id in STATE_COUNTIES[state]:
                rows.append(
                    {
                        "county_id": county_id,
                        "year": 2020,
                        "fold": fold,
                        "split": "test" if state == fold else "train",
                    }
                )
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_run(root: Path, representation: str, state: int) -> Path:
    split_path = _write_split(root)
    run_dir = root / representation / str(state)
    run_dir.mkdir(parents=True)
    observed = np.array([160.0 + state, 180.0 + state])
    offset = 1.0 if representation == "encoder_a" else 2.0
    predicted = observed + np.array([offset, -offset])
    predictions = pd.DataFrame(
        {
            "county_id": STATE_COUNTIES[state],
            "year": [2020, 2020],
            "observed_yield": observed,
            "predicted_yield": predicted,
            "observed_yield_bu_per_acre": observed,
            "predicted_yield_bu_per_acre": predicted,
            "split_or_fold": [f"fold_{state}", f"fold_{state}"],
            "seed": [0, 0],
            "model_name": [representation, representation],
        }
    )
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    result = {
        "schema_version": 1,
        "selection": {
            "source": "validation",
            "metric": "county_rmse",
            "alpha": 10.0,
            "value": 3.0,
        },
        "test": {
            "county_r2": float(r2_score(observed, predicted)),
            "county_rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
            "county_mae": float(mean_absolute_error(observed, predicted)),
            "county_n": len(predictions),
        },
        "experiment": {
            "family": "main_benchmark",
            "representation_type": "frozen_embedding",
            "id": representation,
            "input_modalities": ["Sentinel-2"],
            "climate_fusion": {"source": "none", "stage": "none"},
            "aggregation": {"spatial": "mean_std", "temporal": "mean"},
        },
        "split": {"path": str(split_path), "fold": state},
        "target_and_metric_units": {
            "canonical_yield": "bushels_per_acre",
            "r2": "dimensionless",
            "rmse": "bushels_per_acre",
            "mae": "bushels_per_acre",
        },
        "cohort": {
            "county_years_before_daymet": 4,
            "county_years_after_daymet": 4,
            "feature_dim_before_daymet": 4 if representation == "encoder_a" else 6,
            "feature_dim_after_daymet": 4 if representation == "encoder_a" else 6,
        },
        "daymet": None,
    }
    (run_dir / "result.json").write_text(json.dumps(result))
    return run_dir


def _write_complete_grid(root: Path) -> None:
    for representation in REPRESENTATIONS:
        for state in STATES:
            _write_run(root, representation, state)


def test_loso_aggregate_validates_grid_and_writes_macro_and_pooled_metrics(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs"
    _write_complete_grid(run_root)
    output = aggregate_loso_runs(
        run_root,
        out_dir=tmp_path / "aggregate",
        expected_states=STATES,
        expected_representations=REPRESENTATIONS,
        bootstrap_repetitions=200,
        bootstrap_seed=7,
    )

    assert len(output["results"]) == len(REPRESENTATIONS) * len(STATES)
    assert set(output["summary"]["states"]) == {2}
    assert set(output["summary"]["county_years_tested"]) == {4}
    assert output["contract"]["matched_county_years"] == 4
    assert output["contract"]["climate_fusion"]["source"] == "none"
    assert output["contract"]["aggregation"]["pooled_uncertainty"]["unit"] == "state"
    assert output["contract"]["aggregation"]["pooled_uncertainty"]["repetitions"] == 200
    assert {
        "pooled_r2_bootstrap_std",
        "pooled_rmse_bootstrap_std",
        "pooled_mae_bootstrap_std",
    }.issubset(output["summary"].columns)
    assert np.isfinite(
        output["summary"][
            [
                "pooled_r2_bootstrap_std",
                "pooled_rmse_bootstrap_std",
                "pooled_mae_bootstrap_std",
            ]
        ].to_numpy()
    ).all()
    assert (tmp_path / "aggregate" / "results_by_state.csv").is_file()
    assert (tmp_path / "aggregate" / "summary_across_states.csv").is_file()
    assert (tmp_path / "aggregate" / "predictions.csv").is_file()
    assert (tmp_path / "aggregate" / "aggregation_contract.json").is_file()


def test_loso_aggregate_rejects_incomplete_state_grid(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    for representation in REPRESENTATIONS:
        for state in STATES:
            if (representation, state) != ("encoder_b", 18):
                _write_run(run_root, representation, state)

    with pytest.raises(ValueError, match="state coverage"):
        aggregate_loso_runs(
            run_root,
            out_dir=tmp_path / "aggregate",
            expected_states=STATES,
            expected_representations=REPRESENTATIONS,
        )


def test_loso_aggregate_rejects_prediction_metric_drift(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    _write_complete_grid(run_root)
    path = run_root / "encoder_b" / "18" / "predictions.csv"
    predictions = pd.read_csv(path)
    predictions.loc[0, "predicted_yield"] += 20.0
    predictions.loc[0, "predicted_yield_bu_per_acre"] += 20.0
    predictions.to_csv(path, index=False)

    with pytest.raises(ValueError, match="disagrees with predictions"):
        aggregate_loso_runs(
            run_root,
            out_dir=tmp_path / "aggregate",
            expected_states=STATES,
            expected_representations=REPRESENTATIONS,
        )
