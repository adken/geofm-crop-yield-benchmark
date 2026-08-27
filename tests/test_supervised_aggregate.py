from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from benchmark_embeddings.supervised_aggregate import aggregate_supervised_runs


MODELS = ("supervised_s2_3d_convlstm", "supervised_s2_gru")
FOLDS = (0, 1)
SEEDS = (0, 1)
KEYS = ("17001-2020", "17003-2020", "17005-2020", "17007-2020")


def _write_run(root: Path, model: str, fold: int, seed: int) -> Path:
    run_dir = root / model / f"fold_{fold}" / f"seed_{seed}"
    run_dir.mkdir(parents=True)
    test_keys = KEYS[:2] if fold == 0 else KEYS[2:]
    val_keys = [KEYS[2]] if fold == 0 else [KEYS[0]]
    train_keys = sorted(set(KEYS) - set(test_keys) - set(val_keys))
    split = {
        "mode": "primary",
        "path": "/same/folds.csv",
        "fold": fold,
        "validation_fold": 1 - fold,
        "label": f"fold_{fold}",
    }
    config = {
        "out_dir": str(run_dir),
        "seed": seed,
        "deterministic": True,
        "device": "cuda",
        "data": {"npz_dir": "/same/raw", "expected_spatial_size": [256, 256]},
        "split": {"mode": "primary", "path": "/same/folds.csv", "fold": fold},
        "normalization": {"max_patches_per_county": 8},
        "model": {"name": model.removeprefix("supervised_s2_"), "stem_channels": [4]},
        "training": {"learning_rate": 0.001},
    }
    (run_dir / "config_used.yaml").write_text(yaml.safe_dump(config))
    contract = {
        "num_county_years": len(KEYS),
        "input_contract": "[num_patches,7,10,256,256]",
        "bands": ["B02", "B03"],
        "patch_count_min": 2,
        "patch_count_median": 2.5,
        "patch_count_max": 3,
        "split": split,
        "partition_counts": {
            "train": len(train_keys),
            "val": len(val_keys),
            "test": len(test_keys),
        },
        "partition_keys": {"train": train_keys, "val": val_keys, "test": list(test_keys)},
    }
    (run_dir / "data_contract.json").write_text(json.dumps(contract))
    prediction = pd.DataFrame(
        {
            "county_id": [key.split("-")[0] for key in test_keys],
            "year": [2020, 2020],
            "observed_yield": [8.0 + KEYS.index(key) for key in test_keys],
            "predicted_yield": [
                8.1 + KEYS.index(key) + 0.01 * seed for key in test_keys
            ],
        }
    )
    prediction.to_csv(run_dir / "predictions.csv", index=False)
    observed = prediction["observed_yield"].to_numpy()
    predicted = prediction["predicted_yield"].to_numpy()
    result = {
        "experiment": {"id": model},
        "split": split,
        "seed": seed,
        "selection": {"epoch": 2, "value": 0.2},
        "test": {
            "county_r2": float(r2_score(observed, predicted)),
            "county_rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
            "county_mae": float(mean_absolute_error(observed, predicted)),
            "county_n": len(prediction),
        },
    }
    (run_dir / "result.json").write_text(json.dumps(result))
    return run_dir


def _all_runs(tmp_path: Path) -> list[Path]:
    return [
        _write_run(tmp_path, model, fold, seed)
        for model in MODELS
        for fold in FOLDS
        for seed in SEEDS
    ]


def test_supervised_aggregate_enforces_complete_matched_grid(tmp_path: Path) -> None:
    output = aggregate_supervised_runs(
        _all_runs(tmp_path / "runs"),
        out_dir=tmp_path / "summary",
        expected_models=MODELS,
        expected_folds=FOLDS,
        expected_seeds=SEEDS,
    )

    assert len(output["results"]) == len(MODELS) * len(FOLDS) * len(SEEDS)
    assert set(output["summary"]["folds"]) == {2}
    assert set(output["summary"]["seeds_per_fold"]) == {2}
    assert set(output["summary"]["county_years_tested"]) == {4}
    assert output["data_contract"]["matched_county_years"] == 4
    assert output["data_contract"]["workflow_role"] == (
        "supervised_sentinel2_benchmark"
    )
    assert output["data_contract"]["estimator_family"] == "supervised_deep_learning"
    assert output["data_contract"]["year_policy"] == (
        "all_cohort_years_in_each_train_validation_test_partition"
    )
    assert (tmp_path / "summary" / "predictions.csv").is_file()


def test_supervised_aggregate_rejects_protocol_or_prediction_drift(tmp_path: Path) -> None:
    runs = _all_runs(tmp_path / "protocol")
    config_path = runs[-1] / "config_used.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["training"]["learning_rate"] = 0.1
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="configuration drift"):
        aggregate_supervised_runs(
            runs,
            out_dir=tmp_path / "unused",
            expected_models=MODELS,
            expected_folds=FOLDS,
            expected_seeds=SEEDS,
        )

    runs = _all_runs(tmp_path / "prediction")
    prediction_path = runs[-1] / "predictions.csv"
    prediction = pd.read_csv(prediction_path)
    prediction.loc[0, "predicted_yield"] += 2.0
    prediction.to_csv(prediction_path, index=False)
    with pytest.raises(ValueError, match="disagrees with predictions"):
        aggregate_supervised_runs(
            runs,
            out_dir=tmp_path / "unused2",
            expected_models=MODELS,
            expected_folds=FOLDS,
            expected_seeds=SEEDS,
        )
