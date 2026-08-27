#!/usr/bin/env python3
"""Matched five-fold tabular regressors for main and climate-fusion benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import (
    FoldPartitions,
    load_fold_partitions,
    validate_all_years_in_partitions,
    years_from_keys,
)
from .daymet import fuse_daymet_features, load_daymet_features
from .frozen import read_embeddings
from .loyo import (
    EXPECTED_BACKBONES,
    REPRESENTATIONS as MAIN_REPRESENTATIONS,
    load_embedding_features,
    load_sentinel2_features,
    validate_matched_main_cohort,
)
from .probe import pool_county_embeddings, two_stage_mean_aggregation_contract


FAMILY_MAIN = "main"
FAMILY_CLIMATE = "climate_fusion"
FAMILIES = (FAMILY_MAIN, FAMILY_CLIMATE)
CLIMATE_REPRESENTATIONS = (
    "clay_daymet",
    "prithvi_daymet",
    "terramind_daymet",
    "presto_daymet",
    "presto_era5",
    "sentinel2_indices_daymet",
)
REGRESSOR_RIDGE = "ridge"
REGRESSOR_RF = "random_forest"
REGRESSOR_XGB = "xgboost"
REGRESSOR_EBM = "ebm"
REGRESSORS = (REGRESSOR_RIDGE, REGRESSOR_RF, REGRESSOR_XGB, REGRESSOR_EBM)
DEFAULT_FOLDS = (0, 1, 2, 3, 4)
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)


def regressor_registry(*, n_jobs: int, ebm_interactions: int = 0) -> dict[str, dict[str, Any]]:
    """One explicit configuration shared by every representation."""
    return {
        REGRESSOR_RIDGE: {
            "estimator": "StandardScaler+Ridge",
            "selection": "validation_county_rmse",
            "alpha_grid": list(DEFAULT_RIDGE_ALPHAS),
            "scaling": "fit_on_train_for_selection_then_train_plus_validation_for_refit",
            "seed_policy": "deterministic_once",
        },
        REGRESSOR_RF: {
            "estimator": "sklearn.ensemble.RandomForestRegressor",
            "n_estimators": 600,
            "criterion": "squared_error",
            "max_depth": None,
            "min_samples_leaf": 2,
            "max_features": 1.0,
            "bootstrap": True,
            "n_jobs": int(n_jobs),
            "selection": "fixed_preregistered",
            "seed_policy": "shared_stochastic_seeds",
        },
        REGRESSOR_XGB: {
            "estimator": "xgboost.XGBRegressor",
            "n_estimators": 600,
            "learning_rate": 0.03,
            "max_depth": 6,
            "min_child_weight": 2,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "n_jobs": int(n_jobs),
            "selection": "fixed_preregistered",
            "seed_policy": "shared_stochastic_seeds",
        },
        # Library defaults except interactions, matching scripts/run_main_table.py.
        # The earlier preregistered configuration set twelve hyperparameters
        # explicitly (outer_bags=14, learning_rate=0.04, max_leaves=2, ...) and
        # underfits by roughly 0.09 R2 relative to the defaults. Keeping it here
        # would make the EBM column of the climate-fusion table differ from the
        # EBM column of the main table for reasons unrelated to climate, which
        # is exactly the comparison the fusion table exists to support.
        REGRESSOR_EBM: {
            "estimator": "interpret.glassbox.ExplainableBoostingRegressor",
            "interactions": int(ebm_interactions),
            "hyperparameters": "library_defaults_except_interactions",
            "n_jobs": int(n_jobs),
            "selection": "fixed_preregistered",
            "seed_policy": "shared_stochastic_seeds",
        },
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n")


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    observed = np.asarray(observed, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if observed.shape != predicted.shape or observed.size == 0:
        raise ValueError("metrics require aligned non-empty county predictions")
    return {
        "r2": float(r2_score(observed, predicted)) if observed.size > 1 else None,
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "n": int(observed.size),
    }


def _package_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except ImportError:
        return None
    return str(getattr(module, "__version__", "unknown"))


def require_regressor_dependencies(regressors: Sequence[str]) -> dict[str, str | None]:
    versions = {
        "scikit_learn": sklearn.__version__,
        "xgboost": _package_version("xgboost"),
        "interpret": _package_version("interpret"),
    }
    missing = []
    if REGRESSOR_XGB in regressors and versions["xgboost"] is None:
        missing.append("xgboost")
    if REGRESSOR_EBM in regressors and versions["interpret"] is None:
        missing.append("interpret")
    if missing:
        raise ImportError(
            "selected canonical regressors require missing packages "
            f"{missing}; install benchmark-embeddings[tabular]"
        )
    return versions


def load_all_fold_partitions(
    split_path: str | Path,
    *,
    expected_folds: Sequence[int],
    cohort_keys: Sequence[str],
) -> dict[int, FoldPartitions]:
    folds = tuple(int(value) for value in expected_folds)
    if not folds or len(set(folds)) != len(folds):
        raise ValueError("unique outer folds are required")
    manifest = pd.read_csv(split_path)
    if "fold" not in manifest:
        raise ValueError("split manifest lacks fold column")
    manifest_folds = sorted(pd.to_numeric(manifest["fold"], errors="raise").astype(int).unique())
    if manifest_folds != sorted(folds):
        raise ValueError(f"split folds are {manifest_folds}, expected {sorted(folds)}")
    available = set(cohort_keys)
    cohort_years = years_from_keys(cohort_keys)
    partitions = {}
    test_occurrences = {key: 0 for key in cohort_keys}
    for fold in folds:
        parts = load_fold_partitions(split_path, fold=fold, id_column="fips_year")
        assigned = set(parts.train) | set(parts.val) | set(parts.test)
        missing = sorted(available - assigned)
        extra = sorted(assigned - available)
        if missing or extra:
            raise ValueError(
                f"fold {fold} cohort mismatch: missing={missing[:5]}, extra={extra[:5]}"
            )
        for key in parts.test:
            test_occurrences[key] += 1
        validate_all_years_in_partitions(parts, expected_years=cohort_years)
        partitions[fold] = parts
    bad = sorted(key for key, count in test_occurrences.items() if count != 1)
    if bad:
        raise ValueError(
            "every county-year must occur in exactly one outer test fold; "
            f"violations={bad[:5]}"
        )
    return partitions


def split_manifest_cohort(
    split_path: str | Path,
    *,
    expected_folds: Sequence[int],
) -> list[str]:
    """Return the explicit benchmark cohort shared by every outer fold."""
    manifest = pd.read_csv(split_path, dtype={"fips_year": str})
    required = {"fips_year", "fold", "split"}
    missing_columns = sorted(required - set(manifest.columns))
    if missing_columns:
        raise ValueError(f"split manifest lacks columns {missing_columns}")
    folds = sorted(int(value) for value in expected_folds)
    observed_folds = sorted(
        pd.to_numeric(manifest["fold"], errors="raise").astype(int).unique()
    )
    if observed_folds != folds:
        raise ValueError(f"split folds are {observed_folds}, expected {folds}")
    manifest["fold"] = pd.to_numeric(manifest["fold"], errors="raise").astype(int)
    manifest["fips_year"] = manifest["fips_year"].astype(str).str.strip()
    if manifest["fips_year"].eq("").any():
        raise ValueError("split manifest contains an empty fips_year")
    reference: set[str] | None = None
    for fold in folds:
        fold_rows = manifest.loc[manifest["fold"] == fold]
        if fold_rows["fips_year"].duplicated().any():
            raise ValueError(f"split fold {fold} contains duplicate county-year rows")
        keys = set(fold_rows["fips_year"])
        if reference is None:
            reference = keys
        elif keys != reference:
            raise ValueError(
                f"split fold {fold} uses a different cohort: "
                f"missing={sorted(reference-keys)[:5]}, extra={sorted(keys-reference)[:5]}"
            )
    if not reference:
        raise ValueError("split manifest has an empty cohort")
    return sorted(reference)


def _subset_frame_to_cohort(
    frame: pd.DataFrame,
    *,
    keys: Sequence[str],
    name: str,
) -> tuple[pd.DataFrame, int]:
    indexed = frame.set_index("key", drop=False)
    if not indexed.index.is_unique:
        raise ValueError(f"{name} has duplicate county-year keys")
    requested = set(keys)
    available = set(indexed.index)
    missing = sorted(requested - available)
    if missing:
        raise ValueError(
            f"cohort mismatch for {name}: missing common split keys={missing[:5]}"
        )
    extra = available - requested
    return indexed.loc[list(keys)].reset_index(drop=True), len(extra)


def _validate_presto_era5(
    path: str | Path,
    labels: pd.DataFrame,
    *,
    expected_timesteps: int,
    temporal_pool: str = "mean",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    embeddings = read_embeddings(path)
    backbones = sorted(embeddings["backbone"].astype(str).unique())
    if backbones != ["presto_s2_era5"]:
        raise ValueError(f"native Presto+ERA5 requires presto_s2_era5, got {backbones}")
    if "experiment_family" in embeddings:
        families = sorted(embeddings["experiment_family"].astype(str).str.lower().unique())
        if families != ["auxiliary_climate_fusion"]:
            raise ValueError(f"Presto+ERA5 has invalid experiment family {families}")
    if "input_modalities" in embeddings:
        modalities = sorted(embeddings["input_modalities"].astype(str).unique())
        if len(modalities) != 1 or "era5" not in modalities[0].lower():
            raise ValueError("Presto+ERA5 provenance must record ERA5-Land input")
    data = pool_county_embeddings(
        embeddings,
        labels,
        temporal_pool=temporal_pool,
        spatial_pool="mean_std",
        expected_timesteps=expected_timesteps,
    ).sort_values("key").reset_index(drop=True)
    return data, {
        "backbone": "presto_s2_era5",
        "path": str(Path(path).resolve()),
        "embedding_rows": int(len(embeddings)),
        "county_years": int(len(data)),
        "feature_dim": int(np.asarray(data["features"].iloc[0]).size),
        "spatial_pool": "mean_std_population",
        "temporal_pool": "preencoded_global_pool",
        "climate_source": "ERA5-Land",
        "fusion_stage": "presto_encoder_input",
    }


def _validate_equal_keys_and_targets(
    frames: Mapping[str, pd.DataFrame],
    *,
    reference_name: str,
) -> list[str]:
    reference = frames[reference_name].set_index("key")
    keys = sorted(reference.index.tolist())
    key_set = set(keys)
    for name, frame in frames.items():
        candidate = frame.set_index("key")
        if not candidate.index.is_unique:
            raise ValueError(f"{name} has duplicate county-year keys")
        candidate_keys = set(candidate.index)
        if candidate_keys != key_set:
            raise ValueError(
                f"county-year cohort mismatch for {name}: "
                f"missing={sorted(key_set-candidate_keys)[:5]}, "
                f"extra={sorted(candidate_keys-key_set)[:5]}"
            )
        if not np.allclose(
            reference.loc[keys, "yield_bu_per_acre"].to_numpy(),
            candidate.loc[keys, "yield_bu_per_acre"].to_numpy(),
            atol=0.0,
            rtol=0.0,
        ):
            raise ValueError(f"yield target mismatch between {reference_name} and {name}")
    return keys


def _validate_patch_parity(
    frames: Mapping[str, pd.DataFrame],
    *,
    names: Sequence[str],
    keys: Sequence[str],
) -> None:
    reference_name = names[0]
    reference = frames[reference_name].set_index("key")
    for name in names[1:]:
        candidate = frames[name].set_index("key")
        if not np.array_equal(
            reference.loc[list(keys), "n_patches"].to_numpy(),
            candidate.loc[list(keys), "n_patches"].to_numpy(),
        ):
            raise ValueError(f"complete-patch count mismatch: {reference_name} versus {name}")
        bad = [
            key
            for key in keys
            if tuple(reference.loc[key, "complete_patch_ids"])
            != tuple(candidate.loc[key, "complete_patch_ids"])
        ]
        if bad:
            raise ValueError(
                f"complete-patch identity mismatch: {reference_name} versus {name}: {bad[:5]}"
            )


def load_main_family(
    embedding_paths: Mapping[str, str | Path],
    *,
    sentinel2_indices_path: str | Path,
    sentinel2_fips_map: str | Path | None,
    labels: pd.DataFrame,
    expected_timesteps: int,
    cohort_keys: Sequence[str],
    temporal_pool: str = "mean",
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], list[str], str]:
    expected = set(MAIN_REPRESENTATIONS) - {"sentinel2_indices"}
    if set(embedding_paths) != expected:
        raise ValueError(f"main embedding paths must contain exactly {sorted(expected)}")
    frames: dict[str, pd.DataFrame] = {}
    contracts: dict[str, Any] = {}
    for name in MAIN_REPRESENTATIONS[:-1]:
        frames[name], contracts[name] = load_embedding_features(
            name,
            embedding_paths[name],
            labels,
            expected_timesteps=expected_timesteps,
            temporal_pool=temporal_pool,
        )
        contracts[name]["experiment_family"] = "main_benchmark"
        contracts[name]["added_climate_features"] = []
    frames["sentinel2_indices"], contracts["sentinel2_indices"] = load_sentinel2_features(
        sentinel2_indices_path,
        labels,
        fips_map=sentinel2_fips_map,
        expected_timesteps=expected_timesteps,
    )
    contracts["sentinel2_indices"]["experiment_family"] = "main_benchmark"
    for name in MAIN_REPRESENTATIONS:
        source_rows = len(frames[name])
        frames[name], excluded = _subset_frame_to_cohort(
            frames[name], keys=cohort_keys, name=name
        )
        contracts[name]["source_county_years_before_cohort_filter"] = source_rows
        contracts[name]["county_years"] = len(frames[name])
        contracts[name]["rows_outside_common_cohort_excluded"] = excluded
    keys = validate_matched_main_cohort(frames)
    patch_lookup = frames["clay"].set_index("key")["complete_patch_ids"]
    patch_hash = _patch_hash(keys, patch_lookup)
    return frames, contracts, keys, patch_hash


def _patch_hash(keys: Sequence[str], lookup: pd.Series) -> str:
    separator = "\x1f"
    return hashlib.sha256(
        "\n".join(
            f"{key}{separator}{separator.join(tuple(lookup.loc[key]))}" for key in keys
        ).encode("utf-8")
    ).hexdigest()


def load_climate_family(
    embedding_paths: Mapping[str, str | Path],
    *,
    presto_era5_path: str | Path,
    sentinel2_indices_path: str | Path,
    sentinel2_fips_map: str | Path | None,
    daymet_path: str | Path,
    daymet_fips_map: str | Path | None,
    labels: pd.DataFrame,
    expected_timesteps: int,
    cohort_keys: Sequence[str],
    temporal_pool: str = "mean",
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], list[str], str]:
    expected = {"clay", "prithvi", "terramind", "presto"}
    if set(embedding_paths) != expected:
        raise ValueError(f"climate base paths must contain exactly {sorted(expected)}")
    source_frames: dict[str, pd.DataFrame] = {}
    source_contracts: dict[str, Any] = {}
    for name in ("clay", "prithvi", "terramind", "presto"):
        source_frames[name], source_contracts[name] = load_embedding_features(
            name,
            embedding_paths[name],
            labels,
            expected_timesteps=expected_timesteps,
            temporal_pool=temporal_pool,
        )
    source_frames["presto_era5"], source_contracts["presto_era5"] = (
        _validate_presto_era5(
            presto_era5_path,
            labels,
            expected_timesteps=expected_timesteps,
            temporal_pool=temporal_pool,
        )
    )
    source_frames["sentinel2_indices"], source_contracts["sentinel2_indices"] = (
        load_sentinel2_features(
            sentinel2_indices_path,
            labels,
            fips_map=sentinel2_fips_map,
            expected_timesteps=expected_timesteps,
        )
    )
    for name, frame in list(source_frames.items()):
        source_rows = len(frame)
        source_frames[name], excluded = _subset_frame_to_cohort(
            frame, keys=cohort_keys, name=name
        )
        source_contracts[name]["source_county_years_before_cohort_filter"] = source_rows
        source_contracts[name]["county_years"] = len(source_frames[name])
        source_contracts[name]["rows_outside_common_cohort_excluded"] = excluded
    keys = _validate_equal_keys_and_targets(source_frames, reference_name="clay")
    _validate_patch_parity(
        source_frames,
        names=("clay", "prithvi", "terramind", "presto", "presto_era5"),
        keys=keys,
    )
    daymet = load_daymet_features(
        daymet_path,
        fips_map=daymet_fips_map,
        expected_timesteps=expected_timesteps,
    )
    s2_schedule = source_contracts["sentinel2_indices"]["interval_schedule"]
    if list(daymet.attrs["interval_schedule"]) != list(s2_schedule):
        raise ValueError(
            "Daymet and Sentinel-2 indices use different interval schedules: "
            f"{daymet.attrs['interval_schedule']} versus {s2_schedule}"
        )
    daymet_keys = set(daymet["county_id"] + "-" + daymet["year"].astype(str))
    benchmark_keys = set(keys)
    missing_daymet = sorted(benchmark_keys - daymet_keys)
    if missing_daymet:
        raise ValueError(
            "Daymet must cover every benchmark county-year; "
            f"missing={missing_daymet[:5]}"
        )
    daymet_attrs = dict(daymet.attrs)
    extra_daymet = sorted(daymet_keys - benchmark_keys)
    daymet_key = daymet["county_id"] + "-" + daymet["year"].astype(str)
    daymet = daymet.loc[daymet_key.isin(benchmark_keys)].copy()
    daymet.attrs.update(daymet_attrs)
    daymet.attrs["rows_outside_benchmark_excluded"] = len(extra_daymet)
    frames = {
        "clay_daymet": fuse_daymet_features(source_frames["clay"], daymet),
        "prithvi_daymet": fuse_daymet_features(source_frames["prithvi"], daymet),
        "terramind_daymet": fuse_daymet_features(source_frames["terramind"], daymet),
        "presto_daymet": fuse_daymet_features(source_frames["presto"], daymet),
        "presto_era5": source_frames["presto_era5"].copy(),
        "sentinel2_indices_daymet": fuse_daymet_features(
            source_frames["sentinel2_indices"], daymet
        ),
    }
    keys = _validate_equal_keys_and_targets(frames, reference_name="clay_daymet")
    daymet_names = list(daymet.attrs["feature_names"])
    contracts = {}
    for output_name, source_name in (
        ("clay_daymet", "clay"),
        ("prithvi_daymet", "prithvi"),
        ("terramind_daymet", "terramind"),
        ("presto_daymet", "presto"),
        ("sentinel2_indices_daymet", "sentinel2_indices"),
    ):
        contracts[output_name] = {
            **source_contracts[source_name],
            "backbone": f"{source_contracts[source_name]['backbone']}_daymet_late_fusion",
            "experiment_family": "auxiliary_climate_fusion",
            "climate_source": "Daymet",
            "fusion_stage": "county_year_late",
            "added_regressor_feature_count": len(daymet_names),
            "added_regressor_features": daymet_names,
            "daymet_rows_outside_benchmark_excluded": int(
                daymet.attrs["rows_outside_benchmark_excluded"]
            ),
            "feature_dim_after_fusion": int(np.asarray(frames[output_name]["features"].iloc[0]).size),
        }
    contracts["presto_era5"] = {
        **source_contracts["presto_era5"],
        "experiment_family": "auxiliary_climate_fusion",
        "encoder_climate_variable_count": 2,
        "encoder_climate_variables": ["temperature_2m", "total_precipitation"],
        "added_regressor_feature_count": 0,
        "added_regressor_features": [],
    }
    patch_lookup = source_frames["clay"].set_index("key")["complete_patch_ids"]
    return frames, contracts, keys, _patch_hash(keys, patch_lookup)


def _make_fixed_estimator(
    regressor: str,
    *,
    seed: int,
    registry: Mapping[str, Mapping[str, Any]],
):
    config = registry[regressor]
    if regressor == REGRESSOR_RF:
        return RandomForestRegressor(
            n_estimators=config["n_estimators"],
            criterion=config["criterion"],
            max_depth=config["max_depth"],
            min_samples_leaf=config["min_samples_leaf"],
            max_features=config["max_features"],
            bootstrap=config["bootstrap"],
            n_jobs=config["n_jobs"],
            random_state=int(seed),
        )
    if regressor == REGRESSOR_XGB:
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=config["n_estimators"],
            learning_rate=config["learning_rate"],
            max_depth=config["max_depth"],
            min_child_weight=config["min_child_weight"],
            subsample=config["subsample"],
            colsample_bytree=config["colsample_bytree"],
            reg_alpha=config["reg_alpha"],
            reg_lambda=config["reg_lambda"],
            objective=config["objective"],
            tree_method=config["tree_method"],
            n_jobs=config["n_jobs"],
            random_state=int(seed),
        )
    if regressor == REGRESSOR_EBM:
        from interpret.glassbox import ExplainableBoostingRegressor

        return ExplainableBoostingRegressor(
            interactions=config["interactions"],
            n_jobs=config["n_jobs"],
            random_state=int(seed),
        )
    raise ValueError(f"{regressor} is not a fixed nonlinear regressor")


def evaluate_regressor_fold(
    frame: pd.DataFrame,
    parts: FoldPartitions,
    *,
    regressor: str,
    seed: int,
    registry: Mapping[str, Mapping[str, Any]],
    ridge_alphas: Sequence[float],
) -> tuple[dict[str, Any], pd.DataFrame]:
    indexed = frame.set_index("key", drop=False)
    train = indexed.loc[parts.train]
    val = indexed.loc[parts.val]
    test = indexed.loc[parts.test]
    x_train = np.stack(train["features"]).astype(np.float32)
    y_train = train["yield_bu_per_acre"].to_numpy(dtype=np.float64)
    x_val = np.stack(val["features"]).astype(np.float32)
    y_val = val["yield_bu_per_acre"].to_numpy(dtype=np.float64)
    fit = pd.concat([train, val], ignore_index=True)
    x_fit = np.stack(fit["features"]).astype(np.float32)
    y_fit = fit["yield_bu_per_acre"].to_numpy(dtype=np.float64)
    x_test = np.stack(test["features"]).astype(np.float32)
    y_test = test["yield_bu_per_acre"].to_numpy(dtype=np.float64)
    if regressor == REGRESSOR_RIDGE:
        candidates = []
        for alpha in ridge_alphas:
            candidate = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))
            candidate.fit(x_train, y_train)
            val_prediction = candidate.predict(x_val)
            candidates.append(
                (float(np.sqrt(mean_squared_error(y_val, val_prediction))), float(alpha))
            )
        validation_rmse, selected_alpha = min(candidates)
        model = make_pipeline(StandardScaler(), Ridge(alpha=selected_alpha))
        selection = {
            "source": "shared_validation_partition",
            "metric": "county_rmse",
            "selected_alpha": selected_alpha,
            "value": validation_rmse,
            "candidate_alphas": [float(value) for value in ridge_alphas],
        }
    else:
        model = _make_fixed_estimator(regressor, seed=seed, registry=registry)
        selection = {
            "source": "fixed_preregistered_configuration",
            "metric": None,
            "selected_alpha": None,
            "value": None,
        }
    model.fit(x_fit, y_fit)
    prediction = np.asarray(model.predict(x_test), dtype=np.float64)
    metrics = _metrics(y_test, prediction)
    predictions = test[["county_id", "year", "key"]].copy()
    predictions["observed_yield"] = y_test
    predictions["predicted_yield"] = prediction
    predictions["observed_yield_bu_per_acre"] = y_test
    predictions["predicted_yield_bu_per_acre"] = prediction
    return {"selection": selection, "test": metrics}, predictions


def summarize_results(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_fold = (
        results.groupby(
            ["family", "representation", "backbone", "regressor", "fold"],
            as_index=False,
        )
        .agg(
            seeds=("seed", "nunique"),
            test_n=("test_n", "first"),
            r2=("test_r2", "mean"),
            r2_seed_std=("test_r2", lambda values: values.std(ddof=0)),
            rmse=("test_rmse", "mean"),
            rmse_seed_std=("test_rmse", lambda values: values.std(ddof=0)),
            rmse_bu_per_acre=("test_rmse_bu_per_acre", "mean"),
            rmse_bu_per_acre_seed_std=(
                "test_rmse_bu_per_acre",
                lambda values: values.std(ddof=0),
            ),
            mae=("test_mae", "mean"),
            mae_seed_std=("test_mae", lambda values: values.std(ddof=0)),
        )
        .sort_values(["representation", "regressor", "fold"])
        .reset_index(drop=True)
    )
    summary = (
        by_fold.groupby(
            ["family", "representation", "backbone", "regressor"], as_index=False
        )
        .agg(
            folds=("fold", "nunique"),
            seeds_per_fold=("seeds", "first"),
            test_n_total=("test_n", "sum"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", lambda values: values.std(ddof=0)),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", lambda values: values.std(ddof=0)),
            rmse_bu_per_acre_mean=("rmse_bu_per_acre", "mean"),
            rmse_bu_per_acre_std=(
                "rmse_bu_per_acre",
                lambda values: values.std(ddof=0),
            ),
            mae_mean=("mae", "mean"),
            mae_std=("mae", lambda values: values.std(ddof=0)),
        )
        .sort_values(["regressor", "representation"])
        .reset_index(drop=True)
    )
    return by_fold, summary


def run_regression_benchmark(
    *,
    family: str,
    embedding_paths: Mapping[str, str | Path],
    sentinel2_indices_path: str | Path,
    labels_path: str | Path,
    split_path: str | Path,
    out_dir: str | Path,
    sentinel2_fips_map: str | Path | None = None,
    presto_era5_path: str | Path | None = None,
    daymet_path: str | Path | None = None,
    daymet_fips_map: str | Path | None = None,
    folds: Sequence[int] = DEFAULT_FOLDS,
    regressors: Sequence[str] = REGRESSORS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    ridge_alphas: Sequence[float] = DEFAULT_RIDGE_ALPHAS,
    expected_timesteps: int = 7,
    n_jobs: int = -1,
    temporal_pool: str = "mean",
    ebm_interactions: int = 0,
    preflight_only: bool = False,
) -> dict[str, Any]:
    family = str(family).strip().lower()
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}")
    regressors = tuple(str(value).strip().lower() for value in regressors)
    seeds = tuple(int(value) for value in seeds)
    ridge_alphas = tuple(float(value) for value in ridge_alphas)
    if not regressors or any(value not in REGRESSORS for value in regressors):
        raise ValueError(f"regressors must be selected from {REGRESSORS}")
    if len(set(regressors)) != len(regressors):
        raise ValueError("regressors must not contain duplicates")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("unique stochastic seeds are required")
    if not ridge_alphas or any(value <= 0.0 for value in ridge_alphas):
        raise ValueError("positive Ridge alpha candidates are required")
    versions = require_regressor_dependencies(regressors)
    temporal_pool = str(temporal_pool).strip().lower()
    if temporal_pool not in {"mean", "joint"}:
        raise ValueError("temporal_pool must be 'mean' or 'joint'")
    registry = regressor_registry(n_jobs=n_jobs, ebm_interactions=ebm_interactions)
    registry[REGRESSOR_RIDGE]["alpha_grid"] = list(ridge_alphas)
    cohort_keys = split_manifest_cohort(split_path, expected_folds=folds)
    labels = pd.read_csv(labels_path)
    if family == FAMILY_MAIN:
        frames, contracts, common_keys, patch_hash = load_main_family(
            embedding_paths,
            sentinel2_indices_path=sentinel2_indices_path,
            sentinel2_fips_map=sentinel2_fips_map,
            labels=labels,
            expected_timesteps=expected_timesteps,
            cohort_keys=cohort_keys,
            temporal_pool=temporal_pool,
        )
    else:
        if presto_era5_path is None or daymet_path is None:
            raise ValueError("climate_fusion requires presto_era5_path and daymet_path")
        frames, contracts, common_keys, patch_hash = load_climate_family(
            embedding_paths,
            presto_era5_path=presto_era5_path,
            sentinel2_indices_path=sentinel2_indices_path,
            sentinel2_fips_map=sentinel2_fips_map,
            daymet_path=daymet_path,
            daymet_fips_map=daymet_fips_map,
            labels=labels,
            expected_timesteps=expected_timesteps,
            cohort_keys=cohort_keys,
            temporal_pool=temporal_pool,
        )
    partitions = load_all_fold_partitions(
        split_path,
        expected_folds=folds,
        cohort_keys=common_keys,
    )
    cohort_years = years_from_keys(common_keys)
    split_contract = {
        str(fold): {
            "validation_fold": parts.validation_fold,
            "train_keys": parts.train,
            "validation_keys": parts.val,
            "test_keys": parts.test,
            "years": validate_all_years_in_partitions(
                parts, expected_years=cohort_years
            ),
        }
        for fold, parts in sorted(partitions.items())
    }
    cohort_hash = hashlib.sha256("\n".join(common_keys).encode("utf-8")).hexdigest()
    data_contract = {
        "schema_version": 1,
        "experiment": "matched_tabular_regression_benchmark",
        "workflow_role": "main_encoder_embedding_comparison" if family == FAMILY_MAIN else "auxiliary_climate_fusion_comparison",
        "estimator_family": "classical_ml",
        "family": family,
        "representations": contracts,
        "matched_county_years": len(common_keys),
        "matched_key_sha256": cohort_hash,
        "matched_complete_patch_identity_sha256": patch_hash,
        "labels": str(Path(labels_path).resolve()),
        "split": {
            "path": str(Path(split_path).resolve()),
            "outer_folds": sorted(int(value) for value in folds),
            "grouping": "county_all_years_together",
            "cohort_years": cohort_years,
            "year_policy": "all_cohort_years_in_each_train_validation_test_partition",
            "partitions": split_contract,
        },
        "regressor_registry": {name: registry[name] for name in regressors},
        "stochastic_seeds": seeds,
        "ridge_seed_policy": "deterministic_once_seed_0",
        "feature_aggregation": two_stage_mean_aggregation_contract(temporal_pool),
        "aggregation": {
            "unit": "county_year",
            "within_fold": "mean_across_stochastic_seeds",
            "across_folds": "mean_and_population_standard_deviation",
        },
        "target_and_metric_units": {
            "canonical_yield": "bushels_per_acre",
            "r2": "dimensionless",
            "rmse": "bushels_per_acre",
            "mae": "bushels_per_acre",
        },
        "software": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            **versions,
        },
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "data_contract.json", data_contract)
    if preflight_only:
        return {"data_contract": data_contract, "results": None}

    results = []
    prediction_tables = []
    for representation, frame in frames.items():
        aligned = frame.set_index("key").loc[common_keys].reset_index()
        backbone = contracts[representation]["backbone"]
        for fold, parts in sorted(partitions.items()):
            for regressor in regressors:
                run_seeds = (0,) if regressor == REGRESSOR_RIDGE else seeds
                for seed in run_seeds:
                    output, predictions = evaluate_regressor_fold(
                        aligned,
                        parts,
                        regressor=regressor,
                        seed=seed,
                        registry=registry,
                        ridge_alphas=ridge_alphas,
                    )
                    selection = output["selection"]
                    metrics = output["test"]
                    result = {
                        "family": family,
                        "representation": representation,
                        "backbone": backbone,
                        "regressor": regressor,
                        "fold": int(fold),
                        "validation_fold": parts.validation_fold,
                        "seed": int(seed),
                        "train_n": len(parts.train),
                        "validation_n": len(parts.val),
                        "test_n": metrics["n"],
                        "selected_alpha": selection["selected_alpha"],
                        "validation_rmse": selection["value"],
                        "test_r2": metrics["r2"],
                        "test_rmse": metrics["rmse"],
                        "test_rmse_bu_per_acre": metrics["rmse"],
                        "test_mae": metrics["mae"],
                    }
                    results.append(result)
                    predictions["family"] = family
                    predictions["representation"] = representation
                    predictions["backbone"] = backbone
                    predictions["regressor"] = regressor
                    predictions["fold"] = int(fold)
                    predictions["validation_fold"] = parts.validation_fold
                    predictions["seed"] = int(seed)
                    predictions["model_name"] = f"{family}_{regressor}"
                    prediction_tables.append(predictions)
    result_frame = pd.DataFrame(results)
    prediction_frame = pd.concat(prediction_tables, ignore_index=True)
    by_fold, summary = summarize_results(result_frame)
    expected_seed_counts = {
        name: 1 if name == REGRESSOR_RIDGE else len(seeds) for name in regressors
    }
    for (representation, regressor, fold), group in result_frame.groupby(
        ["representation", "regressor", "fold"]
    ):
        if len(group) != expected_seed_counts[regressor]:
            raise RuntimeError(
                f"incomplete runs for {representation}/{regressor}/fold {fold}"
            )
        if group["test_n"].nunique() != 1:
            raise RuntimeError("test count drift across stochastic seeds")
    prediction_key = ["representation", "regressor", "seed", "key"]
    if prediction_frame.duplicated(prediction_key).any():
        raise RuntimeError("a county-year appears in more than one outer test fold")
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
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--clay", required=True)
    parser.add_argument("--prithvi", required=True)
    parser.add_argument("--terramind", required=True)
    parser.add_argument("--presto", required=True)
    parser.add_argument("--alphaearth")
    parser.add_argument("--presto-era5")
    parser.add_argument("--s2-indices")
    parser.add_argument(
        "--s2-daymet-merged",
        help=(
            "Merged county-year table containing the 21 Sentinel-2 indices and "
            "35 Daymet variables; accepted as CSV, Excel, or Parquet"
        ),
    )
    parser.add_argument("--s2-indices-fips-map")
    parser.add_argument("--daymet")
    parser.add_argument("--daymet-fips-map")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--regressors", nargs="+", choices=REGRESSORS, default=REGRESSORS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--ridge-alphas", nargs="+", type=float, default=DEFAULT_RIDGE_ALPHAS)
    parser.add_argument(
        "--temporal-pool",
        choices=("mean", "joint"),
        default="mean",
        help="County pooling over the patch-timestep axis. 'joint' takes one "
             "mean and population standard deviation over every patch-timestep "
             "row at once, matching the main benchmark, LOYO and LOSO; 'mean' "
             "is the earlier two-stage form. Only Clay, Prithvi and TerraMind "
             "are affected -- Presto and the Sentinel-2 indices have no "
             "temporal axis, so the two coincide for them.",
    )
    parser.add_argument(
        "--ebm-interactions",
        type=int,
        default=0,
        help="Pairwise interaction terms for the EBM head. 0 is the additive "
             "GAM used by the main table.",
    )
    parser.add_argument("--timesteps", type=int, default=7)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if args.family == FAMILY_MAIN:
        if args.alphaearth is None:
            parser.error("--alphaearth is required for the main family")
        if args.s2_indices is None:
            parser.error("--s2-indices is required for the main family")
        if args.s2_daymet_merged is not None:
            parser.error("--s2-daymet-merged belongs only to climate_fusion")
        sentinel2_indices_path = args.s2_indices
        daymet_path = None
        embedding_paths = {
            "clay": args.clay,
            "prithvi": args.prithvi,
            "terramind": args.terramind,
            "presto": args.presto,
            "alphaearth": args.alphaearth,
        }
    else:
        if args.presto_era5 is None:
            parser.error("--presto-era5 is required for climate_fusion")
        if args.s2_daymet_merged is not None:
            if args.s2_indices is not None or args.daymet is not None:
                parser.error(
                    "use --s2-daymet-merged by itself, not with --s2-indices/--daymet"
                )
            sentinel2_indices_path = args.s2_daymet_merged
            daymet_path = args.s2_daymet_merged
        else:
            if args.s2_indices is None or args.daymet is None:
                parser.error(
                    "climate_fusion requires --s2-daymet-merged or both "
                    "--s2-indices and --daymet"
                )
            sentinel2_indices_path = args.s2_indices
            daymet_path = args.daymet
        embedding_paths = {
            "clay": args.clay,
            "prithvi": args.prithvi,
            "terramind": args.terramind,
            "presto": args.presto,
        }
    output = run_regression_benchmark(
        family=args.family,
        embedding_paths=embedding_paths,
        sentinel2_indices_path=sentinel2_indices_path,
        sentinel2_fips_map=args.s2_indices_fips_map,
        presto_era5_path=args.presto_era5,
        daymet_path=daymet_path,
        daymet_fips_map=args.daymet_fips_map,
        labels_path=args.labels,
        split_path=args.split,
        out_dir=args.out_dir,
        folds=args.folds,
        regressors=args.regressors,
        seeds=args.seeds,
        ridge_alphas=args.ridge_alphas,
        expected_timesteps=args.timesteps,
        n_jobs=args.n_jobs,
        temporal_pool=args.temporal_pool,
        ebm_interactions=args.ebm_interactions,
        preflight_only=args.preflight_only,
    )
    if args.preflight_only:
        print(json.dumps(_json_safe(output["data_contract"]), indent=2))
    else:
        print(output["summary"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
