#!/usr/bin/env python3
"""Climate-free LOYO evaluation for the six main benchmark representations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

LOYO_RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .frozen import read_embeddings
from .probe import (
    attach_county_labels,
    pool_county_embeddings,
    two_stage_mean_aggregation_contract,
)
from .s2_indices import load_s2_index_features


REPRESENTATIONS = (
    "clay",
    "prithvi",
    "terramind",
    "presto",
    "alphaearth",
    "sentinel2_indices",
)
PATCH_REPRESENTATIONS = ("clay", "prithvi", "terramind", "presto")
EXPECTED_BACKBONES = {
    "clay": "clay_v1_5_cls",
    "prithvi": "prithvi_eo_v2_300_tl_per_timestep_spatial_mean",
    "terramind": "terramind_v1_base_s2_6_prithvi",
    "presto": "presto_s2",
    "alphaearth": "alphaearth",
}
EXPECTED_REPRESENTATION_SCOPES = {
    "clay": "timestep",
    "prithvi": "timestep",
    "terramind": "timestep",
    "presto": "sequence",
    "alphaearth": "sequence",
}
DEFAULT_YEARS = (2019, 2020, 2021, 2022)
DEFAULT_SEEDS = (0, 1, 2)
LOYO_REGRESSOR = "random_forest"


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


def _uniform_values(frame: pd.DataFrame, column: str) -> list[str]:
    if column not in frame:
        return []
    return sorted(frame[column].dropna().astype(str).str.strip().unique().tolist())


def validate_unfused_main_embedding(
    name: str,
    embeddings: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reject auxiliary Daymet/ERA5 variants and noncanonical main backbones."""
    if name not in EXPECTED_BACKBONES:
        raise ValueError(f"unknown frozen main-benchmark representation {name!r}")
    backbones = _uniform_values(embeddings, "backbone")
    expected = EXPECTED_BACKBONES[name]
    if backbones != [expected]:
        raise ValueError(f"{name} LOYO requires backbone {expected!r}, got {backbones}")
    families = [value.lower() for value in _uniform_values(embeddings, "experiment_family")]
    if families and families != ["main_benchmark"]:
        raise ValueError(f"{name} is not an unfused main-benchmark table: {families}")
    modalities = _uniform_values(embeddings, "input_modalities")
    prohibited = [
        value
        for value in modalities
        if "daymet" in value.lower() or "era5" in value.lower()
    ]
    if prohibited:
        raise ValueError(f"{name} contains added climate inputs: {prohibited}")
    output = embeddings.copy()
    scope_inferred = False
    if name == "alphaearth" and "representation_scope" not in output:
        if set(output["timestep"].astype(int)) != {0}:
            raise ValueError("AlphaEarth annual rows must use timestep 0")
        output["representation_scope"] = "sequence"
        scope_inferred = True
    scopes = [value.lower() for value in _uniform_values(output, "representation_scope")]
    expected_scope = EXPECTED_REPRESENTATION_SCOPES[name]
    if scopes != [expected_scope]:
        raise ValueError(
            f"{name} requires representation_scope={expected_scope!r}, got {scopes}"
        )
    temporal_ingestion = _uniform_values(output, "temporal_ingestion")
    if name == "prithvi" and temporal_ingestion != ["single_timestep_independent"]:
        raise ValueError(
            "Prithvi requires temporal_ingestion='single_timestep_independent'; "
            f"got {temporal_ingestion}"
        )
    return output, {
        "backbone": expected,
        "input_modalities_recorded": modalities,
        "experiment_family_recorded": families,
        "added_climate_features": [],
        "representation_scope": expected_scope,
        "temporal_ingestion": temporal_ingestion,
        "representation_scope_inferred_for_legacy_alphaearth": scope_inferred,
    }


def load_embedding_features(
    name: str,
    path: str | Path,
    labels: pd.DataFrame,
    *,
    expected_timesteps: int,
    temporal_pool: str = "mean",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    print(f"[LOYO] reading {name}: {Path(path).resolve()}", flush=True)
    embeddings = read_embeddings(path)
    embeddings, source_contract = validate_unfused_main_embedding(name, embeddings)
    data = pool_county_embeddings(
        embeddings,
        labels,
        temporal_pool=temporal_pool,
        spatial_pool="mean_std",
        expected_timesteps=expected_timesteps,
    ).sort_values("key").reset_index(drop=True)
    dimensions = sorted({np.asarray(value).size for value in data["features"]})
    if len(dimensions) != 1:
        raise ValueError(f"{name} county features are ragged: {dimensions}")
    if not np.isfinite(np.stack(data["features"])).all():
        raise ValueError(f"{name} county features contain non-finite values")
    contract = {
        **source_contract,
        "path": str(Path(path).resolve()),
        "embedding_rows": int(len(embeddings)),
        "county_years": int(len(data)),
        "feature_dim": int(dimensions[0]),
        "spatial_pool": "mean_std_population",
        "temporal_pool_requested": temporal_pool,
        # Report what was actually applied, not a constant. A sequence-scoped
        # representation has one row per patch sequence, so the temporal pool is
        # an identity whatever was asked for; anything else gets the requested
        # pool. Hard-coding "mean" here made a joint run write a contract
        # claiming two-stage pooling.
        "temporal_pool": (
            "preencoded_global_pool"
            if data["representation_scope"].iloc[0] == "sequence"
            else temporal_pool
        ),
        "complete_patch_sequences_only": True,
    }
    return data, contract


def load_sentinel2_features(
    path: str | Path,
    labels: pd.DataFrame,
    *,
    fips_map: str | Path | None,
    expected_timesteps: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    features = load_s2_index_features(
        path,
        fips_map=fips_map,
        expected_timesteps=expected_timesteps,
    )
    feature_names = list(features.attrs["feature_names"])
    schedule = list(features.attrs["interval_schedule"])
    data = attach_county_labels(features, labels).sort_values("key").reset_index(drop=True)
    return data, {
        "backbone": "sentinel2_indices",
        "path": str(Path(path).resolve()),
        "fips_map": str(Path(fips_map).resolve()) if fips_map else None,
        "county_years": int(len(data)),
        "feature_dim": len(feature_names),
        "feature_names": feature_names,
        "interval_schedule": schedule,
        "unmapped_source_rows_excluded": int(
            features.attrs.get("unmapped_rows_excluded", 0)
        ),
        "incomplete_source_rows_excluded": int(
            features.attrs.get("incomplete_rows_excluded", 0)
        ),
        "spatial_pool": "preaggregated",
        "temporal_pool": "preaggregated_21d",
        "added_climate_features": [],
    }


def validate_matched_main_cohort(frames: Mapping[str, pd.DataFrame]) -> list[str]:
    if set(frames) != set(REPRESENTATIONS):
        raise ValueError(f"LOYO requires exactly {REPRESENTATIONS}")
    reference = frames[REPRESENTATIONS[0]].set_index("key")
    if not reference.index.is_unique:
        raise ValueError("Clay county-year keys are not unique")
    reference_keys = sorted(reference.index.tolist())
    reference_key_set = set(reference_keys)
    for name in REPRESENTATIONS[1:]:
        candidate = frames[name].set_index("key")
        if not candidate.index.is_unique:
            raise ValueError(f"{name} county-year keys are not unique")
        candidate_keys = set(candidate.index)
        if candidate_keys != reference_key_set:
            missing = sorted(reference_key_set - candidate_keys)
            extra = sorted(candidate_keys - reference_key_set)
            raise ValueError(
                f"LOYO county-year cohort mismatch for {name}: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        reference_y = reference.loc[reference_keys, "yield_bu_per_acre"].to_numpy()
        candidate_y = candidate.loc[reference_keys, "yield_bu_per_acre"].to_numpy()
        if not np.allclose(reference_y, candidate_y, atol=0.0, rtol=0.0):
            raise ValueError(f"LOYO target mismatch between clay and {name}")
    for name in PATCH_REPRESENTATIONS[1:]:
        candidate = frames[name].set_index("key")
        reference_counts = reference.loc[reference_keys, "n_patches"].to_numpy()
        candidate_counts = candidate.loc[reference_keys, "n_patches"].to_numpy()
        if not np.array_equal(reference_counts, candidate_counts):
            raise ValueError(f"LOYO complete-patch count mismatch between clay and {name}")
        mismatched = [
            key
            for key in reference_keys
            if tuple(reference.loc[key, "complete_patch_ids"])
            != tuple(candidate.loc[key, "complete_patch_ids"])
        ]
        if mismatched:
            raise ValueError(
                f"LOYO complete-patch identity mismatch between clay and {name}: "
                f"{mismatched[:5]}"
            )
    return reference_keys


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    return {
        "r2": float(r2_score(observed, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "n": int(len(observed)),
    }


def summarize_loyo(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if results.empty:
        raise ValueError("cannot summarize empty LOYO results")
    by_year = (
        results.groupby(
            ["representation", "backbone", "held_out_year"], as_index=False
        )
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
        .sort_values(["representation", "held_out_year"])
        .reset_index(drop=True)
    )
    across_years = (
        by_year.groupby(["representation", "backbone"], as_index=False)
        .agg(
            held_out_years=("held_out_year", "nunique"),
            seeds_per_year=("seeds", "first"),
            county_years_tested=("test_n", "sum"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", lambda values: values.std(ddof=0)),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", lambda values: values.std(ddof=0)),
            mae_mean=("mae", "mean"),
            mae_std=("mae", lambda values: values.std(ddof=0)),
        )
        .sort_values("representation")
        .reset_index(drop=True)
    )
    return by_year, across_years


def run_main_benchmark_loyo(
    embedding_paths: Mapping[str, str | Path],
    *,
    sentinel2_indices_path: str | Path,
    labels_path: str | Path,
    out_dir: str | Path,
    sentinel2_fips_map: str | Path | None = None,
    years: Sequence[int] = DEFAULT_YEARS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    expected_timesteps: int = 7,
    n_estimators: int = 600,
    min_samples_leaf: int = 2,
    max_features: float | str = 1.0,
    n_jobs: int = -1,
    regressor: str = "random_forest",
    temporal_pool: str = "mean",
    preflight_only: bool = False,
) -> dict[str, Any]:
    expected_frozen = set(REPRESENTATIONS) - {"sentinel2_indices"}
    if set(embedding_paths) != expected_frozen:
        raise ValueError(f"embedding_paths must contain exactly {sorted(expected_frozen)}")
    years = tuple(int(value) for value in years)
    seeds = tuple(int(value) for value in seeds)
    if not years or len(set(years)) != len(years):
        raise ValueError("unique held-out years are required")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("unique deterministic seeds are required")
    if int(n_estimators) <= 0 or int(min_samples_leaf) <= 0:
        raise ValueError("n_estimators and min_samples_leaf must be positive")
    regressor = str(regressor).strip().lower()
    if regressor not in {"random_forest", "ridge"}:
        raise ValueError("regressor must be 'random_forest' or 'ridge'")
    if regressor == "ridge":
        # Ridge is deterministic; seeds would produce identical fits and an
        # artificial seed-averaging caveat in every caption.
        seeds = (0,)
    labels = pd.read_csv(labels_path)
    frames: dict[str, pd.DataFrame] = {}
    contracts: dict[str, Any] = {}
    for name in REPRESENTATIONS[:-1]:
        frame, contract = load_embedding_features(
            name,
            embedding_paths[name],
            labels,
            expected_timesteps=expected_timesteps,
            temporal_pool=temporal_pool,
        )
        frames[name] = frame
        contracts[name] = contract
    frames["sentinel2_indices"], contracts["sentinel2_indices"] = (
        load_sentinel2_features(
            sentinel2_indices_path,
            labels,
            fips_map=sentinel2_fips_map,
            expected_timesteps=expected_timesteps,
        )
    )
    common_keys = validate_matched_main_cohort(frames)
    cohort_years = sorted({int(key.rsplit("-", 1)[1]) for key in common_keys})
    if cohort_years != sorted(years):
        raise ValueError(f"LOYO cohort years are {cohort_years}, expected {sorted(years)}")
    year_counts = {
        year: sum(int(key.endswith(f"-{year}")) for key in common_keys) for year in years
    }
    if any(count < 2 for count in year_counts.values()):
        raise ValueError(f"each held-out year needs at least two county-years: {year_counts}")
    cohort_hash = hashlib.sha256("\n".join(common_keys).encode("utf-8")).hexdigest()
    clay_patch_lookup = frames["clay"].set_index("key")["complete_patch_ids"]
    separator = "\x1f"
    patch_hash = hashlib.sha256(
        "\n".join(
            f"{key}{separator}{separator.join(clay_patch_lookup.loc[key])}"
            for key in common_keys
        ).encode("utf-8")
    ).hexdigest()
    data_contract = {
        "schema_version": 1,
        "experiment": "main_benchmark_climate_free_loyo",
        "workflow_role": "temporal_generalization",
        "estimator_family": "classical_ml",
        "representations": contracts,
        "matched_county_years": len(common_keys),
        "matched_key_sha256": cohort_hash,
        "matched_complete_patch_identity_sha256": patch_hash,
        "county_years_by_held_out_year": year_counts,
        "held_out_years": years,
        "labels": str(Path(labels_path).resolve()),
        # Describe the pooling this run performed, not a fixed one. Calling this
        # without an argument recorded two-stage pooling for joint runs, so the
        # contract contradicted the numbers it accompanied.
        "feature_aggregation": two_stage_mean_aggregation_contract(temporal_pool),
        "target_and_metric_units": {
            "canonical_yield": "bushels_per_acre",
            "r2": "dimensionless",
            "rmse": "bushels_per_acre",
            "mae": "bushels_per_acre",
        },
        "fusion_contract": {
            "added_climate_features": [],
            "daymet_late_fusion": False,
            "presto_era5_input_fusion": False,
            "alphaearth_note": "precomputed_multimodal_reference_not_added_fusion",
        },
        "evaluation": {
            "split": "leave_one_year_out",
            "county_pooling": (
                "joint_patch_timestep_mean_std" if temporal_pool == "joint"
                else "spatial_mean_std_per_timestep_then_temporal_mean"
            ),
            "training_years": "all requested years except held_out_year",
            "regressor_key": regressor,
            "regressor": ("Ridge" if regressor == "ridge" else "RandomForestRegressor"),
            "regressor_scope": "loyo_only",
            "model_selection": (
                "alpha_by_inner_leave_one_year_out_on_training_years"
                if regressor == "ridge" else "none_fixed_protocol"
            ),
            "ridge_alphas": (
                list(LOYO_RIDGE_ALPHAS) if regressor == "ridge" else None
            ),
            "feature_standardisation": (
                "fitted_on_training_rows_only" if regressor == "ridge" else None
            ),
            "n_estimators": int(n_estimators),
            "criterion": "squared_error",
            "max_depth": None,
            "min_samples_leaf": int(min_samples_leaf),
            "max_features": max_features,
            "bootstrap": True,
            "seeds": seeds,
            "metrics": ["county_r2", "county_rmse", "county_mae"],
            "summary": "average_seeds_within_year_then_mean_population_std_across_years",
        },
        "software": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "data_contract.json", data_contract)
    if preflight_only:
        return {"data_contract": data_contract, "results": None}

    results = []
    prediction_tables = []
    for representation in REPRESENTATIONS:
        frame = frames[representation].set_index("key").loc[common_keys].reset_index()
        feature_matrix = np.stack(frame["features"]).astype(np.float32)
        target = frame["yield_bu_per_acre"].to_numpy(dtype=np.float64)
        row_years = frame["year"].to_numpy(dtype=int)
        backbone = contracts[representation]["backbone"]
        for held_out_year in years:
            train_indices = np.flatnonzero(row_years != held_out_year)
            test_indices = np.flatnonzero(row_years == held_out_year)
            for seed in seeds:
                if regressor == "ridge":
                    # Select alpha by inner leave-one-year-out over the training
                    # years only, then refit on all of them. The held-out year is
                    # never seen during selection. Standardisation is fitted on
                    # the training rows alone: without it the L2 penalty is
                    # scale-dependent and therefore unequal across
                    # representations of differing magnitude.
                    train_years = sorted(set(row_years[train_indices].tolist()))
                    best_alpha, best_score = None, None
                    for alpha in LOYO_RIDGE_ALPHAS:
                        scores = []
                        for inner_year in train_years:
                            inner_fit = train_indices[
                                row_years[train_indices] != inner_year
                            ]
                            inner_val = train_indices[
                                row_years[train_indices] == inner_year
                            ]
                            if inner_fit.size == 0 or inner_val.size == 0:
                                continue
                            scaler = StandardScaler().fit(feature_matrix[inner_fit])
                            inner = Ridge(alpha=alpha).fit(
                                scaler.transform(feature_matrix[inner_fit]),
                                target[inner_fit],
                            )
                            scores.append(
                                _metrics(
                                    target[inner_val],
                                    inner.predict(
                                        scaler.transform(feature_matrix[inner_val])
                                    ),
                                )["rmse"]
                            )
                        if not scores:
                            continue
                        mean_rmse = float(np.mean(scores))
                        if best_score is None or mean_rmse < best_score:
                            best_alpha, best_score = float(alpha), mean_rmse
                    if best_alpha is None:
                        raise ValueError(
                            "ridge alpha selection needs at least two training years"
                        )
                    scaler = StandardScaler().fit(feature_matrix[train_indices])
                    model = Ridge(alpha=best_alpha)
                    model.fit(
                        scaler.transform(feature_matrix[train_indices]),
                        target[train_indices],
                    )
                    prediction = model.predict(scaler.transform(feature_matrix[test_indices]))
                else:
                    best_alpha = None
                    model = RandomForestRegressor(
                        n_estimators=int(n_estimators),
                        criterion="squared_error",
                        max_depth=None,
                        min_samples_leaf=int(min_samples_leaf),
                        max_features=max_features,
                        bootstrap=True,
                        n_jobs=int(n_jobs),
                        random_state=int(seed),
                    )
                    model.fit(feature_matrix[train_indices], target[train_indices])
                    prediction = model.predict(feature_matrix[test_indices])
                metrics = _metrics(target[test_indices], prediction)
                results.append(
                    {
                        "representation": representation,
                        "backbone": backbone,
                        "regressor": regressor,
                        "selected_alpha": best_alpha,
                        "held_out_year": int(held_out_year),
                        "seed": int(seed),
                        "train_n": int(len(train_indices)),
                        "test_n": metrics["n"],
                        "test_r2": metrics["r2"],
                        "test_rmse": metrics["rmse"],
                        "test_mae": metrics["mae"],
                    }
                )
                predictions = frame.loc[
                    test_indices, ["county_id", "year", "key"]
                ].copy()
                predictions["observed_yield"] = target[test_indices]
                predictions["predicted_yield"] = prediction
                predictions["observed_yield_bu_per_acre"] = target[test_indices]
                predictions["predicted_yield_bu_per_acre"] = prediction
                predictions["representation"] = representation
                predictions["backbone"] = backbone
                predictions["held_out_year"] = int(held_out_year)
                predictions["seed"] = int(seed)
                predictions["regressor"] = regressor
                predictions["model_name"] = f"climate_free_loyo_{regressor}"
                prediction_tables.append(predictions)

    result_frame = pd.DataFrame(results)
    prediction_frame = pd.concat(prediction_tables, ignore_index=True)
    by_year, across_years = summarize_loyo(result_frame)
    result_frame.to_csv(out_dir / "results_by_year_and_seed.csv", index=False)
    by_year.to_csv(out_dir / "results_by_year.csv", index=False)
    across_years.to_csv(out_dir / "summary_across_years.csv", index=False)
    prediction_frame.to_csv(out_dir / "predictions.csv", index=False)
    return {
        "data_contract": data_contract,
        "results": result_frame,
        "by_year": by_year,
        "summary": across_years,
        "predictions": prediction_frame,
    }


def _parse_max_features(value: str) -> float | str:
    text = str(value).strip().lower()
    if text in {"sqrt", "log2"}:
        return text
    number = float(text)
    if not 0.0 < number <= 1.0:
        raise argparse.ArgumentTypeError("max-features must be sqrt, log2, or a float in (0,1]")
    return number


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clay", required=True)
    parser.add_argument("--prithvi", required=True)
    parser.add_argument("--terramind", required=True)
    parser.add_argument("--presto", required=True, help="S2-only Presto; ERA5 is rejected")
    parser.add_argument("--alphaearth", required=True)
    parser.add_argument("--s2-indices", required=True)
    parser.add_argument("--s2-indices-fips-map")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--timesteps", type=int, default=7)
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--max-features", type=_parse_max_features, default=1.0)
    parser.add_argument(
        "--regressor",
        choices=("random_forest", "ridge"),
        default="random_forest",
        help="LOYO head. 'ridge' matches the linear probe of the main benchmark, "
             "selects alpha by inner leave-one-year-out over the training years, "
             "standardises on training rows only, and ignores --seeds because it "
             "is deterministic.",
    )
    parser.add_argument(
        "--temporal-pool",
        choices=("mean", "joint"),
        default="mean",
        help="County aggregation. 'joint' matches scripts/run_main_table.py by "
             "pooling every patch-timestep row as a single set; 'mean' keeps "
             "the two-stage form (spatial mean and standard deviation per "
             "composite, then a temporal mean) used by the probe.",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    output = run_main_benchmark_loyo(
        {
            "clay": args.clay,
            "prithvi": args.prithvi,
            "terramind": args.terramind,
            "presto": args.presto,
            "alphaearth": args.alphaearth,
        },
        sentinel2_indices_path=args.s2_indices,
        sentinel2_fips_map=args.s2_indices_fips_map,
        labels_path=args.labels,
        out_dir=args.out_dir,
        years=args.years,
        seeds=args.seeds,
        expected_timesteps=args.timesteps,
        regressor=args.regressor,
        temporal_pool=args.temporal_pool,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        n_jobs=args.n_jobs,
        preflight_only=args.preflight_only,
    )
    if args.preflight_only:
        print(json.dumps(_json_safe(output["data_contract"]), indent=2))
    else:
        print(output["summary"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
