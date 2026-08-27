#!/usr/bin/env python3
"""Leakage-safe Ridge evaluation for frozen embeddings and S2 indices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_fold_partitions
from .daymet import fuse_daymet_features, load_daymet_features
from .frozen import read_embeddings
from .s2_indices import load_s2_index_features


MAIN_BENCHMARK = "main_benchmark"
AUXILIARY_CLIMATE_FUSION = "auxiliary_climate_fusion"
AGGREGATION_OPERATION_ORDER = (
    "complete_patches_then_spatial_pool_per_timestep_then_temporal_pool"
)
SPATIAL_POOL_MEAN_STD = "mean_and_population_standard_deviation"
TEMPORAL_POOL_MEAN = "mean"


JOINT_AGGREGATION_OPERATION_ORDER = (
    "complete_patches_then_joint_pool_over_patch_timestep_rows"
)


def two_stage_mean_aggregation_contract(
    temporal_pool: str = TEMPORAL_POOL_MEAN,
) -> dict[str, str | int]:
    """Describe the county feature construction used by main, LOYO and LOSO.

    Takes the pool as an argument rather than hard-coding it: the same helper
    now describes both the two-stage form (spatial pool per timestep, then a
    temporal pool) and the joint form (one mean and standard deviation over all
    patch-timestep rows at once). A contract that names a pooling the run did
    not perform is worse than no contract, so callers must pass what they used.
    """
    temporal_pool = str(temporal_pool).strip().lower()
    if temporal_pool not in {"mean", "concat", "joint"}:
        raise ValueError("temporal_pool must be 'mean', 'concat', or 'joint'")
    joint = temporal_pool == "joint"
    return {
        "scope": "patch_embedding_representations",
        "operation_order": (
            JOINT_AGGREGATION_OPERATION_ORDER if joint
            else AGGREGATION_OPERATION_ORDER
        ),
        "spatial_pool": SPATIAL_POOL_MEAN_STD,
        "spatial_pool_axis": (
            "all_patch_timestep_rows_within_county_year" if joint
            else "complete_patches_within_county_year_per_timestep"
        ),
        "spatial_std_ddof": 0,
        "temporal_pool": temporal_pool,
        "temporal_pool_axis": "timestep",
        "sequence_scope_policy": "identity_on_single_preencoded_sequence_row",
    }


def resolve_experiment_contract(backbone: str, *, daymet: bool) -> dict[str, str]:
    """Keep the main benchmark and auxiliary climate study disjoint."""
    backbone = str(backbone).strip().lower()
    if daymet:
        if backbone == "presto_s2_era5":
            raise ValueError(
                "the reported Presto+Daymet experiment starts from presto_s2; "
                "do not stack Daymet onto the separate Presto+ERA5 variant"
            )
        return {
            "family": AUXILIARY_CLIMATE_FUSION,
            "climate_source": "Daymet",
            "fusion_stage": "county_year_late",
        }
    if backbone == "presto_s2_era5":
        return {
            "family": AUXILIARY_CLIMATE_FUSION,
            "climate_source": "ERA5-Land",
            "fusion_stage": "presto_encoder_input",
        }
    return {
        "family": MAIN_BENCHMARK,
        "climate_source": "none",
        "fusion_stage": "none",
    }


def pool_county_embeddings(
    embeddings: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    temporal_pool: str = "mean",
    spatial_pool: str = "mean_std",
    expected_timesteps: int = 7,
) -> pd.DataFrame:
    """Apply manuscript-order spatial pooling, then temporal aggregation."""
    temporal_pool = str(temporal_pool).lower()
    if temporal_pool not in {"mean", "concat", "joint"}:
        raise ValueError("temporal_pool must be 'mean', 'concat', or 'joint'")
    if temporal_pool == "joint" and str(spatial_pool).lower() != "mean_std":
        raise ValueError("temporal_pool='joint' requires spatial_pool='mean_std'")
    sequences = build_county_embedding_sequences(
        embeddings,
        labels,
        spatial_pool=spatial_pool,
        expected_timesteps=expected_timesteps,
    )

    def _joint(sequence: np.ndarray) -> np.ndarray:
        """One mean and standard deviation over all patch-timestep rows.

        Matches scripts/run_main_table.py, which treats every patch-timestep row
        as a member of one unordered set. Recovered exactly from the per-timestep
        statistics because every retained patch has all seven composites, so the
        counts are equal across timesteps:

            mean = mean_t(m_t)
            var  = mean_t(s_t^2 + m_t^2) - mean_t(m_t)^2

        The mean agrees with temporal_pool='mean'; only the dispersion differs,
        because this form absorbs between-timestep variation that the two-stage
        form leaves out.
        """
        width = sequence.shape[1] // 2
        means, spreads = sequence[:, :width], sequence[:, width:]
        overall = means.mean(axis=0)
        variance = np.maximum((spreads ** 2 + means ** 2).mean(axis=0) - overall ** 2, 0.0)
        return np.concatenate([overall, np.sqrt(variance)])

    output = sequences.copy()
    if temporal_pool == "joint":
        output["features"] = output["sequence"].map(_joint)
    else:
        output["features"] = output["sequence"].map(
            lambda sequence: (
                sequence.mean(axis=0)
                if temporal_pool == "mean"
                else sequence.reshape(-1)
            )
        )
    return output.drop(columns="sequence")


def build_county_embedding_sequences(
    embeddings: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    spatial_pool: str = "mean_std",
    expected_timesteps: int = 7,
) -> pd.DataFrame:
    """Build the shared county-level temporal sequence before temporal readout."""
    spatial_pool = str(spatial_pool).lower()
    if spatial_pool not in {"mean", "mean_std"}:
        raise ValueError("spatial_pool must be 'mean' or 'mean_std'")
    frame = embeddings.copy()
    frame["_vector"] = frame["embedding"].map(
        lambda value: np.asarray(value, dtype=np.float32)
    )
    backbones = sorted(frame["backbone"].unique())
    if len(backbones) != 1:
        raise ValueError(f"probe input must contain one backbone, got {backbones}")
    if "representation_scope" in frame:
        scopes = sorted(frame["representation_scope"].astype(str).str.lower().unique())
        if len(scopes) != 1 or scopes[0] not in {"timestep", "sequence"}:
            raise ValueError(
                "representation_scope must be uniformly 'timestep' or 'sequence', "
                f"got {scopes}"
            )
        representation_scope = scopes[0]
    else:
        representation_scope = "timestep"
    schedule = [0] if representation_scope == "sequence" else list(range(expected_timesteps))
    rows = []
    for (county, year), group in frame.groupby(["county_id", "year"], sort=True):
        complete_patch_ids = []
        for patch_id, patch_group in group.groupby("patch_id", sort=True):
            patch_group = patch_group.sort_values("timestep")
            times = patch_group["timestep"].astype(int).tolist()
            if times == schedule:
                complete_patch_ids.append(patch_id)
        if not complete_patch_ids:
            continue
        complete = group[group["patch_id"].isin(complete_patch_ids)]
        spatial_sequence = []
        for timestep in schedule:
            matrix = np.stack(
                complete.loc[complete["timestep"] == timestep, "_vector"].tolist()
            )
            mean = matrix.mean(axis=0)
            spatial_vector = (
                mean
                if spatial_pool == "mean"
                else np.concatenate([mean, matrix.std(axis=0, ddof=0)])
            )
            spatial_sequence.append(spatial_vector)
        rows.append(
            {
                "county_id": str(county),
                "year": int(year),
                "sequence": np.stack(spatial_sequence).astype(np.float32),
                "n_patches": len(complete_patch_ids),
                "complete_patch_ids": tuple(sorted(str(value) for value in complete_patch_ids)),
                "representation_scope": representation_scope,
            }
        )
    sequences = pd.DataFrame(rows)
    if sequences.empty:
        raise ValueError("no county-years have the required complete timestep schedule")
    return attach_county_labels(sequences, labels)


def attach_county_labels(county_features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Attach one target to each pre-aggregated county-year feature vector."""
    labels = labels.copy()
    county_col = next(
        (column for column in ("county_id", "county", "county_fips", "fips") if column in labels),
        None,
    )
    year_col = next((column for column in ("year", "Year") if column in labels), None)
    target_col = next(
        (
            column
            for column in ("yield_bu_per_acre", "yield", "Yield", "observed_yield")
            if column in labels
        ),
        None,
    )
    if county_col is None or year_col is None or target_col is None:
        raise ValueError(
            "labels need county, year, and yield (bushels per acre) columns"
        )
    labels["county_id"] = labels[county_col].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(5)
    labels["year"] = pd.to_numeric(labels[year_col], errors="raise").astype(int)
    labels["yield_bu_per_acre"] = pd.to_numeric(labels[target_col], errors="coerce")
    labels = labels[["county_id", "year", "yield_bu_per_acre"]].dropna()
    if labels.duplicated(["county_id", "year"]).any():
        raise ValueError("labels contain duplicate county-year rows")
    output = county_features.merge(
        labels, on=["county_id", "year"], how="inner", validate="one_to_one"
    )
    output["key"] = output["county_id"] + "-" + output["year"].astype(str)
    return output


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int | None]:
    return {
        "r2": float(r2_score(observed, predicted)) if observed.size > 1 else None,
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "n": int(observed.size),
    }


def evaluate_fold(
    data: pd.DataFrame,
    *,
    split_path: str | Path,
    fold: int,
    alphas: Sequence[float],
) -> tuple[dict, pd.DataFrame]:
    parts = load_fold_partitions(split_path, fold=int(fold), id_column="fips_year")
    by_key = data.set_index("key", drop=False)

    def subset(keys: Sequence[str]) -> pd.DataFrame:
        missing = sorted(set(keys).difference(by_key.index))
        if missing:
            raise ValueError(f"split has {len(missing)} keys absent from embeddings: {missing[:5]}")
        return by_key.loc[list(keys)]

    train, val, test = subset(parts.train), subset(parts.val), subset(parts.test)
    x_train = np.stack(train["features"])
    y_train = train["yield_bu_per_acre"].to_numpy(dtype=np.float64)
    x_val = np.stack(val["features"])
    y_val = val["yield_bu_per_acre"].to_numpy(dtype=np.float64)
    candidates = []
    for alpha in alphas:
        model = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))
        model.fit(x_train, y_train)
        prediction = model.predict(x_val)
        candidates.append((float(np.sqrt(mean_squared_error(y_val, prediction))), float(alpha)))
    validation_rmse, best_alpha = min(candidates)
    fit = pd.concat([train, val], ignore_index=True)
    final_model = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha))
    final_model.fit(
        np.stack(fit["features"]), fit["yield_bu_per_acre"].to_numpy()
    )
    prediction = final_model.predict(np.stack(test["features"]))
    observed = test["yield_bu_per_acre"].to_numpy(dtype=np.float64)
    metrics = _metrics(observed, prediction)
    rows = test[["county_id", "year"]].copy()
    rows["observed_yield"] = observed
    rows["predicted_yield"] = prediction
    rows["observed_yield_bu_per_acre"] = observed
    rows["predicted_yield_bu_per_acre"] = prediction
    rows["split_or_fold"] = f"fold_{fold}"
    return {
        "selection": {
            "source": "validation",
            "metric": "county_rmse",
            "alpha": best_alpha,
            "value": validation_rmse,
        },
        "test": {f"county_{key}": value for key, value in metrics.items()},
    }, rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--embeddings")
    inputs.add_argument(
        "--s2-indices",
        help="County-year EVI/LAI/fPAR table for the 21-D Sentinel-2 baseline",
    )
    parser.add_argument(
        "--s2-indices-fips-map",
        help="County-name/state-to-FIPS table if the index table has no FIPS column",
    )
    parser.add_argument("--labels", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--temporal-pool", choices=("mean", "concat", "joint"), default="mean")
    parser.add_argument("--spatial-pool", choices=("mean", "mean_std"), default="mean_std")
    parser.add_argument("--timesteps", type=int, default=7)
    parser.add_argument("--alphas", nargs="+", type=float, default=(0.01, 0.1, 1.0, 10.0, 100.0))
    parser.add_argument(
        "--daymet-features",
        help="County-year Daymet table for the manuscript's 35-D late-fusion experiment",
    )
    parser.add_argument(
        "--daymet-fips-map",
        help="County-name/state-to-FIPS table, required when Daymet has no FIPS column",
    )
    args = parser.parse_args(argv)
    labels = pd.read_csv(args.labels)
    embeddings: pd.DataFrame | None = None
    s2_index_feature_names: list[str] = []
    if args.embeddings:
        embeddings = read_embeddings(args.embeddings)
        backbone_values = sorted(embeddings["backbone"].astype(str).unique())
        if len(backbone_values) != 1:
            raise ValueError(f"probe input must contain one backbone, got {backbone_values}")
        backbone = backbone_values[0]
        data = pool_county_embeddings(
            embeddings,
            labels,
            temporal_pool=args.temporal_pool,
            spatial_pool=args.spatial_pool,
            expected_timesteps=args.timesteps,
        )
        representation_scope = str(data["representation_scope"].iloc[0])
        temporal_aggregation = (
            "preencoded_global_pool"
            if representation_scope == "sequence"
            else args.temporal_pool
        )
        spatial_aggregation = args.spatial_pool
    else:
        index_features = load_s2_index_features(
            args.s2_indices,
            fips_map=args.s2_indices_fips_map,
            expected_timesteps=args.timesteps,
        )
        s2_index_feature_names = list(index_features.attrs["feature_names"])
        data = attach_county_labels(index_features, labels)
        backbone = "sentinel2_indices"
        representation_scope = "county_year"
        temporal_aggregation = "preaggregated_21d"
        spatial_aggregation = "preaggregated"
    contract = resolve_experiment_contract(
        backbone, daymet=bool(args.daymet_features)
    )
    daymet_feature_names: list[str] = []
    daymet_rows = 0
    daymet_unmapped_rows = 0
    county_years_before_daymet = int(len(data))
    feature_dim_before_daymet = int(len(data["features"].iloc[0]))
    if args.daymet_features:
        daymet = load_daymet_features(
            args.daymet_features,
            fips_map=args.daymet_fips_map,
            expected_timesteps=args.timesteps,
        )
        daymet_feature_names = list(daymet.attrs["feature_names"])
        daymet_rows = int(len(daymet))
        daymet_unmapped_rows = int(daymet.attrs["unmapped_rows_excluded"])
        data = fuse_daymet_features(data, daymet)
    result, predictions = evaluate_fold(
        data,
        split_path=args.split,
        fold=args.fold,
        alphas=args.alphas,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if embeddings is not None and "input_modalities" in embeddings:
        modality_values = sorted(embeddings["input_modalities"].astype(str).unique())
        if len(modality_values) != 1:
            raise ValueError(f"embedding table mixes input modalities: {modality_values}")
        input_modalities = [
            value.strip()
            for value in modality_values[0].split(",")
            if value.strip()
        ]
    elif embeddings is not None:
        input_modalities = ["Sentinel-2"]
    else:
        input_modalities = ["Sentinel-2 indices"]
    if args.daymet_features and "Daymet" not in input_modalities:
        input_modalities.append("Daymet")
    fusion_suffix = "_daymet_late_fusion" if args.daymet_features else ""
    experiment_prefix = "frozen" if embeddings is not None else "handcrafted"
    result.update(
        {
            "schema_version": 1,
            "experiment": {
                "family": contract["family"],
                "representation_type": (
                    "frozen_embedding" if embeddings is not None else "handcrafted_baseline"
                ),
                "id": (
                    f"{experiment_prefix}_{backbone}_ridge_"
                    f"{spatial_aggregation}_{temporal_aggregation}{fusion_suffix}"
                ),
                "input_modalities": input_modalities,
                "climate_fusion": {
                    "source": contract["climate_source"],
                    "stage": contract["fusion_stage"],
                },
                "aggregation": {
                    "spatial": spatial_aggregation,
                    "temporal": temporal_aggregation,
                    "representation_scope": representation_scope,
                    "complete_patch_sequences_only": embeddings is not None,
                    "daymet_fusion_stage": (
                        "county_year_late" if args.daymet_features else "none"
                    ),
                },
            },
            "split": {"path": str(args.split), "fold": args.fold},
            "labels": str(Path(args.labels).resolve()),
            "target_and_metric_units": {
                "canonical_yield": "bushels_per_acre",
                "r2": "dimensionless",
                "rmse": "bushels_per_acre",
                "mae": "bushels_per_acre",
            },
            "cohort": {
                "county_years_before_daymet": county_years_before_daymet,
                "county_years_after_daymet": int(len(data)),
                "feature_dim_before_daymet": feature_dim_before_daymet,
                "feature_dim_after_daymet": int(len(data["features"].iloc[0])),
            },
            "daymet": (
                {
                    "path": str(Path(args.daymet_features).resolve()),
                    "fips_map": (
                        str(Path(args.daymet_fips_map).resolve())
                        if args.daymet_fips_map
                        else None
                    ),
                    "feature_count": len(daymet_feature_names),
                    "feature_names": daymet_feature_names,
                    "mapped_rows": daymet_rows,
                    "unmapped_rows_excluded": daymet_unmapped_rows,
                }
                if args.daymet_features
                else None
            ),
            "sentinel2_indices": (
                {
                    "path": str(Path(args.s2_indices).resolve()),
                    "fips_map": (
                        str(Path(args.s2_indices_fips_map).resolve())
                        if args.s2_indices_fips_map
                        else None
                    ),
                    "feature_count": len(s2_index_feature_names),
                    "feature_names": s2_index_feature_names,
                }
                if args.s2_indices
                else None
            ),
        }
    )
    predictions["seed"] = 0
    predictions["model_name"] = result["experiment"]["id"]
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
