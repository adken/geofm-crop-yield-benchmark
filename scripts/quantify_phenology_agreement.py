#!/usr/bin/env python3
"""Quantify how closely embedding trajectories track vegetation indices.

The seasonal-trajectory figure is a visual comparison. This replaces the eye
with four numbers, per encoder and per index:

* per-county-year Pearson correlation between the embedding's PC1 trajectory
  and the index trajectory, reported as a distribution over county-years rather
  than a single correlation of two mean curves (which would have n=7);
* the lag, in composites, that maximises the mean correlation, which measures
  whether the embedding peaks early or late relative to the index;
* the share of embedding variance carried by PC1, so a reader knows how much of
  the representation the trajectory describes;
* the cross-validated R^2 of predicting each index from the full embedding at
  the same composite, which asks how much of the index is linearly recoverable
  rather than merely correlated in shape.

    python scripts/quantify_phenology_agreement.py \\
        --embeddings clay=outputs/embeddings/clay_v1_5_cls.parquet \\
        --s2-indices outputs/cohort_covered/sentinel2_indices_covered.csv \\
        --split outputs/cohort_covered/group_kfold_county_tabular.csv \\
        --out-dir outputs/phenology_agreement
"""
from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

INDICES = ("LAI", "EVI", "FPAR")


def county_year_timestep_matrix(parquet: str, batch: int = 4096):
    import pyarrow.parquet as pq

    handle = pq.ParquetFile(parquet)
    identity = handle.read(columns=["county_id", "year", "timestep"]).to_pandas()
    identity["key"] = (
        identity.county_id.astype(str).str.zfill(5) + "-" + identity.year.astype(str)
    )
    keys = sorted(identity["key"].unique())
    steps = np.sort(identity["timestep"].unique())
    key_index = {k: i for i, k in enumerate(keys)}
    step_index = {int(t): i for i, t in enumerate(steps)}
    del identity

    total = None
    count = np.zeros((len(keys), len(steps)), dtype=np.int64)
    for chunk in handle.iter_batches(
        batch_size=batch, columns=["county_id", "year", "timestep", "embedding"]
    ):
        listed = chunk.column("embedding")
        width = len(listed) and len(listed[0])
        vectors = (listed.flatten().to_numpy(zero_copy_only=False)
                   .astype(np.float64, copy=False).reshape(len(listed), width))
        if total is None:
            total = np.zeros((len(keys), len(steps), width))
        counties = chunk.column("county_id").to_pylist()
        years = chunk.column("year").to_pylist()
        rows = np.fromiter((key_index[f"{str(c).zfill(5)}-{y}"]
                            for c, y in zip(counties, years)),
                           dtype=np.int64, count=len(listed))
        cols = np.fromiter((step_index[int(t)] for t in chunk.column("timestep").to_pylist()),
                           dtype=np.int64, count=len(listed))
        np.add.at(total, (rows, cols), vectors)
        np.add.at(count, (rows, cols), 1)
        del vectors, listed, counties, years, rows, cols, chunk
        gc.collect()
    return total / count[:, :, None], keys


def pc1(cube: np.ndarray) -> tuple[np.ndarray, float]:
    """Project [N,T,D] on PC1; also return the variance share it carries."""
    flat = cube.reshape(-1, cube.shape[-1])
    centred = flat - flat.mean(0, keepdims=True)
    _, singular, components = np.linalg.svd(centred, full_matrices=False)
    share = float(singular[0] ** 2 / np.sum(singular ** 2))
    return (centred @ components[0]).reshape(cube.shape[:2]), share


def rowwise_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson r per row between two [N,T] arrays."""
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    denominator = np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, (a * b).sum(1) / denominator, np.nan)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embeddings", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--s2-indices", required=True)
    ap.add_argument("--split", help="county-grouped manifest for the recoverability test")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    frame = pd.read_csv(args.s2_indices)
    frame["key"] = (frame.county_id.astype(str).str.zfill(5) + "-" + frame.year.astype(str))

    rows, recovery = [], []
    for spec in args.embeddings:
        name, path = spec.split("=", 1)
        cube, keys = county_year_timestep_matrix(path)
        projected, share = pc1(cube)
        aligned = frame.set_index("key").loc[keys]

        for index_name in INDICES:
            columns = sorted([c for c in aligned.columns
                              if re.fullmatch(rf"{index_name}_\d+", str(c), re.I)],
                             key=lambda c: int(str(c).split("_")[1]))
            if len(columns) != cube.shape[1]:
                continue
            index_curves = aligned[columns].to_numpy(dtype=float)

            # Orient PC1 once per index so the sign convention is explicit.
            oriented = projected
            if np.corrcoef(projected.mean(0), index_curves.mean(0))[0, 1] < 0:
                oriented = -projected

            correlations = rowwise_correlation(oriented, index_curves)
            best_lag, best_mean = 0, np.nanmean(correlations)
            for lag in (-2, -1, 1, 2):
                if lag > 0:
                    shifted = rowwise_correlation(oriented[:, lag:], index_curves[:, :-lag])
                else:
                    shifted = rowwise_correlation(oriented[:, :lag], index_curves[:, -lag:])
                if np.nanmean(shifted) > best_mean:
                    best_lag, best_mean = lag, float(np.nanmean(shifted))

            rows.append({
                "encoder": name, "index": index_name,
                "pc1_variance_share": round(share, 4),
                "r_mean": round(float(np.nanmean(correlations)), 4),
                "r_median": round(float(np.nanmedian(correlations)), 4),
                "r_sd": round(float(np.nanstd(correlations)), 4),
                "share_r_above_0.9": round(float(np.nanmean(correlations > 0.9)), 4),
                "best_lag_composites": best_lag,
                "r_at_best_lag": round(float(best_mean), 4),
                "county_years": len(keys),
            })

            if args.split:
                from sklearn.linear_model import RidgeCV
                from sklearn.preprocessing import StandardScaler
                manifest = pd.read_csv(args.split, dtype={"county_id": str})
                manifest = manifest[manifest.fips_year.isin(keys)]
                position = pd.Series(range(len(keys)), index=keys)
                for step in range(cube.shape[1]):
                    scores = []
                    for fold in sorted(manifest.fold.unique()):
                        part = manifest[manifest.fold == fold].set_index("fips_year")["split"]
                        role = pd.Series(keys).map(part).values
                        train = position[role != "test"].to_numpy()
                        test = position[role == "test"].to_numpy()
                        scaler = StandardScaler().fit(cube[train, step])
                        model = RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0)).fit(
                            scaler.transform(cube[train, step]), index_curves[train, step])
                        prediction = model.predict(scaler.transform(cube[test, step]))
                        truth = index_curves[test, step]
                        scores.append(1 - ((truth - prediction) ** 2).sum()
                                      / ((truth - truth.mean()) ** 2).sum())
                    recovery.append({"encoder": name, "index": index_name,
                                     "composite": step + 1,
                                     "r2_mean": round(float(np.mean(scores)), 4),
                                     "r2_sd": round(float(np.std(scores)), 4)})
        del cube, projected
        gc.collect()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    agreement = pd.DataFrame(rows)
    agreement.to_csv(out / "trajectory_agreement.csv", index=False)
    print(agreement.to_string(index=False))
    if recovery:
        table = pd.DataFrame(recovery)
        table.to_csv(out / "index_recoverability.csv", index=False)
        print("\nlinear recoverability of each index from the embedding (mean over composites)")
        print(table.groupby(["encoder", "index"]).r2_mean.mean().round(3).to_string())
    (out / "contract.json").write_text(json.dumps({
        "correlation": "per county-year Pearson r across the seven composites",
        "lag_search": [-2, -1, 0, 1, 2],
        "recoverability": "RidgeCV per composite, county-grouped folds, test folds only",
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
