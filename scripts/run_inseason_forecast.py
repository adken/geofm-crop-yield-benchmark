#!/usr/bin/env python3
"""In-season forecasting: accuracy as composites accumulate through the season.

Features are truncated to the first k of the seven 28-day composites, for
k = 1..7, and a regressor is fitted on each truncation. Reports RMSE and R2 per
k, so the curve shows how early in the season each representation becomes
useful.

    python scripts/run_inseason_forecast.py \\
        --embeddings clay=outputs/embeddings/clay_v1_5_cls.parquet \\
        --embeddings prithvi=outputs/embeddings/prithvi.parquet \\
        --s2-indices outputs/cohort_covered/sentinel2_indices_covered.csv \\
        --labels data/labels/county_yield.csv \\
        --split outputs/cohort_covered/group_kfold_county_tabular.csv \\
        --out-dir outputs/inseason_covered

Differs from the historical notebook in three ways, all deliberate:

* County-grouped folds from the shared manifest, not a random 80/20 split. A
  random split puts the same county in train and test.
* One set of folds for every representation, so the curves are paired.
* Five folds with mean and standard deviation, not a single draw.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

COMPOSITES = 7


def county_year_timestep_features(parquet: str, pool: str, batch: int = 4096):
    """Stream a [N, T, D or 2D] array of per-composite county-year features."""
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

    total = square = None
    count = np.zeros((len(keys), len(steps)), dtype=np.int64)
    for chunk in handle.iter_batches(batch_size=batch,
                                     columns=["county_id", "year", "timestep", "embedding"]):
        listed = chunk.column("embedding")
        width = len(listed) and len(listed[0])
        vectors = (listed.flatten().to_numpy(zero_copy_only=False)
                   .astype(np.float64, copy=False).reshape(len(listed), width))
        if total is None:
            total = np.zeros((len(keys), len(steps), width))
            square = np.zeros_like(total)
        counties = chunk.column("county_id").to_pylist()
        years = chunk.column("year").to_pylist()
        rows = np.fromiter((key_index[f"{str(c).zfill(5)}-{y}"]
                            for c, y in zip(counties, years)),
                           dtype=np.int64, count=len(listed))
        cols = np.fromiter((step_index[int(t)] for t in chunk.column("timestep").to_pylist()),
                           dtype=np.int64, count=len(listed))
        np.add.at(total, (rows, cols), vectors)
        np.add.at(square, (rows, cols), vectors ** 2)
        np.add.at(count, (rows, cols), 1)
        del vectors, listed, counties, years, rows, cols, chunk
        gc.collect()

    if (count == 0).any():
        raise ValueError(f"{parquet}: some county-years lack a complete timestep set")
    n = count[:, :, None]
    mean = total / n
    if pool == "mean":
        return mean, keys
    variance = np.maximum(square / n - mean ** 2, 0.0)
    return np.concatenate([mean, np.sqrt(variance)], axis=-1), keys


def index_features(path: str, keys: list[str]) -> np.ndarray:
    """[N, T, 3] LAI/EVI/FPAR per composite, aligned to `keys`."""
    frame = pd.read_csv(path)
    frame["key"] = (frame.county_id.astype(str).str.zfill(5) + "-" + frame.year.astype(str))
    frame = frame.set_index("key").loc[keys]
    blocks = []
    for name in ("EVI", "LAI", "FPAR"):
        columns = sorted([c for c in frame.columns
                          if re.fullmatch(rf"{name}_\d+", str(c), re.I)],
                         key=lambda c: int(str(c).split("_")[1]))
        if len(columns) != COMPOSITES:
            raise ValueError(f"{path}: expected {COMPOSITES} {name}_* columns, got {len(columns)}")
        blocks.append(frame[columns].to_numpy(dtype=float))
    return np.stack(blocks, axis=-1)


def make_model(name: str, seed: int, n_jobs: int):
    if name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=6,
                            min_child_weight=2, subsample=0.8, colsample_bytree=0.8,
                            objective="reg:squarederror", tree_method="hist",
                            random_state=seed, n_jobs=n_jobs)
    if name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=600, min_samples_leaf=2,
                                     max_features=1.0, random_state=seed, n_jobs=n_jobs)
    raise ValueError(name)


def fit_predict(name, X, y, train, val, test, seed, n_jobs):
    """Ridge selects alpha on the validation fold; the ensembles use train+val."""
    if name == "ridge":
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        best = None
        for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
            scaler = StandardScaler().fit(X[train])
            model = Ridge(alpha=alpha).fit(scaler.transform(X[train]), y[train])
            rmse = np.sqrt(mean_squared_error(y[val], model.predict(scaler.transform(X[val]))))
            if best is None or rmse < best[0]:
                best = (rmse, alpha)
        full = np.concatenate([train, val])
        scaler = StandardScaler().fit(X[full])
        model = Ridge(alpha=best[1]).fit(scaler.transform(X[full]), y[full])
        return model.predict(scaler.transform(X[test]))
    full = np.concatenate([train, val])
    model = make_model(name, seed, n_jobs).fit(X[full], y[full])
    return model.predict(X[test])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embeddings", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--s2-indices", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--regressor", choices=("xgboost", "ridge", "random_forest"),
                    default="xgboost")
    ap.add_argument("--spatial-pool", choices=("mean", "mean_std"), default="mean_std")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--mode", choices=("progressive", "leave_one_out"),
                    default="progressive",
                    help="progressive: features from the first k composites. "
                         "leave_one_out: all seven composites with one removed, "
                         "which isolates each growth stage's marginal "
                         "contribution rather than its cumulative one.")
    ap.add_argument("--n-jobs", type=int, default=4)
    args = ap.parse_args()

    representations: dict[str, np.ndarray] = {}
    shared: list[str] | None = None
    for spec in args.embeddings:
        name, path = spec.split("=", 1)
        cube, keys = county_year_timestep_features(path, args.spatial_pool)
        if shared is None:
            shared = keys
        elif keys != shared:
            raise SystemExit("encoders cover different county-years")
        representations[name] = cube
        print(f"  {name}: {cube.shape[0]:,} county-years x {cube.shape[1]} composites "
              f"x {cube.shape[2]} features")
    representations["s2_indices"] = index_features(args.s2_indices, shared)

    labels = pd.read_csv(args.labels)
    labels.columns = [str(c).lower() for c in labels.columns]
    ycol = next(c for c in labels.columns if c in ("yield", "yield_bu_ac", "yield_bu_per_acre"))
    ccol = next(c for c in labels.columns if c in ("county_id", "county", "geoid", "fips"))
    labels["key"] = (labels[ccol].astype(str).str.zfill(5) + "-" + labels["year"].astype(str))
    target = labels.set_index("key").loc[shared, ycol].to_numpy(dtype=float)

    manifest = pd.read_csv(args.split, dtype={"county_id": str})
    manifest = manifest[manifest.fips_year.isin(shared)]
    position = pd.Series(range(len(shared)), index=shared)

    rows = []
    for name, cube in representations.items():
        if args.mode == "progressive":
            selections = [(k, list(range(k))) for k in range(1, COMPOSITES + 1)]
        else:
            # Full set first as the reference, then each composite removed.
            selections = [(0, list(range(COMPOSITES)))]
            selections += [(d + 1, [i for i in range(COMPOSITES) if i != d])
                           for d in range(COMPOSITES)]
        for k, chosen in selections:
            X = cube[:, chosen, :].reshape(len(shared), -1)
            per_fold = []
            for fold in sorted(manifest.fold.unique()):
                part = manifest[manifest.fold == fold].set_index("fips_year")["split"]
                role = pd.Series(shared).map(part)
                train = position[role.values == "train"].to_numpy()
                val = position[role.values == "val"].to_numpy()
                test = position[role.values == "test"].to_numpy()
                if not len(test) or not len(val):
                    continue
                scores = []
                for seed in args.seeds:
                    pred = fit_predict(args.regressor, X, target, train, val, test,
                                       seed, args.n_jobs)
                    scores.append((r2_score(target[test], pred),
                                   np.sqrt(mean_squared_error(target[test], pred))))
                per_fold.append(np.mean(scores, axis=0))
            per_fold = np.array(per_fold)
            rows.append({"representation": name,
                         ("composites" if args.mode == "progressive"
                          else "dropped_composite"): k,
                         "r2_mean": per_fold[:, 0].mean(), "r2_std": per_fold[:, 0].std(ddof=0),
                         "rmse_mean": per_fold[:, 1].mean(), "rmse_std": per_fold[:, 1].std(ddof=0),
                         "folds": len(per_fold)})
            print(f"  {name:12} k={k}  R2 {rows[-1]['r2_mean']:.3f} "
                  f"+/-{rows[-1]['r2_std']:.3f}   RMSE {rows[-1]['rmse_mean']:.2f}", flush=True)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    stem = "inseason_results" if args.mode == "progressive" else "stage_importance"
    if args.mode == "leave_one_out":
        # Marginal loss from removing each composite, against the full-set row.
        reference = frame[frame.dropped_composite == 0].set_index("representation")
        frame["delta_r2"] = frame.apply(
            lambda r: round(reference.loc[r.representation, "r2_mean"] - r.r2_mean, 4)
            if r.dropped_composite else 0.0, axis=1)
    frame.round(4).to_csv(out / f"{stem}.csv", index=False)
    (out / "contract.json").write_text(json.dumps({
        "county_years": len(shared),
        "regressor": args.regressor,
        "spatial_pool": args.spatial_pool,
        "seeds": args.seeds,
        "split": str(Path(args.split).resolve()),
        "folds": "county-grouped GroupKFold from the shared manifest",
        "mode": args.mode,
        "note": ("features truncated to the first k of seven composites"
                 if args.mode == "progressive"
                 else "all seven composites with one removed; dropped_composite=0 "
                      "is the full-set reference and delta_r2 is the loss"),
    }, indent=2) + "\n")
    print(f"\nwrote {out/(stem + '.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
