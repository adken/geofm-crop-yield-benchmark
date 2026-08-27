#!/usr/bin/env python3
"""Main table: every representation x the four canonical regressor heads.

Builds the common cohort across whatever representations are supplied, runs
Ridge / Random Forest / XGBoost / EBM on county-grouped folds, and reports
mean +/- population sd across folds with RF/XGB/EBM seed-averaged (0/1/2)
within each fold -- the order the README specifies.

    python scripts/run_main_table.py \\
        --alphaearth-csv   data/sources/embeddings_with_yield_matched.csv \\
        --s2-daymet-merged data/sources/s2_daymet_merged_matched.xlsx \\
        --fips-map         data/geometry/county_fips_map.csv \\
        --split            outputs/cohort_2180/group_kfold_county_tabular.csv \\
        --embeddings presto=outputs/embeddings/presto_s2.parquet \\
        --out-dir    outputs/main_table

Repeat --embeddings for each encoder Parquet. Every representation is
restricted to the cohort common to all of them, so the comparison is paired.

Two deviations from the repo's registry are deliberate and flagged in the
output. Ridge standardizes features; the published run did not, which makes the
L2 penalty scale-dependent and unequal across representations. EBM drops the
registry's max_rounds=1000 / max_bins=128 / no-early-stopping configuration,
which underfits by ~0.09 R2, and uses library defaults except for
interactions=0 -- interpret's '5x' default is 10,240 pairwise terms on a
2048-dim representation fitted on ~1,660 county-years, which neither finishes
nor is supportable at that sample size.
"""
from __future__ import annotations
import argparse, json, re, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

RIDGE_ALPHAS = [0.01, 0.1, 1, 10, 100]
HEADS = ["Ridge", "Random Forest", "XGBoost", "EBM"]


def norm_name(s):
    return (s.astype(str).str.strip().str.lower()
            .str.replace(r"\s+county$", "", regex=True)
            .str.replace(r"\s+", " ", regex=True))


def load_tabular(ae_csv, merged, fips_map):
    look = pd.read_csv(fips_map, dtype={"GEOID": str})
    look["_n"], look["_s"] = norm_name(look.NAME), look.STATEFP.astype(int)
    raw = pd.read_excel(merged) if str(merged).endswith(("xlsx", "xls")) else pd.read_csv(merged)
    k = pd.DataFrame({"_n": norm_name(raw["County"]),
                      "_s": pd.to_numeric(raw["StateFP"]).astype(int)})
    raw["county_id"] = k.merge(look[["_n", "_s", "GEOID"]], on=["_n", "_s"], how="left")["GEOID"].values
    ycol = next(c for c in raw.columns if str(c).lower() == "year")
    raw["k"] = raw.county_id + "-" + raw[ycol].astype(str)
    idx = sorted([c for c in raw.columns if re.fullmatch(r"(EVI|LAI|FPAR)_\d+", str(c), re.I)],
                 key=lambda c: (str(c).split("_")[0].upper(), int(str(c).split("_")[1])))
    # 35 Daymet columns: five variables over the same seven intervals. Late
    # fusion at county level -- the climate result the paper reports. The
    # regression_benchmark climate_fusion family cannot produce this without a
    # Presto+ERA5 table, which does not exist for this cohort.
    dmt = sorted([c for c in raw.columns
                  if re.fullmatch(r"(dayl|prcp|srad|tmax|tmin)_\d+", str(c), re.I)],
                 key=lambda c: (str(c).split("_")[0].lower(), int(str(c).split("_")[1])))
    comp = raw[raw[idx].notna().all(axis=1) & raw.county_id.notna()].copy()
    ae = pd.read_csv(ae_csv)
    ae["county_id"] = ae.GEOID.astype(str).str.zfill(5)
    ae["k"] = ae.county_id + "-" + ae.year.astype(str)
    emb = [c for c in ae.columns if c.startswith("mean_A")]
    ycol_ae = "Yield" if "Yield" in ae.columns else "yield"
    return ae, emb, comp, idx, dmt, ycol_ae


def _as_vector(value):
    """Coerce one stored embedding to a float vector.

    Parquet round-trips a list column as a list or ndarray, but some
    writer/reader combinations hand it back as the string repr of a list. That
    surfaces far downstream as a dtype error inside np.mean, so normalise here
    rather than trusting the column type.
    """
    if isinstance(value, str):
        return np.fromstring(value.strip().lstrip("[").rstrip("]"), sep=",")
    return np.asarray(value, dtype=np.float64)


def pooled(parquet):
    """County-year features from a canonical embedding table: mean+std over rows."""
    d = pd.read_parquet(parquet)
    d["k"] = d.county_id.astype(str).str.zfill(5) + "-" + d.year.astype(str)
    out = {}
    widths = set()
    for key, s in d.groupby("k")["embedding"]:
        e = np.stack([_as_vector(v) for v in s.values])
        widths.add(e.shape[1])
        out[key] = np.concatenate([e.mean(0), e.std(0)])
    if len(widths) != 1:
        raise ValueError(
            f"{parquet}: embeddings have inconsistent widths {sorted(widths)}; "
            "a truncated or mixed-model table would silently corrupt the pooling"
        )
    return out


def make(head, seed, n_jobs, ebm_interactions=0):
    if head == "Random Forest":
        return RandomForestRegressor(n_estimators=600, criterion="squared_error",
            max_depth=None, min_samples_leaf=2, max_features=1.0, bootstrap=True,
            random_state=seed, n_jobs=n_jobs)
    if head == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=6,
            min_child_weight=2, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.0,
            reg_lambda=1.0, objective="reg:squarederror", tree_method="hist",
            random_state=seed, n_jobs=n_jobs)
    from interpret.glassbox import ExplainableBoostingRegressor

    # interpret's default is interactions='5x' -- five pairwise terms per
    # feature, so 10,240 of them on Clay's 2048 dimensions, fitted on ~1,660
    # training county-years and searched across 14 outer bags. It does not
    # finish in reasonable time, the terms cannot be supported by that sample
    # size, and pairwise interactions between anonymous embedding dimensions
    # are not interpretable, which is the only reason to prefer an EBM here.
    # interactions=0 is the additive GAM the method is named for.
    return ExplainableBoostingRegressor(
        interactions=ebm_interactions, random_state=seed, n_jobs=n_jobs
    )


def evaluate(X, Y, keyser, man, head, seeds, n_jobs, ebm_interactions=0):
    per_fold = []
    for f in sorted(man.fold.unique()):
        role = keyser.map(man[man.fold == f].set_index("fips_year")["split"])
        tr = np.flatnonzero((role == "train").values)
        va = np.flatnonzero((role == "val").values)
        te = np.flatnonzero((role == "test").values)
        if len(te) == 0 or len(va) == 0:
            continue
        if head == "Ridge":
            best = None
            for a in RIDGE_ALPHAS:
                sc = StandardScaler().fit(X[tr])
                m = Ridge(alpha=a).fit(sc.transform(X[tr]), Y[tr])
                r = np.sqrt(mean_squared_error(Y[va], m.predict(sc.transform(X[va]))))
                if best is None or r < best[0]:
                    best = (r, a)
            full = np.concatenate([tr, va])
            sc = StandardScaler().fit(X[full])
            m = Ridge(alpha=best[1]).fit(sc.transform(X[full]), Y[full])
            p = m.predict(sc.transform(X[te]))
            per_fold.append((r2_score(Y[te], p), np.sqrt(mean_squared_error(Y[te], p)),
                             mean_absolute_error(Y[te], p)))
        else:
            full = np.concatenate([tr, va])
            a, b, c = [], [], []
            for s in seeds:
                m = make(head, s, n_jobs, ebm_interactions).fit(X[full], Y[full])
                p = m.predict(X[te])
                a.append(r2_score(Y[te], p)); b.append(np.sqrt(mean_squared_error(Y[te], p)))
                c.append(mean_absolute_error(Y[te], p))
            per_fold.append((np.mean(a), np.mean(b), np.mean(c)))
    arr = np.array(per_fold)
    return dict(r2=arr[:, 0].mean(), r2_sd=arr[:, 0].std(ddof=0),
                rmse=arr[:, 1].mean(), rmse_sd=arr[:, 1].std(ddof=0),
                mae=arr[:, 2].mean(), folds=len(per_fold))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alphaearth-csv", required=True)
    ap.add_argument("--s2-daymet-merged", required=True)
    ap.add_argument("--fips-map", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--embeddings", action="append", default=[],
                    metavar="NAME=PATH", help="repeatable, e.g. presto=.../presto_s2.parquet")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--heads", nargs="+", default=HEADS, choices=HEADS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument(
        "--ebm-interactions",
        default=0,
        help="EBM pairwise interaction terms. 0 (default) is the additive "
             "GAM. interpret's own default of '5x' is 10,240 terms on a "
             "2048-dim representation and does not finish.",
    )
    ap.add_argument(
        "--with-daymet",
        action="store_true",
        help="add an 'S2 + Daymet' row: the 21 index columns concatenated with "
             "the 35 Daymet columns. This is county-level late fusion, the "
             "climate result the paper reports, and it needs no ERA5 patches.",
    )
    args = ap.parse_args()

    ae, embc, comp, idxc, dmtc, ycol = load_tabular(args.alphaearth_csv,
                                                    args.s2_daymet_merged, args.fips_map)
    reps = {}
    for spec in args.embeddings:
        name, path = spec.split("=", 1)
        reps[name] = pooled(path)

    common = set(ae.k) & set(comp.k)
    for name, d in reps.items():
        common &= set(d)
    keys = sorted(common)
    print(f"common cohort across all representations: {len(keys):,} county-years, "
          f"{len({k[:5] for k in keys}):,} counties")
    if not keys:
        raise SystemExit("no shared county-years")

    a = ae[ae.k.isin(keys)].sort_values("k").reset_index(drop=True)
    c = comp[comp.k.isin(keys)].sort_values("k").reset_index(drop=True)
    assert list(a.k) == list(c.k) == keys
    Y = a[ycol].to_numpy(float)
    keyser = pd.Series(keys)

    X = {"AlphaEarth": a[embc].to_numpy(float),
         "S2 indices": c[idxc].to_numpy(float)}
    if args.with_daymet:
        missing = c[dmtc].isna().any(axis=1).sum()
        if missing:
            raise SystemExit(
                f"--with-daymet: {missing} of {len(c)} county-years have gaps in "
                f"the {len(dmtc)} Daymet columns; the late-fusion row would not "
                "share the cohort with the others"
            )
        X["S2 + Daymet"] = c[idxc + dmtc].to_numpy(float)
    for name, d in reps.items():
        X[name] = np.stack([d[k] for k in keys])

    man = pd.read_csv(args.split, dtype={"county_id": str})
    man = man[man.fips_year.isin(keys)]

    rows = []
    for rep, Xm in X.items():
        for head in args.heads:
            r = evaluate(Xm, Y, keyser, man, head, args.seeds, args.n_jobs,
                         args.ebm_interactions)
            rows.append(dict(representation=rep, dim=Xm.shape[1], head=head, **r))
            print(f"  {rep:<14}{head:<15}R2 {r['r2']:.3f} +/-{r['r2_sd']:.3f}   "
                  f"RMSE {r['rmse']:.2f}   MAE {r['mae']:.2f}", flush=True)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    d = pd.DataFrame(rows)
    d.round(4).to_csv(out / "main_table.csv", index=False)
    (out / "run_contract.json").write_text(json.dumps({
        "cohort_county_years": len(keys),
        "counties": len({k[:5] for k in keys}),
        "representations": {k: int(v.shape[1]) for k, v in X.items()},
        "heads": args.heads, "seeds": args.seeds,
        "split": str(Path(args.split).resolve()),
        "deviations_from_registry": [
            "Ridge standardizes features; the published run did not",
            "EBM uses library defaults except interactions; the registry "
            "config underfits by ~0.09 R2",
            f"EBM interactions={args.ebm_interactions} (interpret default '5x' "
            "is 10,240 pairwise terms at 2048 features)",
        ],
    }, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out/'main_table.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
