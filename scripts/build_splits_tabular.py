#!/usr/bin/env python3
"""Build a split manifest for the tabular cohort (no Sentinel-2 patch requirement).

`benchmark_embeddings.build_splits` intersects the complete-patch cohort into every
manifest it produces, which is correct for the GeoFM benchmark but collapses the
tabular cohort from 2,180 to 1,038. This mirrors its fold logic exactly -- GroupKFold
by county, validation = (fold + offset) % n_splits, every year required in every
partition -- while omitting the patch term.

Emits the same columns as the canonical manifest, plus a leave-one-state-out variant.
"""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold

def norm_name(s): return (s.astype(str).str.strip().str.lower()
    .str.replace(r"\s+county$","",regex=True).str.replace(r"\s+"," ",regex=True))

def merged_keys(path, fips_map):
    raw = pd.read_excel(path) if str(path).endswith(("xlsx","xls")) else pd.read_csv(path)
    look = pd.read_csv(fips_map, dtype={"GEOID": str})
    look["_n"], look["_s"] = norm_name(look.NAME), look.STATEFP.astype(int)
    ycol = next(c for c in raw.columns if str(c).lower() == "year")
    k = pd.DataFrame({"_n": norm_name(raw["County"]),
                      "_s": pd.to_numeric(raw["StateFP"]).astype(int)})
    cid = k.merge(look[["_n","_s","GEOID"]], on=["_n","_s"], how="left")["GEOID"]
    idx = [c for c in raw.columns if re.fullmatch(r"(EVI|LAI|FPAR)_\d+", str(c), re.I)]
    ok = raw[idx].notna().all(axis=1).to_numpy() & cid.notna().to_numpy()
    return set(cid[ok].astype(str) + "-" + raw[ycol][ok].astype(int).astype(str))

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--alphaearth-csv", required=True)
    ap.add_argument("--s2-daymet-merged", required=True)
    ap.add_argument("--fips-map", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--validation-fold-offset", type=int, default=1)
    ap.add_argument("--min-state-county-years", type=int, default=40)
    ap.add_argument("--restrict-keys",
                    help="optional county-year key list (one per line). Use the "
                         "--out-keys file from match_tiles.py to build the manifest "
                         "on exactly the cohort the Sentinel-2 tiles support.")
    args = ap.parse_args()

    ae = pd.read_csv(args.alphaearth_csv)
    ae_k = set(ae.GEOID.astype(str).str.zfill(5) + "-" + ae.year.astype(str))
    s2_k = merged_keys(args.s2_daymet_merged, args.fips_map)
    keys = sorted(ae_k & s2_k)
    if args.restrict_keys:
        allowed = {l.strip() for l in Path(args.restrict_keys).read_text().splitlines() if l.strip()}
        before = len(keys)
        keys = [k for k in keys if k in allowed]
        print(f"restricted by {args.restrict_keys}: {before:,} -> {len(keys):,} county-years")
        missing = allowed - set(keys)
        if missing:
            print(f"  note: {len(missing):,} keys in the restrict list are absent from "
                  f"the tabular sources and were ignored")
    cohort = pd.DataFrame({"fips_year": keys,
                           "county_id": [k.split("-")[0] for k in keys],
                           "year": [int(k.split("-")[1]) for k in keys]})
    years = set(cohort.year)
    print(f"cohort {len(cohort):,} county-years, {cohort.county_id.nunique():,} counties, "
          f"years {sorted(years)}")

    outer = GroupKFold(n_splits=args.n_splits)
    fold_by_county = {}
    for fold, (_, te) in enumerate(outer.split(np.zeros(len(cohort)),
                                               groups=cohort.county_id.to_numpy())):
        for c in cohort.iloc[te].county_id.unique():
            fold_by_county[c] = fold
    rows = []
    for fold in range(args.n_splits):
        vfold = (fold + args.validation_fold_offset) % args.n_splits
        seen = {"train": set(), "val": set(), "test": set()}
        for it in cohort.itertuples(index=False):
            cf = fold_by_county[it.county_id]
            role = "test" if cf == fold else ("val" if cf == vfold else "train")
            seen[role].add(it.year)
            rows.append({"fips_year": it.fips_year, "county_id": it.county_id,
                         "year": it.year, "fold": fold, "split": role,
                         "county_outer_fold": cf, "validation_fold": vfold})
        for role, ys in seen.items():
            if ys != years:
                raise ValueError(f"fold {fold} {role} years {sorted(ys)} != {sorted(years)}")
    man = pd.DataFrame(rows)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    gk = out / "group_kfold_county_tabular.csv"
    man.to_csv(gk, index=False)

    st = cohort.county_id.str[:2]
    counts = st.value_counts()
    elig = sorted(counts[counts >= args.min_state_county_years].index)
    lrows = []
    for i, s in enumerate(elig):
        vs = elig[(i + 1) % len(elig)]
        for it, state in zip(cohort.itertuples(index=False), st):
            role = "test" if state == s else ("val" if state == vs else "train")
            lrows.append({"fips_year": it.fips_year, "county_id": it.county_id,
                          "year": it.year, "fold": s, "split": role,
                          "held_out_state": s, "validation_state": vs})
    loso = pd.DataFrame(lrows)
    lo = out / "loso_state_tabular.csv"
    loso.to_csv(lo, index=False)

    h = hashlib.sha256("\n".join(keys).encode()).hexdigest()
    contract = {"restrict_keys": str(Path(args.restrict_keys).resolve()) if args.restrict_keys else None,
                "cohort_county_years": len(cohort),
                "counties": int(cohort.county_id.nunique()),
                "years": sorted(int(y) for y in years),
                "cohort_sha256": h,
                "patch_requirement": False,
                "n_splits": args.n_splits,
                "validation_fold_offset": args.validation_fold_offset,
                "loso_states": elig,
                "loso_min_county_years": args.min_state_county_years,
                "sources": {"alphaearth": str(Path(args.alphaearth_csv).resolve()),
                            "s2_daymet_merged": str(Path(args.s2_daymet_merged).resolve()),
                            "fips_map": str(Path(args.fips_map).resolve())}}
    (out / "cohort_contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    print(f"  wrote {gk}   ({len(man):,} rows)")
    print(f"  wrote {lo}   ({len(loso):,} rows, {len(elig)} states)")
    print(f"  wrote {out/'cohort_contract.json'}")
    print(f"  cohort sha256 {h[:16]}...")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
