#!/usr/bin/env python3
"""Audit whatever authoritative source files land in data/sources/.

Reports, for each file: row count, detected county/year columns, key coverage,
and how it intersects the existing patch cohort and yield labels. Then prints the
cohort that would result, so the effect of a source swap is visible before
anything downstream is rebuilt.

    python scripts/audit_sources.py [--sources-dir data/sources]

Read-only. Writes nothing, changes nothing.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]

COUNTY_COLS = ("county_id", "geoid", "fips", "county_fips", "county", "countyfips")
YEAR_COLS = ("year", "yr", "season")
NAME_COLS = ("county_name", "name", "county")
STATE_COLS = ("statefp", "state_fips", "state ansi", "state")


def read_any(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix.lower() in {".csv", ".txt"}:
            return pd.read_csv(path)
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        if path.suffix.lower() in {".parquet", ".pq"}:
            return pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - diagnostic tool
        print(f"    could not read: {type(exc).__name__}: {exc}")
    return None


def pick(frame: pd.DataFrame, names) -> str | None:
    lower = {str(c).strip().lower(): str(c) for c in frame.columns}
    return next((lower[n] for n in names if n in lower), None)


def keys_from(frame: pd.DataFrame) -> set[str] | None:
    """Build county_id-year keys, using a FIPS column or name+state lookup."""
    ycol = pick(frame, YEAR_COLS)
    if ycol is None:
        return None
    ccol = pick(frame, COUNTY_COLS)
    years = pd.to_numeric(frame[ycol], errors="coerce")
    if ccol is not None:
        cid = frame[ccol].astype(str).str.extract(r"(\d+)")[0].str.zfill(5)
        if cid.notna().mean() > 0.5:
            ok = cid.notna() & years.notna()
            return set(cid[ok] + "-" + years[ok].astype(int).astype(str))
    # fall back to county name + state via the generated FIPS map
    ncol, scol = pick(frame, NAME_COLS), pick(frame, STATE_COLS)
    fips_map = REPO / "data/geometry/county_fips_map.csv"
    if ncol and scol and fips_map.exists():
        from benchmark_embeddings.daymet import _county_name, _fips_lookup

        look = _fips_lookup(fips_map)
        k = pd.DataFrame(
            {
                "_county_name": _county_name(frame[ncol]),
                "_statefp": pd.to_numeric(frame[scol], errors="coerce"),
            }
        ).dropna()
        k["_statefp"] = k["_statefp"].astype(int)
        m = k.merge(look, on=["_county_name", "_statefp"], how="left")
        cid = m["county_id"]
        yy = years.loc[k.index]
        ok = cid.notna().to_numpy() & yy.notna().to_numpy()
        return set(cid[ok].to_numpy() + "-" + yy[ok].astype(int).astype(str).to_numpy())
    return None


def complete_patch_keys() -> set[str]:
    pat = re.compile(r"county_(\d+)_year_(\d{4})_x(-?\d+)_y(-?\d+)_interval_(\d+)")
    groups: dict[tuple, set[int]] = defaultdict(set)
    root = REPO / "data/patches/sentinel-2-l2a"
    if not root.exists():
        return set()
    for p in root.rglob("*.npz"):
        m = pat.search(p.name)
        if not m:
            continue
        c, y, x, yy, iv = m.groups()
        if int(iv) > 6:
            continue
        groups[(c.zfill(5), int(y), f"{x}_{yy}")].add(int(iv))
    return {f"{c}-{y}" for (c, y, _), iv in groups.items() if len(iv) == 7}


def label_keys() -> set[str]:
    path = REPO / "data/labels/county_yield.csv"
    lab = pd.read_csv(path).dropna(subset=["yield"])
    return set(lab["county"].astype(str).str.zfill(5) + "-" + lab["year"].astype(str))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources-dir", default=str(REPO / "data/sources"))
    ap.add_argument("--skip-patches", action="store_true",
                    help="skip the patch scan (it walks 77k files)")
    args = ap.parse_args()

    src = Path(args.sources_dir)
    files = sorted(
        p for p in src.glob("*")
        if p.suffix.lower() in {".csv", ".xlsx", ".xls", ".parquet", ".pq"}
    )
    if not files:
        print(f"No data files in {src}. Drop the authoritative exports there first.")
        return 1

    labels = label_keys()
    patches = set() if args.skip_patches else complete_patch_keys()
    if patches:
        print(f"complete-patch county-years (7 timesteps): {len(patches):,}")
    print(f"labelled county-years:                     {len(labels):,}\n")

    found: dict[str, set[str]] = {}
    for f in files:
        print(f"{f.name}")
        frame = read_any(f)
        if frame is None:
            print()
            continue
        print(f"    rows {len(frame):,}   columns {len(frame.columns)}")
        print(f"    first columns: {list(frame.columns)[:8]}")
        keys = keys_from(frame)
        if keys is None:
            print("    could not derive county-year keys "
                  "(need a FIPS or county-name + state column, plus year)\n")
            continue
        found[f.name] = keys
        counties = {k.split("-")[0] for k in keys}
        years = sorted({k.split("-")[1] for k in keys})
        print(f"    county-years {len(keys):,}   counties {len(counties):,}   years {years}")
        print(f"    n labels  {len(keys & labels):,}")
        if patches:
            print(f"    n patches {len(keys & patches):,}")
        print()

    # The raw-key intersection above ignores the 7-interval completeness filter
    # that load_s2_index_features applies, so it OVERSTATES the cohort. Re-run
    # the real loaders on any file that looks like the merged S2/Daymet table.
    filtered: dict[str, set[str]] = {}
    fips_map = REPO / "data/geometry/county_fips_map.csv"
    for f in files:
        frame = read_any(f)
        if frame is None or not any(
            re.fullmatch(r"EVI_\d+", str(c)) for c in frame.columns
        ):
            continue
        from benchmark_embeddings.s2_indices import load_s2_index_features

        try:
            s2 = load_s2_index_features(f, fips_map=fips_map)
        except Exception as exc:
            print(f"{f.name}: 7-interval filter REJECTS this file "
                  f"({type(exc).__name__}: {str(exc)[:70]})\n")
            continue
        k = set(s2["county_id"] + "-" + s2["year"].astype(str))
        filtered[f.name] = k
        raw_n = len(found.get(f.name, ()))
        print(f"{f.name}: after the pipeline's 7-interval completeness filter "
              f"{raw_n:,} -> {len(k):,} county-years")
    if filtered:
        print()

    if len(found) >= 2:
        inter = set.intersection(*found.values())
        # apply the filtered variant wherever we have one
        for name, k in filtered.items():
            inter &= k
        print("Intersection across all readable sources "
              "(with the 7-interval filter applied)")
        print(f"    county-years {len(inter):,}   counties {len({k.split('-')[0] for k in inter}):,}")
        print(f"    n labels                 {len(inter & labels):,}")
        if patches:
            full = inter & labels & patches
            print(f"    n labels n patches       {len(full):,}   <- GeoFM-capable cohort")
            print(f"    tabular-only cohort      {len(inter & labels):,}   "
                  f"(no patch requirement)")
            man = REPO / "data/group_kfold_county_T7.csv"
            if man.exists():
                cur = set(pd.read_csv(man, dtype=str)["fips_year"])
                print(f"\n    current manifest cohort  {len(cur):,}")
                print(f"    would gain               {len(full - cur):,}")
                print(f"    would lose               {len(cur - full):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
