#!/usr/bin/env python3
"""Verify the staged tiles and the three tabular representations agree on one cohort.

Two independent things go wrong between staging and extraction, and both are
silent: a short file copy shrinks the tile corpus, and a representation with a
missing column drops county-years from the paired comparison.  Either changes
every downstream number without raising anything.  This checks both and exits
non-zero on a mismatch.

    python scripts/verify_cohort.py \\
      --alphaearth-csv   data/sources/embeddings_with_yield_matched.csv \\
      --s2-daymet-merged data/sources/s2_daymet_merged_matched.xlsx \\
      --fips-map         data/geometry/county_fips_map.csv \\
      --cohort-keys      /e/project1/3d-abc/adriko1/scripts/cohort_covered_keys.txt \\
      --tile-dir         /e/project1/3d-abc/adriko1/datasets/US/T7 \\
      --expect-files     427049 \\
      --expect-county-years 2076

Key derivation is imported from build_splits_tabular.py rather than
reimplemented, so a cohort that verifies here is the cohort the manifest builds.
--tile-dir may be omitted to check the tabular side alone.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location(
    "_build_splits_tabular", HERE / "build_splits_tabular.py"
)
_bst = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bst)

INDEX_COLUMNS = re.compile(r"(EVI|LAI|FPAR)_\d+", re.I)
DAYMET_COLUMNS = re.compile(r"(dayl|prcp|srad|tmax|tmin)_\d+", re.I)
TILE_NAME = re.compile(
    r"county_(?P<county>\d+)_year_(?P<year>\d{4})"
    r"(?:_\d+)?"
    r"_x(?P<x>-?\d+)_y(?P<y>-?\d+)_interval_(?P<interval>\d+)",
    re.IGNORECASE,
)


def keys_from_merged(path: Path, fips_map: Path, pattern: re.Pattern, label: str) -> set[str]:
    """County-years whose every `pattern` column is present and non-null."""
    raw = pd.read_excel(path) if str(path).endswith(("xlsx", "xls")) else pd.read_csv(path)
    look = pd.read_csv(fips_map, dtype={"GEOID": str})
    look["_n"], look["_s"] = _bst.norm_name(look.NAME), look.STATEFP.astype(int)
    ycol = next(c for c in raw.columns if str(c).lower() == "year")
    frame = pd.DataFrame(
        {"_n": _bst.norm_name(raw["County"]), "_s": pd.to_numeric(raw["StateFP"]).astype(int)}
    )
    cid = frame.merge(look[["_n", "_s", "GEOID"]], on=["_n", "_s"], how="left")["GEOID"]
    cols = [c for c in raw.columns if pattern.fullmatch(str(c))]
    if not cols:
        sys.exit(f"{label}: no columns match {pattern.pattern} in {path}")
    ok = raw[cols].notna().all(axis=1).to_numpy() & cid.notna().to_numpy()
    print(f"  {label}: {len(cols)} feature columns, {int(ok.sum()):,} complete rows")
    return set(cid[ok].astype(str) + "-" + raw[ycol][ok].astype(int).astype(str))


def scan_tiles(tile_dir: Path, timesteps: int) -> tuple[int, int, set[str], int]:
    """Filenames only. Returns (files, complete_tiles, county_years, unparsed)."""
    intervals: dict[tuple[str, str, str], set[int]] = {}
    files = unparsed = 0
    for root, _dirs, names in os.walk(tile_dir):
        for name in names:
            if not name.endswith(".npz"):
                continue
            files += 1
            m = TILE_NAME.search(name)
            if not m:
                unparsed += 1
                continue
            key = (m["county"].zfill(5), m["year"], f"x{m['x']}_y{m['y']}")
            intervals.setdefault(key, set()).add(int(m["interval"]))
    schedule = set(range(timesteps))
    complete = {k for k, seen in intervals.items() if schedule <= seen}
    return files, len(complete), {f"{c}-{y}" for c, y, _ in complete}, unparsed


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--alphaearth-csv", required=True)
    ap.add_argument("--s2-daymet-merged", required=True)
    ap.add_argument("--fips-map", required=True)
    ap.add_argument("--cohort-keys", required=True)
    ap.add_argument("--tile-dir")
    ap.add_argument("--expect-files", type=int)
    ap.add_argument("--expect-county-years", type=int)
    ap.add_argument("--timesteps", type=int, default=7)
    ap.add_argument("--min-state-county-years", type=int, default=40)
    args = ap.parse_args()

    failures: list[str] = []

    def check(label: str, got, want) -> None:
        good = want is None or got == want
        flag = "ok  " if good else "FAIL"
        tail = "" if want is None else f"  (expected {want:,})"
        print(f"  [{flag}] {label}: {got:,}{tail}")
        if not good:
            failures.append(f"{label}: {got:,} != {want:,}")

    cohort = {
        line.strip()
        for line in Path(args.cohort_keys).read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    print(f"\nreference cohort: {len(cohort):,} county-years from {args.cohort_keys}")
    check("cohort key count", len(cohort), args.expect_county_years)

    # ---- 1. tiles ---------------------------------------------------------
    if args.tile_dir:
        print(f"\n1. staged tiles ({args.tile_dir})")
        files, tiles, tile_cy, unparsed = scan_tiles(Path(args.tile_dir), args.timesteps)
        check("npz files", files, args.expect_files)
        check("unparsed filenames", unparsed, 0)
        print(f"  [ok  ] complete {args.timesteps}-interval tiles: {tiles:,}")
        check("county-years with complete tiles", len(tile_cy), args.expect_county_years)
        missing = cohort - tile_cy
        extra = tile_cy - cohort
        check("cohort county-years absent from tiles", len(missing), 0)
        check("tile county-years outside the cohort", len(extra), 0)
        if missing:
            print(f"         e.g. {sorted(missing)[:5]}")
        if extra:
            print(f"         e.g. {sorted(extra)[:5]}")
    else:
        print("\n1. staged tiles: skipped (--tile-dir not given)")

    # ---- 2. the three tabular representations -----------------------------
    print("\n2. tabular representations")
    ae = pd.read_csv(args.alphaearth_csv)
    ae_cols = [c for c in ae.columns if str(c).startswith("mean_A")]
    ae_ok = ae[ae_cols].notna().all(axis=1)
    ae_keys = set(
        ae.GEOID.astype(str).str.zfill(5)[ae_ok] + "-" + ae.year.astype(str)[ae_ok]
    )
    print(f"  AlphaEarth: {len(ae_cols)} feature columns, {int(ae_ok.sum()):,} complete rows")

    s2_keys = keys_from_merged(
        Path(args.s2_daymet_merged), Path(args.fips_map), INDEX_COLUMNS, "S2 indices"
    )
    dm_keys = keys_from_merged(
        Path(args.s2_daymet_merged), Path(args.fips_map), DAYMET_COLUMNS, "Daymet"
    )
    s2dm_keys = s2_keys & dm_keys
    print(f"  S2+Daymet (both complete): {len(s2dm_keys):,} county-years")

    print("\n3. coverage of the reference cohort")
    for label, keys in (
        ("AlphaEarth", ae_keys),
        ("S2 indices", s2_keys),
        ("S2+Daymet", s2dm_keys),
    ):
        check(f"{label} covers the cohort", len(cohort & keys), len(cohort))
        absent = cohort - keys
        if absent:
            print(f"         {len(absent):,} missing, e.g. {sorted(absent)[:5]}")

    paired = cohort & ae_keys & s2_keys & dm_keys
    print("\n4. paired cohort (the N the main table will actually use)")
    check("all three representations + cohort", len(paired), args.expect_county_years)
    counties = {k[:5] for k in paired}
    states = Counter(k[:2] for k in paired)
    eligible = sum(1 for _, n in states.items() if n >= args.min_state_county_years)
    print(f"  counties: {len(counties):,}")
    print(f"  states with >={args.min_state_county_years} county-years: "
          f"{eligible} of {len(states)}")
    by_year = Counter(k.split("-")[1] for k in paired)
    print(f"  by year: {dict(sorted(by_year.items()))}")

    print()
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED")
        for line in failures:
            print(f"  - {line}")
        print("\nDo not extract until these agree; every downstream number keys off this cohort.")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
