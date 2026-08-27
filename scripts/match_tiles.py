#!/usr/bin/env python3
"""Match Sentinel-2 tiles on the cluster against a target county-year cohort.

Scans a tile directory by FILENAME ONLY -- no NPZ is opened -- so it stays fast
over ~1M files on shared storage. Reports how much of the target cohort is
covered by complete seven-interval tile sequences, and writes a file list you can
hand to rsync.

    # report only
    python match_tiles.py --tile-dir /scratch/tiles --keys data/sources/embeddings_with_yield_matched.csv

    # report + emit a transfer list
    python match_tiles.py --tile-dir /scratch/tiles \\
        --keys data/sources/embeddings_with_yield_matched.csv \\
        --out-list tiles_to_copy.txt

    # then, from the cluster:
    rsync -av --files-from=tiles_to_copy.txt / user@host:/dest/

Read-only. Copies nothing, deletes nothing.

--keys accepts a CSV/XLSX with county and year columns (GEOID/county_id/FIPS +
year), or a plain text file of one `county-year` key per line, e.g. 17001-2019.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

FILENAME = re.compile(
    r"county_(?P<county>\d+)_year_(?P<year>\d{4})"
    r"(?:_\d+)?"                       # optional extra segment in the cluster naming
    r"_x(?P<x>-?\d+)_y(?P<y>-?\d+)_interval_(?P<interval>\d+)",
    re.IGNORECASE,
)


def load_keys(path: Path) -> set[str]:
    """county-year keys from a table or a plain key list."""
    if path.suffix.lower() in {".txt", ".list", ""}:
        return {
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        }
    import pandas as pd

    frame = (
        pd.read_excel(path)
        if path.suffix.lower() in {".xlsx", ".xls"}
        else pd.read_csv(path)
    )
    lower = {str(c).strip().lower(): c for c in frame.columns}
    ccol = next(
        (lower[n] for n in ("geoid", "county_id", "fips", "county_fips") if n in lower),
        None,
    )
    ycol = next((lower[n] for n in ("year", "yr") if n in lower), None)
    if ccol is None or ycol is None:
        sys.exit(
            f"{path}: need a county FIPS column (GEOID/county_id/fips) and a year "
            f"column; found {list(frame.columns)[:10]}"
        )
    county = frame[ccol].astype(str).str.extract(r"(\d+)")[0].str.zfill(5)
    year = frame[ycol].astype(int).astype(str)
    return set((county + "-" + year).dropna())


def scan(tile_dir: Path) -> tuple[dict, int, int]:
    """Walk filenames only. Returns tiles -> set(interval), plus counters."""
    tiles: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    paths: dict[tuple[str, str, str], dict[int, str]] = defaultdict(dict)
    seen = unparsed = 0
    for root, _dirs, names in os.walk(tile_dir):
        for name in names:
            if not name.endswith(".npz"):
                continue
            seen += 1
            m = FILENAME.search(name)
            if not m:
                unparsed += 1
                continue
            key = (
                m.group("county").zfill(5),
                m.group("year"),
                f"x{m.group('x')}_y{m.group('y')}",
            )
            iv = int(m.group("interval"))
            tiles[key].add(iv)
            paths[key][iv] = os.path.join(root, name)
    return (tiles, paths), seen, unparsed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tile-dir", required=True)
    ap.add_argument("--keys", required=True)
    ap.add_argument("--out-list", help="write matching file paths here, one per line")
    ap.add_argument("--out-keys",
                    help="write the covered county-year keys here, one per line. "
                         "This is the definitive cohort the tiles support -- feed it "
                         "to build_splits_tabular.py --restrict-keys")
    ap.add_argument("--timesteps", type=int, default=7,
                    help="required in-schedule intervals, 0..N-1 (default 7)")
    ap.add_argument("--link-dir",
                    help="materialise the matched tiles in this directory "
                         "(created if absent). The source is never modified.")
    ap.add_argument("--link-mode", choices=("hardlink", "symlink", "copy", "move"),
                    default="hardlink",
                    help="hardlink (default) costs no extra storage on the same "
                         "filesystem. 'move' renames instead -- cheap within a "
                         "filesystem and the only mode that makes deleting the "
                         "remainder safe without further checks. 'symlink' does NOT: "
                         "deleting the source afterwards destroys the data.")
    ap.add_argument("--link-jobs", type=int, default=16,
                    help="concurrent link operations (default 16). Metadata ops on "
                         "GPFS/Lustre are latency-bound, so this is a large win.")
    ap.add_argument("--dry-run", action="store_true",
                    help="with --link-dir, report what would be created and stop")
    args = ap.parse_args()

    tile_dir = Path(args.tile_dir)
    if not tile_dir.is_dir():
        sys.exit(f"not a directory: {tile_dir}")
    target = load_keys(Path(args.keys))
    print(f"target cohort: {len(target):,} county-years\n")

    print(f"scanning {tile_dir} (filenames only)...")
    (tiles, paths), seen, unparsed = scan(tile_dir)
    print(f"  files seen      {seen:,}")
    print(f"  unparsed names  {unparsed:,}")
    print(f"  distinct tiles  {len(tiles):,}\n")

    schedule = set(range(args.timesteps))
    complete = {k for k, iv in tiles.items() if schedule <= iv}
    cy_all = {f"{c}-{y}" for c, y, _ in tiles}
    cy_complete = {f"{c}-{y}" for c, y, _ in complete}

    covered = target & cy_complete
    partial = (target & cy_all) - cy_complete
    missing = target - cy_all

    print("Coverage of the target cohort")
    print(f"  complete ({args.timesteps} intervals)  {len(covered):>7,}"
          f"   {100*len(covered)/max(len(target),1):>5.1f}%")
    print(f"  present but incomplete            {len(partial):>7,}")
    print(f"  absent from the tile corpus       {len(missing):>7,}")
    print(f"  counties covered                  "
          f"{len({k[:5] for k in covered}):>7,}\n")

    if covered:
        by_year = Counter(k.split("-")[1] for k in covered)
        tgt_year = Counter(k.split("-")[1] for k in target)
        print("  by year (covered / target):")
        for y in sorted(tgt_year):
            print(f"    {y}   {by_year.get(y,0):>5,} / {tgt_year[y]:>5,}")
        st = Counter(k[:2] for k in covered)
        print(f"\n  states with >=40 covered county-years: "
              f"{sum(1 for _, n in st.items() if n >= 40)} of {len(st)}")

    wanted = sorted(
        p
        for key in complete
        if f"{key[0]}-{key[1]}" in target
        for p in paths[key].values()
    )

    if args.out_keys:
        Path(args.out_keys).write_text("\n".join(sorted(covered)) + "\n")
        print(f"\nwrote {len(covered):,} covered county-year keys to {args.out_keys}")
        print(f"  {len({k[:5] for k in covered}):,} counties, "
              f"{sum(1 for _, n in Counter(k[:2] for k in covered).items() if n >= 40)} "
              f"states with >=40 county-years")

    if args.out_list:
        Path(args.out_list).write_text("\n".join(wanted) + "\n")
        approx = sum(os.path.getsize(p) for p in wanted[:200]) / max(min(len(wanted), 200), 1)
        print(f"\nwrote {len(wanted):,} paths to {args.out_list}")
        print(f"  approx size {len(wanted)*approx/1e9:,.1f} GB "
              f"(~{approx/1e6:.2f} MB/file)")
        print(f"  rsync -av --files-from={args.out_list} / <dest>/")

    if args.link_dir:
        dest = Path(args.link_dir)
        approx = sum(os.path.getsize(p) for p in wanted[:200]) / max(min(len(wanted), 200), 1)
        print(f"\n{args.link_mode} {len(wanted):,} files -> {dest}")
        print(f"  logical size {len(wanted)*approx/1e9:,.1f} GB"
              f"{'  (hardlinks consume no extra space)' if args.link_mode=='hardlink' else ''}")
        names = Counter(Path(p).name for p in wanted)
        clash = [n for n, c in names.items() if c > 1]
        if clash:
            sys.exit(f"  ABORT: {len(clash)} duplicate filenames, e.g. {clash[:3]}. "
                     f"A flat directory would lose files.")
        if args.dry_run:
            print("  --dry-run: nothing created")
            return 0
        dest.mkdir(parents=True, exist_ok=True)

        # One syscall per file in the common case. The previous version called
        # out.exists() first, which doubled the metadata operations -- the
        # bottleneck on a parallel filesystem. FileExistsError gives idempotency
        # for free. Threads help because these are syscalls and release the GIL.
        import shutil
        from concurrent.futures import ThreadPoolExecutor

        counts = {"made": 0, "skipped": 0, "fell_back": 0}

        def place(src: str) -> str:
            out = dest / Path(src).name
            try:
                if args.link_mode == "hardlink":
                    os.link(src, out)
                elif args.link_mode == "symlink":
                    os.symlink(os.path.abspath(src), out)
                elif args.link_mode == "move":
                    os.replace(src, out)
                else:
                    shutil.copy2(src, out)
                return "made"
            except FileExistsError:
                return "skipped"
            except OSError:
                if args.link_mode == "move":
                    raise           # cross-device move: never silently degrade
                try:
                    os.symlink(os.path.abspath(src), out)
                    return "fell_back"
                except FileExistsError:
                    return "skipped"

        with ThreadPoolExecutor(max_workers=args.link_jobs) as pool:
            for i, r in enumerate(pool.map(place, wanted, chunksize=256), 1):
                counts["made" if r == "fell_back" else r] += 1
                if r == "fell_back":
                    counts["fell_back"] += 1
                if i % 50000 == 0:
                    print(f"    {i:,} / {len(wanted):,}", flush=True)
        print(f"  created {counts['made']:,}   already present {counts['skipped']:,}"
              + (f"   symlink fallback {counts['fell_back']:,}" if counts["fell_back"] else ""))

        if counts["fell_back"] or args.link_mode == "symlink":
            print()
            print("  " + "!" * 68)
            print("  WARNING: some or all entries are SYMLINKS, not hardlinks.")
            print("  They point at the originals. DELETING THE SOURCE WILL DESTROY")
            print("  THE DATA -- the links will dangle and the tiles are unrecoverable.")
            print("  Use --link-mode move if you intend to remove the remainder.")
            print("  " + "!" * 68)
        elif args.link_mode == "hardlink":
            print("\n  hardlinks: the source and this directory share inodes, so")
            print("  removing the source afterwards is safe -- the data survives here.")
        elif args.link_mode == "move":
            print("\n  moved: the source no longer holds these files. What remains under")
            print("  the tile directory is exactly the set outside the cohort.")
        print(f"\n  point extraction at it:")
        print(f"    export COUNTY_PATCH_TIMESTEPS={dest}")
        print(f"    export CLAY_EXPECTED_INPUT_COUNT={len(wanted)}   # and the "
              f"PRESTO_/PRITHVI_/TERRAMIND_ equivalents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
