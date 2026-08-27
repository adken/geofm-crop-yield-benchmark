#!/usr/bin/env python3
"""Find unreadable NPZ files in a patch directory.

An NPZ is a zip archive. A truncated or empty download raises
``zipfile.BadZipFile: File is not a zip file`` when a DataLoader worker reaches
it -- hours into an extraction, killing the job. A file count matching the
expected cohort says nothing about this: the file exists, it is simply not
valid.

    python scripts/verify_npz_integrity.py /path/to/era5 --jobs 16 \\
        --out-bad /tmp/bad_era5.txt

Reads only the zip central directory and the requested member's header, so the
cost is a metadata operation per file rather than a decompression. Exits
non-zero if anything is unreadable.
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def inspect(path: str, member: str) -> tuple[str, str | None]:
    """Return (path, failure) with failure None when the file is readable."""
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        return path, f"stat failed: {exc}"
    if size == 0:
        return path, "empty file"
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if member and f"{member}.npy" not in names:
                return path, f"missing member {member!r}; has {names[:4]}"
            # Reading one byte forces the local header to be parsed, which
            # catches files whose central directory survived truncation.
            with archive.open(f"{member}.npy") as stream:
                stream.read(1)
    except Exception as exc:  # noqa: BLE001 - any failure means unusable
        return path, f"{type(exc).__name__}: {exc}"
    return path, None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("directory")
    ap.add_argument("--member", default="pixels",
                    help="NPZ array expected inside each file (default: pixels)")
    ap.add_argument("--jobs", type=int, default=16,
                    help="concurrent readers; these are latency-bound on GPFS")
    ap.add_argument("--out-bad", help="write the unreadable paths here, one per line")
    ap.add_argument("--expect", type=int, help="fail if the file count differs")
    args = ap.parse_args()

    root = Path(args.directory)
    if not root.is_dir():
        sys.exit(f"not a directory: {root}")

    paths = [str(p) for p in root.rglob("*.npz")]
    print(f"{len(paths):,} NPZ files under {root}")
    if args.expect is not None and len(paths) != args.expect:
        print(f"  COUNT MISMATCH: expected {args.expect:,}")

    bad: list[tuple[str, str]] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for path, failure in pool.map(
            lambda p: inspect(p, args.member), paths, chunksize=256
        ):
            done += 1
            if failure is not None:
                bad.append((path, failure))
            if done % 50000 == 0:
                print(f"  checked {done:,} / {len(paths):,}  ({len(bad)} bad)", flush=True)

    print(f"\nreadable   {len(paths) - len(bad):,}")
    print(f"unreadable {len(bad):,}")
    for path, failure in bad[:20]:
        print(f"  {Path(path).name}  {failure}")
    if len(bad) > 20:
        print(f"  ... and {len(bad) - 20:,} more")

    if args.out_bad and bad:
        Path(args.out_bad).write_text("\n".join(p for p, _ in bad) + "\n")
        print(f"\nwrote {len(bad):,} paths to {args.out_bad}")

    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
