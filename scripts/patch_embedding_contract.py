#!/usr/bin/env python3
"""Add constant contract columns to an existing embeddings Parquet.

The four frozen extractors do not emit the same metadata columns. Clay writes
none of `representation_scope`, `experiment_family`, `fusion_stage`,
`input_modalities` or `temporal_ingestion`; TerraMind writes only the first;
Prithvi writes all five. The LOYO and temporal-ablation validators require
`representation_scope`, so a Clay table fails with

    ValueError: clay requires representation_scope='timestep', got []

Re-extracting to add a constant string is twelve hours of GPU time, so this
patches the file in place instead. Row groups are streamed, so a 2.6 GB table
does not need to be held in memory.

    python scripts/patch_embedding_contract.py outputs/embeddings/clay_v1_5_cls.parquet \\
        representation_scope=timestep experiment_family=main_benchmark \\
        fusion_stage=none input_modalities=Sentinel-2 \\
        temporal_ingestion=single_timestep_independent

Existing columns are never overwritten; the script reports and skips them.
The original is left as PATH.bak unless --no-backup is given.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("parquet")
    ap.add_argument("assignments", nargs="+", metavar="COLUMN=VALUE")
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    path = Path(args.parquet)
    if not path.is_file():
        sys.exit(f"not a file: {path}")

    additions: dict[str, str] = {}
    for item in args.assignments:
        if "=" not in item:
            sys.exit(f"expected COLUMN=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        additions[key.strip()] = value

    source = pq.ParquetFile(path)
    present = set(source.schema_arrow.names)
    skipped = sorted(k for k in additions if k in present)
    for key in skipped:
        print(f"  already present, skipping: {key}")
        additions.pop(key)
    if not additions:
        print("nothing to add")
        return 0

    rows = source.metadata.num_rows
    print(f"{path.name}: {rows:,} rows, {source.num_row_groups} row groups")
    for key, value in additions.items():
        print(f"  adding {key} = {value!r}")
    if args.dry_run:
        print("  --dry-run: nothing written")
        return 0

    tmp = path.with_suffix(path.suffix + ".tmp")
    writer = None
    written = 0
    try:
        for i in range(source.num_row_groups):
            table = source.read_row_group(i)
            for key, value in additions.items():
                table = table.append_column(
                    key, pa.array([value] * table.num_rows, type=pa.string())
                )
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema)
            writer.write_table(table)
            written += table.num_rows
    finally:
        if writer is not None:
            writer.close()

    if written != rows:
        tmp.unlink(missing_ok=True)
        sys.exit(f"wrote {written:,} rows for {rows:,} expected; original untouched")

    if not args.no_backup:
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    tmp.replace(path)
    check = pq.ParquetFile(path)
    print(f"  wrote {check.metadata.num_rows:,} rows")
    print(f"  columns now: {len(check.schema_arrow.names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
