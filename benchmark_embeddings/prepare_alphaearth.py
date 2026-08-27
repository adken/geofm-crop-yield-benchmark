#!/usr/bin/env python3
"""Convert the matched AlphaEarth county-year CSV to canonical embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .frozen import adapt_alphaearth_csv, write_embeddings


EXPECTED_ALPHAEARTH_DIM = 64


def convert_alphaearth(
    input_path: str | Path,
    output_path: str | Path,
) -> tuple[Path, dict]:
    frame = adapt_alphaearth_csv(input_path, backbone="alphaearth")
    dimensions = sorted({np.asarray(value).size for value in frame["embedding"]})
    if dimensions != [EXPECTED_ALPHAEARTH_DIM]:
        raise ValueError(
            f"AlphaEarth embedding dimensions are {dimensions}, expected "
            f"[{EXPECTED_ALPHAEARTH_DIM}]"
        )
    if frame.duplicated(["county_id", "year"]).any():
        raise ValueError("AlphaEarth source contains duplicate county-year rows")
    frame["experiment_family"] = "main_benchmark"
    frame["input_modalities"] = "precomputed multimodal"
    frame["fusion_stage"] = "none"
    output = write_embeddings(frame, output_path)
    contract = {
        "schema_version": 1,
        "backbone": "alphaearth",
        "source": str(Path(input_path).resolve()),
        "output": str(output.resolve()),
        "county_years": int(len(frame)),
        "embedding_dim": EXPECTED_ALPHAEARTH_DIM,
        "representation_scope": "sequence",
        "experiment_family": "main_benchmark",
        "climate_late_fusion": False,
    }
    sidecar = output.with_suffix(output.suffix + ".provenance.json")
    sidecar.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    return output, contract


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Matched AlphaEarth wide CSV")
    parser.add_argument("--output", required=True, help="Canonical output Parquet")
    args = parser.parse_args(argv)
    output, contract = convert_alphaearth(args.input, args.output)
    print(json.dumps(contract, indent=2, sort_keys=True))
    print(f"Wrote canonical AlphaEarth embeddings to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
