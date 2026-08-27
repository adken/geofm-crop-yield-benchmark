"""Canonical per-patch frozen-representation table.

Image encoders normally emit one row per patch-timestep. Temporal encoders
such as Presto emit one row per complete patch sequence and identify that in
the optional ``representation_scope`` column while retaining ``timestep=0``
as the unique representation index.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "county_id",
    "year",
    "patch_id",
    "timestep",
    "backbone",
    "embedding",
)


def _county(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(5) if text.isdigit() else text


def _vector(value: Any) -> np.ndarray:
    if isinstance(value, (bytes, bytearray, memoryview)):
        vector = np.frombuffer(value, dtype=np.float32)
    else:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("embedding vectors must be non-empty and finite")
    return vector


def validate_embeddings(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize identifiers and reject ambiguous or ragged embedding tables."""
    missing = set(REQUIRED_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"embedding table is missing columns {sorted(missing)}")
    output = frame.copy()
    output["county_id"] = output["county_id"].map(_county)
    output["year"] = pd.to_numeric(output["year"], errors="raise").astype(int)
    output["patch_id"] = output["patch_id"].astype(str)
    output["timestep"] = pd.to_numeric(output["timestep"], errors="raise").astype(int)
    if (output["timestep"] < 0).any():
        raise ValueError("timestep indices must be non-negative")
    output["backbone"] = output["backbone"].astype(str)
    vectors = output["embedding"].map(_vector)
    dimensions = sorted({int(vector.size) for vector in vectors})
    if len(dimensions) != 1:
        raise ValueError(f"ragged embeddings have dimensions {dimensions}")
    output["embedding"] = vectors.map(lambda vector: vector.tolist())
    key = ["county_id", "year", "patch_id", "timestep", "backbone"]
    duplicates = output.duplicated(key, keep=False)
    if duplicates.any():
        raise ValueError(f"duplicate embedding keys: {int(duplicates.sum())} rows")
    return output.sort_values(key).reset_index(drop=True)


def write_embeddings(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_embeddings(frame).to_parquet(path, index=False)

    # Older pandas/pyarrow pairs -- notably the pandas==1.5.3 that openmapflow
    # pins in the Presto environment -- silently store the embedding column as
    # the string repr of a list rather than list<double>. It round-trips, but
    # every downstream consumer then has to know, and the failure surfaces far
    # away as a dtype error inside np.mean. Catch it here, at the source.
    try:
        import pyarrow.parquet as pq

        stored = pq.read_schema(path).field("embedding").type
        if not pa_types_is_list(stored):
            # Repair rather than raise: this fires only after a multi-hour
            # extraction, and failing there would discard the run's provenance
            # for a defect that is fully recoverable. Python's float repr
            # round-trips, so parsing the stored text is lossless.
            import numpy as np

            print(
                f"{path}: 'embedding' stored as {stored} rather than a list "
                "type; this environment's pandas/pyarrow cannot write list "
                "columns natively. Converting in place."
            )
            repaired = pd.read_parquet(path)
            repaired["embedding"] = repaired["embedding"].map(
                lambda value: np.fromstring(
                    str(value).strip().strip("[]"), sep=","
                ).tolist()
            )
            widths = {len(v) for v in repaired["embedding"]}
            if len(widths) != 1:
                raise RuntimeError(
                    f"{path}: parsing the stored strings gave inconsistent "
                    f"embedding widths {sorted(widths)}; the file is left as "
                    "written for inspection"
                )
            temporary = path.with_suffix(path.suffix + ".repair")
            repaired.to_parquet(temporary, index=False)
            if not pa_types_is_list(pq.read_schema(temporary).field("embedding").type):
                temporary.unlink(missing_ok=True)
                raise RuntimeError(
                    f"{path}: conversion did not change the stored type; "
                    "repair this file manually before use"
                )
            temporary.replace(path)
            print(f"{path}: converted, {len(repaired):,} rows, "
                  f"dim {widths.pop()}")
    except (ImportError, KeyError):
        pass
    return path


def pa_types_is_list(dtype) -> bool:
    """True only for a genuine Arrow list type.

    Tested positively rather than by listing the ways it can go wrong. Different
    pandas/pyarrow pairs have stored this column as string, large_string and
    extension<arrow.json>; each round-trips as text and breaks downstream, and a
    negative test would have to be extended for every new variant.
    """
    import pyarrow as pa

    return pa.types.is_list(dtype) or pa.types.is_large_list(dtype)


def read_embeddings(path: str | Path) -> pd.DataFrame:
    return validate_embeddings(pd.read_parquet(path))
