"""Convert historical AlphaEarth/Clay/Presto/Prithvi/TerraMind outputs.

These adapters handle storage differences only. They deliberately do not
repair upstream preprocessing, band-order, or split errors; regenerate an
embedding when its provenance does not satisfy the benchmark contract.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .schema import validate_embeddings

_COUNTY_YEAR = re.compile(
    r"county[_-](?P<county>\d+)(?:[_-]year)?[_-](?P<year>\d{4})",
    re.IGNORECASE,
)
_INTERVAL = re.compile(r"(?:interval|timestep|month)[_-]?(?P<t>\d+)", re.IGNORECASE)


def _pick(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    lower = {column.lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def _decode(value: Any) -> list[float]:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return np.frombuffer(value, dtype=np.float32).tolist()
    return np.asarray(value, dtype=np.float32).reshape(-1).tolist()


def adapt_embedding_parquet(
    path: str | Path,
    *,
    backbone: str,
    embedding_column: str | None = None,
) -> pd.DataFrame:
    """Adapt Clay, Prithvi, or TerraMind list/bytes parquet variants."""
    frame = pd.read_parquet(path)
    embedding_column = embedding_column or _pick(
        frame, ("embedding", "cls_embedding", "emb_cls", "mean_embedding", "emb_mean")
    )
    county = _pick(frame, ("county_id", "county", "county_fips", "fips"))
    year = _pick(frame, ("year", "Year"))
    patch = _pick(frame, ("patch_id", "field_id", "filename", "file"))
    timestep = _pick(frame, ("timestep", "interval_index", "interval", "month"))
    missing = [
        name
        for name, value in {
            "embedding": embedding_column,
            "county": county,
            "year": year,
            "patch": patch,
            "timestep": timestep,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError(f"{path} cannot be adapted; missing semantic fields {missing}")
    output = pd.DataFrame(
        {
            "county_id": frame[county],
            "year": frame[year],
            "patch_id": frame[patch],
            "timestep": frame[timestep],
            "backbone": backbone,
            "embedding": frame[embedding_column].map(_decode),
            "source_path": str(path),
        }
    )
    return validate_embeddings(output)


def adapt_presto_arrays(
    embeddings_path: str | Path,
    metadata_path: str | Path,
    *,
    backbone: str = "presto",
) -> pd.DataFrame:
    """Adapt paired Presto NPY arrays at patch or county-year level."""
    embeddings = np.load(embeddings_path)
    metadata = np.load(metadata_path, allow_pickle=True).tolist()
    if embeddings.ndim != 2 or len(metadata) != embeddings.shape[0]:
        raise ValueError("Presto embeddings and metadata must have matching rows")
    rows = []
    for index, (vector, item) in enumerate(zip(embeddings, metadata)):
        item = dict(item)
        location = str(item.get("location_id", item.get("county_year", "")))
        match = _COUNTY_YEAR.search(location.replace("_year_", "_"))
        county = item.get("county", item.get("county_id"))
        year = item.get("year")
        if match:
            county = county or match.group("county")
            year = year or int(match.group("year"))
        if county is None or year is None:
            parts = location.split("_")
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                county, year = parts[0], int(parts[1])
        if county is None or year is None:
            raise ValueError(f"cannot infer county/year from Presto metadata row {index}")
        rows.append(
            {
                "county_id": county,
                "year": year,
                "patch_id": item.get("patch_id", location or f"presto-{index}"),
                "timestep": int(item.get("timestep", item.get("interval_index", 0))),
                "backbone": backbone,
                "embedding": np.asarray(vector, dtype=np.float32).tolist(),
                "source_path": str(embeddings_path),
            }
        )
    return validate_embeddings(pd.DataFrame(rows))


def adapt_alphaearth_csv(
    path: str | Path,
    *,
    backbone: str = "alphaearth",
) -> pd.DataFrame:
    """Adapt AlphaEarth wide ``mean_A00...`` county-year features."""
    frame = pd.read_csv(path)
    features = sorted(
        (column for column in frame.columns if re.fullmatch(r"mean_A\d+", column)),
        key=lambda column: int(column[6:]),
    )
    # Prefer a stable numeric identifier over a display-name column.  The
    # historical matched CSV contains both ``GEOID`` and ``county``; choosing
    # the latter collapses same-named counties across states.
    county = _pick(
        frame,
        ("county_id", "county_geoid", "GEOID", "county_fips", "fips", "county"),
    )
    year = _pick(frame, ("year", "Year"))
    if not features or county is None or year is None:
        raise ValueError(f"{path} lacks AlphaEarth feature/county/year columns")
    output = pd.DataFrame(
        {
            "county_id": frame[county],
            "year": frame[year],
            "patch_id": [f"alphaearth-county-{index}" for index in frame.index],
            "timestep": 0,
            "backbone": backbone,
            "embedding": frame[features].astype(np.float32).values.tolist(),
            "representation_scope": "sequence",
            "source_path": str(path),
        }
    )
    return validate_embeddings(output)


def adapt_npz_directory(
    path: str | Path,
    *,
    backbone: str,
    embedding_keys: Sequence[str] = (
        "embedding", "embeddings", "cls_embedding", "mean_embedding", "emb_mean"
    ),
) -> pd.DataFrame:
    """Adapt per-file Prithvi/TerraMind NPZ embeddings."""
    rows = []
    for file in sorted(Path(path).rglob("*.npz")):
        with np.load(file, allow_pickle=True) as z:
            key = next((candidate for candidate in embedding_keys if candidate in z.files), None)
            if key is None:
                continue
            value = np.asarray(z[key], dtype=np.float32)
            meta = z["meta"].item() if "meta" in z.files and z["meta"].shape == () else {}
        match = _COUNTY_YEAR.search(file.stem)
        county = meta.get("county", meta.get("county_id")) if isinstance(meta, dict) else None
        year = meta.get("year") if isinstance(meta, dict) else None
        if match:
            county = county or match.group("county")
            year = year or int(match.group("year"))
        if county is None or year is None:
            raise ValueError(f"cannot infer county/year from {file}")
        interval_match = _INTERVAL.search(file.stem)
        if value.ndim == 1:
            vectors = [(int(interval_match.group("t")) if interval_match else 0, value)]
        elif value.ndim == 2:
            vectors = list(enumerate(value))
        else:
            raise ValueError(f"{file}:{key} must be [D] or [T,D], got {value.shape}")
        for timestep, vector in vectors:
            rows.append(
                {
                    "county_id": county,
                    "year": year,
                    "patch_id": file.stem,
                    "timestep": timestep,
                    "backbone": backbone,
                    "embedding": vector.tolist(),
                    "source_path": str(file),
                }
            )
    if not rows:
        raise ValueError(f"no adaptable embeddings found below {path}")
    return validate_embeddings(pd.DataFrame(rows))
