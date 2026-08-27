#!/usr/bin/env python3
"""Seasonal trajectories: embedding PC1 against vegetation indices.

Asks whether frozen embeddings track crop phenology without task supervision.
For each encoder, the per-timestep county-year embeddings are projected onto
their first principal component, and the resulting seasonal curve is compared
with the Sentinel-2 vegetation indices over the same seven composites.

    python scripts/plot_seasonal_trajectories.py \\
        --embeddings clay=outputs/embeddings/clay_v1_5_cls.parquet \\
        --embeddings prithvi=outputs/embeddings/prithvi.parquet \\
        --s2-indices outputs/cohort_covered/sentinel2_indices_covered.csv \\
        --out figures/temporal_evolution_lai_fpar_embeddings.png

Three choices worth knowing, all recorded in the sidecar JSON:

* PC1 is fitted once on the pooled per-timestep vectors across every
  county-year, so a single component describes the whole season rather than a
  different one per composite.
* Its sign is fixed by requiring positive correlation with the first index
  named in --indices (LAI by default). PCA signs
  are arbitrary, and without this the curve can appear inverted between runs.
* Curves are min-max normalised using the mean trajectory's own range, and the
  same affine map is applied to the standard-deviation band so relative spread
  is preserved.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def county_year_timestep_matrix(
    parquet: str, batch_size: int = 4_096
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return [N, T, D] embeddings averaged over patches, plus keys and timesteps.

    Streamed in batches: these tables hold one row per patch and per composite,
    so Clay and Prithvi are 427,049 x 1024 and do not fit in memory as Python
    objects. Only the running per-(county-year, timestep) sum is retained, which
    is 2,076 x 7 x 1024 regardless of how many patches contribute.
    """
    import pyarrow.parquet as pq

    handle = pq.ParquetFile(parquet)
    columns = ["county_id", "year", "timestep", "embedding"]

    # First pass over the small columns only, to fix the output shape.
    identity = handle.read(columns=["county_id", "year", "timestep"]).to_pandas()
    identity["key"] = (
        identity.county_id.astype(str).str.zfill(5) + "-" + identity.year.astype(str)
    )
    keys = sorted(identity["key"].unique())
    timesteps = np.sort(identity["timestep"].unique())
    key_index = {k: i for i, k in enumerate(keys)}
    step_index = {int(t): i for i, t in enumerate(timesteps)}
    del identity

    total = None
    count = np.zeros((len(keys), len(timesteps)), dtype=np.int64)
    for batch in handle.iter_batches(batch_size=batch_size, columns=columns):
        # Flatten the Arrow list column directly rather than via pandas: one
        # Python object per row would be 427,049 lists of 1024 floats and is
        # what pushes peak memory past 3 GB.
        listed = batch.column("embedding")
        width = len(listed) and len(listed[0])
        vectors = (
            listed.flatten().to_numpy(zero_copy_only=False)
            .astype(np.float64, copy=False)
            .reshape(len(listed), width)
        )
        if total is None:
            total = np.zeros((len(keys), len(timesteps), width))
        counties = batch.column("county_id").to_pylist()
        years = batch.column("year").to_pylist()
        rows = np.fromiter(
            (key_index[f"{str(c).zfill(5)}-{y}"] for c, y in zip(counties, years)),
            dtype=np.int64, count=len(listed),
        )
        cols = np.fromiter(
            (step_index[int(t)] for t in batch.column("timestep").to_pylist()),
            dtype=np.int64, count=len(listed),
        )
        np.add.at(total, (rows, cols), vectors)
        np.add.at(count, (rows, cols), 1)
        # These tables are gigabytes; without an explicit release the batch
        # buffers accumulate faster than the collector reclaims them.
        del vectors, listed, counties, years, rows, cols, batch
        gc.collect()

    if (count == 0).any():
        raise ValueError(f"{parquet}: some county-years lack a complete timestep set")
    return total / count[:, :, None], keys, timesteps


def first_component(cube: np.ndarray) -> np.ndarray:
    """Project [N,T,D] onto the PC1 of the pooled [N*T, D] matrix."""
    flat = cube.reshape(-1, cube.shape[-1])
    centred = flat - flat.mean(0, keepdims=True)
    # Economy SVD: only the leading right singular vector is needed.
    _, _, components = np.linalg.svd(centred, full_matrices=False)
    return (centred @ components[0]).reshape(cube.shape[:2])


def index_trajectories(
    path: str, keys: list[str], names: tuple[str, ...] = ("LAI", "FPAR")
) -> dict[str, np.ndarray]:
    """Selected vegetation indices over the seven composites, aligned to `keys`."""
    frame = pd.read_csv(path)
    frame["key"] = (
        frame.county_id.astype(str).str.zfill(5) + "-" + frame.year.astype(str)
    )
    frame = frame.set_index("key").loc[keys]
    out = {}
    for name in names:
        columns = sorted(
            [c for c in frame.columns if re.fullmatch(rf"{name}_\d+", str(c), re.I)],
            key=lambda c: int(str(c).split("_")[1]),
        )
        if not columns:
            raise ValueError(f"{path}: no {name}_* columns")
        out[name] = frame[columns].to_numpy(dtype=float)
    return out


def normalise(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and sd across county-years, min-max scaled by the mean's range."""
    mean = curves.mean(0)
    sd = curves.std(0, ddof=0)
    low, high = float(mean.min()), float(mean.max())
    scale = (high - low) or 1.0
    return (mean - low) / scale, sd / scale


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--embeddings", action="append", default=[], metavar="NAME=PATH",
                    help="repeatable, e.g. clay=outputs/embeddings/clay_v1_5_cls.parquet")
    ap.add_argument("--s2-indices", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", nargs="*", default=None,
                    help="x-axis composite labels; defaults to composite numbers")
    ap.add_argument("--indices", nargs="+", default=["LAI", "FPAR"],
                    help="vegetation indices to plot (default: LAI FPAR). The "
                         "first is also the reference used to fix the PC1 sign.")
    ap.add_argument("--colours", nargs="*", default=None, metavar="NAME=COLOUR",
                    help="override series colours, e.g. clay=#ff7f0e")
    args = ap.parse_args()
    if not args.embeddings:
        ap.error("at least one --embeddings NAME=PATH is required")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    contract: dict[str, object] = {}
    shared_keys: list[str] | None = None
    timesteps = None

    raw: dict[str, np.ndarray] = {}
    for spec in args.embeddings:
        name, path = spec.split("=", 1)
        cube, keys, timesteps = county_year_timestep_matrix(path)
        shared_keys = keys if shared_keys is None else shared_keys
        if keys != shared_keys:
            raise SystemExit("encoders cover different county-years; restrict them first")
        raw[name] = first_component(cube)
        contract[f"{name}_county_years"] = len(keys)

    indices = index_trajectories(args.s2_indices, shared_keys, tuple(args.indices))
    reference_name = args.indices[0]
    reference = indices[reference_name].mean(0)
    contract["pc1_sign_reference"] = f"mean {reference_name}"

    for name, projected in raw.items():
        # Fix the arbitrary PCA sign so the curve is comparable across runs.
        if np.corrcoef(projected.mean(0), reference)[0, 1] < 0:
            projected = -projected
            contract[f"{name}_pc1_sign_flipped"] = True
        series[name] = normalise(projected)
    for name, values in indices.items():
        series[name] = normalise(values)

    x = np.arange(1, len(timesteps) + 1)
    labels = args.labels or [str(v) for v in x]
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=200)
    # Fixed colours so the figure is stable across runs and matches the caption.
    default_colours = {
        "clay": "#ff7f0e", "prithvi": "#d62728",
        "LAI": "#2ca02c", "FPAR": "#1f77b4", "EVI": "#d62728",
    }
    colours = dict(default_colours)
    for item in args.colours or []:
        name, value = item.split("=", 1)
        colours[name] = value
    fallback = ["#9467bd", "#8c564b", "#17becf", "#bcbd22"]
    for position, (name, (mean, sd)) in enumerate(series.items()):
        colour = colours.get(name, fallback[position % len(fallback)])
        ax.fill_between(x, mean - sd, mean + sd, alpha=0.15, color=colour, linewidth=0)
        ax.plot(x, mean, marker="o", linewidth=1.8, color=colour, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("28-day composite")
    ax.set_ylabel("normalised value")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax.legend(frameon=False, ncols=len(series), loc="lower center")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    contract["county_years"] = len(shared_keys)
    contract["timesteps"] = [int(t) for t in timesteps]
    contract["normalisation"] = "min-max on the mean trajectory; sd scaled identically"
    Path(str(out) + ".contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(f"wrote {out}  ({len(shared_keys):,} county-years)")
    for name, (mean, _) in series.items():
        print(f"  {name:12} peak at composite {int(np.argmax(mean)) + 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
