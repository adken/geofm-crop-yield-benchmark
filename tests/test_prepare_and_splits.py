from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_embeddings.build_splits import build_common_split_manifest
from benchmark_embeddings.frozen import read_embeddings
from benchmark_embeddings.prepare_alphaearth import convert_alphaearth


def _alphaearth_source(counties: list[str], years: list[int]) -> pd.DataFrame:
    rows = []
    for county_index, county in enumerate(counties):
        for year in years:
            row = {"GEOID": county, "year": year}
            row.update(
                {
                    f"mean_A{feature:02d}": county_index + year / 10000.0 + feature
                    for feature in range(64)
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _merged_s2_daymet(counties: list[str], years: list[int]) -> pd.DataFrame:
    rows = []
    for county in counties:
        for year in years:
            row = {"county_id": county, "year": year}
            for variable in ("evi", "lai", "fpar"):
                row.update({f"{variable}_{step}": step + 0.1 for step in range(7)})
            for variable in ("dayl", "prcp", "srad", "tmax", "tmin"):
                row.update({f"{variable}_{step}": step + 1.0 for step in range(7)})
            rows.append(row)
    return pd.DataFrame(rows)


def test_alphaearth_conversion_and_grouped_split_manifest(tmp_path: Path) -> None:
    counties = [f"17{index:03d}" for index in range(1, 6)]
    years = [2019, 2020, 2021, 2022]

    source = tmp_path / "alphaearth.csv"
    _alphaearth_source(counties, years).to_csv(source, index=False)
    alpha_path, provenance = convert_alphaearth(
        source, tmp_path / "alphaearth.parquet"
    )
    alpha = read_embeddings(alpha_path)
    assert len(alpha) == 20
    assert alpha["embedding"].map(len).unique().tolist() == [64]
    assert alpha["representation_scope"].unique().tolist() == ["sequence"]
    assert provenance["county_years"] == 20
    assert json.loads(
        alpha_path.with_suffix(".parquet.provenance.json").read_text()
    )["embedding_dim"] == 64

    s2_root = tmp_path / "sentinel2"
    s2_root.mkdir()
    for county in counties:
        for year in years:
            for timestep in range(7):
                np.savez(
                    s2_root
                    / (
                        f"county_{county}_year_{year}_x1_y2_stack_COG_"
                        f"interval_{timestep:02d}.npz"
                    ),
                    pixels=np.zeros((10, 2, 2), dtype=np.float32),
                )

    merged = tmp_path / "s2_daymet.csv"
    _merged_s2_daymet(counties, years).to_csv(merged, index=False)
    labels = tmp_path / "labels.csv"
    pd.DataFrame(
        [
            {"county_id": county, "year": year, "yield": 150.0}
            for county in counties
            for year in years
        ]
    ).to_csv(labels, index=False)

    output = tmp_path / "group_kfold.csv"
    manifest, contract = build_common_split_manifest(
        s2_dir=s2_root,
        s2_daymet_merged=merged,
        s2_fips_map=None,
        alphaearth_path=alpha_path,
        labels_path=labels,
        output_path=output,
        years=years,
        expected_input_count=140,
    )

    assert len(manifest) == 100
    assert contract["county_years"] == 20
    assert contract["counties"] == 5
    assert contract["source_counts"] == {
        "complete_patch_sequences": 20,
        "sentinel2_indices": 20,
        "daymet": 20,
        "alphaearth": 20,
        "yield_labels": 20,
    }
    for fold, rows in manifest.groupby("fold"):
        assert set(rows["split"]) == {"train", "val", "test"}
        for _, county_rows in rows.groupby("county_id"):
            assert county_rows["split"].nunique() == 1
        for _, role_rows in rows.groupby("split"):
            assert set(role_rows["year"]) == set(years)
    test_assignments = manifest.loc[manifest["split"] == "test"]
    assert test_assignments.groupby("county_id")["fold"].nunique().eq(1).all()
    assert output.with_suffix(".csv.contract.json").exists()
