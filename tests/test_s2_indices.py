from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_embeddings.daymet import DAYMET_VARIABLES, fuse_daymet_features, load_daymet_features
from benchmark_embeddings.probe import (
    AUXILIARY_CLIMATE_FUSION,
    resolve_experiment_contract,
)
from benchmark_embeddings.s2_indices import S2_INDEX_VARIABLES, load_s2_index_features


def test_s2_indices_form_21d_baseline_and_support_daymet_late_fusion(
    tmp_path: Path,
) -> None:
    row: dict[str, object] = {"county_id": "17001", "year": 2020}
    value = 0.0
    for variable in S2_INDEX_VARIABLES:
        for interval in range(4, 11):
            row[f"{variable}_{interval}"] = value
            value += 1.0
    path = tmp_path / "s2_indices.csv"
    pd.DataFrame([row]).to_csv(path, index=False)

    s2 = load_s2_index_features(path)

    assert s2.attrs["feature_names"][:7] == [f"evi_{i}" for i in range(4, 11)]
    assert len(s2.loc[0, "features"]) == 21
    np.testing.assert_allclose(s2.loc[0, "features"], np.arange(21))

    daymet = pd.DataFrame(
        {
            "county_id": ["17001"],
            "year": [2020],
            "daymet_features": [np.arange(35, dtype=np.float32)],
        }
    )
    fused = fuse_daymet_features(s2, daymet)
    assert len(fused.loc[0, "features"]) == 56

    contract = resolve_experiment_contract("sentinel2_indices", daymet=True)
    assert contract["family"] == AUXILIARY_CLIMATE_FUSION
    assert contract["fusion_stage"] == "county_year_late"


def test_merged_s2_daymet_table_excludes_incomplete_s2_rows_explicitly(
    tmp_path: Path,
) -> None:
    rows = []
    for county, missing in (("17001", False), ("17003", True)):
        row: dict[str, object] = {"county_id": county, "year": 2020}
        for variable in S2_INDEX_VARIABLES:
            for interval in range(4, 11):
                row[f"{variable}_{interval}"] = float(interval)
        if missing:
            row["evi_4"] = np.nan
        for variable in DAYMET_VARIABLES:
            for interval in range(4, 11):
                row[f"{variable}_{interval}"] = float(interval)
        rows.append(row)
    path = tmp_path / "s2_daymet_merged.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    s2 = load_s2_index_features(path)
    daymet = load_daymet_features(path)

    assert s2[["county_id", "year"]].to_dict("records") == [
        {"county_id": "17001", "year": 2020}
    ]
    assert s2.attrs["incomplete_rows_excluded"] == 1
    assert len(daymet) == 2
