from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmark_embeddings.daymet import (
    DAYMET_VARIABLES,
    fuse_daymet_features,
    load_daymet_features,
)


def _daymet_row(*, county: str = "Adams", statefp: int = 17, year: int = 2020) -> dict:
    row: dict[str, object] = {"county": county, "statefp": statefp, "year": year}
    value = 0.0
    for variable in DAYMET_VARIABLES:
        for interval in range(4, 11):
            row[f"{variable}_{interval}"] = value
            value += 1.0
    return row


def test_daymet_loader_maps_names_to_fips_and_orders_35_features(tmp_path: Path) -> None:
    source = tmp_path / "daymet.csv"
    mapping = tmp_path / "fips.csv"
    pd.DataFrame([_daymet_row()]).to_csv(source, index=False)
    pd.DataFrame(
        [{"county": "Adams", "statefp": 17, "County ANSI": 1}]
    ).to_csv(mapping, index=False)

    daymet = load_daymet_features(source, fips_map=mapping)

    assert daymet.loc[0, "county_id"] == "17001"
    assert daymet.attrs["interval_schedule"] == list(range(4, 11))
    assert daymet.attrs["feature_names"][:7] == [f"dayl_{i}" for i in range(4, 11)]
    np.testing.assert_allclose(daymet.loc[0, "daymet_features"], np.arange(35))


def test_daymet_is_concatenated_only_after_county_pooling(tmp_path: Path) -> None:
    source = tmp_path / "daymet.csv"
    row = _daymet_row()
    row["county_id"] = "17001"
    pd.DataFrame([row]).to_csv(source, index=False)
    daymet = load_daymet_features(source)
    county = pd.DataFrame(
        {
            "county_id": ["17001"],
            "year": [2020],
            "features": [np.asarray([10.0, 20.0], dtype=np.float32)],
        }
    )

    fused = fuse_daymet_features(county, daymet)

    assert len(fused.loc[0, "features"]) == 37
    np.testing.assert_allclose(fused.loc[0, "features"][:2], [10.0, 20.0])
    np.testing.assert_allclose(fused.loc[0, "features"][2:], np.arange(35))


def test_daymet_rejects_an_incomplete_interval_schedule(tmp_path: Path) -> None:
    source = tmp_path / "daymet.csv"
    row = _daymet_row()
    row["county_id"] = "17001"
    del row["tmin_10"]
    pd.DataFrame([row]).to_csv(source, index=False)

    with pytest.raises(ValueError, match="tmin has 6 intervals"):
        load_daymet_features(source)
