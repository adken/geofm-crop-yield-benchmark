from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmark_embeddings.frozen.presto import (
    PRESTO_ERA5_BANDS,
    PRESTO_S2_BANDS,
    PrestoPatchDataset,
    build_presto_batch,
    extract_presto_embeddings,
)
from benchmark_embeddings.frozen.schema import read_embeddings, write_embeddings


S2_NAMES = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
DATES = (
    date(2020, 4, 15),
    date(2020, 5, 13),
    date(2020, 6, 10),
    date(2020, 7, 8),
    date(2020, 8, 5),
    date(2020, 9, 2),
    date(2020, 10, 1),
)


def _write_s2(
    root: Path,
    *,
    county: str,
    x: int,
    y: int,
    timestep: int,
    height: int = 10,
    width: int = 12,
    reflectance: bool = False,
    reverse_bands: bool = False,
    zero_based: bool = False,
) -> Path:
    date_index = timestep if zero_based else timestep - 1
    acquisition = DATES[min(date_index, len(DATES) - 1)]
    canonical = np.stack(
        [
            np.full((height, width), 2000.0 + 100.0 * band, dtype=np.float32)
            + np.arange(height * width, dtype=np.float32).reshape(height, width)
            for band in range(10)
        ]
    )
    if reflectance:
        canonical = canonical / 10000.0
    names = np.asarray(S2_NAMES)
    pixels = canonical
    if reverse_bands:
        names = names[::-1]
        pixels = pixels[::-1]
    path = root / (
        f"county_{county}_year_2020_x{x}_y{y}_stack_COG_t{timestep}_"
        f"{acquisition.isoformat()}.npz"
    )
    np.savez(
        path,
        pixels=pixels,
        band_names=names,
        metadata={
            "county_fips": county,
            "year": 2020,
            "date": acquisition.isoformat(),
            "latitude": 40.0,
            "longitude": -89.0,
        },
    )
    return path


def _write_sequence(
    root: Path,
    *,
    county: str = "17001",
    x: int = 100,
    y: int = 200,
    timesteps: int = 7,
    height: int = 10,
    width: int = 12,
    reflectance: bool = False,
    zero_based: bool = False,
) -> None:
    timestep_values = range(timesteps) if zero_based else range(1, timesteps + 1)
    for timestep in timestep_values:
        _write_s2(
            root,
            county=county,
            x=x,
            y=y,
            timestep=timestep,
            height=height,
            width=width,
            reflectance=reflectance,
            reverse_bands=True,
            zero_based=zero_based,
        )


def _write_era5(
    root: Path,
    *,
    county: str = "17001",
    x: int = 100,
    y: int = 200,
    height: int = 10,
    width: int = 12,
    zero_based: bool = False,
    source_aliases: bool = False,
    include_interval_07: bool = False,
) -> None:
    start = 0 if zero_based else 1
    for timestep, acquisition in enumerate(DATES, start=start):
        # Genuine ERA5-Land source contract: [temperature K, precipitation metres].
        official_pixels = np.stack(
            [
                np.full((height, width), 276.0 + timestep - start, dtype=np.float32),
                np.full((height, width), 0.01 * (timestep - start + 1), dtype=np.float32),
            ]
        )
        pixels = official_pixels
        names = np.asarray(PRESTO_ERA5_BANDS)
        if source_aliases:
            pixels = official_pixels[::-1]
            names = np.asarray(("era5_precip_sum_m", "era5_temp_mean_K"))
        path = root / (
            f"county_{county}_year_2020_x{x}_y{y}_stack_COG_t{timestep}_"
            f"{acquisition.isoformat()}.npz"
        )
        np.savez(path, pixels=pixels, band_names=names)
    if include_interval_07:
        path = root / (
            f"county_{county}_year_2020_x{x}_y{y}_stack_COG_t7_"
            f"{DATES[-1].isoformat()}.npz"
        )
        np.savez(
            path,
            pixels=np.stack(
                [
                    np.full((height, width), 0.07, dtype=np.float32),
                    np.full((height, width), 282.0, dtype=np.float32),
                ]
            ),
            band_names=np.asarray(("era5_precip_sum_m", "era5_temp_mean_K")),
        )


class _FakeEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.months = None

    def forward(
        self,
        *,
        x: torch.Tensor,
        dynamic_world: torch.Tensor,
        mask: torch.Tensor,
        latlons: torch.Tensor,
        month: torch.Tensor,
        eval_task: bool,
    ) -> torch.Tensor:
        assert eval_task is True
        assert x.shape[1:] == (7, 17)
        assert dynamic_world.shape == (x.shape[0], 7)
        assert mask.shape == x.shape
        assert latlons.shape == (x.shape[0], 2)
        self.months = month.detach().cpu()
        base = x.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        return base + torch.arange(128, device=x.device, dtype=x.dtype).unsqueeze(0)


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _FakeEncoder()


class _OfficialHelperStub:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        s2 = kwargs["s2"]
        timesteps = int(s2.shape[0])
        x = torch.zeros(timesteps, 17, dtype=torch.float32)
        # This division represents the normalization owned by the official helper.
        x[:, :10] = s2 / 10000.0
        mask = torch.ones_like(x)
        mask[:, :10] = 0
        if "era5" in kwargs:
            x[:, 10:12] = kwargs["era5"]
            mask[:, 10:12] = 0
        dynamic_world = torch.full((timesteps,), 9, dtype=torch.long)
        return x, mask, dynamic_world


def test_dataset_center_crops_reorders_and_keeps_raw_dn_for_official_helper(
    tmp_path: Path,
) -> None:
    _write_sequence(tmp_path)
    dataset = PrestoPatchDataset(tmp_path, target_size=(8, 8))

    sample = dataset[0]

    # [K,T,C]: K is 1 under spatial_mode='mean' and the sampled pixel count
    # under 'sample', so both modes share one downstream shape contract.
    assert sample["s2_dn"].shape == (1, 7, 10)
    assert sample["patch_id"] == "x100_y200"
    assert sample["months"].tolist() == [3, 4, 5, 6, 7, 8, 9]
    assert sample["latlon"].tolist() == pytest.approx([40.0, -89.0])
    raw = 2000.0 + np.arange(120, dtype=np.float32).reshape(10, 12)[1:9, 2:10]
    assert sample["s2_dn"][0, 0, 0].item() == pytest.approx(float(raw.mean()))
    assert dataset.describe()["oversize_policy"] == "center_crop_before_spatial_mean"
    assert dataset.describe()["spatial_mode"] == "mean"


def test_reflectance_source_is_converted_back_to_dn_exactly_once(tmp_path: Path) -> None:
    _write_sequence(tmp_path, reflectance=True)
    dataset = PrestoPatchDataset(
        tmp_path,
        target_size=(8, 8),
        s2_units="reflectance",
    )
    sample = dataset[0]
    raw = 2000.0 + np.arange(120, dtype=np.float32).reshape(10, 12)[1:9, 2:10]
    assert sample["s2_dn"][0, 0, 0].item() == pytest.approx(float(raw.mean()), rel=1e-6)


def test_dataset_rejects_undersized_patches_and_wrong_cohort_count(tmp_path: Path) -> None:
    _write_sequence(tmp_path, height=7, width=8)
    with pytest.raises(ValueError, match="input cohort has 7 files, expected 8"):
        PrestoPatchDataset(tmp_path, target_size=8, expected_input_count=8)

    dataset = PrestoPatchDataset(tmp_path, target_size=8)
    with pytest.raises(ValueError, match="below benchmark expectation 8x8"):
        _ = dataset[0]


def test_complete_sequences_preserve_variable_county_patch_counts(tmp_path: Path) -> None:
    _write_sequence(tmp_path, county="17001", x=100)
    _write_sequence(tmp_path, county="17001", x=101)
    _write_sequence(tmp_path, county="17003", x=200)
    _write_sequence(tmp_path, county="17003", x=201, timesteps=6)

    dataset = PrestoPatchDataset(tmp_path, target_size=8, expected_input_count=27)
    description = dataset.describe()

    assert len(dataset) == 3
    assert description["incomplete_spatial_patches_excluded"] == 1
    assert description["patch_count_min"] == 1
    assert description["patch_count_max"] == 2


def test_era5_is_passed_through_the_official_era5_slot(
    tmp_path: Path,
) -> None:
    s2_root = tmp_path / "s2"
    era5_root = tmp_path / "era5"
    s2_root.mkdir()
    era5_root.mkdir()
    _write_sequence(s2_root)
    _write_era5(era5_root)
    dataset = PrestoPatchDataset(
        s2_root,
        era5_dir=era5_root,
        target_size=8,
    )
    sample = dataset[0]

    assert dataset.backbone == "presto_s2_era5"
    assert dataset.describe()["era5_model_bands"] == list(PRESTO_ERA5_BANDS)
    assert sample["era5"][0].tolist() == pytest.approx([276.0, 0.01])
    helper = _OfficialHelperStub()
    tensors = {
        key: sample[key].unsqueeze(0)
        for key in ("s2_dn", "era5", "months", "latlon")
    }
    build_presto_batch(tensors, helper, include_era5=True)

    assert helper.calls[0]["s2_bands"] == list(PRESTO_S2_BANDS)
    assert "NDVI" not in helper.calls[0]["s2_bands"]
    assert helper.calls[0]["era5_bands"] == list(PRESTO_ERA5_BANDS)
    torch.testing.assert_close(helper.calls[0]["era5"][0], torch.tensor([276.0, 0.01]))

    frame = extract_presto_embeddings(
        dataset,
        _FakeModel(),
        _OfficialHelperStub(),
        device="cpu",
        batch_size=1,
    )
    assert frame.loc[0, "backbone"] == "presto_s2_era5"
    assert frame.loc[0, "experiment_family"] == "auxiliary_climate_fusion"
    assert frame.loc[0, "fusion_stage"] == "presto_encoder_input"
    assert frame.loc[0, "input_modalities"] == "Sentinel-2,ERA5-Land"


def test_zero_based_interval_07_is_audited_then_excluded_and_era5_metadata_reorders(
    tmp_path: Path,
) -> None:
    s2_root = tmp_path / "s2"
    era5_root = tmp_path / "era5"
    s2_root.mkdir()
    era5_root.mkdir()
    _write_sequence(s2_root, zero_based=True)
    _write_s2(
        s2_root,
        county="17001",
        x=100,
        y=200,
        timestep=7,
        zero_based=True,
        reverse_bands=True,
    )
    _write_era5(
        era5_root,
        zero_based=True,
        source_aliases=True,
        include_interval_07=True,
    )

    dataset = PrestoPatchDataset(
        s2_root,
        era5_dir=era5_root,
        target_size=8,
        expected_input_count=8,
        expected_era5_input_count=8,
    )
    description = dataset.describe()
    sample = dataset[0]

    assert len(dataset) == 1
    assert description["normalized_timestep_schedule"] == list(range(7))
    assert description["sentinel2_out_of_schedule_files_excluded"] == 1
    assert description["era5_out_of_schedule_files_excluded"] == 1
    assert description["sentinel2_schedule_files"] == 7
    assert description["era5_schedule_files"] == 7
    assert sample["era5"][0].tolist() == pytest.approx([276.0, 0.01])


def test_climate_grid_must_match_sentinel2_before_the_shared_center_crop(
    tmp_path: Path,
) -> None:
    s2_root = tmp_path / "s2"
    era5_root = tmp_path / "era5"
    s2_root.mkdir()
    era5_root.mkdir()
    _write_sequence(s2_root, height=10, width=12)
    _write_era5(era5_root, height=9, width=12)
    dataset = PrestoPatchDataset(
        s2_root,
        era5_dir=era5_root,
        target_size=8,
    )

    with pytest.raises(ValueError, match="geospatial resampling must happen upstream"):
        _ = dataset[0]


def test_extraction_emits_one_sequence_row_and_exact_zero_based_months(tmp_path: Path) -> None:
    _write_sequence(tmp_path)
    dataset = PrestoPatchDataset(tmp_path, target_size=8)
    model = _FakeModel()
    helper = _OfficialHelperStub()

    frame = extract_presto_embeddings(
        dataset,
        model,
        helper,
        device="cpu",
        batch_size=1,
    )

    assert len(frame) == 1
    assert frame.loc[0, "timestep"] == 0
    assert frame.loc[0, "representation_scope"] == "sequence"
    assert frame.loc[0, "sequence_timesteps"] == 7
    assert frame.loc[0, "input_modalities"] == "Sentinel-2"
    assert frame.loc[0, "experiment_family"] == "main_benchmark"
    assert frame.loc[0, "fusion_stage"] == "none"
    assert len(frame.loc[0, "embedding"]) == 128
    assert model.encoder.months.tolist() == [[3, 4, 5, 6, 7, 8, 9]]
    assert helper.calls[0]["s2_bands"] == list(PRESTO_S2_BANDS)
    # The extractor has not divided the raw DN before the helper receives it.
    assert float(helper.calls[0]["s2"].max()) > 1000.0

    output = write_embeddings(frame, tmp_path / "presto.parquet")
    restored = read_embeddings(output)
    assert restored.loc[0, "representation_scope"] == "sequence"
    assert restored.loc[0, "months_zero_based"].tolist() == [3, 4, 5, 6, 7, 8, 9]
