from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from benchmark_embeddings.data.io import RAW_S2_12_BANDS, S2_10_BANDS
from benchmark_embeddings.frozen.terramind import (
    EXPERIMENT_S2_10_ZERO_PAD,
    EXPERIMENT_S2_6,
    SOURCE_BANDS_S2_6,
    TERRAMIND_BANDS_S2_6,
    TerraMindPatchDataset,
    extract_terramind_embeddings,
    pool_terramind_tokens,
    terramind_model_kwargs,
)


def _write_patch(
    root: Path,
    *,
    county: str,
    year: int,
    x: int,
    y: int,
    timestep: int,
    height: int = 10,
    width: int = 12,
    reverse_bands: bool = False,
    reflectance: bool = False,
) -> Path:
    grid = np.arange(height * width, dtype=np.float32).reshape(height, width)
    canonical = np.stack(
        [1000.0 * (index + 1) + grid for index in range(10)], axis=0
    )
    if reflectance:
        canonical = canonical / 10000.0
    names = np.asarray(S2_10_BANDS)
    pixels = canonical
    if reverse_bands:
        names = names[::-1]
        pixels = pixels[::-1]
    path = root / f"county_{county}_year_{year}_x{x}_y{y}_stack_COG_t{timestep}.npz"
    np.savez(
        path,
        pixels=pixels,
        band_names=names,
        metadata={
            "county_fips": county,
            "year": year,
            "band_names": names.tolist(),
        },
    )
    return path


def _write_complete_patch(
    root: Path,
    *,
    county: str,
    year: int,
    x: int,
    y: int,
    height: int = 8,
    width: int = 8,
) -> None:
    for timestep in range(1, 8):
        _write_patch(
            root,
            county=county,
            year=year,
            x=x,
            y=y,
            timestep=timestep,
            height=height,
            width=width,
        )


def test_six_band_contract_uses_prithvi_bands_and_two_stage_center_crop(
    tmp_path: Path,
) -> None:
    _write_patch(
        tmp_path,
        county="17001",
        year=2020,
        x=100,
        y=200,
        timestep=1,
        reverse_bands=True,
    )
    dataset = TerraMindPatchDataset(
        tmp_path,
        experiment=EXPERIMENT_S2_6,
        source_size=(10, 10),
        model_size=(8, 8),
        timestep_base=1,
        source_units="dn",
    )

    sample = dataset[0]
    assert sample["pixels"].shape == (6, 8, 8)
    assert sample["patch_id"] == "x100_y200"
    assert sample["timestep"] == 0

    # 10x12 -> central 10x10 removes one column on each side; 10x10 ->
    # central 8x8 removes one row/column on each side.
    grid = np.arange(120, dtype=np.float32).reshape(10, 12)[1:9, 2:10]
    source_indices = [S2_10_BANDS.index(band) for band in SOURCE_BANDS_S2_6]
    for output_index, source_index in enumerate(source_indices):
        expected = np.clip(
            (1000.0 * (source_index + 1) + grid) / 10000.0,
            0.0,
            1.0,
        )
        np.testing.assert_allclose(sample["pixels"][output_index].numpy(), expected)


def test_ten_band_contract_places_observed_bands_and_zero_pads_b01_b09(
    tmp_path: Path,
) -> None:
    _write_patch(
        tmp_path,
        county="17001",
        year=2020,
        x=100,
        y=200,
        timestep=1,
        height=8,
        width=8,
    )
    dataset = TerraMindPatchDataset(
        tmp_path,
        experiment=EXPERIMENT_S2_10_ZERO_PAD,
        source_size=8,
        model_size=8,
        timestep_base=1,
        source_units="dn",
    )

    pixels = dataset[0]["pixels"].numpy()
    assert pixels.shape == (12, 8, 8)
    np.testing.assert_array_equal(pixels[0], 0.0)  # B01
    np.testing.assert_array_equal(pixels[9], 0.0)  # B09
    for source_index, band in enumerate(S2_10_BANDS):
        target_index = RAW_S2_12_BANDS.index(band)
        expected = np.clip((1000.0 * (source_index + 1) + np.arange(64).reshape(8, 8)) / 10000.0, 0, 1)
        np.testing.assert_allclose(pixels[target_index], expected)


def test_official_six_band_interface_is_only_used_for_six_band_experiment() -> None:
    six = terramind_model_kwargs(EXPERIMENT_S2_6)
    padded = terramind_model_kwargs(EXPERIMENT_S2_10_ZERO_PAD)

    assert six["bands"]["S2L2A"] == list(TERRAMIND_BANDS_S2_6)
    assert "bands" not in padded
    assert six["merge_method"] == padded["merge_method"] == "mean"


def test_auto_units_are_detected_per_file(tmp_path: Path) -> None:
    _write_patch(
        tmp_path,
        county="17001",
        year=2020,
        x=100,
        y=200,
        timestep=1,
        height=8,
        width=8,
    )
    _write_patch(
        tmp_path,
        county="17001",
        year=2020,
        x=101,
        y=200,
        timestep=1,
        height=8,
        width=8,
        reflectance=True,
    )
    dataset = TerraMindPatchDataset(
        tmp_path,
        experiment=EXPERIMENT_S2_6,
        source_size=8,
        model_size=8,
        timestep_base=1,
        source_units="auto",
    )
    by_patch = {dataset[index]["patch_id"]: dataset[index]["pixels"] for index in range(2)}
    torch.testing.assert_close(by_patch["x100_y200"], by_patch["x101_y200"])
    assert dataset.describe()["source_units_policy"] == (
        "detect_each_file_then_convert_to_reflectance"
    )


def test_dataset_audits_then_excludes_interval_07(tmp_path: Path) -> None:
    for timestep in range(8):
        _write_patch(
            tmp_path,
            county="17001",
            year=2020,
            x=100,
            y=200,
            timestep=timestep,
            height=8,
            width=8,
        )
    dataset = TerraMindPatchDataset(
        tmp_path,
        experiment=EXPERIMENT_S2_6,
        source_size=8,
        model_size=8,
        expected_input_count=8,
    )
    description = dataset.describe()
    assert len(dataset) == 7
    assert [dataset[index]["timestep"] for index in range(7)] == list(range(7))
    assert description["source_cohort_files"] == 8
    assert description["out_of_schedule_files_excluded"] == 1
    assert description["out_of_schedule_timestep_policy"] == "audit_then_exclude"


def test_pooling_uses_final_layer_and_all_spatial_tokens() -> None:
    early = torch.full((2, 4, 3), 100.0)
    final = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)

    pooled = pool_terramind_tokens(
        [early, final],
        expected_token_count=4,
        expected_embedding_dim=3,
    )
    torch.testing.assert_close(pooled, final.mean(dim=1))
    assert not torch.equal(pooled, final[:, 0])

    with pytest.raises(ValueError, match="expected 196"):
        pool_terramind_tokens([final])


def test_dataset_rejects_undersized_source_and_guards_shared_cohort(
    tmp_path: Path,
) -> None:
    _write_patch(
        tmp_path,
        county="17001",
        year=2020,
        x=100,
        y=200,
        timestep=1,
        height=7,
        width=8,
    )
    with pytest.raises(ValueError, match="input cohort has 1 files, expected 77,813"):
        TerraMindPatchDataset(
            tmp_path,
            experiment=EXPERIMENT_S2_6,
            source_size=8,
            model_size=8,
            timestep_base=1,
            expected_input_count=77813,
        )

    dataset = TerraMindPatchDataset(
        tmp_path,
        experiment=EXPERIMENT_S2_6,
        source_size=8,
        model_size=8,
        timestep_base=1,
    )
    with pytest.raises(ValueError, match="below benchmark source-size expectation"):
        _ = dataset[0]


def test_extraction_preserves_variable_patch_counts_and_every_timestep(
    tmp_path: Path,
) -> None:
    _write_complete_patch(tmp_path, county="17001", year=2020, x=100, y=200)
    _write_complete_patch(tmp_path, county="17001", year=2020, x=101, y=200)
    _write_complete_patch(tmp_path, county="17003", year=2020, x=300, y=400)
    dataset = TerraMindPatchDataset(
        tmp_path,
        experiment=EXPERIMENT_S2_6,
        source_size=8,
        model_size=8,
        expected_input_count=21,
    )

    class FakeTerraMind:
        def __call__(self, inputs: dict[str, torch.Tensor]):
            pixels = inputs["S2L2A"]
            base = pixels.mean(dim=(1, 2, 3))
            tokens = torch.stack(
                [base[:, None] + offset for offset in (0.0, 1.0, 2.0)], dim=1
            )
            return [torch.zeros_like(tokens), tokens]

    frame = extract_terramind_embeddings(
        dataset,
        FakeTerraMind(),
        model_name="test_terramind",
        device="cpu",
        batch_size=5,
        expected_token_count=3,
        expected_embedding_dim=1,
    )

    assert len(frame) == 21
    assert frame["embedding"].map(len).unique().tolist() == [1]
    patch_counts = (
        frame[["county_id", "year", "patch_id"]]
        .drop_duplicates()
        .groupby(["county_id", "year"])
        .size()
        .to_dict()
    )
    assert patch_counts == {("17001", 2020): 2, ("17003", 2020): 1}
    assert frame.groupby(["county_id", "patch_id"])["timestep"].nunique().eq(7).all()
    assert frame["token_pool"].unique().tolist() == ["mean_all_spatial_tokens"]


def test_nonfinite_policy_is_explicit(tmp_path: Path) -> None:
    path = _write_patch(
        tmp_path,
        county="17001",
        year=2020,
        x=100,
        y=200,
        timestep=1,
        height=8,
        width=8,
    )
    with np.load(path, allow_pickle=True) as original:
        pixels = original["pixels"]
        names = original["band_names"]
        metadata = original["metadata"].item()
    pixels[:, 0, 0] = np.nan
    np.savez(path, pixels=pixels, band_names=names, metadata=metadata)

    strict = TerraMindPatchDataset(
        tmp_path,
        experiment=EXPERIMENT_S2_6,
        source_size=8,
        model_size=8,
        timestep_base=1,
    )
    with pytest.raises(ValueError, match="non-finite pixels remain"):
        _ = strict[0]

    zero = TerraMindPatchDataset(
        tmp_path,
        experiment=EXPERIMENT_S2_6,
        source_size=8,
        model_size=8,
        timestep_base=1,
        nonfinite_policy="zero",
    )
    sample = zero[0]
    assert torch.isfinite(sample["pixels"]).all()
    assert sample["valid_fraction"] == pytest.approx(63.0 / 64.0)
