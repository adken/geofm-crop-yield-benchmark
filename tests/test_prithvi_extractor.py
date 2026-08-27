from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmark_embeddings.data.io import S2_10_BANDS
from benchmark_embeddings.frozen.prithvi import (
    INFERENCE_TIMESTEPS,
    POOL_PER_TIMESTEP_SPATIAL_MEAN,
    PRITHVI_BACKBONE,
    PRITHVI_MODEL_BANDS,
    PRITHVI_SOURCE_BANDS,
    PRITHVI_V2_MEAN,
    PRITHVI_V2_STD,
    PrithviPatchDataset,
    expected_token_count,
    extract_prithvi_embeddings,
    load_prithvi,
    pool_prithvi_tokens,
)


DATES = (
    date(2020, 4, 15),
    date(2020, 5, 13),
    date(2020, 6, 10),
    date(2020, 7, 8),
    date(2020, 8, 5),
    date(2020, 9, 2),
    date(2020, 10, 1),
)


def _write_frame(
    root: Path,
    *,
    county: str,
    year: int,
    x: int,
    y: int,
    timestep: int,
    height: int = 10,
    width: int = 12,
    reverse_bands: bool = True,
    reflectance: bool = False,
    include_location: bool = True,
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
    acquisition = DATES[timestep - 1]
    path = root / (
        f"county_{county}_year_{year}_x{x}_y{y}_stack_COG_t{timestep}_"
        f"{acquisition.isoformat()}.npz"
    )
    metadata = {
        "county_fips": county,
        "year": year,
        "date": acquisition.isoformat(),
    }
    if include_location:
        metadata.update(latitude=40.0, longitude=-89.0)
    np.savez(path, pixels=pixels, band_names=names, metadata=metadata)
    return path


def _write_sequence(
    root: Path,
    *,
    county: str = "17001",
    year: int = 2020,
    x: int = 100,
    y: int = 200,
    timesteps: int = 7,
    height: int = 10,
    width: int = 12,
    reflectance: bool = False,
) -> None:
    for timestep in range(1, timesteps + 1):
        _write_frame(
            root,
            county=county,
            year=year,
            x=x,
            y=y,
            timestep=timestep,
            height=height,
            width=width,
            reflectance=reflectance,
        )


def test_six_band_mapping_two_stage_crop_and_official_normalization(
    tmp_path: Path,
) -> None:
    _write_sequence(tmp_path)
    dataset = PrithviPatchDataset(
        tmp_path,
        source_size=(10, 10),
        model_size=(8, 8),
        expected_input_count=7,
    )

    sample = dataset[0]
    assert sample["pixels"].shape == (6, 7, 8, 8)
    assert sample["patch_id"] == "x100_y200"
    assert sample["temporal_coords"].shape == (7, 2)
    assert sample["temporal_coords"][:, 0].tolist() == [2020.0] * 7
    assert sample["temporal_coords"][:, 1].tolist() == [
        float(value.timetuple().tm_yday) for value in DATES
    ]
    assert sample["location_coords"].tolist() == pytest.approx([40.0, -89.0])

    # 10x12 -> 10x10, then 8x8: final rows 1:9 and columns 2:10.
    grid = np.arange(120, dtype=np.float32).reshape(10, 12)[1:9, 2:10]
    for output_index, band in enumerate(PRITHVI_SOURCE_BANDS):
        source_index = S2_10_BANDS.index(band)
        raw = 1000.0 * (source_index + 1) + grid
        expected = (raw - PRITHVI_V2_MEAN[output_index]) / PRITHVI_V2_STD[output_index]
        np.testing.assert_allclose(sample["pixels"][output_index, 0].numpy(), expected)

    description = dataset.describe()
    assert description["zero_padded_bands"] == []
    assert description["model_band_names"] == list(PRITHVI_MODEL_BANDS)
    # The recorded policy must report the crop that actually ran. This fixture
    # uses source_size=10 and model_size=8, so a hardcoded "256_then_224" string
    # would be false provenance -- which is what this assertion used to accept.
    assert description["oversize_policy"] == (
        f"center_crop_to_{dataset.source_size[0]}"
        f"_then_center_crop_to_{dataset.model_size[0]}"
    )
    assert description["source_size"] == list(dataset.source_size)
    assert description["model_size"] == list(dataset.model_size)


def test_reflectance_is_converted_to_dn_before_official_statistics(tmp_path: Path) -> None:
    _write_sequence(tmp_path, reflectance=True)
    reflectance = PrithviPatchDataset(
        tmp_path,
        source_size=8,
        model_size=8,
        source_units="reflectance",
    )[0]["pixels"]

    dn_root = tmp_path / "dn"
    dn_root.mkdir()
    _write_sequence(dn_root)
    dn = PrithviPatchDataset(
        dn_root,
        source_size=8,
        model_size=8,
        source_units="dn",
    )[0]["pixels"]
    torch.testing.assert_close(reflectance, dn)


def test_auto_units_are_detected_per_file(tmp_path: Path) -> None:
    _write_sequence(tmp_path, x=100, height=8, width=8)
    _write_sequence(tmp_path, x=101, height=8, width=8, reflectance=True)
    dataset = PrithviPatchDataset(
        tmp_path,
        source_size=8,
        model_size=8,
        source_units="auto",
    )
    by_patch = {dataset[index]["patch_id"]: dataset[index]["pixels"] for index in range(2)}
    torch.testing.assert_close(by_patch["x100_y200"], by_patch["x101_y200"])
    assert dataset.describe()["source_units_policy"] == (
        "detect_each_file_then_convert_to_dn"
    )


def test_dataset_audits_then_excludes_interval_07(tmp_path: Path) -> None:
    for timestep in range(8):
        _write_frame(
            tmp_path,
            county="17001",
            year=2020,
            x=100,
            y=200,
            timestep=timestep,
            height=8,
            width=8,
        )
    dataset = PrithviPatchDataset(
        tmp_path,
        source_size=8,
        model_size=8,
        expected_input_count=8,
    )
    description = dataset.describe()
    assert len(dataset) == 1
    assert description["sentinel2_input_files"] == 8
    assert description["out_of_schedule_files_excluded"] == 1
    assert description["out_of_schedule_timestep_policy"] == "audit_then_exclude"


def test_dataset_guards_cohort_rejects_undersize_and_requires_location(
    tmp_path: Path,
) -> None:
    _write_sequence(tmp_path, height=7, width=8)
    with pytest.raises(ValueError, match="input cohort has 7 files, expected 77,813"):
        PrithviPatchDataset(
            tmp_path,
            source_size=8,
            model_size=8,
            expected_input_count=77813,
        )
    undersized = PrithviPatchDataset(tmp_path, source_size=8, model_size=8)
    with pytest.raises(ValueError, match="below benchmark source-size expectation"):
        _ = undersized[0]

    missing_root = tmp_path / "missing_location"
    missing_root.mkdir()
    for timestep in range(1, 8):
        _write_frame(
            missing_root,
            county="17001",
            year=2020,
            x=100,
            y=200,
            timestep=timestep,
            height=8,
            width=8,
            include_location=False,
        )
    missing = PrithviPatchDataset(missing_root, source_size=8, model_size=8)
    with pytest.raises(ValueError, match="requires latitude and longitude"):
        _ = missing[0]


def test_complete_sequences_preserve_variable_county_patch_counts(tmp_path: Path) -> None:
    _write_sequence(tmp_path, county="17001", x=100)
    _write_sequence(tmp_path, county="17001", x=101)
    _write_sequence(tmp_path, county="17003", x=200)
    _write_sequence(tmp_path, county="17003", x=201, timesteps=6)

    dataset = PrithviPatchDataset(
        tmp_path,
        source_size=8,
        model_size=8,
        expected_input_count=27,
    )
    description = dataset.describe()
    assert len(dataset) == 3
    assert description["incomplete_spatial_patches_excluded"] == 1
    assert description["patch_count_min"] == 1
    assert description["patch_count_max"] == 2
    assert description["pretraining_timesteps"] == 4
    assert description["expected_timesteps"] == 7


def test_token_count_and_pooling_use_one_frame_without_mixing_cls() -> None:
    assert expected_token_count() == 1 + 14 * 14
    # One CLS plus two spatial tokens for each independently encoded timestep.
    early = torch.full((2, 3, 3), 100.0)
    final = torch.arange(2 * 3 * 3, dtype=torch.float32).reshape(2, 3, 3)

    pooled = pool_prithvi_tokens(
        [early, final],
        pooling=POOL_PER_TIMESTEP_SPATIAL_MEAN,
        expected_tokens=3,
        expected_embedding_dim=3,
    )
    torch.testing.assert_close(pooled, final[:, 1:].mean(dim=1))
    assert not torch.equal(pooled, final.mean(dim=1))


def test_extraction_encodes_independent_frames_and_emits_seven_timestep_rows(
    tmp_path: Path,
) -> None:
    _write_sequence(tmp_path, county="17001", x=100, height=8, width=8)
    _write_sequence(tmp_path, county="17001", x=101, height=8, width=8)
    _write_sequence(tmp_path, county="17003", x=200, height=8, width=8)
    dataset = PrithviPatchDataset(tmp_path, source_size=8, model_size=8)

    class FakePrithvi:
        def __init__(self) -> None:
            self.calls: list[tuple[torch.Size, torch.Size, torch.Size]] = []

        def __call__(self, pixels, temporal_coords, location_coords):
            self.calls.append((pixels.shape, temporal_coords.shape, location_coords.shape))
            base = pixels.mean(dim=(1, 2, 3, 4)) + temporal_coords[:, 0, 1]
            tokens = torch.stack([base[:, None] + offset for offset in range(3)], dim=1)
            return [torch.zeros_like(tokens), tokens]

    model = FakePrithvi()
    frame = extract_prithvi_embeddings(
        dataset,
        model,
        device="cpu",
        batch_size=2,
        expected_tokens=3,
        expected_embedding_dim=1,
    )

    assert len(frame) == 21
    assert model.calls[0] == (
        torch.Size([14, 6, 1, 8, 8]),
        torch.Size([14, 1, 2]),
        torch.Size([14, 2]),
    )
    assert model.calls[1] == (
        torch.Size([7, 6, 1, 8, 8]),
        torch.Size([7, 1, 2]),
        torch.Size([7, 2]),
    )
    assert frame["representation_scope"].unique().tolist() == ["timestep"]
    assert frame["temporal_ingestion"].unique().tolist() == [
        "single_timestep_independent"
    ]
    assert frame["inference_timesteps"].unique().tolist() == [1]
    assert frame["timestep"].unique().tolist() == list(range(7))
    assert frame["token_pool"].unique().tolist() == [
        "mean_2_non_cls_spatial_tokens_per_timestep"
    ]
    assert frame["backbone"].unique().tolist() == [
        f"{PRITHVI_BACKBONE}_per_timestep_spatial_mean"
    ]
    first_patch = frame.loc[frame["patch_id"] == "x100_y200"].sort_values("timestep")
    assert first_patch["embedding"].map(lambda value: value[0]).nunique() == 7
    assert first_patch["source_file"].nunique() == 7
    patch_counts = frame.groupby(["county_id", "year"])["patch_id"].nunique().to_dict()
    assert patch_counts == {("17001", 2020): 2, ("17003", 2020): 1}


def test_loader_explicitly_requests_300m_tl_six_bands_one_frame_and_final_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    class FakeModel:
        embed_dim = 1024
        in_chans = 6
        num_frames = INFERENCE_TIMESTEPS
        blocks = [object()] * 24
        temporal_encoding = True
        location_encoding = True
        out_indices = [23]
        patch_embed = SimpleNamespace(patch_size=(1, 16, 16))

        def to(self, device):
            captured["device"] = device
            return self

        def eval(self):
            return self

    def fake_build(name, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs
        return FakeModel()

    import terratorch.registry

    monkeypatch.setattr(terratorch.registry.BACKBONE_REGISTRY, "build", fake_build)
    model = load_prithvi(device=torch.device("cpu"))
    assert isinstance(model, FakeModel)
    assert captured["name"] == PRITHVI_BACKBONE
    assert captured["kwargs"] == {
        "pretrained": True,
        "bands": list(PRITHVI_MODEL_BANDS),
        "num_frames": INFERENCE_TIMESTEPS,
        "out_indices": [23],
    }

    with pytest.raises(ValueError, match="requires num_frames=1"):
        load_prithvi(device=torch.device("cpu"), num_frames=7)


def test_nonfinite_fallback_is_explicit_and_fills_raw_dn_before_normalizing(
    tmp_path: Path,
) -> None:
    _write_sequence(tmp_path, height=8, width=8)
    path = sorted(tmp_path.glob("*.npz"))[0]
    with np.load(path, allow_pickle=True) as original:
        pixels = original["pixels"]
        names = original["band_names"]
        metadata = original["metadata"].item()
    pixels[:, 0, 0] = np.nan
    np.savez(path, pixels=pixels, band_names=names, metadata=metadata)

    strict = PrithviPatchDataset(tmp_path, source_size=8, model_size=8)
    with pytest.raises(ValueError, match="non-finite pixels remain"):
        _ = strict[0]

    fallback = PrithviPatchDataset(
        tmp_path,
        source_size=8,
        model_size=8,
        nonfinite_policy="zero",
    )[0]
    assert torch.isfinite(fallback["pixels"]).all()
    assert fallback["valid_fraction_min"] == pytest.approx(63.0 / 64.0)
    # Reversed source order is corrected first; every selected band had NaN at [0,0].
    expected = -torch.from_numpy(PRITHVI_V2_MEAN / PRITHVI_V2_STD)
    torch.testing.assert_close(fallback["pixels"][:, 0, 0, 0], expected)
