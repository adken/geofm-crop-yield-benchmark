from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from benchmark_embeddings.frozen.clay import (
    ClayPatchDataset,
    ClaySensorMetadata,
    encode_clay_location,
    encode_clay_time,
    extract_clay_embeddings,
    pool_clay_tokens,
)


CLAY_NAMES = (
    "blue",
    "green",
    "red",
    "rededge1",
    "rededge2",
    "rededge3",
    "nir",
    "nir08",
    "swir16",
    "swir22",
)
S2_NAMES = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")


def _write_metadata(path: Path) -> Path:
    document = {
        "sentinel-2-l2a": {
            "band_order": list(CLAY_NAMES),
            "gsd": 10,
            "bands": {
                "mean": {name: 1000.0 + 100.0 * index for index, name in enumerate(CLAY_NAMES)},
                "std": {name: 500.0 + 10.0 * index for index, name in enumerate(CLAY_NAMES)},
                "wavelength": {
                    name: value
                    for name, value in zip(
                        CLAY_NAMES,
                        (0.493, 0.560, 0.665, 0.704, 0.740, 0.783, 0.842, 0.865, 1.610, 2.190),
                    )
                },
            },
        }
    }
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    return path


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
) -> Path:
    # Every canonical band has a distinct DN value, making reorder/crop and
    # official normalization independently observable.
    canonical = np.stack(
        [
            np.full((height, width), 2000.0 + 100.0 * index, dtype=np.float32)
            + np.arange(height * width, dtype=np.float32).reshape(height, width)
            for index in range(10)
        ]
    )
    names = np.asarray(S2_NAMES)
    pixels = canonical
    if reverse_bands:
        names = names[::-1]
        pixels = canonical[::-1]
    path = root / (
        f"county_{county}_year_{year}_x{x}_y{y}_stack_COG_t{timestep}.npz"
    )
    np.savez(
        path,
        pixels=pixels,
        band_names=names,
        metadata={
            "county_fips": county,
            "year": year,
            "date": "2020-04-15",
            "latitude": 40.0,
            "longitude": -89.0,
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


def test_official_metadata_is_aligned_and_converted_to_reflectance(tmp_path: Path) -> None:
    sensor = ClaySensorMetadata.from_yaml(_write_metadata(tmp_path / "metadata.yaml"))

    assert sensor.bands == S2_NAMES
    assert sensor.mean[0] == pytest.approx(0.1)
    assert sensor.std[0] == pytest.approx(0.05)
    assert sensor.wavelengths_um[0] == pytest.approx(0.493)
    assert sensor.wavelengths_um[-1] == pytest.approx(2.190)


def test_dataset_reorders_normalizes_and_center_crops_without_erasing_patch_id(
    tmp_path: Path,
) -> None:
    metadata_path = _write_metadata(tmp_path / "metadata.yaml")
    first_path = _write_patch(
        tmp_path,
        county="17001",
        year=2020,
        x=100,
        y=200,
        timestep=1,
        reverse_bands=True,
    )
    _write_patch(
        tmp_path,
        county="17001",
        year=2020,
        x=100,
        y=200,
        timestep=7,
    )
    dataset = ClayPatchDataset(
        tmp_path,
        metadata_path=metadata_path,
        target_size=(8, 8),
        expected_timesteps=7,
    )

    sample = dataset[0]
    assert sample["pixels"].shape == (10, 8, 8)
    assert sample["patch_id"] == "x100_y200"
    assert sample["source_id"] == first_path.stem
    assert dataset[1]["patch_id"] == sample["patch_id"]
    assert [dataset[index]["timestep"] for index in range(2)] == [0, 6]

    # Canonical B02 is 2000 DN plus the central rows/columns. Clay's first-band
    # mean/std are 1000/500 DN, equivalently 0.1/0.05 reflectance.
    raw = 2000.0 + np.arange(120, dtype=np.float32).reshape(10, 12)[1:9, 2:10]
    expected = ((raw / 10000.0) - 0.1) / 0.05
    np.testing.assert_allclose(sample["pixels"][0].numpy(), expected, rtol=1e-5)


def test_dataset_rejects_undersized_and_duplicate_spatial_keys(tmp_path: Path) -> None:
    metadata_path = _write_metadata(tmp_path / "metadata.yaml")
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
    _write_patch(
        tmp_path,
        county="17001",
        year=2020,
        x=100,
        y=200,
        timestep=7,
        height=7,
        width=8,
    )
    dataset = ClayPatchDataset(
        tmp_path,
        metadata_path=metadata_path,
        target_size=8,
        expected_timesteps=7,
    )
    with pytest.raises(ValueError, match="below Clay benchmark expectation"):
        _ = dataset[0]


def test_dataset_audits_then_excludes_interval_07(tmp_path: Path) -> None:
    metadata_path = _write_metadata(tmp_path / "metadata.yaml")
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
    dataset = ClayPatchDataset(
        tmp_path,
        metadata_path=metadata_path,
        target_size=8,
        expected_input_count=8,
    )

    description = dataset.describe()
    assert len(dataset) == 7
    assert [dataset[index]["timestep"] for index in range(7)] == list(range(7))
    assert description["source_cohort_files"] == 8
    assert description["out_of_schedule_files_excluded"] == 1
    assert description["out_of_schedule_timestep_policy"] == "audit_then_exclude"


def test_official_cyclic_time_and_location_encodings() -> None:
    time, day, timestamp = encode_clay_time("2020-04-15T06:00:00")
    week_angle = 16 * 2.0 * math.pi / 52.0
    expected = torch.tensor(
        [math.sin(week_angle), math.cos(week_angle), 1.0, 0.0], dtype=torch.float32
    )
    torch.testing.assert_close(time, expected)
    assert day == 106
    assert timestamp.startswith("2020-04-15T06:00:00")

    location = encode_clay_location(30.0, -90.0)
    torch.testing.assert_close(
        location,
        torch.tensor([0.5, math.sqrt(3) / 2, -1.0, 0.0], dtype=torch.float32),
        atol=1e-6,
        rtol=1e-6,
    )


def test_token_pooling_is_explicit_and_shape_checked() -> None:
    tokens = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    torch.testing.assert_close(
        pool_clay_tokens(
            tokens,
            "cls",
            expected_token_count=5,
            expected_embedding_dim=3,
        ),
        tokens[:, 0],
    )
    torch.testing.assert_close(
        pool_clay_tokens(
            tokens,
            "spatial_mean",
            expected_token_count=5,
            expected_embedding_dim=3,
        ),
        tokens[:, 1:].mean(dim=1),
    )
    with pytest.raises(ValueError, match="expected 1025"):
        pool_clay_tokens(tokens, "cls")


def test_clay_input_does_not_clip_valid_reflectance_above_one(tmp_path: Path) -> None:
    metadata_path = _write_metadata(tmp_path / "metadata.yaml")
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
    _write_patch(
        tmp_path,
        county="17001",
        year=2020,
        x=100,
        y=200,
        timestep=7,
        height=8,
        width=8,
    )
    with np.load(path, allow_pickle=True) as original:
        metadata = original["metadata"].item()
        names = original["band_names"]
        pixels = original["pixels"]
    pixels[0, 0, 0] = 12000.0
    np.savez(path, pixels=pixels, band_names=names, metadata=metadata)
    dataset = ClayPatchDataset(
        tmp_path,
        metadata_path=metadata_path,
        target_size=8,
    )

    # (1.2 reflectance - 0.1 Clay mean) / 0.05 Clay std = 22.
    assert dataset[0]["pixels"][0, 0, 0].item() == pytest.approx(22.0)


def test_nonfinite_pixels_fail_by_default_and_zero_fill_is_explicit(tmp_path: Path) -> None:
    metadata_path = _write_metadata(tmp_path / "metadata.yaml")
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
    _write_patch(
        tmp_path,
        county="17001",
        year=2020,
        x=100,
        y=200,
        timestep=7,
        height=8,
        width=8,
    )
    with np.load(path, allow_pickle=True) as original:
        metadata = original["metadata"].item()
        names = original["band_names"]
        pixels = original["pixels"]
    pixels[:, 0, 0] = np.nan
    np.savez(path, pixels=pixels, band_names=names, metadata=metadata)

    strict = ClayPatchDataset(tmp_path, metadata_path=metadata_path, target_size=8)
    with pytest.raises(ValueError, match="non-finite pixels remain"):
        _ = strict[0]

    zero_fill = ClayPatchDataset(
        tmp_path,
        metadata_path=metadata_path,
        target_size=8,
        nonfinite_policy="zero",
    )
    sample = zero_fill[0]
    assert torch.isfinite(sample["pixels"]).all()
    assert sample["valid_fraction"] == pytest.approx(63.0 / 64.0)


def test_extraction_preserves_variable_patch_counts_and_every_source_row(
    tmp_path: Path,
) -> None:
    metadata_path = _write_metadata(tmp_path / "metadata.yaml")
    _write_complete_patch(tmp_path, county="17001", year=2020, x=100, y=200)
    _write_complete_patch(tmp_path, county="17001", year=2020, x=101, y=200)
    _write_complete_patch(tmp_path, county="17003", year=2020, x=300, y=400)
    dataset = ClayPatchDataset(
        tmp_path,
        metadata_path=metadata_path,
        target_size=8,
        expected_timesteps=7,
        expected_input_count=21,
    )

    def fake_encoder(cube: dict[str, torch.Tensor]):
        batch = cube["pixels"].shape[0]
        tokens = torch.zeros(batch, 2, 3, device=cube["pixels"].device)
        tokens[:, 0, 0] = cube["pixels"].mean(dim=(1, 2, 3))
        tokens[:, 0, 1] = cube["time"][:, 0]
        tokens[:, 0, 2] = cube["latlon"][:, 0]
        return tokens, None

    frame = extract_clay_embeddings(
        dataset,
        fake_encoder,
        device="cpu",
        pooling="cls",
        batch_size=5,
        expected_token_count=2,
        expected_embedding_dim=3,
    )

    assert len(frame) == 21
    assert frame["source_id"].nunique() == 21
    counts = (
        frame[["county_id", "year", "patch_id"]]
        .drop_duplicates()
        .groupby("county_id")
        .size()
        .to_dict()
    )
    assert counts == {"17001": 2, "17003": 1}
    assert frame.groupby(["county_id", "patch_id"])["timestep"].nunique().eq(7).all()


def test_expected_77813_style_cohort_guard_fails_on_mismatch(tmp_path: Path) -> None:
    metadata_path = _write_metadata(tmp_path / "metadata.yaml")
    _write_complete_patch(tmp_path, county="17001", year=2020, x=100, y=200)

    with pytest.raises(ValueError, match="expected 77,813"):
        ClayPatchDataset(
            tmp_path,
            metadata_path=metadata_path,
            target_size=8,
            expected_input_count=77813,
        )
