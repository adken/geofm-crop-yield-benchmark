from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from benchmark_embeddings.data import (
    BandNormalizer,
    CountyPatchRecord,
    CountyPatchStore,
    TargetScaler,
)
from benchmark_embeddings.models import (
    SupervisedS23DConvLSTM,
    aggregate_patch_representations,
    build_supervised_model,
)
from benchmark_embeddings.train import (
    _partition_records,
    assert_disjoint_partitions,
    predict_county_record,
)


BANDS = np.array(
    ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
)


def _write_county(
    root: Path,
    county: str,
    year: int,
    patch_count: int,
    height: int,
    width: int,
) -> Path:
    rng = np.random.default_rng(int(county) + year)
    patches = rng.uniform(
        0.01, 0.9, size=(patch_count, 7, 10, height, width)
    ).astype(np.float32)
    path = root / f"county_{county}_year_{year}.npz"
    np.savez(
        path,
        patches=patches,
        county_fips=county,
        year=year,
        crop="corn",
        yield_bu_per_acre=np.float32(140.0 + year % 3),
        band_names=BANDS,
    )
    return path


def _small_model() -> SupervisedS23DConvLSTM:
    return SupervisedS23DConvLSTM(
        input_channels=10,
        stem_channels=(4,),
        stem_spatial_strides=(1,),
        stem_norm_groups=2,
        convlstm_hidden_channels=(4,),
        head_hidden=4,
        dropout=0.0,
    )


def _small_registered_model(name: str):
    common = {
        "input_channels": 10,
        "stem_channels": (4,),
        "stem_spatial_strides": (1,),
        "stem_norm_groups": 2,
        "head_hidden": 4,
        "dropout": 0.0,
    }
    if name == "3d_convlstm":
        common["convlstm_hidden_channels"] = (4,)
    else:
        common["recurrent_hidden"] = 4
    return build_supervised_model(name, **common)


def test_dataset_center_crops_oversized_inputs_and_keeps_variable_patch_counts(
    tmp_path: Path,
) -> None:
    _write_county(tmp_path, "17001", 2020, 2, 8, 9)
    oversized_path = _write_county(tmp_path, "17003", 2021, 5, 10, 11)

    store = CountyPatchStore(
        tmp_path,
        expected_timesteps=7,
        expected_spatial_size=(8, 9),
        oversize_policy="center_crop",
    )
    first, second = store.records

    assert store.load_patches(first).shape == (2, 7, 10, 8, 9)
    cropped = store.load_patches(second, [1, 3]).numpy()
    assert cropped.shape == (2, 7, 10, 8, 9)
    with np.load(oversized_path) as z:
        expected = z["patches"][[1, 3], ..., 1:9, 1:10]
    np.testing.assert_allclose(cropped, expected)

    description = store.describe()
    assert description["patch_count_min"] == 2
    assert description["patch_count_max"] == 5
    assert description["expected_spatial_size"] == [8, 9]
    assert description["input_contract"] == "[num_patches,7,10,8,9]"
    assert description["indexed_county_years_requiring_crop"] == 1
    assert description["indexed_source_per_patch_shapes"] == [
        [7, 10, 8, 9],
        [7, 10, 10, 11],
    ]
    assert description["observed_per_patch_shapes"] == [[7, 10, 8, 9]]


def test_direct_yield_csv_is_authoritative_and_unconverted(tmp_path: Path) -> None:
    _write_county(tmp_path, "17001", 2020, 1, 8, 8)
    labels_path = tmp_path / "county_yield.csv"
    pd.DataFrame(
        {"county": [17001], "year": [2020], "yield": [187.4]}
    ).to_csv(labels_path, index=False)

    store = CountyPatchStore(
        tmp_path,
        yield_csv=labels_path,
        expected_timesteps=7,
        expected_spatial_size=8,
    )

    assert store.records[0].target_bu_per_acre == pytest.approx(187.4)
    assert store.describe()["yield_labels"] == str(labels_path.resolve())
    assert store.describe()["target_units"] == "bushels_per_acre"


def test_default_256_contract_rejects_undersized_patches(tmp_path: Path) -> None:
    _write_county(tmp_path, "17001", 2020, 1, 15, 16)

    with pytest.raises(ValueError, match=r"below required 256x256"):
        CountyPatchStore(tmp_path, expected_timesteps=7)


def test_exact_spatial_policy_can_reject_oversized_patches(tmp_path: Path) -> None:
    _write_county(tmp_path, "17001", 2020, 1, 9, 8)

    with pytest.raises(ValueError, match=r"expected exactly 8x8"):
        CountyPatchStore(
            tmp_path,
            expected_timesteps=7,
            expected_spatial_size=8,
            oversize_policy="error",
        )


def test_per_patch_layout_crops_each_interval_before_stacking(tmp_path: Path) -> None:
    raw_interval_zero = None
    for interval in range(7):
        height, width = ((10, 11) if interval % 2 == 0 else (12, 13))
        frame = np.arange(10 * height * width, dtype=np.float32).reshape(
            10, height, width
        )
        frame /= float(frame.max())
        if interval == 0:
            raw_interval_zero = frame.copy()
        metadata = {
            "county_fips": "17001",
            "year": 2020,
            "interval_index": interval,
            "patch_id": "A",
            "crop": "corn",
            "yield_bu_per_acre": 150.0,
        }
        np.savez(
            tmp_path
            / f"county_17001_year_2020_patch_A_interval_{interval}.npz",
            pixels=frame,
            metadata=metadata,
            band_names=BANDS,
        )

    store = CountyPatchStore(
        tmp_path,
        expected_timesteps=7,
        expected_spatial_size=(8, 9),
        oversize_policy="center_crop",
    )
    loaded = store.load_patches(store.records[0]).numpy()

    assert loaded.shape == (1, 7, 10, 8, 9)
    assert raw_interval_zero is not None
    np.testing.assert_allclose(loaded[0, 0], raw_interval_zero[:, 1:9, 1:10])


@pytest.mark.parametrize("name", ["3d_convlstm", "gru", "lstm"])
def test_model_forward_shapes_with_grouped_counties(name: str) -> None:
    model = _small_registered_model(name)
    patches = torch.randn(5, 7, 10, 8, 9)
    groups = torch.tensor([0, 0, 1, 1, 1])

    predictions = model(patches, groups)
    representations = model.encode_patches(patches)

    assert representations.shape == (5, 4)
    assert predictions.shape == (2,)


def test_county_aggregation_is_invariant_to_patch_order() -> None:
    representations = torch.randn(7, 5)
    groups = torch.tensor([0, 0, 0, 1, 1, 2, 2])
    permutation = torch.tensor([5, 1, 4, 0, 6, 3, 2])

    expected = aggregate_patch_representations(representations, groups)
    permuted = aggregate_patch_representations(
        representations[permutation], groups[permutation]
    )

    torch.testing.assert_close(permuted, expected)


@pytest.mark.parametrize("name", ["3d_convlstm", "gru", "lstm"])
def test_one_optimizer_step_has_finite_loss_and_gradients(name: str) -> None:
    model = _small_registered_model(name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    patches = torch.randn(4, 7, 10, 8, 8)
    groups = torch.tensor([0, 0, 1, 1])
    target = torch.tensor([0.2, -0.4])

    loss = torch.nn.functional.mse_loss(model(patches, groups), target)
    loss.backward()

    assert torch.isfinite(loss)
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    optimizer.step()


def test_split_overlap_is_rejected() -> None:
    def record(key: str) -> CountyPatchRecord:
        county, year = key.split("-")
        return CountyPatchRecord(
            key=key,
            county_id=county,
            year=int(year),
            target_bu_per_acre=150.0,
            patch_ids=("p0",),
            shape=(1, 7, 10, 8, 8),
            layout="bundled",
            source=Path("unused.npz"),
            intervals=tuple(range(7)),
            input_bands=tuple(BANDS.tolist()),
        )

    shared = record("17001-2020")
    try:
        assert_disjoint_partitions([shared], [shared], [record("17003-2020")])
    except ValueError as exc:
        assert "leakage" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("overlap was not rejected")


def test_supervised_trainer_rejects_loyo_mode() -> None:
    with pytest.raises(ValueError, match="fixed Random Forest"):
        _partition_records(object(), {"mode": "loyo"})


def test_validation_prediction_uses_all_patches_deterministically(tmp_path: Path) -> None:
    _write_county(tmp_path, "17001", 2020, 5, 8, 8)
    store = CountyPatchStore(
        tmp_path,
        expected_timesteps=7,
        expected_spatial_size=8,
    )
    record = store.records[0]
    model = _small_model().eval()
    normalizer = BandNormalizer(mean=(0.0,) * 10, std=(1.0,) * 10)
    target_scaler = TargetScaler(mean=10.0, std=2.0)

    first = predict_county_record(
        model,
        store,
        record,
        normalizer,
        target_scaler,
        device=torch.device("cpu"),
        patch_chunk_size=2,
    )
    second = predict_county_record(
        model,
        store,
        record,
        normalizer,
        target_scaler,
        device=torch.device("cpu"),
        patch_chunk_size=2,
    )

    assert first == second


def test_target_inverse_transform_precedes_reported_values() -> None:
    scaler = TargetScaler(mean=8.0, std=2.5)
    standardized = np.array([-1.0, 0.0, 1.0])

    restored = scaler.inverse_array(standardized)

    np.testing.assert_allclose(restored, [5.5, 8.0, 10.5])
