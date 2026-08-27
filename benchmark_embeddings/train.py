#!/usr/bin/env python3
"""Train county-supervised Sentinel-2 ConvLSTM, GRU, or LSTM models."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from .data import (
    BandNormalizer,
    CountyPatchRecord,
    CountyPatchStore,
    load_fold_partitions,
    TargetScaler,
    deterministic_patch_sample,
    fit_band_normalizer,
    validate_all_years_in_partitions,
)
from .models import build_supervised_model


def set_deterministic_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as stream:
        cfg = yaml.safe_load(stream) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return _expand_environment(cfg)


def _partition_records(
    store: CountyPatchStore,
    split_cfg: dict[str, Any],
) -> tuple[list[CountyPatchRecord], list[CountyPatchRecord], list[CountyPatchRecord], dict[str, Any]]:
    mode = str(split_cfg.get("mode", "primary")).lower()
    if mode != "primary":
        raise ValueError(
            "supervised models use only the primary county-grouped folds; "
            "LOYO is the separate fixed Random Forest embedding experiment"
        )
    split_path = split_cfg.get("path")
    if not split_path:
        raise KeyError("primary split mode requires split.path")
    parts = load_fold_partitions(
        split_path,
        fold=int(split_cfg.get("fold", 0)),
        id_column="fips_year",
        validation_fold=split_cfg.get("validation_fold"),
        validation_fold_offset=int(split_cfg.get("validation_fold_offset", 1)),
    )
    train = store.subset(parts.train)
    val = store.subset(parts.val)
    test = store.subset(parts.test)
    cohort_years = sorted({int(record.year) for record in store.records})
    require_all_years = bool(
        split_cfg.get("require_all_years_in_each_partition", True)
    )
    partition_years = {
        "train": sorted({int(record.year) for record in train}),
        "validation": sorted({int(record.year) for record in val}),
        "test": sorted({int(record.year) for record in test}),
    }
    if require_all_years:
        partition_years = validate_all_years_in_partitions(
            parts, expected_years=cohort_years
        )
    metadata = {
        "mode": mode,
        "path": str(split_path),
        "fold": parts.outer_fold,
        "validation_fold": parts.validation_fold,
        "grouping": "county_all_years_together",
        "cohort_years": cohort_years,
        "year_policy": (
            "all_cohort_years_in_each_train_validation_test_partition"
            if require_all_years
            else "not_enforced_synthetic_or_diagnostic_run"
        ),
        "partition_years": partition_years,
        "label": f"fold_{parts.outer_fold}",
    }
    assert_disjoint_partitions(train, val, test)
    if not train or not val or not test:
        raise ValueError(
            f"empty partition: train={len(train)}, val={len(val)}, test={len(test)}"
        )
    return train, val, test, metadata


def assert_disjoint_partitions(
    train: Sequence[CountyPatchRecord],
    val: Sequence[CountyPatchRecord],
    test: Sequence[CountyPatchRecord],
) -> None:
    sets = {
        "train": {record.key for record in train},
        "val": {record.key for record in val},
        "test": {record.key for record in test},
    }
    overlaps = {
        "train/val": sets["train"] & sets["val"],
        "train/test": sets["train"] & sets["test"],
        "val/test": sets["val"] & sets["test"],
    }
    bad = {name: values for name, values in overlaps.items() if values}
    if bad:
        raise ValueError(
            "county-year leakage across partitions: "
            + ", ".join(f"{name}={len(values)}" for name, values in bad.items())
        )


def _device_from_config(value: str) -> torch.device:
    value = str(value).strip().lower()
    if value == "auto":
        if torch.cuda.is_available():
            value = "cuda"
        elif torch.backends.mps.is_available():
            value = "mps"
        else:
            value = "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def _amp_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)


def encode_county_record(
    model: torch.nn.Module,
    store: CountyPatchStore,
    record: CountyPatchRecord,
    indices: Sequence[int] | np.ndarray,
    normalizer: BandNormalizer,
    *,
    device: torch.device,
    patch_chunk_size: int,
    mixed_precision: bool,
) -> torch.Tensor:
    """Encode patches in chunks and return their exact arithmetic mean."""
    selected = np.asarray(indices, dtype=np.int64)
    if selected.size == 0:
        raise ValueError(f"county {record.key} has no selected patches")
    representation_sum = None
    count = 0
    for start in range(0, len(selected), max(1, int(patch_chunk_size))):
        chunk_indices = selected[start : start + patch_chunk_size]
        patches = store.load_patches(record, chunk_indices).to(device)
        patches = normalizer.transform(patches)
        with _amp_context(device, mixed_precision):
            representations = model.encode_patches(patches)
        chunk_sum = representations.float().sum(dim=0)
        representation_sum = (
            chunk_sum if representation_sum is None else representation_sum + chunk_sum
        )
        count += representations.shape[0]
    assert representation_sum is not None
    return representation_sum / count


def predict_county_record(
    model: torch.nn.Module,
    store: CountyPatchStore,
    record: CountyPatchRecord,
    normalizer: BandNormalizer,
    target_scaler: TargetScaler,
    *,
    device: torch.device,
    patch_chunk_size: int,
) -> float:
    """Deterministically predict from every eligible patch in a county-year."""
    indices = np.arange(record.num_patches, dtype=np.int64)
    representation = encode_county_record(
        model,
        store,
        record,
        indices,
        normalizer,
        device=device,
        patch_chunk_size=patch_chunk_size,
        mixed_precision=False,
    )
    standardized = model.regress_counties(representation.unsqueeze(0))
    return float(target_scaler.inverse_tensor(standardized)[0].item())


def county_metrics(observed: Sequence[float], predicted: Sequence[float]) -> dict[str, Any]:
    target = np.asarray(observed, dtype=np.float64)
    prediction = np.asarray(predicted, dtype=np.float64)
    if target.shape != prediction.shape or target.size == 0:
        raise ValueError("metrics require non-empty observed/predicted arrays of equal shape")
    error = prediction - target
    rmse = float(np.sqrt(np.mean(np.square(error))))
    mae = float(np.mean(np.abs(error)))
    denominator = float(np.sum(np.square(target - target.mean())))
    r2 = float(1.0 - np.sum(np.square(error)) / denominator) if denominator > 0 else None
    mean_abs = float(np.mean(np.abs(target)))
    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "nrmse": rmse / mean_abs if mean_abs > 0 else None,
        "n": int(target.size),
    }


@torch.no_grad()
def evaluate_records(
    model: torch.nn.Module,
    store: CountyPatchStore,
    records: Sequence[CountyPatchRecord],
    normalizer: BandNormalizer,
    target_scaler: TargetScaler,
    *,
    device: torch.device,
    patch_chunk_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    rows = []
    for record in sorted(records, key=lambda item: item.key):
        prediction = predict_county_record(
            model,
            store,
            record,
            normalizer,
            target_scaler,
            device=device,
            patch_chunk_size=patch_chunk_size,
        )
        rows.append(
            {
                "county_id": record.county_id,
                "year": record.year,
                "observed_yield": record.target_bu_per_acre,
                "predicted_yield": prediction,
                "observed_yield_bu_per_acre": record.target_bu_per_acre,
                "predicted_yield_bu_per_acre": prediction,
            }
        )
    metrics = county_metrics(
        [row["observed_yield"] for row in rows],
        [row["predicted_yield"] for row in rows],
    )
    return metrics, rows


def _build_model(cfg: dict[str, Any]) -> torch.nn.Module:
    options = dict(cfg)
    name = str(options.pop("name", "3d_convlstm"))
    if name in {"gru", "lstm"}:
        options.pop("convlstm_hidden_channels", None)
        options.pop("convlstm_kernel_sizes", None)
    else:
        options.pop("recurrent_hidden", None)
        options.pop("recurrent_layers", None)
        options.pop("bidirectional", None)
    return build_supervised_model(name, **options)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n")


def train_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    seed = int(cfg.get("seed", 20260614))
    deterministic = bool(cfg.get("deterministic", True))
    set_deterministic_seed(seed, deterministic=deterministic)
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    data_cfg = dict(cfg.get("data", {}))
    store = CountyPatchStore(
        data_cfg["npz_dir"],
        yield_csv=data_cfg.get("yield_csv"),
        expected_timesteps=int(data_cfg.get("expected_timesteps", 7)),
        expected_spatial_size=data_cfg.get("expected_spatial_size", 256),
        oversize_policy=str(data_cfg.get("oversize_policy", "center_crop")),
        undersize_policy=str(data_cfg.get("undersize_policy", "error")),
        input_bands=data_cfg.get("input_bands"),
        require_complete_schedule=bool(data_cfg.get("require_complete_schedule", True)),
        fast_filename_index=bool(data_cfg.get("fast_filename_index", False)),
        max_counties=data_cfg.get("max_counties"),
    )
    train_records, val_records, test_records, split_metadata = _partition_records(
        store, dict(cfg.get("split", {}))
    )

    normalization_cfg = dict(cfg.get("normalization", {}))
    band_normalizer = fit_band_normalizer(
        store,
        train_records,
        seed=seed,
        max_patches_per_county=normalization_cfg.get("max_patches_per_county"),
        max_pixels_per_patch=normalization_cfg.get("max_pixels_per_patch"),
        chunk_size=int(normalization_cfg.get("chunk_size", 4)),
    )
    target_scaler = TargetScaler.fit(train_records)
    normalization = {
        "fit_partition": "train",
        "bands": list(store.target_bands),
        "input": band_normalizer.to_dict(),
        "target_bu_per_acre": target_scaler.to_dict(),
    }
    _write_json(out_dir / "normalization.json", normalization)

    data_contract = store.describe()
    data_contract.update(
        {
            "workflow_role": "supervised_sentinel2_benchmark",
            "estimator_family": "supervised_deep_learning",
            "split": split_metadata,
            "partition_counts": {
                "train": len(train_records),
                "val": len(val_records),
                "test": len(test_records),
            },
            "partition_keys": {
                "train": [record.key for record in train_records],
                "val": [record.key for record in val_records],
                "test": [record.key for record in test_records],
            },
        }
    )
    _write_json(out_dir / "data_contract.json", data_contract)

    device = _device_from_config(str(cfg.get("device", "auto")))
    model = _build_model(dict(cfg.get("model", {}))).to(device)
    model_name = str(model.model_name)
    training_cfg = dict(cfg.get("training", {}))
    learning_rate = float(training_cfg.get("learning_rate", 3e-4))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(training_cfg.get("weight_decay", 1e-5)),
    )
    amp_enabled = bool(training_cfg.get("mixed_precision", True)) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    except (AttributeError, TypeError):  # PyTorch < 2.3
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    batch_size = int(training_cfg.get("batch_size", 2))
    accumulation_steps = int(training_cfg.get("gradient_accumulation_steps", 1))
    patch_chunk_size = int(training_cfg.get("patch_chunk_size", 4))
    patches_per_county = training_cfg.get("patches_per_county", 8)
    max_epochs = int(training_cfg.get("max_epochs", 50))
    patience = int(training_cfg.get("patience", 8))
    grad_clip = float(training_cfg.get("gradient_clip", 1.0))
    if min(batch_size, accumulation_steps, patch_chunk_size, max_epochs, patience) <= 0:
        raise ValueError("batch, accumulation, chunk, epoch, and patience values must be positive")

    best_validation_rmse = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    checkpoint_path = out_dir / "best.pt"
    log: list[dict[str, Any]] = []

    # Evaluation-only: skip training and score the existing validation-selected
    # checkpoint on the held-out test fold. This exists because a fold whose
    # training was stopped by a scheduler wall clock still has a usable
    # checkpoint, but never reaches the evaluation below, so it produces no
    # result. Selection remains what it was during training -- the epoch with
    # the lowest validation county RMSE -- so the reported number is a
    # fixed-budget result rather than a converged one, and must be described
    # that way.
    eval_only = bool(cfg.get("eval_only", False))
    if eval_only:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"--eval-only requires an existing checkpoint at {checkpoint_path}"
            )
        try:
            saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:  # older PyTorch
            saved = torch.load(checkpoint_path, map_location=device)
        best_epoch = int(saved.get("epoch", 0))
        best_validation_rmse = float(
            saved.get("validation_metrics", {}).get("rmse", float("nan"))
        )
        existing_log = out_dir / "log.json"
        if existing_log.exists():
            log = json.loads(existing_log.read_text())
        print(
            f"eval-only: checkpoint from epoch {best_epoch}, "
            f"validation RMSE {best_validation_rmse:.6f}, "
            f"{len(log)} epochs trained",
            flush=True,
        )
        max_epochs = 0

    for epoch in range(max_epochs):
        model.train()
        generator = np.random.default_rng(seed + epoch)
        order = generator.permutation(len(train_records))
        epoch_losses = []
        optimizer.zero_grad(set_to_none=True)
        county_batches = [order[start : start + batch_size] for start in range(0, len(order), batch_size)]
        for batch_index, batch_indices in enumerate(county_batches):
            county_losses = []
            for record_index in batch_indices:
                record = train_records[int(record_index)]
                selected = deterministic_patch_sample(
                    record,
                    patches_per_county,
                    seed=seed,
                    epoch=epoch,
                )
                representation = encode_county_record(
                    model,
                    store,
                    record,
                    selected,
                    band_normalizer,
                    device=device,
                    patch_chunk_size=patch_chunk_size,
                    mixed_precision=amp_enabled,
                )
                with _amp_context(device, amp_enabled):
                    prediction = model.regress_counties(representation.unsqueeze(0))[0]
                    target = torch.tensor(record.target_bu_per_acre, device=device)
                    target = target_scaler.transform_tensor(target)
                    county_losses.append(F.mse_loss(prediction, target))
            county_loss_tensor = torch.stack(county_losses)
            epoch_losses.append(float(county_loss_tensor.mean().detach().cpu()))
            window_start = (batch_index // accumulation_steps) * accumulation_steps
            window_end = min(window_start + accumulation_steps, len(county_batches))
            counties_in_window = sum(
                len(county_batches[index]) for index in range(window_start, window_end)
            )
            scaler.scale(county_loss_tensor.sum() / counties_in_window).backward()
            should_step = (
                (batch_index + 1) % accumulation_steps == 0
                or batch_index + 1 == len(county_batches)
            )
            if should_step:
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        validation_metrics, _ = evaluate_records(
            model,
            store,
            val_records,
            band_normalizer,
            target_scaler,
            device=device,
            patch_chunk_size=patch_chunk_size,
        )
        entry = {
            "epoch": epoch,
            "train_county_mse_scaled": float(np.mean(epoch_losses)),
            "validation_county_r2": validation_metrics["r2"],
            "validation_county_rmse": validation_metrics["rmse"],
            "validation_county_mae": validation_metrics["mae"],
        }
        log.append(entry)
        _write_json(out_dir / "log.json", log)
        validation_rmse = float(validation_metrics["rmse"])
        if validation_rmse < best_validation_rmse:
            best_validation_rmse = validation_rmse
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "validation_metrics": validation_metrics,
                    "normalization": normalization,
                    "model_config": cfg.get("model", {}),
                    "seed": seed,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        print(
            f"epoch={epoch:03d} train_mse_scaled={entry['train_county_mse_scaled']:.6f} "
            f"val_rmse={validation_rmse:.6f} val_r2={validation_metrics['r2']}",
            flush=True,
        )
        if epochs_without_improvement >= patience:
            break

    if best_epoch < 0 or not checkpoint_path.exists():
        raise RuntimeError("training did not produce a validation-selected checkpoint")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:  # older PyTorch
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics, prediction_rows = evaluate_records(
        model,
        store,
        test_records,
        band_normalizer,
        target_scaler,
        device=device,
        patch_chunk_size=patch_chunk_size,
    )
    for row in prediction_rows:
        row.update(
            {
                "split_or_fold": split_metadata["label"],
                "seed": seed,
                "model_name": model_name,
            }
        )
    pd.DataFrame(prediction_rows).to_csv(out_dir / "predictions.csv", index=False)
    result = {
        "schema_version": 2,
        "experiment": {
            "id": model_name,
            "model_label": {
                "supervised_s2_3d_convlstm": "YieldSAT-inspired supervised S2 3D-ConvLSTM",
                "supervised_s2_gru": "Supervised S2 spatial-stem GRU",
                "supervised_s2_lstm": "Supervised S2 spatial-stem LSTM",
            }[model_name],
            "variant": "countyT7_full",
            "input_modalities": ["Sentinel-2"],
        },
        "selection": {
            "source": "validation",
            "metric": "county_rmse",
            "direction": "minimize",
            "epoch": best_epoch,
            "value": best_validation_rmse,
            "epochs_trained": len(log),
            "max_epochs_configured": int(training_cfg.get("max_epochs", 50)),
            "patience": patience,
            "termination": (
                "fixed_budget_not_converged"
                if eval_only or len(log) < int(training_cfg.get("max_epochs", 50))
                else "early_stopping"
            ),
        },
        "split": split_metadata,
        "seed": seed,
        "test": {
            "county_r2": test_metrics["r2"],
            "county_rmse": test_metrics["rmse"],
            "county_mae": test_metrics["mae"],
            "county_nrmse": test_metrics["nrmse"],
            "county_n": test_metrics["n"],
            "field_r2": None,
            "subfield_r2": None,
        },
        "target_and_metric_units": {
            "canonical_yield": "bushels_per_acre",
            "r2": "dimensionless",
            "rmse": "bushels_per_acre",
            "mae": "bushels_per_acre",
        },
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "predictions": str(out_dir / "predictions.csv"),
            "normalization": str(out_dir / "normalization.json"),
            "data_contract": str(out_dir / "data_contract.json"),
        },
    }
    _write_json(out_dir / "result.json", result)
    print(
        f"test county RMSE={test_metrics['rmse']:.6f} "
        f"R2={test_metrics['r2']} n={test_metrics['n']}"
    )
    return result


def inspect_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    data_cfg = dict(cfg.get("data", {}))
    store = CountyPatchStore(
        data_cfg["npz_dir"],
        yield_csv=data_cfg.get("yield_csv"),
        expected_timesteps=int(data_cfg.get("expected_timesteps", 7)),
        expected_spatial_size=data_cfg.get("expected_spatial_size", 256),
        oversize_policy=str(data_cfg.get("oversize_policy", "center_crop")),
        undersize_policy=str(data_cfg.get("undersize_policy", "error")),
        input_bands=data_cfg.get("input_bands"),
        require_complete_schedule=bool(data_cfg.get("require_complete_schedule", True)),
        fast_filename_index=bool(data_cfg.get("fast_filename_index", False)),
        max_counties=data_cfg.get("max_counties"),
    )
    train, val, test, split = _partition_records(store, dict(cfg.get("split", {})))
    description = store.describe()
    description["split"] = split
    description["partition_counts"] = {
        "train": len(train),
        "val": len(val),
        "test": len(test),
    }
    print(json.dumps(description, indent=2))
    return description


def _synthetic_config(out_dir: Path, root: Path) -> dict[str, Any]:
    data_dir = root / "county_npz" / "T7"
    data_dir.mkdir(parents=True)
    rng = np.random.default_rng(91)
    rows = []
    for index in range(8):
        county = f"{17001 + index:05d}"
        year = 2018 + index % 4
        patch_count = 2 + index % 3
        patches = rng.uniform(0.02, 0.8, size=(patch_count, 7, 10, 14, 12)).astype(np.float32)
        target = 140.0 + 2.0 * index + float(patches.mean())
        np.savez(
            data_dir / f"county_{county}_year_{year}.npz",
            patches=patches,
            county_fips=county,
            year=year,
            crop="corn",
            yield_bu_per_acre=np.float32(target),
            band_names=np.array(
                ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
            ),
        )
        split = "train" if index < 4 else "val" if index < 6 else "test"
        rows.append({"fips_year": f"{county}-{year}", "fold": 0, "split": split})
    split_path = root / "group_kfold_county_T7.csv"
    pd.DataFrame(rows).to_csv(split_path, index=False)
    return {
        "out_dir": str(out_dir),
        "seed": 42,
        "deterministic": True,
        "device": "cpu",
        "data": {
            "npz_dir": str(data_dir),
            "expected_timesteps": 7,
            "expected_spatial_size": [12, 10],
            "oversize_policy": "center_crop",
            "require_complete_schedule": True,
        },
        "split": {
            "mode": "primary",
            "path": str(split_path),
            "fold": 0,
            "require_all_years_in_each_partition": False,
        },
        "normalization": {
            "max_patches_per_county": 2,
            "max_pixels_per_patch": 64,
            "chunk_size": 2,
        },
        "model": {
            "name": "3d_convlstm",
            "input_channels": 10,
            "stem_channels": [4],
            "stem_spatial_strides": [1],
            "convlstm_hidden_channels": [4],
            "head_hidden": 4,
            "dropout": 0.0,
        },
        "training": {
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "batch_size": 2,
            "patches_per_county": 2,
            "patch_chunk_size": 2,
            "gradient_accumulation_steps": 1,
            "max_epochs": 2,
            "patience": 2,
            "mixed_precision": False,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="YAML configuration path")
    parser.add_argument("--out-dir", help="Override output directory")
    parser.add_argument("--seed", type=int, help="Override deterministic seed")
    parser.add_argument("--fold", type=int, help="Override primary outer fold")
    parser.add_argument("--device", help="Override device (auto/cpu/cuda/mps)")
    parser.add_argument(
        "--model",
        choices=("3d_convlstm", "gru", "lstm"),
        help="Override model.name from the configuration",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and evaluate the existing best.pt on the test fold. "
             "Use when a run was stopped by the scheduler before reaching "
             "evaluation; the reported result is a fixed-budget number, not a "
             "converged one.",
    )
    parser.add_argument("--inspect-only", action="store_true", help="Validate files, shapes, and splits only")
    parser.add_argument("--synthetic-smoke", action="store_true", help="Run a tiny generated CPU experiment")
    args = parser.parse_args(argv)

    if args.synthetic_smoke:
        if not args.out_dir:
            raise SystemExit("--synthetic-smoke requires --out-dir")
        with tempfile.TemporaryDirectory(prefix="supervised_s2_smoke_") as temporary:
            cfg = _synthetic_config(Path(args.out_dir), Path(temporary))
            if args.model:
                cfg["model"]["name"] = args.model
            train_from_config(cfg)
        return 0
    if not args.config:
        raise SystemExit("--config is required unless --synthetic-smoke is used")
    cfg = load_config(args.config)
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device:
        cfg["device"] = args.device
    if args.model:
        cfg.setdefault("model", {})["name"] = args.model
    split_cfg = cfg.setdefault("split", {})
    if args.fold is not None:
        split_cfg["fold"] = args.fold
    if args.eval_only:
        cfg["eval_only"] = True
    if "out_dir" not in cfg:
        raise KeyError("configuration requires out_dir or CLI --out-dir")
    if args.inspect_only:
        inspect_from_config(cfg)
    else:
        train_from_config(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
