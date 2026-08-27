#!/usr/bin/env python3
"""Matched neural temporal-readout ablation for frozen GeoFM embeddings.

Clay, Prithvi, and TerraMind first use the canonical shared operation order:
population mean and standard deviation across complete spatial patches at each
of seven timesteps.  The resulting county sequence [T,F] is then passed to one
of three learned readouts:

``mean``
    Temporal mean followed by a learned projection.
``concat``
    Flatten all timesteps followed by a learned projection.
``conv1d``
    Two learned temporal convolutions followed by temporal global mean.

Every readout produces the same latent dimension and uses the same MLP yield
head, optimizer, train-only input/target scaling, validation early stopping,
test fold, and deterministic seeds.  The three encoder cohorts and per-county
patch counts must match exactly; the program never silently intersects them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import (
    FoldPartitions,
    load_fold_partitions,
    validate_all_years_in_partitions,
    years_from_keys,
)
from .frozen import read_embeddings
from .loyo import validate_unfused_main_embedding
from .probe import (
    AGGREGATION_OPERATION_ORDER,
    SPATIAL_POOL_MEAN_STD,
    build_county_embedding_sequences,
)


ENCODERS = ("clay", "prithvi", "terramind")
STRATEGY_MEAN = "mean"
STRATEGY_CONCAT = "concat"
STRATEGY_CONV1D = "conv1d"
STRATEGIES = (STRATEGY_MEAN, STRATEGY_CONCAT, STRATEGY_CONV1D)
PREDICTION_HEAD = "mlp"


def set_deterministic_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _device(value: str) -> torch.device:
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


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n")


@dataclass(frozen=True)
class SequenceStandardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, train: np.ndarray) -> "SequenceStandardizer":
        values = np.asarray(train, dtype=np.float64)
        if values.ndim != 3 or values.shape[0] == 0:
            raise ValueError(f"sequence scaler expects non-empty [N,T,F], got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("training sequences contain non-finite values")
        mean = values.mean(axis=0)
        scale = values.std(axis=0, ddof=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        return cls(mean.astype(np.float32), scale.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.shape[-2:] != self.mean.shape:
            raise ValueError(
                f"sequence shape {values.shape[-2:]} disagrees with scaler {self.mean.shape}"
            )
        output = (values - self.mean) / self.scale
        if not np.isfinite(output).all():
            raise ValueError("sequence standardization produced non-finite values")
        return output.astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fit_partition": "train",
            "axis_contract": "per_timestep_per_feature",
            "mean": self.mean,
            "scale": self.scale,
        }


@dataclass(frozen=True)
class TargetStandardizer:
    mean: float
    scale: float

    @classmethod
    def fit(cls, train: np.ndarray) -> "TargetStandardizer":
        values = np.asarray(train, dtype=np.float64).reshape(-1)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError("target scaler needs finite training targets")
        scale = float(values.std(ddof=0))
        return cls(float(values.mean()), scale if scale >= 1e-8 else 1.0)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((np.asarray(values, dtype=np.float32) - self.mean) / self.scale).astype(
            np.float32
        )

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) * self.scale + self.mean).astype(
            np.float32
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fit_partition": "train",
            "units": "bushels_per_acre",
            "mean": self.mean,
            "scale": self.scale,
        }


class TemporalAblationRegressor(nn.Module):
    """Three temporal readouts with a shared latent size and MLP head."""

    def __init__(
        self,
        *,
        strategy: str,
        timesteps: int,
        feature_dim: int,
        readout_dim: int = 256,
        conv_channels: int = 128,
        conv_kernel_size: int = 3,
        mlp_hidden: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.strategy = str(strategy).strip().lower()
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        self.timesteps = int(timesteps)
        self.feature_dim = int(feature_dim)
        self.readout_dim = int(readout_dim)
        if min(self.timesteps, self.feature_dim, self.readout_dim, int(mlp_hidden)) <= 0:
            raise ValueError("timesteps and model dimensions must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if self.strategy == STRATEGY_MEAN:
            self.readout = nn.Sequential(
                nn.Linear(self.feature_dim, self.readout_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            )
        elif self.strategy == STRATEGY_CONCAT:
            self.readout = nn.Sequential(
                nn.Linear(self.timesteps * self.feature_dim, self.readout_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            )
        else:
            kernel = int(conv_kernel_size)
            if kernel <= 0 or kernel % 2 == 0:
                raise ValueError("conv_kernel_size must be a positive odd integer")
            channels = int(conv_channels)
            if channels <= 0:
                raise ValueError("conv_channels must be positive")
            padding = kernel // 2
            self.readout = nn.Sequential(
                nn.Conv1d(self.feature_dim, channels, kernel, padding=padding),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Conv1d(channels, self.readout_dim, kernel, padding=padding),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(start_dim=1),
            )
        self.head = nn.Sequential(
            nn.LayerNorm(self.readout_dim),
            nn.Linear(self.readout_dim, int(mlp_hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(mlp_hidden), 1),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3:
            raise ValueError(f"temporal input must be [B,T,F], got {tuple(sequence.shape)}")
        if tuple(sequence.shape[1:]) != (self.timesteps, self.feature_dim):
            raise ValueError(
                f"temporal input is {tuple(sequence.shape[1:])}, expected "
                f"{(self.timesteps, self.feature_dim)}"
            )
        if self.strategy == STRATEGY_MEAN:
            latent = self.readout(sequence.mean(dim=1))
        elif self.strategy == STRATEGY_CONCAT:
            latent = self.readout(sequence.reshape(sequence.shape[0], -1))
        else:
            latent = self.readout(sequence.transpose(1, 2))
        return self.head(latent).squeeze(-1)


def county_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    observed = np.asarray(observed, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if observed.shape != predicted.shape or observed.size == 0:
        raise ValueError("metrics need non-empty observed/predicted arrays of equal shape")
    residual = predicted - observed
    denominator = float(np.square(observed - observed.mean()).sum())
    return {
        "r2": (
            float(1.0 - np.square(residual).sum() / denominator)
            if denominator > 0.0
            else None
        ),
        "rmse": float(np.sqrt(np.square(residual).mean())),
        "mae": float(np.abs(residual).mean()),
        "n": int(observed.size),
    }


def load_encoder_sequences(
    name: str,
    path: str | Path,
    labels: pd.DataFrame,
    *,
    spatial_pool: str,
    expected_timesteps: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    embeddings = read_embeddings(path)
    embeddings, main_contract = validate_unfused_main_embedding(name, embeddings)
    backbones = sorted(embeddings["backbone"].astype(str).unique())
    if len(backbones) != 1:
        raise ValueError(f"{name} input mixes backbones: {backbones}")
    sequences = build_county_embedding_sequences(
        embeddings,
        labels,
        spatial_pool=spatial_pool,
        expected_timesteps=expected_timesteps,
    ).sort_values("key").reset_index(drop=True)
    scopes = sorted(sequences["representation_scope"].astype(str).str.lower().unique())
    if scopes != ["timestep"]:
        raise ValueError(
            f"{name} temporal ablation requires per-timestep representations, got {scopes}"
        )
    shapes = sorted({tuple(np.asarray(value).shape) for value in sequences["sequence"]})
    if len(shapes) != 1 or shapes[0][0] != int(expected_timesteps):
        raise ValueError(f"{name} has inconsistent temporal sequence shapes {shapes}")
    contract = {
        **main_contract,
        "encoder": name,
        "path": str(Path(path).resolve()),
        "backbone": backbones[0],
        "embedding_rows": int(len(embeddings)),
        "county_years": int(len(sequences)),
        "timesteps": int(shapes[0][0]),
        "embedding_dim": int(shapes[0][1] // (2 if spatial_pool == "mean_std" else 1)),
        "county_timestep_feature_dim": int(shapes[0][1]),
        "patch_count_min": int(sequences["n_patches"].min()),
        "patch_count_median": float(sequences["n_patches"].median()),
        "patch_count_max": int(sequences["n_patches"].max()),
        "spatial_pool": spatial_pool,
        "spatial_std_ddof": 0 if spatial_pool == "mean_std" else None,
    }
    del embeddings
    return sequences, contract


def validate_matched_cohorts(frames: Mapping[str, pd.DataFrame]) -> list[str]:
    if set(frames) != set(ENCODERS):
        raise ValueError(f"temporal ablation requires exactly {ENCODERS}")
    reference_name = ENCODERS[0]
    reference = frames[reference_name].set_index("key")
    reference_keys = set(reference.index)
    for name in ENCODERS[1:]:
        candidate = frames[name].set_index("key")
        candidate_keys = set(candidate.index)
        if candidate_keys != reference_keys:
            missing = sorted(reference_keys - candidate_keys)
            extra = sorted(candidate_keys - reference_keys)
            raise ValueError(
                f"county-year cohort mismatch for {name}: missing={missing[:5]}, "
                f"extra={extra[:5]}"
            )
        reference_counts = reference.loc[sorted(reference_keys), "n_patches"].to_numpy()
        candidate_counts = candidate.loc[sorted(reference_keys), "n_patches"].to_numpy()
        if not np.array_equal(reference_counts, candidate_counts):
            bad = np.flatnonzero(reference_counts != candidate_counts)
            examples = [sorted(reference_keys)[int(index)] for index in bad[:5]]
            raise ValueError(
                f"spatial patch-count mismatch for {name} at county-years {examples}"
            )
        reference_patch_ids = reference.loc[
            sorted(reference_keys), "complete_patch_ids"
        ].tolist()
        candidate_patch_ids = candidate.loc[
            sorted(reference_keys), "complete_patch_ids"
        ].tolist()
        patch_identity_mismatch = [
            key
            for key, expected, observed in zip(
                sorted(reference_keys),
                reference_patch_ids,
                candidate_patch_ids,
                strict=True,
            )
            if tuple(expected) != tuple(observed)
        ]
        if patch_identity_mismatch:
            raise ValueError(
                f"spatial patch-identity mismatch for {name} at county-years "
                f"{patch_identity_mismatch[:5]}"
            )
        reference_y = reference.loc[
            sorted(reference_keys), "yield_bu_per_acre"
        ].to_numpy()
        candidate_y = candidate.loc[
            sorted(reference_keys), "yield_bu_per_acre"
        ].to_numpy()
        if not np.allclose(reference_y, candidate_y, atol=0.0, rtol=0.0):
            raise ValueError(f"yield target mismatch between {reference_name} and {name}")
    return sorted(reference_keys)


def validate_split_coverage(common_keys: Sequence[str], parts: FoldPartitions) -> None:
    available = set(common_keys)
    required = set(parts.train) | set(parts.val) | set(parts.test)
    missing = sorted(required - available)
    if missing:
        raise ValueError(
            f"split requires {len(missing)} county-years absent from the matched cohort: "
            f"{missing[:5]}"
        )
    extra = sorted(available - required)
    if extra:
        raise ValueError(
            f"matched encoder cohort contains {len(extra)} county-years absent from "
            f"fold {parts.outer_fold}: {extra[:5]}"
        )
    if (set(parts.train) & set(parts.val)) or (set(parts.train) & set(parts.test)) or (
        set(parts.val) & set(parts.test)
    ):
        raise ValueError("split partitions are not disjoint")


def _indices_for_keys(frame: pd.DataFrame, keys: Sequence[str]) -> np.ndarray:
    lookup = {key: index for index, key in enumerate(frame["key"].tolist())}
    missing = sorted(set(keys) - set(lookup))
    if missing:
        raise ValueError(f"county sequence table is missing split keys {missing[:5]}")
    return np.asarray([lookup[key] for key in keys], dtype=np.int64)


@torch.no_grad()
def _predict_standardized(
    model: nn.Module,
    sequence: np.ndarray,
    indices: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(indices), int(batch_size)):
        selected = indices[start : start + int(batch_size)]
        tensor = torch.from_numpy(sequence[selected]).to(device)
        outputs.append(model(tensor).float().cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def train_one_readout(
    sequence: np.ndarray,
    target: np.ndarray,
    keys: Sequence[str],
    parts: FoldPartitions,
    *,
    strategy: str,
    seed: int,
    device: torch.device,
    readout_dim: int,
    conv_channels: int,
    conv_kernel_size: int,
    mlp_hidden: int,
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    mixed_precision: bool,
) -> tuple[nn.Module, dict[str, Any], pd.DataFrame, dict[str, Any]]:
    set_deterministic_seed(seed)
    sequence = np.asarray(sequence, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32).reshape(-1)
    if sequence.ndim != 3 or sequence.shape[0] != target.size:
        raise ValueError(
            f"training data must be aligned [N,T,F] and [N], got "
            f"{sequence.shape} and {target.shape}"
        )
    if sequence.shape[0] != len(keys) or len(set(keys)) != len(keys):
        raise ValueError("county-year keys must be unique and aligned with the data")
    if not np.isfinite(sequence).all() or not np.isfinite(target).all():
        raise ValueError("temporal training data contain non-finite values")
    if int(batch_size) <= 0 or int(max_epochs) <= 0 or int(patience) <= 0:
        raise ValueError("batch_size, max_epochs, and patience must be positive")
    if float(learning_rate) <= 0.0 or float(weight_decay) < 0.0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    key_frame = pd.DataFrame({"key": list(keys)})
    train_indices = _indices_for_keys(key_frame, parts.train)
    val_indices = _indices_for_keys(key_frame, parts.val)
    test_indices = _indices_for_keys(key_frame, parts.test)

    sequence_scaler = SequenceStandardizer.fit(sequence[train_indices])
    target_scaler = TargetStandardizer.fit(target[train_indices])
    scaled_sequence = sequence_scaler.transform(sequence)
    scaled_target = target_scaler.transform(target)

    model = TemporalAblationRegressor(
        strategy=strategy,
        timesteps=int(sequence.shape[1]),
        feature_dim=int(sequence.shape[2]),
        readout_dim=readout_dim,
        conv_channels=conv_channels,
        conv_kernel_size=conv_kernel_size,
        mlp_hidden=mlp_hidden,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    amp_enabled = bool(mixed_precision) and device.type == "cuda"
    try:
        amp_scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    except (AttributeError, TypeError):
        amp_scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_validation_rmse = float("inf")
    epochs_without_improvement = 0
    history = []
    for epoch in range(int(max_epochs)):
        model.train()
        order = np.random.default_rng(int(seed) + epoch).permutation(train_indices)
        losses = []
        for start in range(0, len(order), int(batch_size)):
            selected = order[start : start + int(batch_size)]
            x = torch.from_numpy(scaled_sequence[selected]).to(device)
            y = torch.from_numpy(scaled_target[selected]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with _amp_context(device, amp_enabled):
                prediction = model(x)
                loss = F.mse_loss(prediction, y)
            if not torch.isfinite(loss):
                raise RuntimeError("temporal ablation produced a non-finite training loss")
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            losses.append(float(loss.detach().cpu()))

        validation_scaled = _predict_standardized(
            model,
            scaled_sequence,
            val_indices,
            device=device,
            batch_size=batch_size,
        )
        validation_prediction = target_scaler.inverse(validation_scaled)
        validation_metrics = county_metrics(target[val_indices], validation_prediction)
        validation_rmse = float(validation_metrics["rmse"])
        history.append(
            {
                "epoch": epoch,
                "train_mse_scaled": float(np.mean(losses)),
                "validation_r2": validation_metrics["r2"],
                "validation_rmse": validation_rmse,
                "validation_mae": validation_metrics["mae"],
            }
        )
        if validation_rmse < best_validation_rmse - 1e-12:
            best_validation_rmse = validation_rmse
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= int(patience):
            break

    if best_state is None:
        raise RuntimeError("no validation-selected temporal checkpoint was produced")
    model.load_state_dict(best_state)
    test_scaled = _predict_standardized(
        model,
        scaled_sequence,
        test_indices,
        device=device,
        batch_size=batch_size,
    )
    test_prediction = target_scaler.inverse(test_scaled)
    metrics = county_metrics(target[test_indices], test_prediction)
    predictions = pd.DataFrame(
        {
            "key": [keys[index] for index in test_indices],
            "observed_yield": target[test_indices],
            "predicted_yield": test_prediction,
        }
    )
    extracted = predictions["key"].str.extract(r"^(?P<county_id>.+)-(?P<year>\d{4})$")
    predictions.insert(0, "year", extracted["year"].astype(int))
    predictions.insert(0, "county_id", extracted["county_id"])
    predictions["observed_yield_bu_per_acre"] = predictions["observed_yield"]
    predictions["predicted_yield_bu_per_acre"] = predictions["predicted_yield"]
    normalization = {
        "sequence": sequence_scaler.to_dict(),
        "target": target_scaler.to_dict(),
    }
    selection = {
        "source": "validation",
        "metric": "county_rmse",
        "direction": "minimize",
        "epoch": best_epoch,
        "value": best_validation_rmse,
        "history": history,
    }
    return model, metrics, predictions, {
        "normalization": normalization,
        "selection": selection,
    }


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        raise ValueError("cannot summarize empty temporal results")
    summary = (
        results.groupby(["encoder", "backbone", "strategy", "fold"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            test_n=("test_n", "first"),
            r2_mean=("test_r2", "mean"),
            r2_std=("test_r2", lambda values: values.std(ddof=0)),
            rmse_mean=("test_rmse", "mean"),
            rmse_std=("test_rmse", lambda values: values.std(ddof=0)),
            mae_mean=("test_mae", "mean"),
            mae_std=("test_mae", lambda values: values.std(ddof=0)),
            parameter_count=("parameter_count", "first"),
        )
        .sort_values(["encoder", "strategy"])
        .reset_index(drop=True)
    )
    return summary


def run_temporal_ablation(
    embedding_paths: Mapping[str, str | Path],
    *,
    labels_path: str | Path,
    split_path: str | Path,
    fold: int,
    out_dir: str | Path,
    seeds: Sequence[int] = (0, 1, 2),
    strategies: Sequence[str] = STRATEGIES,
    spatial_pool: str = "mean_std",
    expected_timesteps: int = 7,
    readout_dim: int = 256,
    conv_channels: int = 128,
    conv_kernel_size: int = 3,
    mlp_hidden: int = 128,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    max_epochs: int = 300,
    patience: int = 30,
    device: str | torch.device = "auto",
    mixed_precision: bool = False,
    preflight_only: bool = False,
) -> dict[str, Any]:
    if set(embedding_paths) != set(ENCODERS):
        raise ValueError(f"embedding_paths must contain exactly {ENCODERS}")
    strategies = tuple(str(value).strip().lower() for value in strategies)
    if set(strategies) != set(STRATEGIES):
        raise ValueError(f"canonical temporal ablation requires exactly {STRATEGIES}")
    if len(set(strategies)) != len(strategies):
        raise ValueError("strategies must not contain duplicates")
    spatial_pool = str(spatial_pool).strip().lower()
    if spatial_pool != "mean_std":
        raise ValueError(
            "canonical temporal ablation requires mean_std spatial pooling to "
            "match the main benchmark"
        )
    seeds = tuple(int(value) for value in seeds)
    if not seeds:
        raise ValueError("at least one deterministic seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must not contain duplicates")
    labels = pd.read_csv(labels_path)
    frames: dict[str, pd.DataFrame] = {}
    contracts: dict[str, Any] = {}
    for name in ENCODERS:
        frame, contract = load_encoder_sequences(
            name,
            embedding_paths[name],
            labels,
            spatial_pool=spatial_pool,
            expected_timesteps=expected_timesteps,
        )
        frames[name] = frame
        contracts[name] = contract
    common_keys = validate_matched_cohorts(frames)
    parts = load_fold_partitions(split_path, fold=int(fold), id_column="fips_year")
    validate_split_coverage(common_keys, parts)
    cohort_years = years_from_keys(common_keys)
    partition_years = validate_all_years_in_partitions(
        parts, expected_years=cohort_years
    )
    matched_key_sha256 = hashlib.sha256(
        "\n".join(common_keys).encode("utf-8")
    ).hexdigest()
    patch_lookup = frames[ENCODERS[0]].set_index("key")["complete_patch_ids"]
    identity_separator = "\x1f"
    matched_patch_sha256 = hashlib.sha256(
        "\n".join(
            f"{key}{identity_separator}"
            f"{identity_separator.join(patch_lookup.loc[key])}"
            for key in common_keys
        ).encode("utf-8")
    ).hexdigest()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_contract = {
        "schema_version": 1,
        "experiment": "temporal_readout_ablation",
        "workflow_role": "temporal_pooling_ablation",
        "estimator_family": "neural_ablation",
        "prediction_head": PREDICTION_HEAD,
        "encoders": contracts,
        "matched_county_years": len(common_keys),
        "matched_key_sha256": matched_key_sha256,
        "matched_complete_patch_identity_sha256": matched_patch_sha256,
        "spatial_pool": spatial_pool,
        "spatial_operation_order": AGGREGATION_OPERATION_ORDER,
        "feature_aggregation": {
            "scope": "patch_embedding_representations",
            "operation_order": AGGREGATION_OPERATION_ORDER,
            "spatial_pool": SPATIAL_POOL_MEAN_STD,
            "spatial_pool_axis": "complete_patches_within_county_year_per_timestep",
            "spatial_std_ddof": 0,
            "temporal_pool_strategies": list(strategies),
            "temporal_pool_axis": "timestep",
        },
        "expected_timesteps": int(expected_timesteps),
        "protocol": {
            "strategies": strategies,
            "prediction_head": PREDICTION_HEAD,
            "prediction_head_scope": "temporal_pooling_ablation_only",
            "seeds": seeds,
            "readout_dim": int(readout_dim),
            "conv_channels": int(conv_channels),
            "conv_kernel_size": int(conv_kernel_size),
            "mlp_hidden": int(mlp_hidden),
            "dropout": float(dropout),
            "optimizer": "AdamW",
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "loss": "county_mse_on_train_standardized_target",
            "batch_size": int(batch_size),
            "max_epochs": int(max_epochs),
            "early_stopping_patience": int(patience),
            "early_stopping_metric": "validation_county_rmse",
            "sequence_scaling": "train_only_per_timestep_per_feature",
            "target_scaling": "train_only",
            "mixed_precision_requested": bool(mixed_precision),
        },
        "software": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
        },
        "split": {
            "path": str(Path(split_path).resolve()),
            "fold": int(fold),
            "validation_fold": parts.validation_fold,
            "grouping": "county_all_years_together",
            "cohort_years": cohort_years,
            "year_policy": "all_cohort_years_in_each_train_validation_test_partition",
            "partition_years": partition_years,
            "train_keys": parts.train,
            "val_keys": parts.val,
            "test_keys": parts.test,
        },
        "labels": str(Path(labels_path).resolve()),
        "target_and_metric_units": {
            "canonical_yield": "bushels_per_acre",
            "r2": "dimensionless",
            "rmse": "bushels_per_acre",
            "mae": "bushels_per_acre",
        },
    }
    _write_json(out_dir / "data_contract.json", data_contract)
    if preflight_only:
        return {"data_contract": data_contract, "results": None}

    torch_device = device if isinstance(device, torch.device) else _device(device)
    results = []
    prediction_tables = []
    for encoder in ENCODERS:
        frame = frames[encoder].set_index("key").loc[common_keys].reset_index()
        sequence = np.stack(frame["sequence"].tolist()).astype(np.float32)
        target = frame["yield_bu_per_acre"].to_numpy(dtype=np.float32)
        backbone = contracts[encoder]["backbone"]
        for strategy in strategies:
            for seed in seeds:
                model, metrics, predictions, details = train_one_readout(
                    sequence,
                    target,
                    common_keys,
                    parts,
                    strategy=strategy,
                    seed=seed,
                    device=torch_device,
                    readout_dim=readout_dim,
                    conv_channels=conv_channels,
                    conv_kernel_size=conv_kernel_size,
                    mlp_hidden=mlp_hidden,
                    dropout=dropout,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    batch_size=batch_size,
                    max_epochs=max_epochs,
                    patience=patience,
                    mixed_precision=mixed_precision,
                )
                parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
                run_id = f"{encoder}_{strategy}_fold{fold}_seed{seed}"
                checkpoint_path = out_dir / "checkpoints" / f"{run_id}.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "encoder": encoder,
                        "backbone": backbone,
                        "strategy": strategy,
                        "seed": seed,
                        "fold": int(fold),
                        "model_config": {
                            "timesteps": expected_timesteps,
                            "feature_dim": int(sequence.shape[2]),
                            "readout_dim": readout_dim,
                            "conv_channels": conv_channels,
                            "conv_kernel_size": conv_kernel_size,
                            "mlp_hidden": mlp_hidden,
                            "dropout": dropout,
                        },
                        "normalization": details["normalization"],
                        "selection": details["selection"],
                    },
                    checkpoint_path,
                )
                predictions["encoder"] = encoder
                predictions["backbone"] = backbone
                predictions["strategy"] = strategy
                predictions["seed"] = seed
                predictions["split_or_fold"] = f"fold_{fold}"
                predictions["model_name"] = f"temporal_ablation_{strategy}_mlp"
                predictions["prediction_head"] = PREDICTION_HEAD
                prediction_tables.append(predictions)
                result = {
                    "encoder": encoder,
                    "backbone": backbone,
                    "strategy": strategy,
                    "prediction_head": PREDICTION_HEAD,
                    "fold": int(fold),
                    "seed": seed,
                    "test_r2": metrics["r2"],
                    "test_rmse": metrics["rmse"],
                    "test_mae": metrics["mae"],
                    "test_n": metrics["n"],
                    "validation_rmse": details["selection"]["value"],
                    "best_epoch": details["selection"]["epoch"],
                    "parameter_count": parameter_count,
                    "sequence_feature_dim": int(sequence.shape[2]),
                    "readout_dim": int(readout_dim),
                    "checkpoint": str(checkpoint_path),
                }
                results.append(result)
                _write_json(
                    out_dir / "runs" / f"{run_id}.json",
                    {
                        "schema_version": 1,
                        "experiment": "temporal_readout_ablation",
                        "result": result,
                        "selection": details["selection"],
                        "normalization": details["normalization"],
                    },
                )

    result_frame = pd.DataFrame(results)
    prediction_frame = pd.concat(prediction_tables, ignore_index=True)
    summary = summarize_results(result_frame)
    result_frame.to_csv(out_dir / "results_by_seed.csv", index=False)
    prediction_frame.to_csv(out_dir / "predictions.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    return {
        "data_contract": data_contract,
        "results": result_frame,
        "predictions": prediction_frame,
        "summary": summary,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clay", required=True, help="Canonical Clay embedding Parquet")
    parser.add_argument("--prithvi", required=True, help="Corrected Prithvi embedding Parquet")
    parser.add_argument("--terramind", required=True, help="Canonical TerraMind embedding Parquet")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=STRATEGIES)
    parser.add_argument("--spatial-pool", choices=("mean_std",), default="mean_std")
    parser.add_argument("--timesteps", type=int, default=7)
    parser.add_argument("--readout-dim", type=int, default=256)
    parser.add_argument("--conv-channels", type=int, default=128)
    parser.add_argument("--conv-kernel-size", type=int, default=3)
    parser.add_argument("--mlp-hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)

    output = run_temporal_ablation(
        {"clay": args.clay, "prithvi": args.prithvi, "terramind": args.terramind},
        labels_path=args.labels,
        split_path=args.split,
        fold=args.fold,
        out_dir=args.out_dir,
        seeds=args.seeds,
        strategies=args.strategies,
        spatial_pool=args.spatial_pool,
        expected_timesteps=args.timesteps,
        readout_dim=args.readout_dim,
        conv_channels=args.conv_channels,
        conv_kernel_size=args.conv_kernel_size,
        mlp_hidden=args.mlp_hidden,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        device=args.device,
        mixed_precision=args.mixed_precision,
        preflight_only=args.preflight_only,
    )
    if args.preflight_only:
        print(json.dumps(_json_safe(output["data_contract"]), indent=2))
    else:
        print(output["summary"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
