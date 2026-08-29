"""Supervised Sentinel-2 county models trained from random initialization.

The public YieldSAT repository does not currently expose the precise model
used for its reported 3D-ConvLSTM results.  This implementation therefore
uses the more conservative ``YieldSAT-inspired`` designation.  It combines a
All variants share a 3D-convolutional stem whose temporal stride is one,
global spatial pooling, exact county patch means, and an MLP regressor. The
temporal backend is selected from spatial ConvLSTM, GRU, or LSTM.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _as_list(value: int | Sequence[int], length: int, name: str) -> list[int]:
    if isinstance(value, int):
        return [value] * length
    out = [int(v) for v in value]
    if len(out) != length:
        raise ValueError(f"{name} must have {length} entries, got {out}")
    return out


def _group_count(channels: int, requested: int) -> int:
    for groups in range(min(requested, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _normalization(kind: str, channels: int, groups: int) -> nn.Module:
    kind = kind.lower()
    if kind == "group":
        return nn.GroupNorm(_group_count(channels, groups), channels)
    if kind == "batch":
        return nn.BatchNorm3d(channels)
    if kind == "instance":
        return nn.InstanceNorm3d(channels, affine=True)
    if kind in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"unknown stem normalization {kind!r}")


def _activation(kind: str) -> nn.Module:
    kind = kind.lower()
    if kind == "gelu":
        return nn.GELU()
    if kind == "relu":
        return nn.ReLU(inplace=True)
    raise ValueError(f"unknown activation {kind!r}")


class ConvLSTMCell2d(nn.Module):
    """A standard ConvLSTM cell whose states retain two spatial axes."""

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("ConvLSTM kernel sizes must be positive odd integers")
        self.hidden_channels = int(hidden_channels)
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            shape = (x.shape[0], self.hidden_channels, x.shape[-2], x.shape[-1])
            hidden = x.new_zeros(shape)
            cell = x.new_zeros(shape)
        else:
            hidden, cell = state
        input_gate, forget_gate, candidate, output_gate = self.gates(
            torch.cat((x, hidden), dim=1)
        ).chunk(4, dim=1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        candidate = torch.tanh(candidate)
        output_gate = torch.sigmoid(output_gate)
        cell = forget_gate * cell + input_gate * candidate
        hidden = output_gate * torch.tanh(cell)
        return hidden, cell


class ConvLSTMStack(nn.Module):
    """Stack ConvLSTM layers over ``[N,T,C,H,W]`` feature sequences."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: Sequence[int],
        kernel_sizes: int | Sequence[int] = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden = [int(v) for v in hidden_channels]
        if not hidden or any(v <= 0 for v in hidden):
            raise ValueError("at least one positive ConvLSTM hidden width is required")
        kernels = _as_list(kernel_sizes, len(hidden), "convlstm_kernel_sizes")
        cells = []
        in_channels = input_channels
        for width, kernel in zip(hidden, kernels):
            cells.append(ConvLSTMCell2d(in_channels, width, kernel))
            in_channels = width
        self.cells = nn.ModuleList(cells)
        self.dropout = nn.Dropout2d(float(dropout))
        self.output_channels = hidden[-1]

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 5:
            raise ValueError(f"expected [N,T,C,H,W], got {tuple(sequence.shape)}")
        layer_input = sequence
        final_hidden = None
        for layer_index, cell in enumerate(self.cells):
            state = None
            outputs = []
            for timestep in range(layer_input.shape[1]):
                state = cell(layer_input[:, timestep], state)
                hidden, _ = state
                outputs.append(hidden)
            layer_input = torch.stack(outputs, dim=1)
            if layer_index + 1 < len(self.cells):
                batch, time, channels, height, width = layer_input.shape
                layer_input = self.dropout(
                    layer_input.reshape(batch * time, channels, height, width)
                ).reshape(batch, time, channels, height, width)
            final_hidden = state[0]
        assert final_hidden is not None
        return final_hidden


def aggregate_patch_representations(
    patch_representations: torch.Tensor,
    group_index: torch.Tensor,
    num_groups: int | None = None,
) -> torch.Tensor:
    """Compute an order-invariant mean representation for each county-year."""
    if patch_representations.ndim != 2:
        raise ValueError(
            "patch_representations must be [num_patches, representation_dim]"
        )
    if group_index.ndim != 1 or group_index.shape[0] != patch_representations.shape[0]:
        raise ValueError("group_index must contain one group id per patch")
    if group_index.numel() == 0:
        raise ValueError("cannot aggregate an empty patch set")
    group_index = group_index.to(device=patch_representations.device, dtype=torch.long)
    inferred_groups = int(group_index.max().item()) + 1
    num_groups = inferred_groups if num_groups is None else int(num_groups)
    if num_groups < inferred_groups or int(group_index.min().item()) < 0:
        raise ValueError("group indices must be non-negative and below num_groups")
    sums = patch_representations.new_zeros(
        (num_groups, patch_representations.shape[1])
    )
    sums.index_add_(0, group_index, patch_representations)
    counts = patch_representations.new_zeros(num_groups)
    counts.index_add_(0, group_index, patch_representations.new_ones(group_index.shape[0]))
    if torch.any(counts == 0):
        raise ValueError("every requested county group must contain at least one patch")
    return sums / counts.unsqueeze(1)


class SupervisedS23DConvLSTM(nn.Module):
    """County-level Sentinel-2-only regression model trained from scratch."""

    model_name = "supervised_s2_3d_convlstm"

    def __init__(
        self,
        input_channels: int = 10,
        stem_channels: Sequence[int] = (32, 64),
        stem_kernel_size: Sequence[int] = (3, 3, 3),
        stem_spatial_strides: int | Sequence[int] = 2,
        stem_normalization: str = "group",
        stem_norm_groups: int = 8,
        activation: str = "gelu",
        dropout: float = 0.1,
        convlstm_hidden_channels: Sequence[int] = (64,),
        convlstm_kernel_sizes: int | Sequence[int] = 3,
        head_hidden: int = 64,
    ):
        super().__init__()
        stem_widths = [int(v) for v in stem_channels]
        if not stem_widths or any(v <= 0 for v in stem_widths):
            raise ValueError("stem_channels must contain positive widths")
        kernel = tuple(int(v) for v in stem_kernel_size)
        if len(kernel) != 3 or any(v <= 0 or v % 2 == 0 for v in kernel):
            raise ValueError("stem_kernel_size must contain three positive odd values")
        strides = _as_list(
            stem_spatial_strides, len(stem_widths), "stem_spatial_strides"
        )
        layers: list[nn.Module] = []
        in_channels = int(input_channels)
        for width, stride in zip(stem_widths, strides):
            if stride <= 0:
                raise ValueError("spatial strides must be positive")
            layers.extend(
                [
                    nn.Conv3d(
                        in_channels,
                        width,
                        kernel_size=kernel,
                        stride=(1, stride, stride),
                        padding=tuple(v // 2 for v in kernel),
                        bias=stem_normalization.lower() in {"none", "identity"},
                    ),
                    _normalization(stem_normalization, width, stem_norm_groups),
                    _activation(activation),
                    nn.Dropout3d(float(dropout)),
                ]
            )
            in_channels = width
        self.input_channels = int(input_channels)
        self.stem = nn.Sequential(*layers)
        self.convlstm = ConvLSTMStack(
            input_channels=stem_widths[-1],
            hidden_channels=convlstm_hidden_channels,
            kernel_sizes=convlstm_kernel_sizes,
            dropout=dropout,
        )
        representation_dim = self.convlstm.output_channels
        self.representation_dim = representation_dim
        self.head = nn.Sequential(
            nn.LayerNorm(representation_dim),
            nn.Linear(representation_dim, int(head_hidden)),
            _activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(int(head_hidden), 1),
        )

    def encode_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """Encode ``[P,T,C,H,W]`` patches into ``[P,D]`` representations."""
        if patches.ndim != 5:
            raise ValueError(f"expected patches [P,T,C,H,W], got {tuple(patches.shape)}")
        if patches.shape[2] != self.input_channels:
            raise ValueError(
                f"expected {self.input_channels} S2 channels, got {patches.shape[2]}"
            )
        temporal_length = patches.shape[1]
        features = self.stem(patches.permute(0, 2, 1, 3, 4).contiguous())
        if features.shape[2] != temporal_length:
            raise RuntimeError(
                "3D stem changed the temporal length; temporal stride must remain one"
            )
        sequence = features.permute(0, 2, 1, 3, 4).contiguous()
        final_hidden = self.convlstm(sequence)
        return final_hidden.mean(dim=(-2, -1))

    def regress_counties(self, county_representations: torch.Tensor) -> torch.Tensor:
        return self.head(county_representations).squeeze(-1)

    def forward(
        self,
        patches: torch.Tensor,
        group_index: torch.Tensor,
        num_groups: int | None = None,
    ) -> torch.Tensor:
        patch_representations = self.encode_patches(patches)
        county_representations = aggregate_patch_representations(
            patch_representations, group_index, num_groups=num_groups
        )
        return self.regress_counties(county_representations)


class _SupervisedS2VectorRNN(nn.Module):
    """Shared 3D spatial stem followed by a vector GRU or LSTM.

    The stem sees ``[P,C,T,H,W]`` and never downsamples time. Each resulting
    timestep is globally pooled over space before the recurrent layer. This
    keeps GRU/LSTM parameter counts and input contracts comparable while
    making their distinction from the spatial-state ConvLSTM explicit.
    """

    model_name = "supervised_s2_rnn"

    def __init__(
        self,
        *,
        recurrent_type: str,
        input_channels: int = 10,
        stem_channels: Sequence[int] = (32, 64),
        stem_kernel_size: Sequence[int] = (3, 3, 3),
        stem_spatial_strides: int | Sequence[int] = 2,
        stem_normalization: str = "group",
        stem_norm_groups: int = 8,
        activation: str = "gelu",
        dropout: float = 0.1,
        recurrent_hidden: int = 64,
        recurrent_layers: int = 1,
        bidirectional: bool = False,
        head_hidden: int = 64,
    ):
        super().__init__()
        recurrent_type = str(recurrent_type).lower()
        if recurrent_type not in {"gru", "lstm"}:
            raise ValueError("recurrent_type must be 'gru' or 'lstm'")
        stem_widths = [int(value) for value in stem_channels]
        if not stem_widths or any(value <= 0 for value in stem_widths):
            raise ValueError("stem_channels must contain positive widths")
        kernel = tuple(int(value) for value in stem_kernel_size)
        if len(kernel) != 3 or any(value <= 0 or value % 2 == 0 for value in kernel):
            raise ValueError("stem_kernel_size must contain three positive odd values")
        strides = _as_list(
            stem_spatial_strides, len(stem_widths), "stem_spatial_strides"
        )
        layers: list[nn.Module] = []
        in_channels = int(input_channels)
        for width, stride in zip(stem_widths, strides):
            if stride <= 0:
                raise ValueError("spatial strides must be positive")
            layers.extend(
                [
                    nn.Conv3d(
                        in_channels,
                        width,
                        kernel_size=kernel,
                        stride=(1, stride, stride),
                        padding=tuple(value // 2 for value in kernel),
                        bias=stem_normalization.lower() in {"none", "identity"},
                    ),
                    _normalization(stem_normalization, width, stem_norm_groups),
                    _activation(activation),
                    nn.Dropout3d(float(dropout)),
                ]
            )
            in_channels = width
        hidden = int(recurrent_hidden)
        recurrent_layers = int(recurrent_layers)
        if hidden <= 0 or recurrent_layers <= 0:
            raise ValueError("recurrent_hidden and recurrent_layers must be positive")
        rnn_cls = nn.GRU if recurrent_type == "gru" else nn.LSTM
        self.recurrent_type = recurrent_type
        self.input_channels = int(input_channels)
        self.stem = nn.Sequential(*layers)
        self.recurrent = rnn_cls(
            input_size=stem_widths[-1],
            hidden_size=hidden,
            num_layers=recurrent_layers,
            batch_first=True,
            dropout=float(dropout) if recurrent_layers > 1 else 0.0,
            bidirectional=bool(bidirectional),
        )
        self.representation_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(self.representation_dim),
            nn.Linear(self.representation_dim, int(head_hidden)),
            _activation(activation),
            nn.Dropout(float(dropout)),
            nn.Linear(int(head_hidden), 1),
        )

    def encode_patches(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 5:
            raise ValueError(f"expected patches [P,T,C,H,W], got {tuple(patches.shape)}")
        if patches.shape[2] != self.input_channels:
            raise ValueError(
                f"expected {self.input_channels} S2 channels, got {patches.shape[2]}"
            )
        temporal_length = int(patches.shape[1])
        features = self.stem(patches.permute(0, 2, 1, 3, 4).contiguous())
        if features.shape[2] != temporal_length:
            raise RuntimeError("3D stem changed temporal length")
        sequence = features.mean(dim=(-2, -1)).transpose(1, 2).contiguous()
        output, _ = self.recurrent(sequence)
        return output[:, -1]

    def regress_counties(self, county_representations: torch.Tensor) -> torch.Tensor:
        return self.head(county_representations).squeeze(-1)

    def forward(
        self,
        patches: torch.Tensor,
        group_index: torch.Tensor,
        num_groups: int | None = None,
    ) -> torch.Tensor:
        patch_representations = self.encode_patches(patches)
        county_representations = aggregate_patch_representations(
            patch_representations, group_index, num_groups=num_groups
        )
        return self.regress_counties(county_representations)


class SupervisedS2GRU(_SupervisedS2VectorRNN):
    model_name = "supervised_s2_gru"

    def __init__(self, **kwargs):
        super().__init__(recurrent_type="gru", **kwargs)


class SupervisedS2LSTM(_SupervisedS2VectorRNN):
    model_name = "supervised_s2_lstm"

    def __init__(self, **kwargs):
        super().__init__(recurrent_type="lstm", **kwargs)


SUPERVISED_MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "3d_convlstm": SupervisedS23DConvLSTM,
    "convlstm": SupervisedS23DConvLSTM,
    "gru": SupervisedS2GRU,
    "lstm": SupervisedS2LSTM,
}


def build_supervised_model(name: str, **kwargs) -> nn.Module:
    key = str(name).strip().lower()
    if key not in SUPERVISED_MODEL_REGISTRY:
        raise KeyError(
            f"unknown supervised model {name!r}; choose from "
            f"{sorted(SUPERVISED_MODEL_REGISTRY)}"
        )
    return SUPERVISED_MODEL_REGISTRY[key](**kwargs)
