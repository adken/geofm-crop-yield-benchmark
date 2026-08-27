from __future__ import annotations

import pytest
import torch

from benchmark_embeddings.temporal_ablation import _device as temporal_device
from benchmark_embeddings.train import _device_from_config as supervised_device


@pytest.mark.parametrize("resolver", (supervised_device, temporal_device))
def test_auto_prefers_mps_when_cuda_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    resolver,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolver("auto").type == "mps"


@pytest.mark.parametrize("resolver", (supervised_device, temporal_device))
def test_explicit_unavailable_mps_fails_early(
    monkeypatch: pytest.MonkeyPatch,
    resolver,
) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS was requested but is unavailable"):
        resolver("mps")
