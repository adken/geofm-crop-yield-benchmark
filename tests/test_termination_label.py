"""The supervised ``selection.termination`` label and the archived artefacts.

An earlier version of ``benchmark_embeddings.train`` derived the label from
``len(log) < max_epochs``, which is the condition early stopping produces, so
the two outcomes were swapped. Nothing tested the field, so the inverted label
reached the published artefacts. These tests pin the arithmetic and check the
committed result files against it.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "fix_termination_label.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("fix_termination_label", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


infer_termination = _load_script().infer_termination


def _selection(epoch: int, trained: int, patience: int = 8, cap: int = 50) -> dict:
    return {
        "epoch": epoch,
        "epochs_trained": trained,
        "patience": patience,
        "max_epochs_configured": cap,
    }


def test_patience_break_leaves_best_epoch_plus_patience_plus_one():
    # Epochs are zero-indexed: the best epoch, then `patience` epochs that did
    # not improve on it, then the break.
    assert infer_termination(_selection(epoch=17, trained=26)) == "early_stopping"
    assert infer_termination(_selection(epoch=0, trained=9)) == "early_stopping"
    assert infer_termination(_selection(epoch=4, trained=7, patience=2)) == (
        "early_stopping"
    )


def test_exhausting_the_budget_is_not_early_stopping():
    # Improvement late in training, so patience never fires and the loop ends
    # at the cap. This is the case the old code labelled "early_stopping".
    assert infer_termination(_selection(epoch=48, trained=50)) == "epoch_cap_reached"


def test_counts_matching_neither_pattern_are_reported_not_guessed():
    assert infer_termination(_selection(epoch=10, trained=15)) is None
    assert infer_termination({"epoch": 3}) is None


@pytest.mark.parametrize(
    "result_path",
    sorted((REPO_ROOT / "outputs" / "supervised_covered").rglob("result.json")),
    ids=lambda p: p.parent.parent.name,
)
def test_archived_results_agree_with_their_epoch_counts(result_path: Path):
    selection = json.loads(result_path.read_text())["selection"]
    inferred = infer_termination(selection)
    assert inferred is not None, f"{result_path} has counts matching no known exit"
    assert selection["termination"] == inferred
