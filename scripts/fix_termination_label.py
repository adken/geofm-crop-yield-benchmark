"""Recompute the ``selection.termination`` field in supervised result.json files.

Runs before commit 2bc4317 were written by a version of
``benchmark_embeddings.train`` that derived the label like this::

    "fixed_budget_not_converged" if eval_only or len(log) < max_epochs
    else "early_stopping"

``len(log) < max_epochs`` is exactly the condition early stopping produces, so
the two outcomes were swapped: a run that stopped on patience was recorded as
``fixed_budget_not_converged`` and a run that exhausted its epoch budget was
recorded as ``early_stopping``. Only the label was affected; every metric in
those files was written by code that is unchanged.

The correct value is recoverable from fields the same file already carries.
The training loop breaks when ``epochs_without_improvement >= patience``, and
epochs are zero-indexed, so a patience break leaves::

    epochs_trained == epoch + patience + 1

The best epoch, then ``patience`` epochs that did not improve on it. A run that
instead reached the cap has ``epochs_trained == max_epochs_configured`` and no
such relationship to ``epoch``. Both are checked, and anything matching neither
is reported rather than guessed at.

Usage::

    python scripts/fix_termination_label.py outputs/supervised_covered
    python scripts/fix_termination_label.py outputs/supervised_covered --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def infer_termination(selection: dict) -> str | None:
    """Return the label implied by the recorded epoch counts, or None."""
    epoch = selection.get("epoch")
    trained = selection.get("epochs_trained")
    patience = selection.get("patience")
    cap = selection.get("max_epochs_configured")
    if None in (epoch, trained, patience, cap):
        return None
    if trained == epoch + patience + 1:
        return "early_stopping"
    if trained >= cap:
        return "epoch_cap_reached"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="directory searched for result.json")
    parser.add_argument(
        "--check",
        action="store_true",
        help="report what would change and exit non-zero if anything would",
    )
    args = parser.parse_args()

    paths = sorted(args.root.rglob("result.json"))
    if not paths:
        print(f"no result.json under {args.root}", file=sys.stderr)
        return 2

    changed = 0
    unresolved = 0
    for path in paths:
        result = json.loads(path.read_text())
        selection = result.get("selection")
        if not isinstance(selection, dict):
            continue
        current = selection.get("termination")
        inferred = infer_termination(selection)
        rel = path.relative_to(args.root)
        if inferred is None:
            unresolved += 1
            print(f"  {rel}: cannot infer from epoch counts, left as {current!r}")
            continue
        if current == inferred:
            print(f"  {rel}: {current} (unchanged)")
            continue
        changed += 1
        detail = (
            f"epoch {selection['epoch']}, {selection['epochs_trained']} trained, "
            f"patience {selection['patience']}, cap {selection['max_epochs_configured']}"
        )
        print(f"  {rel}: {current} -> {inferred}  ({detail})")
        if not args.check:
            selection["termination"] = inferred
            path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    verb = "would change" if args.check else "changed"
    print(f"{verb} {changed} of {len(paths)} files; {unresolved} unresolved")
    if unresolved:
        return 2
    return 1 if (args.check and changed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
