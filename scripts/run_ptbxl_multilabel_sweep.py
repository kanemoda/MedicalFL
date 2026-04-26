#!/usr/bin/env python
"""Phase 6 BONUS — multi-label PTB-XL FL.

DEFERRED — see DECISIONS.md "Phase 6 — Cross-dataset benchmark".

This sweep would evaluate FedAvg/FedProx/FedBN/FedPerf on multi-label
PTB-XL (5 superclasses, BCEWithLogitsLoss, macro-AUC). The infrastructure
needed but not yet present:

    1. ``data/processed/ptbxl/y_multilabel.npy`` — [21388, 5] binary
       matrix derived from PTB-XL ``scp_codes`` (each record can carry
       multiple superclasses). Build step would extend
       ``src/data/ptbxl.py`` and re-run preprocessing.

    2. A ``label_mode='multilabel'`` branch in ``src/training/federated.py``
       (currently only ``binary`` and ``5class`` are wired). It needs to
       swap CrossEntropyLoss → BCEWithLogitsLoss, change ``y`` dtype from
       int64 → float32, partition on the row index rather than the label,
       and report macro-AUC instead of macro-F1 on the central path.

    3. A multilabel-aware ``_evaluate_local_test`` so per-client AUC is
       comparable across runs.

This script intentionally exits 0 without launching any runs so the
sequential master pipeline (`6A → 6B → bonus`) completes cleanly when
the upstream sweeps finish overnight. Implementing the bonus is a
~1-day standalone task once Phase 6A/6B results are in hand and we
know whether the cross-task signal is worth the extra investment.

To launch this stub manually:
    .venv/bin/python scripts/run_ptbxl_multilabel_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = REPO_ROOT / "data" / "processed" / "ptbxl"


def main() -> int:
    print("=" * 70)
    print("Phase 6 BONUS — multi-label PTB-XL sweep")
    print("=" * 70)

    missing: list[str] = []

    multilabel_y = PROCESSED / "y_multilabel.npy"
    if not multilabel_y.exists():
        missing.append(
            f"{multilabel_y.relative_to(REPO_ROOT)} — needs build step "
            "in src/data/ptbxl.py (multi-label superclass extraction "
            "from scp_codes)"
        )

    # Probe whether label_mode='multilabel' is wired into federated.py.
    fed_path = REPO_ROOT / "src" / "training" / "federated.py"
    if fed_path.exists():
        text = fed_path.read_text()
        if "multilabel" not in text:
            missing.append(
                "src/training/federated.py — no `label_mode='multilabel'` "
                "branch (BCEWithLogitsLoss + macro-AUC path)"
            )

    if missing:
        print("Bonus deferred — required infrastructure missing:\n")
        for item in missing:
            print(f"  - {item}")
        print(
            "\nThis is a planned deferral, documented in DECISIONS.md "
            "(Phase 6 — Cross-dataset benchmark). Phase 6A and 6B "
            "results stand on their own."
        )
        return 0

    print(
        "y_multilabel.npy and the multilabel training branch are both "
        "present, but this script is still a stub. Implement the sweep "
        "loop here once the infrastructure is verified."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
