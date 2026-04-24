#!/usr/bin/env python
"""Retrain the 2 FedBN runs needed for the local-test comparison.

The Phase 4 FedBN runs for `dirichlet_a01` and `label_skew_c2` finished all
50 training rounds successfully but crashed at the reporting step, so no
checkpoint was written. This script retrains those two runs using the exact
Phase 4 sweep config (same seed, same partitioner kwargs), with `_v2`
appended to the run_id so the Phase 4 artifacts are left untouched.

The extended `train_federated` (see src/training/federated.py) now emits a
`local_test` block into the final JSON, so these two retrains automatically
produce the per-client local test metrics needed to complete the 4×2
central-vs-local comparison.

Total wall clock: ~1 hour on a single GPU.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_noniid_sweep import _build_run_config  # noqa: E402
from src.training.federated import train_federated  # noqa: E402
from src.utils.config import load_config  # noqa: E402


TARGETS = [
    {"name": "dirichlet_a01", "fn": "dirichlet", "kwargs": {"alpha": 0.1}},
    {"name": "label_skew_c2", "fn": "label_skew", "kwargs": {"classes_per_client": 2}},
]


def main() -> int:
    cfg = load_config(REPO_ROOT / "configs" / "sweep_noniid.yaml")

    t0 = time.time()
    for i, partition in enumerate(TARGETS, 1):
        run_cfg, run_id = _build_run_config(cfg, partition, "fedbn")
        run_id_v2 = f"{run_id}_v2"
        run_cfg["run_id"] = run_id_v2

        result_json = REPO_ROOT / "results" / "metrics" / f"{run_id_v2}.json"
        if result_json.exists():
            print(f"[SKIP {i}/{len(TARGETS)}] {run_id_v2} — JSON already exists")
            continue

        print(f"\n[START {i}/{len(TARGETS)}] {run_id_v2}", flush=True)
        t_run = time.time()
        result = train_federated(run_cfg, "mitbih")
        elapsed = time.time() - t_run
        lt = result.get("local_test", {})
        print(
            f"[DONE {i}/{len(TARGETS)}] {run_id_v2} · "
            f"best_central_f1={result.get('best_central_f1_macro', float('nan')):.4f} · "
            f"mean_local_f1={lt.get('local_test_f1_mean', float('nan')):.4f} · "
            f"{elapsed / 60:.1f} min",
            flush=True,
        )

    total = time.time() - t0
    print(f"\n[COMPLETE] 2 v2 retrains in {total / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
