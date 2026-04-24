#!/usr/bin/env python
"""Dry-run all 6 Phase 4 partition strategies on MIT-BIH.

Loads `data/processed/mitbih/X.npy + y.npy`, carves the standard 85/15
stratified central-test split, then partitions the 85% pool across 5 clients
with each strategy. Prints per-client class counts to stdout and saves one
stacked-bar figure per strategy under `results/figures/`.

Sanity check: fails (exit 1) if any client ends up with zero total samples
under any strategy — that would be a partition-function bug worth fixing
before the 24-run sweep.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.partition import (  # noqa: E402
    partition_dirichlet,
    partition_iid,
    partition_label_skew,
    partition_quantity_skew,
    plot_partition_distribution,
)
from src.utils.config import load_config  # noqa: E402

CLASS_NAMES = ["N", "S", "V", "F", "Q"]
NUM_CLIENTS = 5
SEED = 42
CENTRAL_TEST_FRAC = 0.15


STRATEGIES = [
    ("iid",            partition_iid,            {}),
    ("label_skew_c2",  partition_label_skew,     {"classes_per_client": 2}),
    ("quantity_skew",  partition_quantity_skew,  {"beta": 0.5}),
    ("dirichlet_a01",  partition_dirichlet,      {"alpha": 0.1}),
    ("dirichlet_a05",  partition_dirichlet,      {"alpha": 0.5}),
    ("dirichlet_a10",  partition_dirichlet,      {"alpha": 1.0}),
]


def _load_pool() -> tuple[np.ndarray, np.ndarray]:
    cfg = load_config(REPO_ROOT / "configs" / "base.yaml")
    processed = Path(cfg["data"]["processed_root"]) / "mitbih"
    X = np.load(processed / "X.npy")
    y = np.load(processed / "y.npy")
    # 85/15 stratified to match the federated driver's split logic.
    X_pool, _X_test, y_pool, _y_test = train_test_split(
        X, y, test_size=CENTRAL_TEST_FRAC, stratify=y, random_state=SEED,
    )
    return X_pool, y_pool


def _print_counts(name: str, partitions, num_classes: int) -> int:
    print(f"\n=== {name} ===")
    empty_clients = 0
    for c_id, (_, y_c, _) in enumerate(partitions):
        if y_c.size == 0:
            print(f"  client {c_id}: EMPTY ({y_c.size} samples)")
            empty_clients += 1
            continue
        counts = np.bincount(y_c, minlength=num_classes).tolist()
        frac = np.bincount(y_c, minlength=num_classes) / max(y_c.size, 1)
        frac_str = " ".join(f"{f:.2%}" for f in frac)
        print(
            f"  client {c_id}: N_total={y_c.size:>6d}  "
            f"counts={counts}  frac=[{frac_str}]"
        )
    return empty_clients


def main() -> int:
    X, y = _load_pool()
    num_classes = int(y.max()) + 1
    print(f"pool size={len(y)}  classes={num_classes}  seed={SEED}  num_clients={NUM_CLIENTS}")
    print(f"global counts: {np.bincount(y, minlength=num_classes).tolist()}")

    fig_dir = REPO_ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    total_empty = 0
    for name, fn, kwargs in STRATEGIES:
        partitions = fn(X, y, num_clients=NUM_CLIENTS, seed=SEED, **kwargs)
        empty_count = _print_counts(name, partitions, num_classes)
        total_empty += empty_count
        fig_path = fig_dir / f"phase4_partition_{name}.png"
        plot_partition_distribution(
            partitions, CLASS_NAMES, fig_path,
            title=f"Phase 4 partition: {name}",
        )
        print(f"  → {fig_path.relative_to(REPO_ROOT)}")

    if total_empty > 0:
        print(f"\nFAIL: {total_empty} client(s) received zero total samples")
        return 1
    print("\nOK: all strategies produced non-empty clients.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
