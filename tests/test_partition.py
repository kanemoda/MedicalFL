"""Tests for non-IID partitioning strategies."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.partition import (  # noqa: E402
    partition_dirichlet,
    partition_iid,
    partition_label_skew,
    partition_quantity_skew,
)


def _synthetic(n: int = 5000, num_classes: int = 5, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 10)).astype(np.float32)
    y = rng.integers(0, num_classes, size=n, dtype=np.int64)
    return X, y


# ---------------------------------------------------------------------------
# IID
# ---------------------------------------------------------------------------

def test_iid_preserves_samples():
    X, y = _synthetic()
    parts = partition_iid(X, y, num_clients=5, seed=42)
    total = sum(len(p[1]) for p in parts)
    assert total == len(y)
    # No overlap between clients.
    all_idx = np.concatenate([p[2] for p in parts])
    assert len(all_idx) == len(y)
    assert len(np.unique(all_idx)) == len(y)


def test_iid_class_balance_within_5_percent():
    X, y = _synthetic(n=20_000)
    parts = partition_iid(X, y, num_clients=5, seed=42)
    num_classes = int(y.max()) + 1
    global_dist = np.bincount(y, minlength=num_classes) / len(y)
    for _, y_c, _ in parts:
        assert y_c.size > 0
        local_dist = np.bincount(y_c, minlength=num_classes) / len(y_c)
        assert np.max(np.abs(local_dist - global_dist)) < 0.05


# ---------------------------------------------------------------------------
# Label skew
# ---------------------------------------------------------------------------

def test_label_skew_limited_classes():
    X, y = _synthetic()
    parts = partition_label_skew(X, y, num_clients=5, seed=42, classes_per_client=2)
    for _, y_c, _ in parts:
        if y_c.size == 0:
            continue
        assert len(np.unique(y_c)) <= 2


# ---------------------------------------------------------------------------
# Quantity skew
# ---------------------------------------------------------------------------

def test_quantity_skew_preserves_samples():
    X, y = _synthetic()
    parts = partition_quantity_skew(X, y, num_clients=5, seed=42, beta=0.5)
    total = sum(len(p[1]) for p in parts)
    assert total == len(y)


# ---------------------------------------------------------------------------
# Dirichlet
# ---------------------------------------------------------------------------

def test_dirichlet_alpha_large_approaches_iid():
    X, y = _synthetic(n=20_000)
    parts = partition_dirichlet(X, y, num_clients=5, seed=42, alpha=100.0)
    num_classes = int(y.max()) + 1
    global_dist = np.bincount(y, minlength=num_classes) / len(y)
    for _, y_c, _ in parts:
        assert y_c.size > 0
        local_dist = np.bincount(y_c, minlength=num_classes) / len(y_c)
        assert np.max(np.abs(local_dist - global_dist)) < 0.10


def test_dirichlet_alpha_small_is_skewed():
    X, y = _synthetic(n=20_000)
    parts = partition_dirichlet(X, y, num_clients=5, seed=42, alpha=0.1)
    num_classes = int(y.max()) + 1

    # For at least one class, a single client must hold > 80% of that class.
    per_class_max_share = []
    for cls in range(num_classes):
        cls_counts = np.array([(y_c == cls).sum() for _, y_c, _ in parts])
        total = cls_counts.sum()
        if total == 0:
            per_class_max_share.append(0.0)
        else:
            per_class_max_share.append(cls_counts.max() / total)
    assert max(per_class_max_share) > 0.8


def test_partitioning_is_deterministic_given_seed():
    X, y = _synthetic()
    p1 = partition_dirichlet(X, y, num_clients=5, seed=7, alpha=0.3)
    p2 = partition_dirichlet(X, y, num_clients=5, seed=7, alpha=0.3)
    for (_, y1, i1), (_, y2, i2) in zip(p1, p2):
        assert np.array_equal(y1, y2)
        assert np.array_equal(i1, i2)
