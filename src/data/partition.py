"""Client-data partitioning strategies for federated learning.

Each partitioner returns `list[tuple[X_i, y_i, idx_i]]` — one tuple per client,
where `idx_i` indexes into the original arrays so later code can recover the
global position of every sample (needed for patient/record-wise logic).

Empty client shards (can happen under extreme skew) are logged as warnings
rather than raised as errors.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

Partition = list[tuple[np.ndarray, np.ndarray, np.ndarray]]

_logger = logging.getLogger(__name__)


def _gather(X: np.ndarray, y: np.ndarray, assignments: list[np.ndarray]) -> Partition:
    """Convert a list-of-index-arrays (one per client) into full (X, y, idx) tuples."""
    out: Partition = []
    for client_id, idx in enumerate(assignments):
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            _logger.warning("client %d received zero samples", client_id)
        out.append((X[idx], y[idx], idx))
    return out


# ---------------------------------------------------------------------------
# 1. IID — stratified equal split
# ---------------------------------------------------------------------------

def partition_iid(
    X: np.ndarray, y: np.ndarray, num_clients: int, seed: int
) -> Partition:
    """Stratified equal split: each class is split into `num_clients` roughly-equal chunks."""
    rng = np.random.default_rng(seed)
    assignments: list[list[int]] = [[] for _ in range(num_clients)]
    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        rng.shuffle(cls_idx)
        for chunk_id, chunk in enumerate(np.array_split(cls_idx, num_clients)):
            assignments[chunk_id].extend(chunk.tolist())
    return _gather(X, y, [np.asarray(a, dtype=np.int64) for a in assignments])


# ---------------------------------------------------------------------------
# 2. Label skew — each client sees only `classes_per_client` classes
# ---------------------------------------------------------------------------

def partition_label_skew(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    seed: int,
    classes_per_client: int = 2,
) -> Partition:
    """Assign each client a random subset of `classes_per_client` classes, then split.

    Each class's samples are divided equally across the clients that hold it.
    Classes that end up with no client (unlikely for small K) are logged.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    if classes_per_client > len(classes):
        raise ValueError(
            f"classes_per_client={classes_per_client} > num_classes={len(classes)}"
        )

    client_classes: list[np.ndarray] = [
        rng.choice(classes, size=classes_per_client, replace=False)
        for _ in range(num_clients)
    ]

    assignments: list[list[int]] = [[] for _ in range(num_clients)]
    for cls in classes:
        holders = [c for c in range(num_clients) if cls in client_classes[c]]
        if not holders:
            _logger.warning("class %s has no client; skipping", cls)
            continue
        cls_idx = np.flatnonzero(y == cls)
        rng.shuffle(cls_idx)
        for chunk_id, chunk in enumerate(np.array_split(cls_idx, len(holders))):
            assignments[holders[chunk_id]].extend(chunk.tolist())
    return _gather(X, y, [np.asarray(a, dtype=np.int64) for a in assignments])


# ---------------------------------------------------------------------------
# 3. Quantity skew — IID labels, but client sizes drawn from Dir(beta)
# ---------------------------------------------------------------------------

def partition_quantity_skew(
    X: np.ndarray, y: np.ndarray, num_clients: int, seed: int, beta: float = 0.5
) -> Partition:
    """Stratified split where client *sizes* follow Dir(beta, num_clients).

    Class proportions per client remain roughly IID; only the total amount of
    data per client is skewed.
    """
    rng = np.random.default_rng(seed)
    size_weights = rng.dirichlet([beta] * num_clients)

    assignments: list[list[int]] = [[] for _ in range(num_clients)]
    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        rng.shuffle(cls_idx)
        cum = np.cumsum(size_weights)[:-1]
        splits = (cum * len(cls_idx)).astype(int)
        chunks = np.split(cls_idx, splits)
        for c_id, chunk in enumerate(chunks):
            assignments[c_id].extend(chunk.tolist())
    return _gather(X, y, [np.asarray(a, dtype=np.int64) for a in assignments])


# ---------------------------------------------------------------------------
# 4. Dirichlet — per-class proportions ~ Dir(alpha) (Yurochkin et al. 2019)
# ---------------------------------------------------------------------------

def partition_dirichlet(
    X: np.ndarray, y: np.ndarray, num_clients: int, seed: int, alpha: float = 0.5
) -> Partition:
    """Classic Dir(alpha) non-IID split.

    For each class, sample a length-`num_clients` proportion vector from
    Dir(alpha, ..., alpha) and distribute that class's indices accordingly.
    Small alpha → heavily skewed; large alpha → approaches IID.
    """
    rng = np.random.default_rng(seed)
    assignments: list[list[int]] = [[] for _ in range(num_clients)]
    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        rng.shuffle(cls_idx)
        props = rng.dirichlet([alpha] * num_clients)
        cum = np.cumsum(props)[:-1]
        splits = (cum * len(cls_idx)).astype(int)
        chunks = np.split(cls_idx, splits)
        for c_id, chunk in enumerate(chunks):
            assignments[c_id].extend(chunk.tolist())
    return _gather(X, y, [np.asarray(a, dtype=np.int64) for a in assignments])


# ---------------------------------------------------------------------------
# Visualization — stacked bar chart of per-client class distribution
# ---------------------------------------------------------------------------

def plot_partition_distribution(
    partitions: Partition,
    class_names: list[str],
    save_path: str | Path,
    title: str = "Client class distribution",
) -> None:
    """Save a stacked bar chart with one bar per client, stacked by class count."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    num_clients = len(partitions)
    num_classes = len(class_names)
    counts = np.zeros((num_clients, num_classes), dtype=np.int64)
    for c_id, (_, y_c, _) in enumerate(partitions):
        if y_c.size:
            bc = np.bincount(y_c, minlength=num_classes)
            counts[c_id] = bc[:num_classes]

    fig, ax = plt.subplots(figsize=(max(6, num_clients * 0.8), 4))
    bottom = np.zeros(num_clients, dtype=np.int64)
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    for cls_id in range(num_classes):
        ax.bar(
            range(num_clients),
            counts[:, cls_id],
            bottom=bottom,
            label=class_names[cls_id],
            color=colors[cls_id],
        )
        bottom += counts[:, cls_id]
    ax.set_xlabel("client")
    ax.set_ylabel("sample count")
    ax.set_title(title)
    ax.set_xticks(range(num_clients))
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
