#!/usr/bin/env python
"""Post-preprocessing sanity checks.

Asserts invariants on the cached arrays and emits two inspection figures:
    results/figures/sanity_class_balance.png
    results/figures/sanity_samples.png
Exits with code 1 on any assertion failure so `make sanity` fails loudly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

FIGURES_DIR = REPO_ROOT / "results" / "figures"


def _load_mitbih(processed_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = processed_root / "mitbih"
    return (
        np.load(d / "X.npy"),
        np.load(d / "y.npy"),
        np.load(d / "record_ids.npy"),
    )


def _load_ptbxl(processed_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = processed_root / "ptbxl"
    return (
        np.load(d / "X.npy"),
        np.load(d / "y.npy"),
        np.load(d / "strat_fold.npy"),
        np.load(d / "ecg_id.npy"),
    )


def _assert_clean(name: str, X: np.ndarray, y: np.ndarray, num_classes: int) -> None:
    assert not np.isnan(X).any(), f"{name}: NaN in X"
    assert not np.isinf(X).any(), f"{name}: inf in X"
    counts = np.bincount(y, minlength=num_classes)
    empty = [i for i, c in enumerate(counts) if c == 0]
    assert not empty, f"{name}: empty classes at indices {empty}"


def _plot_class_balance(
    mitbih_counts: np.ndarray,
    mitbih_names: list[str],
    ptbxl_counts: np.ndarray,
    ptbxl_names: list[str],
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, counts, names, title in [
        (axes[0], mitbih_counts, mitbih_names, "MIT-BIH"),
        (axes[1], ptbxl_counts, ptbxl_names, "PTB-XL"),
    ]:
        bars = ax.bar(names, counts, color="steelblue")
        ax.set_ylabel("count")
        ax.set_title(title)
        for b, c in zip(bars, counts):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), str(int(c)),
                    ha="center", va="bottom", fontsize=8)
    fig.suptitle("Class balance after preprocessing")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def _plot_samples(
    X_mit: np.ndarray, y_mit: np.ndarray, mit_names: list[str],
    X_ptb: np.ndarray, y_ptb: np.ndarray, ptb_names: list[str],
    save_path: Path, n_per_class: int = 5, seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)
    num_classes = len(mit_names)
    fig, axes = plt.subplots(
        2 * num_classes, n_per_class, figsize=(2 * n_per_class, 1.5 * 2 * num_classes),
        sharex=False, sharey=False,
    )
    for cls in range(num_classes):
        # MIT-BIH row
        idx_pool = np.flatnonzero(y_mit == cls)
        sample_idx = rng.choice(idx_pool, size=min(n_per_class, len(idx_pool)), replace=False)
        for j in range(n_per_class):
            ax = axes[cls, j]
            if j < len(sample_idx):
                ax.plot(X_mit[sample_idx[j]], linewidth=0.8, color="navy")
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"MIT-{mit_names[cls]}", fontsize=8)

        # PTB-XL row
        idx_pool = np.flatnonzero(y_ptb == cls)
        sample_idx = rng.choice(idx_pool, size=min(n_per_class, len(idx_pool)), replace=False)
        for j in range(n_per_class):
            ax = axes[num_classes + cls, j]
            if j < len(sample_idx):
                ax.plot(X_ptb[sample_idx[j]], linewidth=0.8, color="darkred")
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"PTB-{ptb_names[cls]}", fontsize=8)

    fig.suptitle("Sample waveforms per class (MIT-BIH top, PTB-XL bottom)")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def main() -> int:
    cfg = load_config(REPO_ROOT / "configs" / "base.yaml")
    logger = get_logger("sanity")

    processed_root = Path(cfg["data"]["processed_root"])
    mit_names = list(cfg["mitbih"]["class_names"])
    ptb_names = list(cfg["ptbxl"]["class_names"])

    X_mit, y_mit, recs = _load_mitbih(processed_root)
    X_ptb, y_ptb, folds, ecg_ids = _load_ptbxl(processed_root)

    # ---- MIT-BIH ----
    mit_total = X_mit.shape[0]
    mit_unique_records = len(np.unique(recs))
    mit_counts = np.bincount(y_mit, minlength=len(mit_names))
    logger.info(
        "MIT-BIH total=%d  records=%d  class_counts=%s",
        mit_total, mit_unique_records, dict(zip(mit_names, mit_counts.tolist())),
    )
    _assert_clean("mitbih", X_mit, y_mit, num_classes=len(mit_names))
    assert X_mit.shape[1] == cfg["mitbih"]["window_samples"], (
        f"mitbih window mismatch: {X_mit.shape[1]} vs {cfg['mitbih']['window_samples']}"
    )
    assert mit_unique_records >= 40, (
        f"expected ≥40 MIT-BIH records processed, got {mit_unique_records}"
    )

    # ---- PTB-XL ----
    ptb_total = X_ptb.shape[0]
    ptb_counts = np.bincount(y_ptb, minlength=len(ptb_names))
    fold_counts = {int(f): int((folds == f).sum()) for f in sorted(np.unique(folds).tolist())}
    logger.info(
        "PTB-XL total=%d  class_counts=%s  fold_counts=%s",
        ptb_total, dict(zip(ptb_names, ptb_counts.tolist())), fold_counts,
    )
    _assert_clean("ptbxl", X_ptb, y_ptb, num_classes=len(ptb_names))
    assert X_ptb.shape[1] == cfg["ptbxl"]["window_samples"], (
        f"ptbxl window mismatch: {X_ptb.shape[1]} vs {cfg['ptbxl']['window_samples']}"
    )
    assert set(fold_counts.keys()) >= set(range(1, 11)), (
        f"PTB-XL folds 1-10 expected; got {sorted(fold_counts.keys())}"
    )

    # ---- Figures ----
    _plot_class_balance(
        mit_counts, mit_names, ptb_counts, ptb_names,
        FIGURES_DIR / "sanity_class_balance.png",
    )
    _plot_samples(
        X_mit, y_mit, mit_names, X_ptb, y_ptb, ptb_names,
        FIGURES_DIR / "sanity_samples.png", n_per_class=5, seed=cfg["seed"],
    )

    summary = {
        "mitbih": {
            "total": int(mit_total),
            "records_processed": int(mit_unique_records),
            "class_counts": dict(zip(mit_names, mit_counts.tolist())),
        },
        "ptbxl": {
            "total": int(ptb_total),
            "class_counts": dict(zip(ptb_names, ptb_counts.tolist())),
            "fold_counts": fold_counts,
        },
    }
    (REPO_ROOT / "results" / "metrics").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "results" / "metrics" / "sanity_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    logger.info("Sanity summary written → results/metrics/sanity_summary.json")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"SANITY FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
