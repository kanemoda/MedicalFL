"""Centralized (non-federated) training loop for MIT-BIH and PTB-XL.

`train_centralized(config, dataset_name)` is the single entry point used by
`scripts/run_experiment.py`. It handles:
  - deterministic seeding,
  - dataset loading + split (70/15/15 stratified for MIT-BIH by sample,
    canonical fold 1-8/9/10 for PTB-XL),
  - class-weighted cross-entropy,
  - best-by-val-F1 checkpointing,
  - JSON metrics + two-panel training-curve figure.
"""
from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.data.augment import PTBXLAugment
from src.evaluation.metrics import (
    compute_metrics,
    format_classification_report,
    metrics_to_jsonable,
)
from src.models.cnn1d import CNN1D
from src.utils.logging import get_logger
from src.utils.seeding import set_all_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"


def _load_mitbih(processed_root: Path):
    d = processed_root / "mitbih"
    X = np.load(d / "X.npy")
    y = np.load(d / "y.npy")
    return X, y


def _load_ptbxl(processed_root: Path, label_mode: str = "binary"):
    d = processed_root / "ptbxl"
    X = np.load(d / "X.npy")
    if label_mode == "binary":
        y_path = d / "y_binary.npy"
    elif label_mode == "5class":
        y_path = d / "y_5class.npy"
    else:
        raise ValueError(f"unknown ptbxl label_mode: {label_mode!r}")
    if not y_path.exists():
        legacy = d / "y.npy"
        if label_mode == "5class" and legacy.exists():
            y_path = legacy
        else:
            raise FileNotFoundError(
                f"expected {y_path}; regenerate via scripts/preprocess_data.py"
            )
    y = np.load(y_path)
    folds = np.load(d / "strat_fold.npy")
    return X, y, folds


def _split_mitbih(X: np.ndarray, y: np.ndarray, seed: int):
    """70/15/15 stratified split by sample (per Phase 2 spec)."""
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def _split_ptbxl(X: np.ndarray, y: np.ndarray, folds: np.ndarray):
    """Canonical PTB-XL split: folds 1-8 train, 9 val, 10 test."""
    train_mask = (folds >= 1) & (folds <= 8)
    val_mask = folds == 9
    test_mask = folds == 10
    return (
        (X[train_mask], y[train_mask]),
        (X[val_mask], y[val_mask]),
        (X[test_mask], y[test_mask]),
    )


def _class_weights_from_train(y_train: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
    safe = np.maximum(counts, 1.0)
    return counts.sum() / (num_classes * safe)


def _make_loader(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, num_workers: int,
    drop_last: bool = False,
) -> DataLoader:
    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(y).long()
    ds = TensorDataset(Xt, yt)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
              criterion: nn.Module, class_names: list[str],
              aug: nn.Module | None = None) -> tuple[dict, float]:
    model.eval()
    if aug is not None:
        aug.eval()
    all_probs: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    total_loss = 0.0
    total_n = 0
    for Xb, yb in loader:
        Xb = Xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        if aug is not None:
            Xb = aug(Xb)
        logits = model(Xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * Xb.size(0)
        total_n += Xb.size(0)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        all_probs.append(probs)
        all_pred.append(preds)
        all_true.append(yb.cpu().numpy())
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_probs)
    metrics = compute_metrics(y_true, y_pred, y_prob, class_names)
    return metrics, total_loss / max(total_n, 1)


def _plot_curves(history: list[dict], out_path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_f1 = [h["train_f1_macro"] for h in history]
    val_f1 = [h["val_f1_macro"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, train_loss, label="train")
    axes[0].plot(epochs, val_loss, label="val")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("CE loss"); axes[0].set_title("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, train_f1, label="train")
    axes[1].plot(epochs, val_f1, label="val")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("F1 macro"); axes[1].set_title("F1 macro")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def train_centralized(config: dict, dataset_name: str) -> dict:
    set_all_seeds(config["seed"], strict=bool(config.get("strict_determinism", False)))
    run_id = config.get("run_id", f"centralized_{dataset_name}_seed{config['seed']}")
    logger = get_logger(run_id)

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"run_id={run_id}  device={device}  dataset={dataset_name}")

    processed_root = Path(config["data"]["processed_root"])
    if dataset_name == "mitbih":
        X, y = _load_mitbih(processed_root)
        (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = _split_mitbih(X, y, config["seed"])
        class_names = list(config["mitbih"]["class_names"])
        num_classes = config["mitbih"]["num_classes"]
    elif dataset_name == "ptbxl":
        label_mode = str(config["ptbxl"].get("label_mode", "binary"))
        X, y, folds = _load_ptbxl(processed_root, label_mode=label_mode)
        (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = _split_ptbxl(X, y, folds)
        class_names = list(config["ptbxl"]["class_names"])
        num_classes = config["ptbxl"]["num_classes"]
        logger.info(f"ptbxl label_mode={label_mode} num_classes={num_classes}")
    else:
        raise ValueError(f"unknown dataset_name: {dataset_name}")

    logger.info(
        f"split sizes: train={len(y_tr)}  val={len(y_va)}  test={len(y_te)}"
    )

    train_loader = _make_loader(
        X_tr, y_tr, batch_size=config["training"]["batch_size"], shuffle=True,
        num_workers=config["num_workers"], drop_last=True,
    )
    val_loader = _make_loader(
        X_va, y_va, batch_size=config["training"]["batch_size"], shuffle=False,
        num_workers=config["num_workers"], drop_last=False,
    )
    test_loader = _make_loader(
        X_te, y_te, batch_size=config["training"]["batch_size"], shuffle=False,
        num_workers=config["num_workers"], drop_last=False,
    )

    class_weights = _class_weights_from_train(y_tr, num_classes)
    cw_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    logger.info(f"class_weights={class_weights.tolist()}")

    model = CNN1D(num_classes=num_classes).to(device)
    logger.info(repr(model))

    aug: nn.Module | None = None
    if dataset_name == "ptbxl" and bool(config["training"].get("augment", False)):
        aug = PTBXLAugment(
            crop_length=int(config["training"].get("crop_length", 900)),
            noise_std=float(config["training"].get("noise_std", 0.02)),
            baseline_max=float(config["training"].get("baseline_max", 0.05)),
            prob=float(config["training"].get("aug_prob", 0.5)),
        ).to(device)
        logger.info(
            f"PTBXLAugment enabled: crop={aug.crop_length} noise_std={aug.noise_std} "
            f"baseline_max={aug.baseline_max} prob={aug.prob}"
        )

    optim_name = config["training"].get("optimizer", "adamw").lower()
    if optim_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["training"]["lr"],
            weight_decay=config["training"].get("weight_decay", 1e-4),
        )
    elif optim_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["training"]["lr"],
            weight_decay=config["training"].get("weight_decay", 1e-4),
        )
    else:
        raise ValueError(f"unsupported optimizer: {optim_name}")
    criterion = nn.CrossEntropyLoss(weight=cw_tensor)

    epochs = int(config["training"]["epochs"])
    scheduler_name = config["training"].get("scheduler", "none").lower()
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == "none":
        scheduler = None
    else:
        raise ValueError(f"unsupported scheduler: {scheduler_name}")

    ckpt_dir = RESULTS / "checkpoints" / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_f1 = -math.inf
    history: list[dict] = []
    patience = int(config["training"].get("early_stop_patience", 0))
    epochs_since_improvement = 0
    early_stopped_at: int | None = None

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        if aug is not None:
            aug.train()
        running_loss = 0.0
        running_n = 0
        preds_buf: list[np.ndarray] = []
        true_buf: list[np.ndarray] = []
        prob_buf: list[np.ndarray] = []
        for Xb, yb in train_loader:
            Xb = Xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if aug is not None:
                Xb = aug(Xb)
            optimizer.zero_grad(set_to_none=True)
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * Xb.size(0)
            running_n += Xb.size(0)
            with torch.no_grad():
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            preds_buf.append(probs.argmax(axis=1))
            true_buf.append(yb.detach().cpu().numpy())
            prob_buf.append(probs)

        train_y_true = np.concatenate(true_buf)
        train_y_pred = np.concatenate(preds_buf)
        train_y_prob = np.concatenate(prob_buf)
        train_metrics = compute_metrics(
            train_y_true, train_y_pred, train_y_prob, class_names
        )
        train_loss = running_loss / max(running_n, 1)

        val_metrics, val_loss = _evaluate(
            model, val_loader, device, criterion, class_names, aug=aug
        )

        if scheduler is not None:
            scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_f1_macro": train_metrics["f1_macro"],
                "val_f1_macro": val_metrics["f1_macro"],
                "val_accuracy": val_metrics["accuracy"],
            }
        )
        logger.info(
            f"epoch {epoch:3d}/{epochs}  train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  train_f1={train_metrics['f1_macro']:.4f}  "
            f"val_f1={val_metrics['f1_macro']:.4f}"
        )

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            epochs_since_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_f1_macro": best_val_f1,
                    "class_weights": class_weights.tolist(),
                },
                ckpt_dir / "best.pt",
            )
        else:
            epochs_since_improvement += 1
            if patience > 0 and epochs_since_improvement >= patience:
                logger.info(
                    f"early stop at epoch {epoch}: no val_f1 improvement "
                    f"for {patience} epochs (best {best_val_f1:.4f})"
                )
                early_stopped_at = epoch
                break

    runtime = time.time() - t0
    logger.info(f"training done in {runtime:.1f} s  best_val_f1={best_val_f1:.4f}")

    # Reload best, test.
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_metrics, test_loss = _evaluate(
        model, test_loader, device, criterion, class_names, aug=aug
    )
    logger.info("TEST classification report:\n" + format_classification_report(test_metrics))

    # Figure + JSON.
    fig_path = RESULTS / "figures" / f"{run_id}_training.png"
    _plot_curves(history, fig_path)

    cuda_version = torch.version.cuda or "none"
    result = {
        "run_id": run_id,
        "dataset": dataset_name,
        "config": config,
        "train_history": history,
        "test_metrics": metrics_to_jsonable(test_metrics),
        "test_loss": test_loss,
        "best_val_f1_macro": best_val_f1,
        "best_checkpoint_epoch": int(ckpt["epoch"]),
        "runtime_seconds": runtime,
        "early_stopped_at": early_stopped_at,
        "git_sha": _git_sha(),
        "torch_version": torch.__version__,
        "cuda_version": cuda_version,
        "class_weights": class_weights.tolist(),
        "train_size": int(len(y_tr)),
        "val_size": int(len(y_va)),
        "test_size": int(len(y_te)),
        "curves_figure": str(fig_path.relative_to(REPO_ROOT)),
    }
    metrics_path = RESULTS / "metrics" / f"{run_id}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2))
    logger.info(f"wrote metrics → {metrics_path}")
    return result
