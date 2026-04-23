"""Federated training driver — single entry point `train_federated`.

Input: a YAML-loaded config dict plus dataset name ("mitbih" or "ptbxl").

Flow:
  1. Load processed arrays.
  2. Carve a *central* test set (15% stratified for MIT-BIH, PTB-XL fold 10 for
     the binary task). Remaining data is available for client partitioning.
  3. Partition the remainder across `num_clients` clients using the strategy
     named by `config['federation']['partition']` (IID by default).
  4. Per-client 80/20 stratified train/val split.
  5. Instantiate `FederatedClient` objects + `FederationServer`, run
     `rounds` rounds of `local_epochs` each.
  6. Evaluate on the central test set after every `central_eval_every` rounds
     (and always at the end). For FedBN we log both 'representative' and
     'client_avg' evaluation modes.
  7. Write `rounds.csv`, `partition.png`, training-curves figure, and final
     metrics JSON.

Results layout (per `run_id`):
    results/
      checkpoints/<run_id>/final.pt
      metrics/<run_id>.json
      metrics/<run_id>_rounds.csv
      figures/<run_id>_partition.png
      figures/<run_id>_fed_curves.png
      logs/<run_id>.log
"""
from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.data.partition import (
    partition_dirichlet,
    partition_iid,
    partition_label_skew,
    partition_quantity_skew,
    plot_partition_distribution,
)
from src.evaluation.metrics import (
    compute_metrics,
    format_classification_report,
    metrics_to_jsonable,
)
from src.federation.client import FederatedClient
from src.federation.server import FederationServer
from src.models.cnn1d import CNN1D
from src.utils.logging import get_logger
from src.utils.seeding import set_all_seeds

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"

CENTRAL_TEST_FRAC_MITBIH = 0.15


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_mitbih(processed_root: Path) -> tuple[np.ndarray, np.ndarray]:
    d = processed_root / "mitbih"
    X = np.load(d / "X.npy")
    y = np.load(d / "y.npy")
    return X, y


def _load_ptbxl(
    processed_root: Path, label_mode: str = "binary"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


# ---------------------------------------------------------------------------
# Central test / remainder carving
# ---------------------------------------------------------------------------

def _carve_mitbih(X: np.ndarray, y: np.ndarray, seed: int):
    """Stratified 85/15 split: 85% → clients, 15% → central test."""
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y,
        test_size=CENTRAL_TEST_FRAC_MITBIH,
        stratify=y,
        random_state=seed,
    )
    return (X_pool, y_pool), (X_test, y_test)


def _carve_ptbxl(X: np.ndarray, y: np.ndarray, folds: np.ndarray):
    """PTB-XL federated split: folds 1-9 → clients, fold 10 → central test."""
    client_mask = (folds >= 1) & (folds <= 9)
    test_mask = folds == 10
    return (X[client_mask], y[client_mask]), (X[test_mask], y[test_mask])


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------

def _partition(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    strategy: str,
    seed: int,
    *,
    alpha: float = 0.5,
    beta: float = 0.5,
    classes_per_client: int = 2,
):
    s = strategy.lower()
    if s == "iid":
        return partition_iid(X, y, num_clients, seed)
    if s == "label_skew":
        return partition_label_skew(
            X, y, num_clients, seed, classes_per_client=classes_per_client
        )
    if s == "quantity_skew":
        return partition_quantity_skew(X, y, num_clients, seed, beta=beta)
    if s == "dirichlet":
        return partition_dirichlet(X, y, num_clients, seed, alpha=alpha)
    raise ValueError(f"unknown partition strategy: {strategy!r}")


def _client_train_val_split(
    X_c: np.ndarray, y_c: np.ndarray, seed: int, val_frac: float = 0.2,
):
    """80/20 stratified per-client split. Falls back to random split if a class
    has fewer than 2 samples (stratify rejects that)."""
    if y_c.size < 5:
        cut = max(1, int(round(y_c.size * (1 - val_frac))))
        return (X_c[:cut], y_c[:cut]), (X_c[cut:], y_c[cut:])
    classes, counts = np.unique(y_c, return_counts=True)
    if counts.min() < 2:
        return train_test_split(
            X_c, y_c, test_size=val_frac, random_state=seed,
        )[::2], train_test_split(
            X_c, y_c, test_size=val_frac, random_state=seed,
        )[1::2]  # defensive fallback — rarely hit
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_c, y_c, test_size=val_frac, stratify=y_c, random_state=seed,
    )
    return (X_tr, y_tr), (X_va, y_va)


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def _class_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    safe = np.maximum(counts, 1.0)
    return counts.sum() / (num_classes * safe)


def _make_loader(
    X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, num_workers: int,
) -> DataLoader:
    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(y).long()
    ds = TensorDataset(Xt, yt)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def _plot_fed_curves(rounds: list[dict], out_path: Path) -> None:
    rs = [r["round"] for r in rounds]
    val_f1 = [r["val_f1_weighted"] for r in rounds]
    test_f1 = [r.get("central_f1_macro", float("nan")) for r in rounds]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(rs, val_f1, label="client-weighted val")
    axes[0].plot(
        [r for r, f in zip(rs, test_f1) if f == f],
        [f for f in test_f1 if f == f],
        label="central test", marker="o",
    )
    axes[0].set_xlabel("round"); axes[0].set_ylabel("F1 macro")
    axes[0].set_title("Federated F1 macro"); axes[0].grid(alpha=0.3); axes[0].legend()

    per_client = np.array([r["val_f1_per_client"] for r in rounds])  # (R, K)
    for k in range(per_client.shape[1]):
        axes[1].plot(rs, per_client[:, k], label=f"client {k}", alpha=0.8)
    axes[1].set_xlabel("round"); axes[1].set_ylabel("val F1 macro")
    axes[1].set_title("Per-client val F1"); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def train_federated(config: dict, dataset_name: str) -> dict:
    set_all_seeds(config["seed"], strict=bool(config.get("strict_determinism", False)))
    run_id = config.get(
        "run_id", f"federated_{dataset_name}_{config['federation']['strategy']}_seed{config['seed']}"
    )
    logger = get_logger(run_id)

    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"run_id={run_id}  device={device}  dataset={dataset_name}")

    processed_root = Path(config["data"]["processed_root"])
    if dataset_name == "mitbih":
        X, y = _load_mitbih(processed_root)
        (X_pool, y_pool), (X_test, y_test) = _carve_mitbih(X, y, config["seed"])
        class_names = list(config["mitbih"]["class_names"])
        num_classes = int(config["mitbih"]["num_classes"])
    elif dataset_name == "ptbxl":
        label_mode = str(config["ptbxl"].get("label_mode", "binary"))
        X, y, folds = _load_ptbxl(processed_root, label_mode=label_mode)
        (X_pool, y_pool), (X_test, y_test) = _carve_ptbxl(X, y, folds)
        class_names = list(config["ptbxl"]["class_names"])
        num_classes = int(config["ptbxl"]["num_classes"])
        logger.info(f"ptbxl label_mode={label_mode}")
    else:
        raise ValueError(f"unknown dataset_name: {dataset_name}")

    fed_cfg = config["federation"]
    num_clients = int(fed_cfg.get("num_clients", 5))
    rounds = int(fed_cfg.get("rounds", 50))
    local_epochs = int(fed_cfg.get("local_epochs", 5))
    strategy = str(fed_cfg.get("strategy", "fedavg")).lower()
    partition_strategy = str(fed_cfg.get("partition", "iid")).lower()
    proximal_mu = float(fed_cfg.get("proximal_mu", 0.0))
    fedperf_alpha = float(fed_cfg.get("fedperf_alpha", 0.5))
    fedbn_central_eval = str(fed_cfg.get("fedbn_central_eval", "representative"))
    central_eval_every = int(fed_cfg.get("central_eval_every", 1))
    partition_kwargs = {
        "alpha": float(fed_cfg.get("dirichlet_alpha", 0.5)),
        "beta": float(fed_cfg.get("quantity_beta", 0.5)),
        "classes_per_client": int(fed_cfg.get("classes_per_client", 2)),
    }

    logger.info(
        f"federation: strategy={strategy} partition={partition_strategy} "
        f"clients={num_clients} rounds={rounds} local_epochs={local_epochs} "
        f"proximal_mu={proximal_mu}"
    )
    logger.info(
        f"pool size={len(y_pool)}  central test size={len(y_test)}  "
        f"class_names={class_names}"
    )

    # --- Partition ---
    partitions = _partition(
        X_pool, y_pool, num_clients, partition_strategy,
        seed=config["seed"], **partition_kwargs,
    )
    partition_fig = RESULTS / "figures" / f"{run_id}_partition.png"
    plot_partition_distribution(
        partitions, class_names, partition_fig,
        title=f"{run_id} — {partition_strategy} partition",
    )

    # --- Class weights from pooled train portion across all clients ---
    # We split each client 80/20 first, then compute weights on the union of
    # train shards (matches centralized semantics: weights computed on train).
    train_ys: list[np.ndarray] = []
    client_splits: list[tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]] = []
    for c_id, (X_c, y_c, _idx_c) in enumerate(partitions):
        if y_c.size == 0:
            logger.warning(f"client {c_id} empty — skipping")
            client_splits.append(((np.empty((0, X_c.shape[1]), dtype=X_c.dtype),
                                   np.empty((0,), dtype=np.int64)),
                                  (np.empty((0, X_c.shape[1]), dtype=X_c.dtype),
                                   np.empty((0,), dtype=np.int64))))
            continue
        (X_tr, y_tr), (X_va, y_va) = _client_train_val_split(
            X_c, y_c, seed=config["seed"] + c_id,
        )
        client_splits.append(((X_tr, y_tr), (X_va, y_va)))
        train_ys.append(y_tr)
    pooled_y_train = (
        np.concatenate(train_ys) if train_ys else np.empty((0,), dtype=np.int64)
    )
    class_weights = _class_weights(pooled_y_train, num_classes)
    logger.info(f"pooled class weights: {class_weights.tolist()}")

    # --- Build clients ---
    model_fn: Callable[[], torch.nn.Module] = lambda: CNN1D(num_classes=num_classes)
    clients: list[FederatedClient] = []
    for c_id, ((X_tr, y_tr), (X_va, y_va)) in enumerate(client_splits):
        if y_tr.size == 0:
            continue
        c = FederatedClient(
            client_id=c_id,
            X_train=X_tr, y_train=y_tr,
            X_val=X_va, y_val=y_va,
            model_fn=model_fn,
            num_classes=num_classes,
            class_names=class_names,
            device=device,
            batch_size=int(config["training"]["batch_size"]),
            lr=float(config["training"]["lr"]),
            weight_decay=float(config["training"].get("weight_decay", 1e-4)),
            optimizer=str(config["training"].get("optimizer", "adamw")),
            num_workers=int(config.get("num_workers", 0)),
            class_weights=class_weights,
        )
        clients.append(c)
        logger.info(
            f"client {c_id}: train={len(y_tr)}  val={len(y_va)}  "
            f"train_class_counts={np.bincount(y_tr, minlength=num_classes).tolist()}"
        )
    if not clients:
        raise RuntimeError("no non-empty clients after partitioning")

    # --- Central test loader ---
    test_loader = _make_loader(
        X_test, y_test,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
    )

    # --- Server ---
    server = FederationServer(
        clients=clients,
        model_fn=model_fn,
        strategy=strategy,
        device=device,
        class_names=class_names,
        proximal_mu=proximal_mu,
        fedperf_alpha=fedperf_alpha,
        fedbn_central_eval=fedbn_central_eval,
        logger=logger,
    )

    # --- Rounds ---
    round_records: list[dict] = []
    csv_path = RESULTS / "metrics" / f"{run_id}_rounds.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Write header upfront; we re-open append mode per round so partial runs
    # still leave usable CSV on disk.
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "round", "val_f1_weighted",
            *[f"client{i}_val_f1" for i in range(len(clients))],
            *[f"client{i}_train_loss" for i in range(len(clients))],
            "central_f1_macro", "central_accuracy", "central_auc_macro",
            "central_mode",
        ])

    best_central_f1 = -float("inf")
    best_round = 0
    final_central_metrics: dict | None = None

    t0 = time.time()
    for r in range(1, rounds + 1):
        round_info = server.run_round(local_epochs=local_epochs)
        record: dict = {"round": r, **round_info}

        do_central = (r % central_eval_every == 0) or (r == rounds)
        central_summary = {
            "central_f1_macro": float("nan"),
            "central_accuracy": float("nan"),
            "central_auc_macro": float("nan"),
            "central_mode": "",
        }
        if do_central:
            central = server.evaluate_central(test_loader)
            record["central_metrics"] = central
            central_summary = {
                "central_f1_macro": float(central["f1_macro"]),
                "central_accuracy": float(central["accuracy"]),
                "central_auc_macro": (
                    float(central["auc_macro"])
                    if central["auc_macro"] == central["auc_macro"]
                    else float("nan")
                ),
                "central_mode": str(central.get("mode", "")),
            }
            if central["f1_macro"] > best_central_f1:
                best_central_f1 = float(central["f1_macro"])
                best_round = r
                final_central_metrics = central

        record.update(central_summary)
        round_records.append(record)

        with csv_path.open("a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                r,
                f"{round_info['val_f1_weighted']:.6f}",
                *[f"{v:.6f}" for v in round_info["val_f1_per_client"]],
                *[f"{v:.6f}" for v in round_info["train_loss_per_client"]],
                f"{central_summary['central_f1_macro']:.6f}",
                f"{central_summary['central_accuracy']:.6f}",
                f"{central_summary['central_auc_macro']:.6f}",
                central_summary["central_mode"],
            ])

        logger.info(
            f"round {r:3d}/{rounds}  val_f1_weighted={round_info['val_f1_weighted']:.4f}  "
            f"central_f1={central_summary['central_f1_macro']:.4f}  "
            f"mode={central_summary['central_mode']}"
        )

    runtime = time.time() - t0
    logger.info(f"federated training done in {runtime:.1f} s  best central f1={best_central_f1:.4f} @ round {best_round}")

    # --- Final central eval (+ dual for FedBN) ---
    final_eval = server.evaluate_central(test_loader)
    logger.info("FINAL central classification report:\n" + format_classification_report(final_eval))
    dual: dict | None = None
    if strategy == "fedbn":
        dual = server.dual_evaluate_fedbn(test_loader)
        logger.info(
            f"FedBN representative f1_macro={dual['representative']['f1_macro']:.4f}  "
            f"client_avg f1_macro={dual['client_avg']['f1_macro']:.4f}"
        )

    # --- Save checkpoint ---
    ckpt_dir = RESULTS / "checkpoints" / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "round": rounds,
        "global_params": server.global_params,
        "client_states": [c.snapshot() for c in clients],
        "strategy": strategy,
        "bn_keys": list(server.bn_keys),
    }
    torch.save(ckpt, ckpt_dir / "final.pt")

    # --- Figures ---
    curves_fig = RESULTS / "figures" / f"{run_id}_fed_curves.png"
    _plot_fed_curves(round_records, curves_fig)

    # --- Metrics JSON ---
    result = {
        "run_id": run_id,
        "dataset": dataset_name,
        "strategy": strategy,
        "partition_strategy": partition_strategy,
        "num_clients": len(clients),
        "rounds": rounds,
        "local_epochs": local_epochs,
        "config": config,
        "client_info": [
            {
                "client_id": c.client_id,
                "num_train": c.num_samples,
                "num_val": len(c.val_loader.dataset),  # type: ignore[arg-type]
            }
            for c in clients
        ],
        "class_weights": class_weights.tolist(),
        "final_central_metrics": metrics_to_jsonable(final_eval),
        "best_central_f1_macro": best_central_f1,
        "best_round": best_round,
        "runtime_seconds": runtime,
        "round_csv": str(csv_path.relative_to(REPO_ROOT)),
        "partition_figure": str(partition_fig.relative_to(REPO_ROOT)),
        "curves_figure": str(curves_fig.relative_to(REPO_ROOT)),
        "git_sha": _git_sha(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "none",
        "central_test_size": int(len(y_test)),
        "pool_size": int(len(y_pool)),
    }
    if dual is not None:
        result["fedbn_dual_eval"] = {
            k: metrics_to_jsonable(v) for k, v in dual.items()
        }
    if final_central_metrics is not None and final_central_metrics is not final_eval:
        result["best_round_central_metrics"] = metrics_to_jsonable(final_central_metrics)

    metrics_path = RESULTS / "metrics" / f"{run_id}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2))
    logger.info(f"wrote metrics → {metrics_path}")
    return result
