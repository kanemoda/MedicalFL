#!/usr/bin/env python
"""Evaluate saved Phase 4 checkpoints on each client's own val split.

No retraining — this is the shortcut path for the 18 DONE Phase 4 runs.
For each run we:
  1. Read `results/metrics/{run_id}.json` to recover the config that produced it.
  2. Replay the data carve + partition + per-client 80/20 split using the same
     seed logic, so the reconstructed val_loaders are bit-identical to the ones
     the training run held in memory.
  3. Load `results/checkpoints/{run_id}/final.pt`.
  4. Evaluate each client locally:
        - FedAvg/FedProx/FedPerf → load `global_params` into every client
        - FedBN                   → load `client_states[i]` into client i
  5. Append the `local_test` block to `{run_id}.json`.

RECOVERED FedBN runs have no saved checkpoint (they crashed at the reporting
step before torch.save). They must be retrained end-to-end instead — see
scripts/run_fedbn_v2.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.federation.client import FederatedClient  # noqa: E402
from src.models.cnn1d import CNN1D  # noqa: E402
from src.training.federated import (  # noqa: E402
    _carve_mitbih,
    _carve_ptbxl,
    _class_weights,
    _client_train_val_split,
    _evaluate_local_test,
    _load_mitbih,
    _load_ptbxl,
    _partition,
)
from src.utils.seeding import set_all_seeds  # noqa: E402


def _resolve_processed_root(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def evaluate_run(run_id: str, *, verbose: bool = True) -> dict:
    """Append a `local_test` block to `results/metrics/{run_id}.json` and return it."""
    json_path = REPO_ROOT / "results" / "metrics" / f"{run_id}.json"
    ckpt_path = REPO_ROOT / "results" / "checkpoints" / run_id / "final.pt"

    if not json_path.exists():
        raise FileNotFoundError(f"{json_path} — no Phase 4 artifact for this run_id")
    existing = json.loads(json_path.read_text())
    if existing.get("status") == "RECOVERED" or not ckpt_path.exists():
        raise FileNotFoundError(
            f"{ckpt_path} not present (likely a RECOVERED run); retrain "
            f"with the _v2 suffix to produce a fresh checkpoint."
        )

    config = existing["config"]
    dataset_name = existing["dataset"]
    strategy = str(existing["strategy"]).lower()

    set_all_seeds(config["seed"], strict=bool(config.get("strict_determinism", False)))
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    processed_root = _resolve_processed_root(config["data"]["processed_root"])
    if dataset_name == "mitbih":
        X, y = _load_mitbih(processed_root)
        (X_pool, y_pool), _ = _carve_mitbih(X, y, config["seed"])
        class_names = list(config["mitbih"]["class_names"])
        num_classes = int(config["mitbih"]["num_classes"])
    elif dataset_name == "ptbxl":
        label_mode = str(config["ptbxl"].get("label_mode", "binary"))
        X, y, folds = _load_ptbxl(processed_root, label_mode=label_mode)
        (X_pool, y_pool), _ = _carve_ptbxl(X, y, folds)
        class_names = list(config["ptbxl"]["class_names"])
        num_classes = int(config["ptbxl"]["num_classes"])
    else:
        raise ValueError(f"unknown dataset_name: {dataset_name!r}")

    fed_cfg = config["federation"]
    num_clients = int(fed_cfg.get("num_clients", 5))
    partition_strategy = str(fed_cfg.get("partition", "iid")).lower()
    partition_kwargs = {
        "alpha": float(fed_cfg.get("dirichlet_alpha", 0.5)),
        "beta": float(fed_cfg.get("quantity_beta", 0.5)),
        "classes_per_client": int(fed_cfg.get("classes_per_client", 2)),
    }

    partitions = _partition(
        X_pool, y_pool, num_clients, partition_strategy,
        seed=config["seed"], **partition_kwargs,
    )

    client_splits = []
    train_ys: list[np.ndarray] = []
    for c_id, (X_c, y_c, _idx) in enumerate(partitions):
        if y_c.size == 0:
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

    model_fn = lambda: CNN1D(num_classes=num_classes)  # noqa: E731
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
    if not clients:
        raise RuntimeError(f"no non-empty clients for {run_id}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    global_params = ckpt["global_params"]
    client_states = ckpt["client_states"]
    if len(client_states) != len(clients):
        raise RuntimeError(
            f"{run_id}: checkpoint has {len(client_states)} client states but "
            f"reconstructed {len(clients)} clients — seed/config mismatch?"
        )

    local_test = _evaluate_local_test(
        clients=clients,
        strategy=strategy,
        global_params=global_params,
        client_states=client_states,
    )

    existing["local_test"] = local_test
    json_path.write_text(json.dumps(existing, indent=2))

    if verbose:
        per = [f"{v:.3f}" for v in local_test["per_client_local_test_f1"]]
        print(
            f"  {run_id}\n"
            f"    mean_local_f1={local_test['local_test_f1_mean']:.4f}  "
            f"std={local_test['local_test_f1_std']:.4f}\n"
            f"    per_client={per}  regime={local_test['eval_regime']}"
        )
    return local_test


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_ids", nargs="+", help="run_ids to evaluate")
    args = parser.parse_args()

    failed: list[str] = []
    for run_id in args.run_ids:
        try:
            evaluate_run(run_id)
        except FileNotFoundError as e:
            print(f"  SKIP {run_id}: {e}", file=sys.stderr)
            failed.append(run_id)
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL {run_id}: {type(e).__name__}: {e}", file=sys.stderr)
            failed.append(run_id)
    if failed:
        print(f"\n{len(failed)} run(s) not processed: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
