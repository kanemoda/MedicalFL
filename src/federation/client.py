"""Federated client: local training + evaluation for a single participant.

A `FederatedClient` owns its own data (train + val loaders), a model instance,
and an optimizer. Each round:
  1. Server broadcasts global params → `set_parameters`.
  2. Client runs `local_epochs` of SGD (+ optional FedProx proximal term) →
     `local_train`.
  3. Client reports back `(state_dict, num_samples, train_metrics, val_metrics)`
     → `get_parameters` + `evaluate`.

The proximal term (`mu > 0`) is FedProx-style: a penalty
    (mu / 2) * || w - w_global ||²
added to the local loss. `mu == 0` collapses to FedAvg/FedBN.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import compute_metrics


def _numpy_to_tensor_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    Xt = torch.from_numpy(np.ascontiguousarray(X)).float()
    yt = torch.from_numpy(np.ascontiguousarray(y)).long()
    ds = TensorDataset(Xt, yt)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )


class FederatedClient:
    def __init__(
        self,
        client_id: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_fn,
        num_classes: int,
        class_names: list[str],
        device: torch.device,
        *,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer: str = "adamw",
        num_workers: int = 0,
        class_weights: np.ndarray | None = None,
    ) -> None:
        self.client_id = client_id
        self.device = device
        self.num_classes = num_classes
        self.class_names = list(class_names)
        self.num_samples = int(len(y_train))

        self.train_loader = _numpy_to_tensor_loader(
            X_train, y_train, batch_size=batch_size, shuffle=True,
            drop_last=len(y_train) > batch_size, num_workers=num_workers,
        )
        self.val_loader = _numpy_to_tensor_loader(
            X_val, y_val, batch_size=batch_size, shuffle=False,
            drop_last=False, num_workers=num_workers,
        )

        self.model: nn.Module = model_fn().to(device)

        cw = None
        if class_weights is not None:
            cw = torch.tensor(class_weights, dtype=torch.float32, device=device)
        self.criterion = nn.CrossEntropyLoss(weight=cw)

        optim_name = optimizer.lower()
        if optim_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay,
            )
        elif optim_name == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=weight_decay,
            )
        elif optim_name == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"unsupported optimizer: {optimizer}")

    # ------------------------------------------------------------------
    # Parameter exchange
    # ------------------------------------------------------------------

    def get_parameters(self, keys: Iterable[str] | None = None) -> dict[str, torch.Tensor]:
        """Return a CPU-side copy of `state_dict` (optionally restricted to `keys`)."""
        sd = self.model.state_dict()
        if keys is not None:
            keys = set(keys)
            return {k: v.detach().cpu().clone() for k, v in sd.items() if k in keys}
        return {k: v.detach().cpu().clone() for k, v in sd.items()}

    def set_parameters(self, params: dict[str, torch.Tensor]) -> None:
        """Load `params` into the model; missing keys are left untouched.

        This is the crux of FedBN: the server never sends BN keys, so those stay
        at whatever the client held locally after the previous local step.
        """
        own = self.model.state_dict()
        for k, v in params.items():
            if k not in own:
                raise KeyError(f"param {k!r} not in client model")
            own[k] = v.to(own[k].device, dtype=own[k].dtype)
        self.model.load_state_dict(own, strict=True)

    # ------------------------------------------------------------------
    # Local training
    # ------------------------------------------------------------------

    def local_train(
        self,
        local_epochs: int,
        *,
        proximal_mu: float = 0.0,
        global_params: dict[str, torch.Tensor] | None = None,
    ) -> dict:
        """Run `local_epochs` of SGD over the client's train set.

        If `proximal_mu > 0`, requires `global_params` (the FedProx anchor
        weights) and adds `(mu/2) * || w - w_global ||²` to each step's loss.
        """
        if proximal_mu > 0 and global_params is None:
            raise ValueError("FedProx requires global_params when proximal_mu > 0")

        anchor: dict[str, torch.Tensor] | None = None
        if proximal_mu > 0 and global_params is not None:
            anchor = {
                k: v.detach().to(self.device) for k, v in global_params.items()
            }

        self.model.train()
        running_loss = 0.0
        running_n = 0
        preds_buf: list[np.ndarray] = []
        true_buf: list[np.ndarray] = []
        prob_buf: list[np.ndarray] = []

        for _ in range(local_epochs):
            for Xb, yb in self.train_loader:
                Xb = Xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(Xb)
                loss = self.criterion(logits, yb)

                if anchor is not None:
                    prox = torch.zeros((), device=self.device)
                    for name, p in self.model.named_parameters():
                        if name in anchor and p.requires_grad:
                            prox = prox + ((p - anchor[name]) ** 2).sum()
                    loss = loss + (proximal_mu / 2.0) * prox

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * Xb.size(0)
                running_n += Xb.size(0)
                with torch.no_grad():
                    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
                preds_buf.append(probs.argmax(axis=1))
                true_buf.append(yb.detach().cpu().numpy())
                prob_buf.append(probs)

        y_true = np.concatenate(true_buf) if true_buf else np.zeros(0, dtype=np.int64)
        y_pred = np.concatenate(preds_buf) if preds_buf else np.zeros(0, dtype=np.int64)
        y_prob = (
            np.concatenate(prob_buf)
            if prob_buf
            else np.zeros((0, self.num_classes), dtype=np.float64)
        )
        metrics = compute_metrics(y_true, y_pred, y_prob, self.class_names)
        metrics["loss"] = running_loss / max(running_n, 1)
        return metrics

    # ------------------------------------------------------------------
    # Local evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, loader: DataLoader | None = None) -> dict:
        loader = loader if loader is not None else self.val_loader
        self.model.eval()
        all_true: list[np.ndarray] = []
        all_pred: list[np.ndarray] = []
        all_prob: list[np.ndarray] = []
        total_loss = 0.0
        total_n = 0
        for Xb, yb in loader:
            Xb = Xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)
            logits = self.model(Xb)
            loss = self.criterion(logits, yb)
            total_loss += loss.item() * Xb.size(0)
            total_n += Xb.size(0)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_prob.append(probs)
            all_pred.append(probs.argmax(axis=1))
            all_true.append(yb.cpu().numpy())
        if total_n == 0:
            return {"loss": float("nan"), "accuracy": float("nan"),
                    "f1_macro": float("nan"), "auc_macro": float("nan"),
                    "num_samples": 0}
        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)
        y_prob = np.concatenate(all_prob)
        m = compute_metrics(y_true, y_pred, y_prob, self.class_names)
        m["loss"] = total_loss / total_n
        m["num_samples"] = int(total_n)
        return m

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, torch.Tensor]:
        """Deep-copy of the model state for external inspection/checkpointing."""
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def restore(self, snap: dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(deepcopy(snap), strict=True)
