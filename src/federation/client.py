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

Phase 5 — differential privacy
------------------------------
When a ``dp_config`` dict is passed in, the client delegates optimizer /
model construction to ``src.federation.dp.setup_dp_training``.  The
training loop then:
  - wraps the loader in Opacus's ``BatchMemoryManager`` (physical batch
    ≤ ``max_physical_batch_size``),
  - calls ``opt_dp.step()`` each (physical) iteration — Opacus
    accumulates per-sample grads and only performs the DP update at
    the logical batch boundary,
  - calls ``opt_nondp.step()`` each iteration when present (DP-FedBN
    BN-params optimiser — plain AdamW, no noise).
Parameter exchange ignores the Opacus ``_module.`` prefix so FedAvg /
FedBN aggregation keeps the original state-dict key space.
"""
from __future__ import annotations

import logging
import math
from copy import deepcopy
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import compute_metrics


# ---------------------------------------------------------------------------
# Class-weight utility (zero-safe for non-IID partitions)
# ---------------------------------------------------------------------------

_MIN_CLASS_COUNT_WARN = 5


def compute_class_weights(
    y: np.ndarray,
    num_classes: int,
    *,
    logger: logging.Logger | None = None,
    client_id: int | None = None,
) -> np.ndarray:
    """Inverse-frequency class weights that survive zero counts.

    Under extreme non-IID (Dirichlet α=0.1, label-skew C=2) a client may hold
    zero samples of a rare class. `np.bincount` reports count=0 for those,
    which makes a naive `1/count` weight blow up. We floor the denominator at
    1 *for the weight computation only* — the numerator uses the true sum so
    class ratios for present classes remain correct.

    Consequences for an absent class at a client:
      - Its weight is inflated (`total / (K * 1)`) but it contributes zero to
        the per-batch loss because no samples of it exist locally.
      - Federated aggregation propagates that class's parameters from clients
        that do hold it. No need to collapse num_classes per client.
    """
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    safe_counts = np.maximum(counts, 1.0)
    total = float(counts.sum())
    # If the client is empty, uniform weights.
    if total <= 0.0:
        return np.ones(num_classes, dtype=np.float64)
    weights = total / (num_classes * safe_counts)

    if logger is not None:
        prefix = f"client {client_id}: " if client_id is not None else ""
        for c in range(num_classes):
            if counts[c] < _MIN_CLASS_COUNT_WARN:
                logger.warning(
                    f"{prefix}class {c} has only {int(counts[c])} samples; "
                    f"inflated weight {weights[c]:.2f}"
                )
    return weights


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


def _inner_module(model: nn.Module) -> nn.Module:
    """Unwrap Opacus's ``GradSampleModule`` layers for state-dict access."""
    inner = model
    while hasattr(inner, "_module") and isinstance(inner._module, nn.Module):
        inner = inner._module
    return inner


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
        dp_config: dict[str, Any] | None = None,
    ) -> None:
        self.client_id = client_id
        self.device = device
        self.num_classes = num_classes
        self.class_names = list(class_names)
        self.num_samples = int(len(y_train))
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

        self.train_loader = _numpy_to_tensor_loader(
            X_train, y_train, batch_size=batch_size, shuffle=True,
            drop_last=len(y_train) > batch_size, num_workers=num_workers,
        )
        self.val_loader = _numpy_to_tensor_loader(
            X_val, y_val, batch_size=batch_size, shuffle=False,
            drop_last=False, num_workers=num_workers,
        )

        raw_model: nn.Module = model_fn().to(device)

        cw = None
        if class_weights is not None:
            cw = torch.tensor(class_weights, dtype=torch.float32, device=device)
        self.criterion = nn.CrossEntropyLoss(weight=cw)

        # ------------------------------------------------------------------
        # DP wiring — when requested
        # ------------------------------------------------------------------
        self.dp_enabled: bool = False
        self.dp_mode: str | None = None
        self.privacy_engine = None
        self.opt_nondp: torch.optim.Optimizer | None = None
        self.dp_target_epsilon: float | None = None
        self.dp_target_delta: float | None = None
        self.dp_max_grad_norm: float | None = None
        self.dp_max_physical_batch_size: int = batch_size
        self.spent_epsilon: float | None = None

        if dp_config is not None and bool(dp_config.get("enabled", False)):
            # Lazy import avoids forcing opacus on non-DP runs.
            from src.federation.dp import setup_dp_training

            total_epochs = int(dp_config["total_epochs"])
            target_eps = dp_config.get("target_epsilon", None)
            target_delta = float(dp_config.get("target_delta", 1e-5))
            max_grad_norm = float(dp_config.get("max_grad_norm", 1.0))
            mode = str(dp_config.get("mode", "fedbn"))
            self.dp_max_physical_batch_size = int(
                dp_config.get("max_physical_batch_size", batch_size)
            )

            setup = setup_dp_training(
                model=raw_model,
                train_loader=self.train_loader,
                target_epsilon=target_eps,
                target_delta=target_delta,
                max_grad_norm=max_grad_norm,
                total_epochs=total_epochs,
                mode=mode,
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
            self.model = setup.model
            self.optimizer = setup.opt_dp
            self.opt_nondp = setup.opt_nondp
            self.train_loader = setup.train_loader
            self.privacy_engine = setup.privacy_engine
            self.dp_enabled = self.privacy_engine is not None
            self.dp_mode = setup.mode
            self.dp_target_epsilon = setup.target_epsilon
            self.dp_target_delta = setup.target_delta
            self.dp_max_grad_norm = setup.max_grad_norm
        else:
            self.model = raw_model
            self.optimizer = _build_plain_optimizer(
                raw_model.parameters(), optimizer, lr, weight_decay,
            )

    # ------------------------------------------------------------------
    # Model-inner access (Opacus-safe)
    # ------------------------------------------------------------------

    def _inner(self) -> nn.Module:
        return _inner_module(self.model)

    # ------------------------------------------------------------------
    # Parameter exchange
    # ------------------------------------------------------------------

    def get_parameters(
        self, keys: Iterable[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return a CPU-side copy of ``state_dict`` (optionally restricted to
        ``keys``).

        Uses the *inner* (Opacus-unwrapped) model so FedAvg/FedBN keys are
        the plain ``conv1.weight`` variety, not ``_module.conv1.weight``.
        """
        sd = self._inner().state_dict()
        if keys is not None:
            keys = set(keys)
            return {k: v.detach().cpu().clone() for k, v in sd.items() if k in keys}
        return {k: v.detach().cpu().clone() for k, v in sd.items()}

    def set_parameters(self, params: dict[str, torch.Tensor]) -> None:
        """Load ``params`` into the inner model; missing keys left untouched.

        This is the crux of FedBN: the server never sends BN keys, so those
        stay at whatever the client held locally after the previous local
        step.
        """
        inner = self._inner()
        own = inner.state_dict()
        for k, v in params.items():
            if k not in own:
                raise KeyError(f"param {k!r} not in client model")
            own[k] = v.to(own[k].device, dtype=own[k].dtype)
        inner.load_state_dict(own, strict=True)

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
        """Run ``local_epochs`` of SGD over the client's train set.

        When DP is enabled the Opacus ``BatchMemoryManager`` splits each
        logical (Poisson-sampled) batch into physical chunks ≤
        ``dp_max_physical_batch_size`` and ``opt_dp.step()`` is called per
        chunk — Opacus accumulates per-sample grads internally and only
        performs the DP update (clip + noise) at the logical-batch
        boundary. The separate ``opt_nondp`` (DP-FedBN BN optimiser, when
        present) steps each physical chunk too; with our sizing (physical
        == logical) that's one update per batch, matching standard
        training.
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
            iterator = self._epoch_iterator()
            for Xb, yb in iterator:
                Xb = Xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                if yb.numel() == 0:
                    continue
                self.optimizer.zero_grad(set_to_none=True)
                if self.opt_nondp is not None:
                    self.opt_nondp.zero_grad(set_to_none=True)
                logits = self.model(Xb)
                loss = self.criterion(logits, yb)

                if anchor is not None:
                    prox = torch.zeros((), device=self.device)
                    for name, p in self._inner().named_parameters():
                        if name in anchor and p.requires_grad:
                            prox = prox + ((p - anchor[name]) ** 2).sum()
                    loss = loss + (proximal_mu / 2.0) * prox

                loss.backward()
                self.optimizer.step()
                if self.opt_nondp is not None:
                    self.opt_nondp.step()

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

    def _epoch_iterator(self):
        """Yield (Xb, yb) batches, using Opacus ``BatchMemoryManager`` under DP.

        Falls back to ``self.train_loader`` directly when no privacy engine
        is active.
        """
        if self.privacy_engine is None:
            yield from self.train_loader
            return

        from opacus.utils.batch_memory_manager import BatchMemoryManager

        with BatchMemoryManager(
            data_loader=self.train_loader,
            max_physical_batch_size=self.dp_max_physical_batch_size,
            optimizer=self.optimizer,
        ) as safe_loader:
            yield from safe_loader

    # ------------------------------------------------------------------
    # Privacy accounting
    # ------------------------------------------------------------------

    def update_spent_epsilon(self) -> float | None:
        """Query Opacus for cumulative ε; cache in ``self.spent_epsilon``."""
        if self.privacy_engine is None or self.dp_target_delta is None:
            return None
        try:
            eps = float(self.privacy_engine.get_epsilon(self.dp_target_delta))
        except Exception:  # noqa: BLE001
            eps = math.nan
        self.spent_epsilon = eps
        return eps

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
        """Deep-copy of the (inner) model state for external inspection."""
        return {k: v.detach().cpu().clone() for k, v in self._inner().state_dict().items()}

    def restore(self, snap: dict[str, torch.Tensor]) -> None:
        self._inner().load_state_dict(deepcopy(snap), strict=True)


# ---------------------------------------------------------------------------
# Plain optimiser factory (non-DP path)
# ---------------------------------------------------------------------------

def _build_plain_optimizer(
    params, name: str, lr: float, weight_decay: float,
) -> torch.optim.Optimizer:
    n = name.lower()
    if n == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if n == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if n == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"unsupported optimizer: {name}")
