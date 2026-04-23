"""Federation server: round orchestration + central evaluation.

`FederationServer` holds the authoritative `global_params` dict and a reference
model (the same architecture every client uses). On each round it:

  1. Broadcasts `global_params` (or the FedBN-restricted view) to every client.
  2. Calls `client.local_train(...)` → gathers updates + sample counts +
     val-F1s (FedPerf needs the last).
  3. Delegates to `aggregation.aggregate(...)`.
  4. For central test evaluation:
       - FedAvg/FedProx/FedPerf → load `global_params` into `ref_model` and run.
       - FedBN → cannot use the server state directly because BN stats are
         per-client. We provide two modes:
           - `central_eval="representative"`: pick client 0, restore its full
             state_dict, evaluate on the central test loader.
           - `central_eval="client_avg"`: evaluate each client on the central
             test loader, average metrics.
         For Track A (primary) we log *both* and pick one as "the" number in
         the headline table, with both reported in DECISIONS.
"""
from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_metrics
from src.federation.aggregation import (
    AGGREGATION_STRATEGIES,
    aggregate,
    get_bn_key_set,
)
from src.federation.client import FederatedClient


class FederationServer:
    def __init__(
        self,
        clients: list[FederatedClient],
        model_fn: Callable[[], nn.Module],
        *,
        strategy: str,
        device: torch.device,
        class_names: list[str],
        proximal_mu: float = 0.0,
        fedperf_alpha: float = 0.5,
        fedbn_central_eval: str = "representative",
        logger=None,
    ) -> None:
        if strategy.lower() not in AGGREGATION_STRATEGIES:
            raise ValueError(
                f"unknown strategy: {strategy!r}; choose from {AGGREGATION_STRATEGIES}"
            )
        if fedbn_central_eval not in {"representative", "client_avg"}:
            raise ValueError(
                f"fedbn_central_eval must be 'representative' or 'client_avg'; "
                f"got {fedbn_central_eval!r}"
            )

        self.clients = clients
        self.strategy = strategy.lower()
        self.device = device
        self.class_names = list(class_names)
        self.proximal_mu = float(proximal_mu)
        self.fedperf_alpha = float(fedperf_alpha)
        self.fedbn_central_eval = fedbn_central_eval
        self.logger = logger

        # Reference model — same architecture as every client's, lives on
        # `device` so central evaluation is fast.
        self.ref_model: nn.Module = model_fn().to(device)
        self.bn_keys: set[str] = get_bn_key_set(self.ref_model)
        # Global params = the full state_dict of the reference model.
        # BN keys are tracked but for non-FedBN strategies they are simply
        # part of the aggregate like any other tensor.
        self.global_params: dict[str, torch.Tensor] = {
            k: v.detach().cpu().clone()
            for k, v in self.ref_model.state_dict().items()
        }

        # Seed every client with the same initial params so round 0 is
        # deterministic.
        for c in self.clients:
            c.set_parameters(self.global_params)

    # ------------------------------------------------------------------
    # Round mechanics
    # ------------------------------------------------------------------

    def _broadcast(self) -> None:
        """Push current `global_params` (minus BN keys for FedBN) to every client."""
        if self.strategy == "fedbn":
            shared = {
                k: v for k, v in self.global_params.items() if k not in self.bn_keys
            }
        else:
            shared = self.global_params
        for c in self.clients:
            c.set_parameters(shared)

    def run_round(self, local_epochs: int) -> dict:
        """Execute one federated round; return per-client + aggregated metrics."""
        self._broadcast()

        client_updates: list[dict[str, torch.Tensor]] = []
        num_samples: list[int] = []
        val_f1s: list[float] = []
        train_losses: list[float] = []
        train_f1s: list[float] = []
        val_losses: list[float] = []

        for c in self.clients:
            train_metrics = c.local_train(
                local_epochs=local_epochs,
                proximal_mu=self.proximal_mu,
                global_params=self.global_params if self.proximal_mu > 0 else None,
            )
            val_metrics = c.evaluate()

            client_updates.append(c.get_parameters())
            num_samples.append(c.num_samples)
            val_f1 = val_metrics.get("f1_macro", float("nan"))
            val_f1s.append(float(val_f1) if val_f1 == val_f1 else 0.0)
            train_losses.append(float(train_metrics.get("loss", float("nan"))))
            train_f1s.append(float(train_metrics.get("f1_macro", float("nan"))))
            val_losses.append(float(val_metrics.get("loss", float("nan"))))

        new_global = aggregate(
            self.strategy,
            client_updates,
            num_samples,
            bn_keys=self.bn_keys,
            val_f1=val_f1s,
            fedperf_alpha=self.fedperf_alpha,
        )
        # Merge back into the full state_dict so keys that weren't aggregated
        # (e.g. BN for FedBN) survive as server-side defaults from round 0.
        merged = dict(self.global_params)
        merged.update(new_global)
        self.global_params = merged

        return {
            "num_samples": num_samples,
            "train_loss_per_client": train_losses,
            "train_f1_per_client": train_f1s,
            "val_loss_per_client": val_losses,
            "val_f1_per_client": val_f1s,
            "val_f1_weighted": _weighted_mean(val_f1s, num_samples),
        }

    # ------------------------------------------------------------------
    # Central evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate_central(self, test_loader: DataLoader) -> dict:
        """Run central evaluation on `test_loader` using current server state.

        For FedBN, selects between two modes controlled by
        `fedbn_central_eval` (see module docstring). The result dict is
        augmented with a `mode` field identifying which path produced it.
        """
        if self.strategy != "fedbn":
            self.ref_model.load_state_dict(self.global_params, strict=True)
            metrics = _eval_model(self.ref_model, test_loader, self.device, self.class_names)
            metrics["mode"] = "global"
            return metrics

        if self.fedbn_central_eval == "representative":
            client = self.clients[0]
            client.model.eval()
            metrics = _eval_model(client.model, test_loader, self.device, self.class_names)
            metrics["mode"] = "fedbn_representative_client_0"
            return metrics

        # client_avg: evaluate every client; return mean over clients.
        per_client: list[dict] = []
        for c in self.clients:
            c.model.eval()
            m = _eval_model(c.model, test_loader, self.device, self.class_names)
            per_client.append(m)

        def _avg(field: str) -> float:
            vals = [m[field] for m in per_client if not _is_nan(m[field])]
            return float(np.mean(vals)) if vals else float("nan")

        agg = {
            "accuracy": _avg("accuracy"),
            "precision_macro": _avg("precision_macro"),
            "recall_macro": _avg("recall_macro"),
            "f1_macro": _avg("f1_macro"),
            "auc_macro": _avg("auc_macro"),
            "per_client_f1_macro": [m["f1_macro"] for m in per_client],
            "mode": "fedbn_client_avg",
            "class_names": self.class_names,
        }
        return agg

    def dual_evaluate_fedbn(self, test_loader: DataLoader) -> dict:
        """Return *both* FedBN central-eval modes for logging."""
        if self.strategy != "fedbn":
            raise ValueError("dual_evaluate_fedbn is only meaningful for FedBN")
        saved_mode = self.fedbn_central_eval
        out: dict = {}
        try:
            self.fedbn_central_eval = "representative"
            out["representative"] = self.evaluate_central(test_loader)
            self.fedbn_central_eval = "client_avg"
            out["client_avg"] = self.evaluate_central(test_loader)
        finally:
            self.fedbn_central_eval = saved_mode
        return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_nan(x: float) -> bool:
    return isinstance(x, float) and x != x


def _weighted_mean(values: list[float], weights: list[int]) -> float:
    total = float(sum(weights))
    if total <= 0:
        return float("nan")
    clean = [
        (v, w) for v, w in zip(values, weights)
        if isinstance(v, float) and v == v
    ]
    if not clean:
        return float("nan")
    num = sum(v * w for v, w in clean)
    den = sum(w for _, w in clean)
    return num / den if den > 0 else float("nan")


@torch.no_grad()
def _eval_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
) -> dict:
    model.eval()
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_prob: list[np.ndarray] = []
    for Xb, yb in loader:
        Xb = Xb.to(device, non_blocking=True)
        logits = model(Xb)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_prob.append(probs)
        all_pred.append(probs.argmax(axis=1))
        all_true.append(yb.numpy())
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)
    return compute_metrics(y_true, y_pred, y_prob, class_names)
