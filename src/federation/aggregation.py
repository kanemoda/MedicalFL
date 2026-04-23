"""Parameter-aggregation strategies for federated rounds.

Each strategy takes
  - `client_updates`: list[dict[str, Tensor]] — one state_dict per client
  - `num_samples`: list[int] — dataset size per client (for size-weighted avg)
  - optional keyword args (e.g. `val_f1`, `global_params`)
and returns a single dict[str, Tensor] that the server then broadcasts.

FedBN differs from the rest by *excluding* BatchNorm layers from the aggregate:
clients keep their own BN running stats + affine params across rounds.
`get_bn_key_set(model)` is the single source of truth for which keys count as BN.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

BN_MODULES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
)


# ---------------------------------------------------------------------------
# BN key discovery
# ---------------------------------------------------------------------------

def get_bn_key_set(model: nn.Module) -> set[str]:
    """All state_dict keys that belong to BN modules in `model`.

    Covers both learnable params (`weight`, `bias`) and buffers
    (`running_mean`, `running_var`, `num_batches_tracked`). Keys are matched by
    the `module_name.attr` convention that `state_dict()` uses.
    """
    bn_keys: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, BN_MODULES):
            prefix = f"{name}." if name else ""
            for param_name, _ in module.named_parameters(recurse=False):
                bn_keys.add(prefix + param_name)
            for buf_name, _ in module.named_buffers(recurse=False):
                bn_keys.add(prefix + buf_name)
    return bn_keys


# ---------------------------------------------------------------------------
# Weighted parameter average
# ---------------------------------------------------------------------------

def _weighted_average(
    client_updates: list[dict[str, torch.Tensor]],
    weights: list[float],
    keys: Iterable[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Weighted mean of state_dicts across clients.

    - `weights` are normalized internally.
    - `keys=None` → average every key present in the first update.
    - Float-dtype tensors are averaged; integer tensors (e.g. BN's
      `num_batches_tracked`) take the first client's value verbatim.
    """
    if len(client_updates) != len(weights):
        raise ValueError("len(client_updates) must equal len(weights)")
    if not client_updates:
        raise ValueError("no client updates to aggregate")

    total = float(sum(weights))
    if total <= 0:
        raise ValueError(f"weights must sum to a positive value; got {total}")
    norm_weights = [w / total for w in weights]

    first = client_updates[0]
    if keys is None:
        keys = list(first.keys())
    else:
        keys = list(keys)

    agg: dict[str, torch.Tensor] = {}
    for k in keys:
        ref = first[k]
        if ref.is_floating_point():
            acc = torch.zeros_like(ref, dtype=torch.float32)
            for w, upd in zip(norm_weights, client_updates):
                acc += upd[k].to(dtype=torch.float32) * w
            agg[k] = acc.to(dtype=ref.dtype)
        else:
            agg[k] = ref.clone()
    return agg


# ---------------------------------------------------------------------------
# FedAvg — McMahan et al. 2017
# ---------------------------------------------------------------------------

def fedavg_aggregate(
    client_updates: list[dict[str, torch.Tensor]],
    num_samples: list[int],
) -> dict[str, torch.Tensor]:
    """Sample-size-weighted mean of all parameters."""
    return _weighted_average(client_updates, [float(n) for n in num_samples])


# ---------------------------------------------------------------------------
# FedProx — Li et al. 2020. Same aggregation as FedAvg;
# the proximal term lives on the *client* (src/federation/client.py).
# ---------------------------------------------------------------------------

def fedprox_aggregate(
    client_updates: list[dict[str, torch.Tensor]],
    num_samples: list[int],
) -> dict[str, torch.Tensor]:
    return fedavg_aggregate(client_updates, num_samples)


# ---------------------------------------------------------------------------
# FedBN — Li et al. 2021: average everything *except* BN layers
# ---------------------------------------------------------------------------

def fedbn_aggregate(
    client_updates: list[dict[str, torch.Tensor]],
    num_samples: list[int],
    bn_keys: set[str],
) -> dict[str, torch.Tensor]:
    """Sample-weighted mean of non-BN parameters. BN keys are omitted from
    the server's broadcast so each client keeps its own BN stats."""
    first = client_updates[0]
    shared_keys = [k for k in first.keys() if k not in bn_keys]
    return _weighted_average(
        client_updates, [float(n) for n in num_samples], keys=shared_keys
    )


# ---------------------------------------------------------------------------
# FedPerf — performance-weighted average
# Clients with higher validation F1 contribute more. Simple mixing:
#   w_i = alpha * size_i + (1 - alpha) * f1_i
# Both size and F1 are L1-normalized before mixing; alpha ∈ [0, 1].
# ---------------------------------------------------------------------------

def fedperf_aggregate(
    client_updates: list[dict[str, torch.Tensor]],
    num_samples: list[int],
    val_f1: list[float],
    alpha: float = 0.5,
) -> dict[str, torch.Tensor]:
    if len(num_samples) != len(val_f1) or len(num_samples) != len(client_updates):
        raise ValueError("num_samples, val_f1, client_updates must match in length")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0,1]; got {alpha}")

    size = [float(n) for n in num_samples]
    # Replace NaN/inf F1 (e.g. client with one class in its val set) with 0 so
    # it contributes only via the size term.
    perf = [
        (f if isinstance(f, float) and f == f and f != float("inf") else 0.0)
        for f in val_f1
    ]
    # Floor at 0 so a negative F1 (shouldn't happen) can't subtract mass.
    perf = [max(0.0, p) for p in perf]

    size_sum = sum(size) or 1.0
    perf_sum = sum(perf)
    size_norm = [s / size_sum for s in size]
    if perf_sum > 0:
        perf_norm = [p / perf_sum for p in perf]
        weights = [alpha * s + (1 - alpha) * p for s, p in zip(size_norm, perf_norm)]
    else:
        # All-zero F1 → fall back to pure FedAvg weighting.
        weights = size_norm

    return _weighted_average(client_updates, weights)


# ---------------------------------------------------------------------------
# Strategy registry / factory
# ---------------------------------------------------------------------------

AGGREGATION_STRATEGIES = {"fedavg", "fedprox", "fedbn", "fedperf"}


def aggregate(
    strategy: str,
    client_updates: list[dict[str, torch.Tensor]],
    num_samples: list[int],
    *,
    bn_keys: set[str] | None = None,
    val_f1: list[float] | None = None,
    fedperf_alpha: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Unified strategy dispatch used by `FederationServer`."""
    s = strategy.lower()
    if s == "fedavg":
        return fedavg_aggregate(client_updates, num_samples)
    if s == "fedprox":
        return fedprox_aggregate(client_updates, num_samples)
    if s == "fedbn":
        if bn_keys is None:
            raise ValueError("FedBN requires bn_keys (pass get_bn_key_set(model))")
        return fedbn_aggregate(client_updates, num_samples, bn_keys)
    if s == "fedperf":
        if val_f1 is None:
            raise ValueError("FedPerf requires val_f1 per client")
        return fedperf_aggregate(
            client_updates, num_samples, val_f1, alpha=fedperf_alpha
        )
    raise ValueError(f"unknown aggregation strategy: {strategy!r}")
