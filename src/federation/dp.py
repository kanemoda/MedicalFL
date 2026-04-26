"""Differential-privacy wiring for federated training (Phase 5).

Three DP composition modes:

  ``fedavg``
      Naive DP-SGD over *all* parameters, BN kept intact. Opacus's
      ``ModuleValidator`` raises ``ShouldReplaceModuleError`` for any
      ``BatchNorm*`` module in the model (because BN couples samples
      across a batch, which breaks per-sample privacy). We catch the
      failure and re-raise as ``DPModeUnsupportedError`` so the sweep
      records it as an empirical finding rather than crashing.

  ``fedavg_groupnorm``
      Replace every BN with GroupNorm up-front via
      ``ModuleValidator.fix`` (Opacus's default replaces BN_k with
      ``GroupNorm(gcd(32, k), k)``), then wrap the whole model with
      Opacus. This is the Opacus-recommended workaround.

  ``fedbn``
      Freeze BN parameters (``requires_grad=False``) *before* wrapping
      the model with Opacus. Because Opacus's
      ``ModuleValidator.validate`` iterates only ``trainable_modules``
      and ``GradSampleModule`` only registers hooks on modules with
      trainable params, frozen BN is skipped on both paths. After
      ``make_private_with_epsilon`` returns we unfreeze BN so the
      plain second optimiser can update it. The split gives us:
        - ``opt_dp``      AdamW over conv/FC params, wrapped by Opacus
                          (per-sample clip + Gaussian noise)
        - ``opt_nondp``   plain AdamW over BN params (regular .grad
                          from autograd, no DP)
      BN running stats + affine updates stay fully local; the server
      never sees them, matching the FedBN threat model.

For ``target_epsilon`` None / +inf we short-circuit to a plain AdamW
(no Opacus, no privacy engine). ``fedavg_groupnorm`` still replaces
BN with GN in that path because the model topology is fixed by mode,
not by the privacy budget.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader

from src.federation.aggregation import BN_MODULES

DP_MODES = {"fedavg", "fedavg_groupnorm", "fedbn"}


class DPModeUnsupportedError(RuntimeError):
    """Raised when Opacus refuses a DP mode (e.g. naive BN+DP-SGD)."""


class DPSetup(NamedTuple):
    model: nn.Module
    opt_dp: torch.optim.Optimizer
    opt_nondp: torch.optim.Optimizer | None
    train_loader: DataLoader
    privacy_engine: PrivacyEngine | None
    mode: str
    target_epsilon: float | None
    target_delta: float
    max_grad_norm: float


# ---------------------------------------------------------------------------
# Parameter inspection
# ---------------------------------------------------------------------------


def split_bn_and_non_bn_params(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Partition ``model`` params into (BN, non-BN).

    BN = every learnable tensor owned by a ``BatchNormNd`` /
    ``SyncBatchNorm`` module (uses the same module-type list as
    ``aggregation.get_bn_key_set``).
    """
    bn_params: list[nn.Parameter] = []
    bn_ids: set[int] = set()
    for _, module in model.named_modules():
        if isinstance(module, BN_MODULES):
            for p in module.parameters(recurse=False):
                bn_params.append(p)
                bn_ids.add(id(p))
    non_bn_params = [p for p in model.parameters() if id(p) not in bn_ids]
    return bn_params, non_bn_params


def replace_bn_with_groupnorm(model: nn.Module) -> nn.Module:
    """Replace every BN module with GroupNorm (Opacus default).

    Idempotent: returns the input unchanged when no BN module is present,
    otherwise returns a cloned model with BN replaced by GroupNorm.
    """
    has_bn = any(
        isinstance(m, BN_MODULES) for _, m in model.named_modules()
    )
    if not has_bn:
        return model
    return ModuleValidator.fix(model)


def inner_module(model: nn.Module) -> nn.Module:
    """Return the non-Opacus-wrapped module for state-dict access.

    Opacus wraps the model in ``GradSampleModule`` which stores the
    original at ``._module``; its ``state_dict`` is keyed with a
    ``_module.`` prefix that would break FedAvg/FedBN parameter
    exchange. We strip the wrapper when we need the "plain" view.
    """
    inner = model
    while hasattr(inner, "_module") and isinstance(inner._module, nn.Module):
        inner = inner._module
    return inner


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------


def _make_adamw(params, lr: float, weight_decay: float) -> torch.optim.AdamW:
    return torch.optim.AdamW(list(params), lr=lr, weight_decay=weight_decay)


def _is_no_dp(target_epsilon: float | None) -> bool:
    if target_epsilon is None:
        return True
    if isinstance(target_epsilon, float) and math.isinf(target_epsilon):
        return True
    return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def setup_dp_training(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    target_epsilon: float | None,
    target_delta: float,
    max_grad_norm: float,
    total_epochs: int,
    mode: str,
    lr: float = 1e-3,
    weight_decay: float = 5e-4,
) -> DPSetup:
    """Return the (model, optimizer(s), loader, engine) tuple for a mode.

    Args:
        model: unwrapped CNN (has BN modules).
        train_loader: the client's training ``DataLoader``.
        target_epsilon: per-client ε target; ``None`` or ``+inf`` = no DP.
        target_delta: per-client δ target.
        max_grad_norm: per-sample L2 clip threshold.
        total_epochs: ``rounds * local_epochs`` — what Opacus uses to
            calibrate the noise multiplier.
        mode: one of ``DP_MODES``.
        lr, weight_decay: AdamW hyperparameters.

    Raises:
        DPModeUnsupportedError: Opacus refused the requested mode
            (happens for ``fedavg`` with BN present).
    """
    if mode not in DP_MODES:
        raise ValueError(f"unknown DP mode: {mode!r}; choose from {DP_MODES}")

    if _is_no_dp(target_epsilon):
        if mode == "fedavg_groupnorm":
            model = replace_bn_with_groupnorm(model)
        optimizer = _make_adamw(model.parameters(), lr, weight_decay)
        return DPSetup(
            model=model,
            opt_dp=optimizer,
            opt_nondp=None,
            train_loader=train_loader,
            privacy_engine=None,
            mode=mode,
            target_epsilon=None,
            target_delta=target_delta,
            max_grad_norm=max_grad_norm,
        )

    # Finite-ε paths beyond this point.
    if mode == "fedavg":
        privacy_engine = PrivacyEngine()
        optimizer = _make_adamw(model.parameters(), lr, weight_decay)
        try:
            model_priv, opt_priv, loader_priv = (
                privacy_engine.make_private_with_epsilon(
                    module=model,
                    optimizer=optimizer,
                    data_loader=train_loader,
                    target_epsilon=float(target_epsilon),
                    target_delta=target_delta,
                    epochs=total_epochs,
                    max_grad_norm=max_grad_norm,
                )
            )
        except Exception as e:  # noqa: BLE001 — deliberate broad catch
            raise DPModeUnsupportedError(
                f"DP-FedAvg (naive BN) rejected by Opacus: {type(e).__name__}: {e}. "
                "Expected for models containing BatchNorm; use "
                "mode='fedavg_groupnorm' (swap BN→GN) or mode='fedbn' "
                "(freeze BN from DP)."
            ) from e
        return DPSetup(
            model=model_priv,
            opt_dp=opt_priv,
            opt_nondp=None,
            train_loader=loader_priv,
            privacy_engine=privacy_engine,
            mode=mode,
            target_epsilon=float(target_epsilon),
            target_delta=target_delta,
            max_grad_norm=max_grad_norm,
        )

    if mode == "fedavg_groupnorm":
        model_fixed = replace_bn_with_groupnorm(model)
        privacy_engine = PrivacyEngine()
        optimizer = _make_adamw(model_fixed.parameters(), lr, weight_decay)
        # NOTE: tried grad_sample_mode='ew' (ExpandedWeights, Opacus's
        # vectorised per-sample backend) on 2026-04-25 to speed up
        # groupnorm runs. Crashed at the second optimiser.step() with
        # "Current Expanded Weights accumulates the gradients, which
        # will be incorrect for multiple calls without clearing
        # gradients" — incompatible with how BatchMemoryManager hands
        # us logical-batch chunks and our `set_to_none=True`
        # zero_grad pattern. Reverted to the default hooks backend.
        model_priv, opt_priv, loader_priv = (
            privacy_engine.make_private_with_epsilon(
                module=model_fixed,
                optimizer=optimizer,
                data_loader=train_loader,
                target_epsilon=float(target_epsilon),
                target_delta=target_delta,
                epochs=total_epochs,
                max_grad_norm=max_grad_norm,
            )
        )
        return DPSetup(
            model=model_priv,
            opt_dp=opt_priv,
            opt_nondp=None,
            train_loader=loader_priv,
            privacy_engine=privacy_engine,
            mode=mode,
            target_epsilon=float(target_epsilon),
            target_delta=target_delta,
            max_grad_norm=max_grad_norm,
        )

    # fedbn — the core contribution.
    bn_params, non_bn_params = split_bn_and_non_bn_params(model)
    if not bn_params:
        raise ValueError(
            "mode='fedbn' but model contains no BatchNorm modules; "
            "use mode='fedavg' or 'fedavg_groupnorm' instead"
        )

    # Freeze BN so Opacus (a) skips it in ModuleValidator and (b) doesn't
    # register grad-sample hooks on it.
    for p in bn_params:
        p.requires_grad_(False)

    privacy_engine = PrivacyEngine()
    opt_dp = _make_adamw(non_bn_params, lr, weight_decay)
    try:
        model_priv, opt_dp_priv, loader_priv = (
            privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=opt_dp,
                data_loader=train_loader,
                target_epsilon=float(target_epsilon),
                target_delta=target_delta,
                epochs=total_epochs,
                max_grad_norm=max_grad_norm,
            )
        )
    finally:
        # Always unfreeze — BN must be updated locally each step.
        for p in bn_params:
            p.requires_grad_(True)

    opt_nondp = _make_adamw(bn_params, lr, weight_decay)

    return DPSetup(
        model=model_priv,
        opt_dp=opt_dp_priv,
        opt_nondp=opt_nondp,
        train_loader=loader_priv,
        privacy_engine=privacy_engine,
        mode=mode,
        target_epsilon=float(target_epsilon),
        target_delta=target_delta,
        max_grad_norm=max_grad_norm,
    )


# ---------------------------------------------------------------------------
# Spent-ε helper
# ---------------------------------------------------------------------------


def get_spent_epsilon(engine: PrivacyEngine | None, delta: float) -> float | None:
    """``engine.get_epsilon(delta)`` with a None-safe fallback."""
    if engine is None:
        return None
    try:
        return float(engine.get_epsilon(delta))
    except Exception:  # noqa: BLE001
        return None
