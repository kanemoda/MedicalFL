"""DP wiring unit tests (Phase 5).

Four tests covering the privacy plumbing that the sweep relies on:

  1. ``test_dp_bn_identification`` — ``split_bn_and_non_bn_params`` counts
     match what CNN1D actually carries (5 BN modules × 2 learnables + 12
     non-BN tensors from 4 convs / fc1 / fc2).
  2. ``test_dp_fedbn_training_step`` — one full forward / backward / step
     under DP-FedBN actually updates BN (via ``opt_nondp``) *and* non-BN
     (via the Opacus-wrapped ``opt_dp``) parameters.
  3. ``test_dp_privacy_accounting`` — after the declared training horizon,
     ``privacy_engine.get_epsilon(δ)`` lands within 10% of the target ε
     that was handed to Opacus.
  4. ``test_dp_groupnorm_replacement`` — ``ModuleValidator.fix`` yields a
     BN-free model that still forward-passes and is accepted by Opacus.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from opacus.validators import ModuleValidator
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.federation.aggregation import BN_MODULES  # noqa: E402
from src.federation.dp import (  # noqa: E402
    DPModeUnsupportedError,
    replace_bn_with_groupnorm,
    setup_dp_training,
    split_bn_and_non_bn_params,
)
from src.models.cnn1d import CNN1D  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIGNAL_LEN = 250
NUM_CLASSES = 5
BATCH_SIZE = 8
DATASET_SIZE = 64  # keeps the unit tests fast while giving Opacus headroom


def _make_loader(seed: int = 0) -> DataLoader:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((DATASET_SIZE, SIGNAL_LEN)).astype(np.float32)
    y = rng.integers(0, NUM_CLASSES, size=DATASET_SIZE, dtype=np.int64)
    y[:NUM_CLASSES] = np.arange(NUM_CLASSES)  # ensure all classes present
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 1. BN / non-BN parameter split
# ---------------------------------------------------------------------------

def test_dp_bn_identification():
    model = CNN1D(num_classes=NUM_CLASSES)

    bn_params, non_bn_params = split_bn_and_non_bn_params(model)

    bn_module_count = sum(
        1 for _, m in model.named_modules() if isinstance(m, BN_MODULES)
    )
    assert bn_module_count == 5, (
        f"CNN1D should expose 5 BN modules (bn1..bn4 + bn_fc); got {bn_module_count}"
    )
    # 5 BN modules × {weight, bias} = 10 tensors.
    assert len(bn_params) == 10, (
        f"expected 10 BN tensors, got {len(bn_params)}"
    )
    # Non-BN: 4 convs × {weight, bias} + fc1 × {weight, bias}
    #       + fc2 × {weight, bias} = 12 tensors.
    assert len(non_bn_params) == 12, (
        f"expected 12 non-BN tensors, got {len(non_bn_params)}"
    )

    # The two sets must partition the model parameters — no overlap, no loss.
    bn_ids = {id(p) for p in bn_params}
    non_bn_ids = {id(p) for p in non_bn_params}
    all_ids = {id(p) for p in model.parameters()}
    assert bn_ids.isdisjoint(non_bn_ids)
    assert bn_ids | non_bn_ids == all_ids


# ---------------------------------------------------------------------------
# 2. DP-FedBN training step actually moves the right params
# ---------------------------------------------------------------------------

def test_dp_fedbn_training_step():
    device = _device()
    torch.manual_seed(0)
    model = CNN1D(num_classes=NUM_CLASSES).to(device)
    loader = _make_loader(seed=1)

    setup = setup_dp_training(
        model=model,
        train_loader=loader,
        target_epsilon=3.0,
        target_delta=1e-5,
        max_grad_norm=1.0,
        total_epochs=1,
        mode="fedbn",
        lr=1e-3,
        weight_decay=0.0,
    )

    # Snapshot a BN weight (updated by opt_nondp) and a conv weight
    # (updated by the Opacus-wrapped opt_dp).
    inner = setup.model
    while hasattr(inner, "_module") and isinstance(inner._module, nn.Module):
        inner = inner._module

    bn_before = inner.bn1.weight.detach().cpu().clone()
    conv_before = inner.conv1.weight.detach().cpu().clone()

    setup.model.train()
    criterion = nn.CrossEntropyLoss()

    # A single step through the Opacus-wrapped loader — Poisson sampling
    # occasionally returns zero-sample batches; loop until we get one.
    stepped = False
    for Xb, yb in setup.train_loader:
        if yb.numel() == 0:
            continue
        Xb = Xb.to(device)
        yb = yb.to(device)
        setup.opt_dp.zero_grad(set_to_none=True)
        assert setup.opt_nondp is not None, "DP-FedBN must expose opt_nondp"
        setup.opt_nondp.zero_grad(set_to_none=True)
        logits = setup.model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        setup.opt_dp.step()
        setup.opt_nondp.step()
        stepped = True
        break
    assert stepped, "never reached a non-empty Poisson batch"

    bn_after = inner.bn1.weight.detach().cpu().clone()
    conv_after = inner.conv1.weight.detach().cpu().clone()

    # BN params should have moved via the plain (non-DP) optimiser.
    assert not torch.equal(bn_before, bn_after), (
        "BN weights didn't update — opt_nondp.step() may be missing"
    )
    # Non-BN params move via the DP-wrapped optimiser (clip + noise).
    assert not torch.equal(conv_before, conv_after), (
        "Conv weights didn't update — opt_dp.step() may be missing"
    )
    # Non-BN params now carry Opacus-maintained per-sample grads.
    assert hasattr(inner.conv1.weight, "grad_sample") or (
        inner.conv1.weight.grad is not None
    )
    # Confirm the privacy engine was attached — the whole point of the mode.
    assert setup.privacy_engine is not None
    assert setup.mode == "fedbn"


# ---------------------------------------------------------------------------
# 3. Privacy accounting — achieved ε ≈ target ε
# ---------------------------------------------------------------------------

def test_dp_privacy_accounting():
    device = _device()
    torch.manual_seed(0)
    model = CNN1D(num_classes=NUM_CLASSES).to(device)
    loader = _make_loader(seed=2)

    target_eps = 3.0
    target_delta = 1e-5
    epochs = 3

    setup = setup_dp_training(
        model=model,
        train_loader=loader,
        target_epsilon=target_eps,
        target_delta=target_delta,
        max_grad_norm=1.0,
        total_epochs=epochs,
        mode="fedbn",
        lr=1e-3,
        weight_decay=0.0,
    )
    setup.model.train()
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for Xb, yb in setup.train_loader:
            if yb.numel() == 0:
                continue
            Xb = Xb.to(device)
            yb = yb.to(device)
            setup.opt_dp.zero_grad(set_to_none=True)
            assert setup.opt_nondp is not None
            setup.opt_nondp.zero_grad(set_to_none=True)
            loss = criterion(setup.model(Xb), yb)
            loss.backward()
            setup.opt_dp.step()
            setup.opt_nondp.step()

    achieved = float(setup.privacy_engine.get_epsilon(target_delta))
    # Opacus calibrates the noise multiplier so achieved ε ≈ target ε at
    # the end of the declared horizon. Being within 10% confirms (a) the
    # accountant is active and (b) we iterated the expected number of
    # Poisson-sampled batches.
    assert abs(achieved - target_eps) / target_eps < 0.10, (
        f"achieved ε={achieved:.3f} too far from target ε={target_eps} "
        f"(|Δ|/target ≥ 10%)"
    )


# ---------------------------------------------------------------------------
# 4. BatchNorm → GroupNorm replacement
# ---------------------------------------------------------------------------

def test_dp_groupnorm_replacement():
    device = _device()
    torch.manual_seed(0)
    model = CNN1D(num_classes=NUM_CLASSES).to(device)

    fixed = replace_bn_with_groupnorm(model)

    # No BatchNorm modules must remain in the fixed model.
    remaining_bn = [
        name for name, m in fixed.named_modules() if isinstance(m, BN_MODULES)
    ]
    assert not remaining_bn, (
        f"GN replacement should have removed all BN; found {remaining_bn}"
    )
    # At least one GroupNorm should have been injected.
    gn_count = sum(
        1 for _, m in fixed.named_modules() if isinstance(m, nn.GroupNorm)
    )
    assert gn_count >= 1, "expected GroupNorm modules after BN→GN swap"

    # Model must still forward-pass end-to-end.
    fixed = fixed.to(device)
    fixed.eval()
    with torch.no_grad():
        x = torch.randn(2, 1, SIGNAL_LEN, device=device)
        out = fixed(x)
    assert out.shape == (2, NUM_CLASSES)

    # Opacus must accept the GN model (this is the whole reason we swap).
    # ModuleValidator insists on train() mode for its checks.
    fixed.train()
    errors = ModuleValidator.validate(fixed, strict=False)
    assert not errors, (
        f"Opacus ModuleValidator still rejects the GN model: {errors}"
    )

    # And the setup factory itself must now succeed on fedavg_groupnorm.
    loader = _make_loader(seed=3)
    torch.manual_seed(0)
    fresh = CNN1D(num_classes=NUM_CLASSES).to(device)
    setup = setup_dp_training(
        model=fresh,
        train_loader=loader,
        target_epsilon=3.0,
        target_delta=1e-5,
        max_grad_norm=1.0,
        total_epochs=1,
        mode="fedavg_groupnorm",
        lr=1e-3,
        weight_decay=0.0,
    )
    assert setup.privacy_engine is not None
    assert setup.mode == "fedavg_groupnorm"
    # After swap the inner module should contain no BN anywhere.
    inner = setup.model
    while hasattr(inner, "_module") and isinstance(inner._module, nn.Module):
        inner = inner._module
    bn_after = [n for n, m in inner.named_modules() if isinstance(m, BN_MODULES)]
    assert not bn_after, f"fedavg_groupnorm setup left BN modules: {bn_after}"


# ---------------------------------------------------------------------------
# 5. Sanity: naive BN + DP is correctly refused
# ---------------------------------------------------------------------------

def test_dp_fedavg_naive_bn_refused():
    """We explicitly want this path to raise — the sweep uses the error as
    a first-class signal that naive DP-FedAvg is incompatible with BN."""
    device = _device()
    model = CNN1D(num_classes=NUM_CLASSES).to(device)
    loader = _make_loader(seed=4)
    with pytest.raises(DPModeUnsupportedError):
        setup_dp_training(
            model=model,
            train_loader=loader,
            target_epsilon=3.0,
            target_delta=1e-5,
            max_grad_norm=1.0,
            total_epochs=1,
            mode="fedavg",
            lr=1e-3,
            weight_decay=0.0,
        )
