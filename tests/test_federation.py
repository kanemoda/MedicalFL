"""Federation-core unit tests.

Seven tests covering:
  1. BN key discovery for CNN1D.
  2. FedAvg = weighted mean of non-BN state.
  3. FedBN skips BN keys.
  4. FedPerf weights move mass toward higher-F1 clients.
  5. FedProx proximal term actually pulls weights toward the global anchor.
  6. One client round runs end-to-end with consistent loader shapes.
  7. FederationServer runs a round with each of the four strategies and returns
     the expected metric keys.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.federation.aggregation import (  # noqa: E402
    AGGREGATION_STRATEGIES,
    aggregate,
    fedavg_aggregate,
    fedbn_aggregate,
    fedperf_aggregate,
    get_bn_key_set,
)
from src.federation.client import FederatedClient  # noqa: E402
from src.federation.server import FederationServer  # noqa: E402
from src.models.cnn1d import CNN1D  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _toy_client_arrays(
    seed: int, num_classes: int = 5, signal_len: int = 250, n: int = 128
):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, signal_len)).astype(np.float32)
    y = rng.integers(0, num_classes, size=n, dtype=np.int64)
    return X, y


def _make_client(client_id: int, seed: int, device: torch.device,
                 num_classes: int = 5, signal_len: int = 250) -> FederatedClient:
    X_tr, y_tr = _toy_client_arrays(seed, num_classes, signal_len, n=64)
    X_va, y_va = _toy_client_arrays(seed + 1000, num_classes, signal_len, n=32)
    # Ensure all classes appear somewhere (so metrics don't get weird).
    y_tr[:num_classes] = np.arange(num_classes)
    y_va[:num_classes] = np.arange(num_classes)
    return FederatedClient(
        client_id=client_id,
        X_train=X_tr, y_train=y_tr,
        X_val=X_va, y_val=y_va,
        model_fn=lambda: CNN1D(num_classes=num_classes),
        num_classes=num_classes,
        class_names=[f"c{i}" for i in range(num_classes)],
        device=device,
        batch_size=16,
        lr=1e-3,
        weight_decay=0.0,
        optimizer="adamw",
    )


# ---------------------------------------------------------------------------
# 1. BN key discovery
# ---------------------------------------------------------------------------

def test_get_bn_key_set_on_cnn1d():
    m = CNN1D(num_classes=5)
    bn_keys = get_bn_key_set(m)

    # CNN1D has bn1..bn4 + bn_fc → 5 BN modules. Each contributes weight, bias,
    # running_mean, running_var, num_batches_tracked = 5 keys. Total 25.
    assert len(bn_keys) == 25

    for stem in ["bn1", "bn2", "bn3", "bn4", "bn_fc"]:
        for attr in ["weight", "bias", "running_mean", "running_var", "num_batches_tracked"]:
            assert f"{stem}.{attr}" in bn_keys, f"missing {stem}.{attr}"

    # Must NOT include conv / fc keys.
    sd = m.state_dict()
    conv_keys = [k for k in sd if k.startswith(("conv", "fc1", "fc2"))]
    assert all(k not in bn_keys for k in conv_keys), "conv/fc keys leaked into bn_keys"


# ---------------------------------------------------------------------------
# 2. FedAvg = weighted mean
# ---------------------------------------------------------------------------

def test_fedavg_is_weighted_mean():
    sd_a = {"w": torch.tensor([1.0, 2.0, 3.0])}
    sd_b = {"w": torch.tensor([4.0, 4.0, 4.0])}
    # 100 vs 300 samples → weights 0.25, 0.75.
    agg = fedavg_aggregate([sd_a, sd_b], [100, 300])
    expected = 0.25 * sd_a["w"] + 0.75 * sd_b["w"]
    assert torch.allclose(agg["w"], expected, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. FedBN skips BN keys
# ---------------------------------------------------------------------------

def test_fedbn_excludes_bn_keys():
    m = CNN1D(num_classes=5)
    bn_keys = get_bn_key_set(m)

    sd_a = {k: v.detach().clone() for k, v in m.state_dict().items()}
    sd_b = {k: v.detach().clone() + 1.0 if v.is_floating_point() else v.clone()
            for k, v in m.state_dict().items()}

    agg = fedbn_aggregate([sd_a, sd_b], [1, 1], bn_keys)

    # No BN key is in the aggregate.
    assert all(k not in bn_keys for k in agg.keys())
    # Every non-BN key is present.
    non_bn_keys = set(sd_a.keys()) - bn_keys
    assert set(agg.keys()) == non_bn_keys
    # Conv weight is the mean of the two.
    k = "conv1.weight"
    assert torch.allclose(agg[k], (sd_a[k] + sd_b[k]) / 2.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. FedPerf weights move toward higher-F1 clients
# ---------------------------------------------------------------------------

def test_fedperf_weights_shift_toward_high_f1():
    sd_lo = {"w": torch.tensor([0.0])}
    sd_hi = {"w": torch.tensor([1.0])}
    sizes = [1000, 1000]  # equal — isolates the F1 term
    # alpha=0 → pure performance weighting.
    agg_perf = fedperf_aggregate([sd_lo, sd_hi], sizes, val_f1=[0.1, 0.9], alpha=0.0)
    assert agg_perf["w"].item() > 0.5, "high-F1 client should dominate"
    # alpha=1 → pure size weighting (equal sizes → 0.5).
    agg_size = fedperf_aggregate([sd_lo, sd_hi], sizes, val_f1=[0.1, 0.9], alpha=1.0)
    assert abs(agg_size["w"].item() - 0.5) < 1e-6
    # Mixed: between 0.5 and agg_perf.
    agg_mix = fedperf_aggregate([sd_lo, sd_hi], sizes, val_f1=[0.1, 0.9], alpha=0.5)
    assert 0.5 < agg_mix["w"].item() < agg_perf["w"].item()


# ---------------------------------------------------------------------------
# 5. FedProx proximal term pulls toward anchor
# ---------------------------------------------------------------------------

def test_fedprox_proximal_pulls_weights():
    """With a large mu, one local step should change weights LESS than mu=0
    because the proximal term penalizes deviation from the global anchor."""
    torch.manual_seed(0)
    device = torch.device("cpu")
    client_no = _make_client(0, seed=0, device=device)
    client_yes = _make_client(0, seed=0, device=device)

    # Identical starting point.
    init_state = {k: v.detach().cpu().clone() for k, v in client_no.model.state_dict().items()}
    client_yes.set_parameters(init_state)

    # One epoch, no prox.
    client_no.local_train(local_epochs=1, proximal_mu=0.0, global_params=None)
    # One epoch, heavy prox anchored to init.
    client_yes.local_train(local_epochs=1, proximal_mu=10.0, global_params=init_state)

    no_diff = 0.0
    yes_diff = 0.0
    for k, v0 in init_state.items():
        if not v0.is_floating_point():
            continue
        no_diff += (client_no.model.state_dict()[k].cpu() - v0).norm().item()
        yes_diff += (client_yes.model.state_dict()[k].cpu() - v0).norm().item()
    assert yes_diff < no_diff, (
        f"proximal mu=10 drift ({yes_diff:.4f}) should be smaller than "
        f"mu=0 drift ({no_diff:.4f})"
    )


# ---------------------------------------------------------------------------
# 6. End-to-end client round
# ---------------------------------------------------------------------------

def test_client_round_end_to_end():
    torch.manual_seed(0)
    device = torch.device("cpu")
    client = _make_client(0, seed=0, device=device)
    # Train one local epoch.
    train_metrics = client.local_train(local_epochs=1)
    assert "loss" in train_metrics
    assert np.isfinite(train_metrics["loss"])
    # Evaluate.
    val_metrics = client.evaluate()
    assert val_metrics["num_samples"] > 0
    assert np.isfinite(val_metrics["loss"])
    # get_parameters returns cpu tensors with same keys as the model state_dict.
    params = client.get_parameters()
    sd_keys = set(client.model.state_dict().keys())
    assert set(params.keys()) == sd_keys
    for v in params.values():
        assert v.device.type == "cpu"


# ---------------------------------------------------------------------------
# 7. Server round with each strategy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", sorted(AGGREGATION_STRATEGIES))
def test_server_round_all_strategies(strategy: str):
    torch.manual_seed(0)
    device = torch.device("cpu")
    clients = [_make_client(i, seed=i, device=device) for i in range(3)]
    proximal_mu = 0.01 if strategy == "fedprox" else 0.0
    server = FederationServer(
        clients=clients,
        model_fn=lambda: CNN1D(num_classes=5),
        strategy=strategy,
        device=device,
        class_names=[f"c{i}" for i in range(5)],
        proximal_mu=proximal_mu,
    )
    info = server.run_round(local_epochs=1)

    for key in [
        "num_samples", "train_loss_per_client", "train_f1_per_client",
        "val_loss_per_client", "val_f1_per_client", "val_f1_weighted",
    ]:
        assert key in info, f"missing {key} from round info (strategy={strategy})"
    assert len(info["val_f1_per_client"]) == len(clients)


# ---------------------------------------------------------------------------
# Unified aggregate() dispatch sanity
# ---------------------------------------------------------------------------

def test_aggregate_dispatch_rejects_unknown():
    sd = {"w": torch.zeros(3)}
    with pytest.raises(ValueError):
        aggregate("notastrategy", [sd], [1])
