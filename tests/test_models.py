"""Unit tests for CNN1D model and metrics helpers."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.models.cnn1d import CNN1D  # noqa: E402


def test_cnn1d_forward_mitbih_shape():
    model = CNN1D(num_classes=5).eval()
    x = torch.randn(4, 1, 250)
    out = model(x)
    assert out.shape == (4, 5)


def test_cnn1d_forward_ptbxl_shape():
    model = CNN1D(num_classes=5).eval()
    x = torch.randn(4, 1, 1000)
    out = model(x)
    assert out.shape == (4, 5)


def test_cnn1d_input_transpose_handling():
    model = CNN1D(num_classes=5).eval()
    x_chw = torch.randn(4, 1, 250)
    x_hwc = x_chw.transpose(1, 2).contiguous()  # (4, 250, 1)
    with torch.no_grad():
        out_chw = model(x_chw)
        out_hwc = model(x_hwc)
    assert torch.allclose(out_chw, out_hwc, atol=1e-5)


def test_cnn1d_grad_flow():
    model = CNN1D(num_classes=5).train()
    x = torch.randn(8, 1, 250, requires_grad=False)
    y = torch.randint(0, 5, (8,))
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


def test_cnn1d_param_count():
    model = CNN1D(num_classes=5)
    total = sum(p.numel() for p in model.parameters())
    assert 200_000 < total < 800_000, f"Unexpected param count: {total}"


def test_metrics_handles_missing_class():
    class_names = ["N", "S", "V", "F", "Q"]
    rng = np.random.default_rng(0)
    # Only classes 0..3 present in ground truth; class 4 is absent.
    y_true = rng.integers(0, 4, size=200)
    y_pred = rng.integers(0, 5, size=200)
    y_prob = rng.random(size=(200, 5))
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    m = compute_metrics(y_true, y_pred, y_prob, class_names)
    # Per-class F1 for the absent class (index 4) must be NaN.
    assert math.isnan(m["per_class_f1"][4])
    # Macro averages stay finite (they exclude the absent class).
    assert math.isfinite(m["f1_macro"])


def test_metrics_single_class_returns_nan_auc():
    class_names = ["N", "S", "V", "F", "Q"]
    y_true = np.zeros(50, dtype=np.int64)
    y_pred = np.zeros(50, dtype=np.int64)
    y_prob = np.zeros((50, 5), dtype=np.float64)
    y_prob[:, 0] = 1.0
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = compute_metrics(y_true, y_pred, y_prob, class_names)
    assert math.isnan(m["auc_macro"])
