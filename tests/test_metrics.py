"""Tests for src/evaluation/metrics.py.

Focus on `format_classification_report` tolerance — it must not crash on
slim metric dicts (e.g. FedBN ``client_avg`` aggregates) that omit
per-class keys.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import (  # noqa: E402
    compute_metrics,
    format_classification_report,
)


def test_format_report_full_metrics_contains_per_class_block():
    y_true = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    y_pred = np.array([0, 1, 1, 1, 2, 2], dtype=np.int64)
    y_prob = np.eye(3, dtype=np.float64)[y_pred]
    m = compute_metrics(y_true, y_pred, y_prob, class_names=["N", "S", "V"])

    report = format_classification_report(m)
    assert "accuracy" in report
    assert "f1_macro" in report
    assert "per-class (P/R/F1):" in report
    for cn in ["N", "S", "V"]:
        assert cn in report


def test_format_report_slim_fedbn_client_avg_does_not_crash():
    # Shape matches what FederationServer.evaluate_central returns for
    # FedBN with fedbn_central_eval="client_avg".
    slim = {
        "accuracy": 0.982,
        "precision_macro": 0.88,
        "recall_macro": 0.91,
        "f1_macro": 0.899,
        "auc_macro": 0.994,
        "per_client_f1_macro": [0.95, 0.88, 0.92, 0.91, 0.91],
        "mode": "fedbn_client_avg",
        "class_names": ["N", "S", "V", "F", "Q"],
    }
    report = format_classification_report(slim)
    assert "accuracy: 0.9820" in report
    assert "f1_macro: 0.8990" in report
    assert "auc_macro: 0.9940" in report
    # No per-class keys → the per-class block must be absent.
    assert "per-class" not in report


def test_format_report_per_class_f1_only_falls_back_to_f1_block():
    partial = {
        "accuracy": 0.9,
        "f1_macro": 0.85,
        "auc_macro": float("nan"),
        "per_class_f1": [0.9, 0.8, 0.75],
        "class_names": ["N", "S", "V"],
    }
    report = format_classification_report(partial)
    assert "per-class F1:" in report
    assert "per-class (P/R/F1):" not in report
    assert "N: F1=0.9000" in report
    assert "auc_macro: nan" in report


def test_format_report_handles_nan_scalars():
    metrics = {
        "accuracy": float("nan"),
        "f1_macro": 0.5,
    }
    report = format_classification_report(metrics)
    assert "accuracy: nan" in report
    assert "f1_macro: 0.5000" in report


def test_format_report_missing_class_names_uses_defaults():
    partial = {
        "f1_macro": 0.5,
        "per_class_f1": [0.6, 0.4],
    }
    report = format_classification_report(partial)
    assert "c0: F1=0.6000" in report
    assert "c1: F1=0.4000" in report
