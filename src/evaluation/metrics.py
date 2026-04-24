"""Classification metrics for ECG experiments.

`compute_metrics` returns a dict with accuracy, macro-averaged precision /
recall / F1, per-class values (NaN where a class is absent from `y_true`),
OVR AUC, and a confusion matrix. Designed to be the single source of truth
for every centralized + federated run so reports stay comparable.
"""
from __future__ import annotations

import math
import warnings
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def _per_class_or_nan(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class (precision, recall, f1); NaN for classes not seen in y_true."""
    labels = list(range(num_classes))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    present = np.isin(labels, np.unique(y_true))
    prec = np.where(present, prec, np.nan)
    rec = np.where(present, rec, np.nan)
    f1 = np.where(present, f1, np.nan)
    return prec, rec, f1


def _macro_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
) -> float:
    """OVR macro AUC; NaN if fewer than two classes are present in y_true.

    For binary (num_classes == 2), sklearn wants the positive-class score,
    not a multi_class='ovr' call with labels.
    """
    present = np.unique(y_true)
    if present.size < 2:
        warnings.warn("macro AUC undefined with <2 classes in y_true; returning NaN")
        return float("nan")
    try:
        if num_classes == 2:
            return float(roc_auc_score(y_true, y_prob[:, 1]))
        return float(
            roc_auc_score(
                y_true,
                y_prob,
                multi_class="ovr",
                average="macro",
                labels=list(range(num_classes)),
            )
        )
    except ValueError as e:
        warnings.warn(f"roc_auc_score raised {type(e).__name__}: {e}; returning NaN")
        return float("nan")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str],
) -> dict:
    """Compute the standard metric bundle reported for every run.

    Args:
        y_true: (N,) int64 ground-truth labels in [0, num_classes).
        y_pred: (N,) int64 predicted labels.
        y_prob: (N, num_classes) float softmax probabilities.
        class_names: length-num_classes list of class names for the report.

    Returns a JSON-serializable dict; NaN values are preserved as `float('nan')`
    (callers that write JSON should convert to `None` or string as needed).
    """
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    num_classes = len(class_names)

    accuracy = float((y_true == y_pred).mean()) if y_true.size else float("nan")
    per_prec, per_rec, per_f1 = _per_class_or_nan(y_true, y_pred, num_classes)

    # Macro = mean across classes present in y_true only.
    def _nanmean(a: np.ndarray) -> float:
        a = a[~np.isnan(a)]
        return float(a.mean()) if a.size else float("nan")

    precision_macro = _nanmean(per_prec)
    recall_macro = _nanmean(per_rec)
    f1_macro = _nanmean(per_f1)
    auc_macro = _macro_auc(y_true, y_prob, num_classes)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # JSON-safe per-class entries: NaN → None when the caller serializes; here
    # we keep float NaN so matplotlib/printouts work naturally.
    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "auc_macro": auc_macro,
        "per_class_precision": per_prec.tolist(),
        "per_class_recall": per_rec.tolist(),
        "per_class_f1": per_f1.tolist(),
        "class_names": list(class_names),
        "confusion_matrix": cm.tolist(),
        "support": np.bincount(y_true, minlength=num_classes).tolist(),
    }


def metrics_to_jsonable(metrics: dict) -> dict:
    """Replace NaN entries with None so `json.dumps` works without NaNs."""
    def _clean(v):
        if isinstance(v, float) and math.isnan(v):
            return None
        if isinstance(v, list):
            return [_clean(x) for x in v]
        return v

    return {k: _clean(v) for k, v in metrics.items()}


def format_classification_report(metrics: dict) -> str:
    """Pretty-print the available classification metrics.

    Tolerates slim metric dicts (e.g. FedBN ``client_avg`` aggregates) that
    omit per-class breakdowns or support counts. Whatever keys are present
    are printed; whatever is missing is simply skipped.
    """
    def _fmt_f(v) -> str:
        return f"{v:.4f}" if isinstance(v, (int, float)) and not (
            isinstance(v, float) and math.isnan(v)
        ) else "nan"

    lines: list[str] = []
    if "accuracy" in metrics:
        lines.append(f"accuracy: {_fmt_f(metrics['accuracy'])}")
    if "f1_macro" in metrics:
        lines.append(f"f1_macro: {_fmt_f(metrics['f1_macro'])}")
    if "auc_macro" in metrics:
        lines.append(f"auc_macro: {_fmt_f(metrics['auc_macro'])}")

    if "per_class_f1" in metrics:
        per_f1 = metrics["per_class_f1"]
        names = metrics.get(
            "class_names", [f"c{i}" for i in range(len(per_f1))]
        )
        has_pr = "per_class_precision" in metrics and "per_class_recall" in metrics
        if has_pr:
            lines.append("per-class (P/R/F1):")
            for i, name in enumerate(names):
                p = metrics["per_class_precision"][i]
                r = metrics["per_class_recall"][i]
                f = per_f1[i]
                lines.append(
                    f"  {name}: P={_fmt_f(p)} R={_fmt_f(r)} F1={_fmt_f(f)}"
                )
        else:
            lines.append("per-class F1:")
            for i, name in enumerate(names):
                lines.append(f"  {name}: F1={_fmt_f(per_f1[i])}")

    return "\n".join(lines)
