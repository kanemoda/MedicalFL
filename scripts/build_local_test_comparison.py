#!/usr/bin/env python
"""Build the central-vs-local comparison for Dir α=0.1 and label_skew_c2.

Outputs:
  - results/metrics/central_vs_local_dirichlet_a01.csv
  - results/metrics/central_vs_local_label_skew_c2.csv
  - results/figures/fig_central_vs_local.{png,pdf}
  - stdout: two markdown tables + 8 local test F1 values + one-line verdict.

For FedAvg / FedProx / FedPerf the original Phase 4 `{run_id}.json` is used.
For FedBN the `{run_id}_v2.json` (fresh retrain with checkpoint) is used —
the Phase 4 FedBN artifacts were recovered from rounds.csv only and lack
saved model states, so the local-test shortcut cannot run against them.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = REPO_ROOT / "results" / "metrics"
FIG_DIR = REPO_ROOT / "results" / "figures"

AGGREGATIONS = ["fedavg", "fedprox", "fedbn", "fedperf"]
AGGREGATION_LABELS = {
    "fedavg": "FedAvg", "fedprox": "FedProx",
    "fedbn": "FedBN", "fedperf": "FedPerf",
}
PARTITIONS = [
    ("dirichlet_a01", "Dirichlet α=0.1"),
    ("label_skew_c2", "Label skew (C=2)"),
]


def _json_path_for(aggregation: str, partition: str) -> Path:
    base = f"{aggregation}_{partition}_mitbih_seed42"
    if aggregation == "fedbn":
        # v2 retrain — has saved checkpoint and emits local_test block
        return METRICS_DIR / f"{base}_v2.json"
    return METRICS_DIR / f"{base}.json"


def _load_row(aggregation: str, partition: str) -> dict:
    path = _json_path_for(aggregation, partition)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    data = json.loads(path.read_text())
    central = float(data.get("best_central_f1_macro", float("nan")))
    lt = data.get("local_test") or {}
    per_client = lt.get("per_client_local_test_f1") or []
    local_mean = float(lt.get("local_test_f1_mean", float("nan")))
    local_std = float(lt.get("local_test_f1_std", float("nan")))
    return {
        "aggregation": aggregation,
        "partition": partition,
        "central_f1": central,
        "local_f1_mean": local_mean,
        "local_f1_std": local_std,
        "delta": local_mean - central,
        "per_client_local_f1": per_client,
        "json": str(path.relative_to(REPO_ROOT)),
    }


def _print_table(partition: str, label: str, rows: list[dict]) -> None:
    print(f"\n### {label} — central vs local test F1 (macro)")
    print()
    print("| Aggregation | Central F1 | Local F1 (mean ± std) | Δ (local − central) |")
    print("|---|---:|---:|---:|")
    for row in rows:
        d = row["delta"]
        arrow = "↑" if d > 0.01 else ("↓" if d < -0.01 else "≈")
        print(
            f"| {AGGREGATION_LABELS[row['aggregation']]} "
            f"| {row['central_f1']:.3f} "
            f"| {row['local_f1_mean']:.3f} ± {row['local_f1_std']:.3f} "
            f"| {arrow} {d:+.3f} |"
        )


def _write_csv(partition: str, rows: list[dict]) -> Path:
    path = METRICS_DIR / f"central_vs_local_{partition}.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "aggregation", "partition", "central_f1",
            "local_f1_mean", "local_f1_std", "delta",
            "per_client_local_f1",
        ])
        for row in rows:
            w.writerow([
                row["aggregation"], row["partition"],
                f"{row['central_f1']:.6f}",
                f"{row['local_f1_mean']:.6f}",
                f"{row['local_f1_std']:.6f}",
                f"{row['delta']:.6f}",
                ";".join(f"{v:.6f}" for v in row["per_client_local_f1"]),
            ])
    return path


def _plot_grouped_bars(
    rows_by_partition: dict[str, list[dict]], out_base: Path,
) -> None:
    plt.rcParams.update({
        "font.size": 10,
        "font.family": "serif",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, (part, label) in zip(axes, PARTITIONS):
        rows = rows_by_partition[part]
        x = np.arange(len(AGGREGATIONS))
        central = np.array([r["central_f1"] for r in rows])
        local = np.array([r["local_f1_mean"] for r in rows])
        local_std = np.array([r["local_f1_std"] for r in rows])
        width = 0.38
        ax.bar(x - width / 2, central, width=width, color="#4477AA",
               label="Central test F1 (best round)")
        ax.bar(x + width / 2, local, width=width, yerr=local_std, color="#EE6677",
               capsize=3, label="Local test F1 (per-client mean ± std)")
        ax.set_xticks(x)
        ax.set_xticklabels([AGGREGATION_LABELS[a] for a in AGGREGATIONS])
        ax.set_title(label)
        ax.set_ylabel("F1 macro")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        for xi, (c, lmean) in enumerate(zip(central, local)):
            ax.text(xi - width / 2, c + 0.01, f"{c:.2f}", ha="center",
                    va="bottom", fontsize=8)
            ax.text(xi + width / 2, lmean + 0.01, f"{lmean:.2f}",
                    ha="center", va="bottom", fontsize=8)
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle(
        "Central vs per-client local test F1 under severe non-IID (MIT-BIH)",
        y=1.02,
    )
    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _verdict(rows_by_partition: dict[str, list[dict]]) -> str:
    """One-sentence assessment of whether the 'eval regime' story holds."""
    ls = {r["aggregation"]: r for r in rows_by_partition["label_skew_c2"]}
    if "fedbn" not in ls or "fedavg" not in ls:
        return "Verdict: inconclusive (missing FedBN or FedAvg label_skew rows)."
    fedbn_local = ls["fedbn"]["local_f1_mean"]
    fedavg_local = ls["fedavg"]["local_f1_mean"]
    if np.isnan(fedbn_local) or np.isnan(fedavg_local):
        return "Verdict: inconclusive (NaN local F1 in target row)."
    if fedbn_local > fedavg_local:
        return (
            f"Verdict: YES — FedBN local F1 ({fedbn_local:.3f}) exceeds FedAvg "
            f"local F1 ({fedavg_local:.3f}) on label_skew_c2, so the 'eval "
            f"regime' frame holds: FedBN's central-test collapse is not a "
            f"training failure, it is a mismatch between a personalized model "
            f"and a uniform global test distribution."
        )
    return (
        f"Verdict: NO — FedBN local F1 ({fedbn_local:.3f}) does NOT exceed "
        f"FedAvg local F1 ({fedavg_local:.3f}) on label_skew_c2; the eval "
        f"regime story alone cannot explain the central-test collapse."
    )


def main() -> int:
    rows_by_partition: dict[str, list[dict]] = {}
    all_local: list[tuple[str, str, float]] = []

    missing: list[str] = []
    for part, label in PARTITIONS:
        rows: list[dict] = []
        for agg in AGGREGATIONS:
            try:
                row = _load_row(agg, part)
            except FileNotFoundError as e:
                missing.append(str(e))
                continue
            rows.append(row)
            all_local.append((agg, part, row["local_f1_mean"]))
        rows_by_partition[part] = rows

    if missing:
        print("MISSING JSONs:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        if len(missing) == len(AGGREGATIONS) * len(PARTITIONS):
            return 2

    # -------- 8 local test F1 values --------
    print("\n## 8 local test F1 values")
    print()
    for agg, part, val in all_local:
        print(f"  {AGGREGATION_LABELS[agg]:>7s}  {part:>14s}  local_f1_mean={val:.4f}")

    # -------- 2 comparison tables --------
    for part, label in PARTITIONS:
        rows = rows_by_partition.get(part, [])
        if rows:
            _print_table(part, label, rows)
            path = _write_csv(part, rows)
            print(f"\n(wrote {path.relative_to(REPO_ROOT)})")

    # -------- figure --------
    if all(rows_by_partition.get(p) for p, _ in PARTITIONS):
        out_base = FIG_DIR / "fig_central_vs_local"
        _plot_grouped_bars(rows_by_partition, out_base)
        print(
            f"\n(wrote {out_base.relative_to(REPO_ROOT)}.png and "
            f"{out_base.relative_to(REPO_ROOT)}.pdf)"
        )

    # -------- one-line verdict --------
    print()
    print(_verdict(rows_by_partition))
    return 0


if __name__ == "__main__":
    sys.exit(main())
