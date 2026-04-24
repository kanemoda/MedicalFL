"""Phase 4 figure generator.

`generate_phase4_figures(results_dir, out_dir)` produces the four Phase 4
paper figures from the sweep artifacts on disk. Call it after
`scripts/run_noniid_sweep.py` has written `noniid_summary.csv` plus one
`<run_id>_rounds.csv` per run.

Outputs (both .png at 300 DPI and .pdf vector):
  - fig_noniid_heatmap             — 6 × 4 F1 matrix
  - fig_noniid_convergence         — 6-panel convergence grid (1 per partition)
  - fig_partition_distributions_mitbih — 6-panel client class bars
  - fig_degradation_from_iid       — grouped bars of ΔF1 vs IID
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Canonical Phase 4 orderings.
PARTITION_ORDER = [
    "iid",
    "dirichlet_a10",
    "dirichlet_a05",
    "dirichlet_a01",
    "quantity_skew",
    "label_skew_c2",
]
PARTITION_LABELS = {
    "iid": "IID",
    "dirichlet_a10": "Dir(α=1.0)",
    "dirichlet_a05": "Dir(α=0.5)",
    "dirichlet_a01": "Dir(α=0.1)",
    "quantity_skew": "Quantity skew (β=0.5)",
    "label_skew_c2": "Label skew (C=2)",
}
AGGREGATION_ORDER = ["fedavg", "fedprox", "fedbn", "fedperf"]
AGGREGATION_LABELS = {
    "fedavg": "FedAvg", "fedprox": "FedProx",
    "fedbn": "FedBN",  "fedperf": "FedPerf",
}


def _style() -> None:
    plt.rcParams.update({
        "font.size": 10,
        "font.family": "serif",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def _save(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary loader
# ---------------------------------------------------------------------------

def _load_summary(path: Path) -> dict[tuple[str, str], dict]:
    """Return {(partition, aggregation) → row}.

    The summary CSV schema is:
      run_id, partition, aggregation, status, final_central_f1,
      best_central_f1, best_round, wall_clock_s

    Rows with status ``DONE`` or ``RECOVERED`` are both included — the
    recovered FedBN runs have the same ``central_f1_macro`` data quality
    as the DONE runs, just sourced from rounds.csv rather than a final
    evaluation that crashed.

    ``f1_primary`` is the value used by the heatmap / degradation bars —
    we report the best central-round F1 (consistent with the figure
    title "best central round").
    """
    out: dict[tuple[str, str], dict] = {}
    valid_statuses = {"DONE", "RECOVERED"}
    with path.open() as f:
        for row in csv.DictReader(f):
            if row.get("status", "DONE") not in valid_statuses:
                continue
            part = row["partition"]
            agg = row["aggregation"]

            def _to_float(key: str) -> float:
                raw = row.get(key, "")
                try:
                    return float(raw) if raw else float("nan")
                except ValueError:
                    return float("nan")

            best = _to_float("best_central_f1")
            final = _to_float("final_central_f1")
            out[(part, agg)] = {
                **row,
                "best_central_f1": best,
                "final_central_f1": final,
                "f1_primary": best,
            }
    return out


# ---------------------------------------------------------------------------
# Figure: 6×4 heatmap
# ---------------------------------------------------------------------------

def _heatmap(summary: dict[tuple[str, str], dict], out_base: Path) -> None:
    grid = np.full((len(PARTITION_ORDER), len(AGGREGATION_ORDER)), np.nan)
    for i, part in enumerate(PARTITION_ORDER):
        for j, agg in enumerate(AGGREGATION_ORDER):
            row = summary.get((part, agg))
            if row and np.isfinite(row["f1_primary"]):
                grid[i, j] = row["f1_primary"]

    finite = grid[np.isfinite(grid)]
    vmin = float(finite.min()) if finite.size else 0.0
    vmax = float(finite.max()) if finite.size else 1.0

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(AGGREGATION_ORDER)))
    ax.set_xticklabels([AGGREGATION_LABELS[a] for a in AGGREGATION_ORDER])
    ax.set_yticks(range(len(PARTITION_ORDER)))
    ax.set_yticklabels([PARTITION_LABELS[p] for p in PARTITION_ORDER])
    ax.set_xlabel("Aggregation")
    ax.set_ylabel("Partition strategy")
    ax.set_title("MIT-BIH test F1 macro — non-IID × aggregation (best central round)")
    ax.grid(False)

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if np.isfinite(v):
                # pick text color for contrast against the cell color
                lum = (v - vmin) / (vmax - vmin + 1e-12)
                txt_color = "white" if lum < 0.55 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=txt_color)
            else:
                ax.text(j, i, "—", ha="center", va="center", color="#888")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Test F1 macro")
    _save(fig, out_base)


# ---------------------------------------------------------------------------
# Figure: 6-panel convergence grid
# ---------------------------------------------------------------------------

def _convergence_grid(
    results_dir: Path, summary: dict[tuple[str, str], dict], out_base: Path,
) -> None:
    # collect per-run central_f1_macro series
    palette = plt.get_cmap("tab10").colors

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    all_vals: list[float] = []
    series_cache: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for part in PARTITION_ORDER:
        for agg in AGGREGATION_ORDER:
            row = summary.get((part, agg))
            if not row:
                continue
            rounds_csv = results_dir / f"{row['run_id']}_rounds.csv"
            if not rounds_csv.exists():
                continue
            pts: list[tuple[int, float]] = []
            with rounds_csv.open() as f:
                for r in csv.DictReader(f):
                    try:
                        rr = int(r["round"])
                        cf = float(r["central_f1_macro"])
                    except (ValueError, KeyError):
                        continue
                    if np.isfinite(cf):
                        pts.append((rr, cf))
                        all_vals.append(cf)
            series_cache[(part, agg)] = pts

    if not all_vals:
        return
    ymin = max(0.0, min(all_vals) - 0.02)
    ymax = min(1.0, max(all_vals) + 0.02)

    legend_handles = []
    for idx, part in enumerate(PARTITION_ORDER):
        ax = axes[idx // 3, idx % 3]
        for j, agg in enumerate(AGGREGATION_ORDER):
            pts = series_cache.get((part, agg), [])
            if not pts:
                continue
            xs, ys = zip(*pts)
            line, = ax.plot(
                xs, ys, marker=".", markersize=3, linewidth=1.4,
                color=palette[j % len(palette)], label=AGGREGATION_LABELS[agg],
            )
            if idx == 0:
                legend_handles.append(line)
        ax.set_title(PARTITION_LABELS[part])
        ax.set_xlabel("Round")
        ax.set_ylabel("Central test F1 macro")
        ax.set_ylim(ymin, ymax)
        ax.grid(alpha=0.3)

    if legend_handles:
        axes[0, 2].legend(handles=legend_handles, loc="lower right", fontsize=9)
    fig.suptitle("MIT-BIH federated convergence per non-IID regime", y=1.02)
    fig.tight_layout()
    _save(fig, out_base)


# ---------------------------------------------------------------------------
# Figure: 6-panel partition distributions (consolidate per-strategy figures)
# ---------------------------------------------------------------------------

def _partition_distributions(
    fig_dir: Path, out_base: Path,
) -> None:
    """Consolidate the 6 phase4_partition_*.png images into one figure.

    Reads the PNGs the `validate_partitions.py` dry-run produced and stitches
    them into a 2×3 grid so the paper figure is a single file.
    """
    from matplotlib.image import imread
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for idx, part in enumerate(PARTITION_ORDER):
        ax = axes[idx // 3, idx % 3]
        png = fig_dir / f"phase4_partition_{part}.png"
        if png.exists():
            ax.imshow(imread(png))
        else:
            ax.text(0.5, 0.5, f"missing {png.name}",
                    ha="center", va="center", transform=ax.transAxes)
        ax.set_title(PARTITION_LABELS[part])
        ax.axis("off")
    fig.suptitle("MIT-BIH per-client class distributions under the 6 partitions", y=1.02)
    fig.tight_layout()
    _save(fig, out_base)


# ---------------------------------------------------------------------------
# Figure: ΔF1 from IID
# ---------------------------------------------------------------------------

def _degradation_bars(summary: dict[tuple[str, str], dict], out_base: Path) -> None:
    # IID baseline per aggregation
    iid_baseline = {
        agg: summary[("iid", agg)]["f1_primary"]
        for agg in AGGREGATION_ORDER
        if ("iid", agg) in summary
    }
    non_iid = [p for p in PARTITION_ORDER if p != "iid"]
    palette = plt.get_cmap("tab10").colors

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(non_iid))
    width = 0.18
    for j, agg in enumerate(AGGREGATION_ORDER):
        deltas = []
        for part in non_iid:
            row = summary.get((part, agg))
            base = iid_baseline.get(agg, float("nan"))
            if row and np.isfinite(row["f1_primary"]) and np.isfinite(base):
                deltas.append(row["f1_primary"] - base)
            else:
                deltas.append(float("nan"))
        offset = (j - (len(AGGREGATION_ORDER) - 1) / 2) * width
        ax.bar(
            x + offset, deltas, width=width,
            color=palette[j % len(palette)], label=AGGREGATION_LABELS[agg],
        )
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([PARTITION_LABELS[p] for p in non_iid], rotation=20, ha="right")
    ax.set_ylabel("ΔF1 macro vs. IID baseline (same aggregation)")
    ax.set_title("Non-IID degradation — MIT-BIH 5-client 50-round")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    _save(fig, out_base)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_phase4_figures(results_metrics_dir, out_dir) -> None:
    """Produce all Phase 4 figures from sweep artifacts on disk."""
    _style()
    results_metrics_dir = Path(results_metrics_dir)
    out_dir = Path(out_dir)

    summary_csv = results_metrics_dir / "noniid_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(
            f"{summary_csv} not found — run scripts/run_noniid_sweep.py first"
        )
    summary = _load_summary(summary_csv)
    print(f"loaded {len(summary)} sweep entries from {summary_csv}")

    _heatmap(summary, out_dir / "fig_noniid_heatmap")
    _convergence_grid(results_metrics_dir, summary, out_dir / "fig_noniid_convergence")
    _partition_distributions(out_dir, out_dir / "fig_partition_distributions_mitbih")
    _degradation_bars(summary, out_dir / "fig_degradation_from_iid")

    print(f"wrote Phase 4 figures to {out_dir}")
