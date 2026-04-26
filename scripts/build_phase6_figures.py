#!/usr/bin/env python
"""Phase 6 figures — cross-task validation (MIT-BIH 5-class + PTB-XL binary).

Inputs
------
- ``results/metrics/ptbxl_summary.csv`` (Phase 6A non-DP)
- ``results/metrics/ptbxl_dp_summary.csv`` (Phase 6B-v2 DP, dirichlet_a01)
- ``results/metrics/central_vs_local_dirichlet_a01.csv`` (Phase 4 MIT-BIH non-DP)

Outputs
-------
- ``results/figures/fig_phase6_central_vs_local_ptbxl.{png,pdf}``
- ``results/figures/fig_phase6_dp_privacy_utility_ptbxl.{png,pdf}``
- ``results/figures/fig_phase6_cross_task_summary.{png,pdf}``  ← money figure
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = REPO_ROOT / "results" / "metrics"
FIG_DIR = REPO_ROOT / "results" / "figures"

# Paul Tol bright-palette colors, shared with build_phase5_figures.py.
COLOR = {
    "fedavg": "#4477AA",
    "fedprox": "#CCBB44",
    "fedbn": "#228833",
    "fedperf": "#AA3377",
    "dp_fedavg": "#4477AA",
    "dp_fedavg_groupnorm": "#EE6677",
    "dp_fedbn": "#228833",
    "central": "#4477AA",
    "local": "#EE6677",
}
MARKER = {
    "dp_fedavg": "o",
    "dp_fedavg_groupnorm": "s",
    "dp_fedbn": "D",
}
DP_METHOD_LABEL = {
    "dp_fedavg": "DP-FedAvg (raw BN)",
    "dp_fedavg_groupnorm": "DP-FedAvg + GroupNorm",
    "dp_fedbn": "DP-FedBN (ours)",
}
AGG_LABEL = {
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "fedbn": "FedBN",
    "fedperf": "FedPerf",
}
AGG_ORDER = ["fedavg", "fedprox", "fedbn", "fedperf"]

# x-axis layout for ε (log scale): 1, 3 then a separated slot for ∞.
EPS_POS = {1.0: 1.0, 3.0: 3.0, math.inf: 10.0}
EPS_TICKS = [1.0, 3.0, 10.0]
EPS_TICKLABELS = ["1", "3", "∞"]


def _setup_style() -> None:
    plt.rcParams.update({
        "font.size": 10,
        "font.family": "serif",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def _save(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_ptbxl_nondp() -> pd.DataFrame:
    """Phase 6A PTB-XL non-DP summary, dirichlet_a01 partition only."""
    df = pd.read_csv(METRICS_DIR / "ptbxl_summary.csv")
    return df[df["partition"] == "dirichlet_a01"].reset_index(drop=True)


def _load_ptbxl_dp() -> pd.DataFrame:
    """Phase 6B-v2 PTB-XL DP summary, dirichlet_a01 partition only.

    target_epsilon is normalised to float (∞ → math.inf).
    """
    df = pd.read_csv(METRICS_DIR / "ptbxl_dp_summary.csv")
    df = df[df["partition"] == "dirichlet_a01"].reset_index(drop=True)
    df["target_epsilon"] = df["target_epsilon"].apply(
        lambda v: math.inf if str(v).strip().lower() in ("inf", "+inf") else float(v)
    )
    return df


def _load_mitbih_nondp_dirichlet() -> pd.DataFrame:
    """Phase 4 MIT-BIH non-DP central-vs-local CSV, dirichlet_a01 only."""
    return pd.read_csv(METRICS_DIR / "central_vs_local_dirichlet_a01.csv")


# ---------------------------------------------------------------------------
# Figure 1: PTB-XL central vs local (no-DP, dirichlet_a01)
# ---------------------------------------------------------------------------


def fig_central_vs_local_ptbxl(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.6))

    aggs = [a for a in AGG_ORDER if a in set(df["aggregation"])]
    central = [
        float(df.loc[df["aggregation"] == a, "central_f1_best"].iloc[0]) for a in aggs
    ]
    local_mean = [
        float(df.loc[df["aggregation"] == a, "local_f1_mean"].iloc[0]) for a in aggs
    ]
    local_std = [
        float(df.loc[df["aggregation"] == a, "local_f1_std"].iloc[0]) for a in aggs
    ]

    xs = np.arange(len(aggs))
    width = 0.36
    ax.bar(
        xs - width / 2, central, width=width,
        color=COLOR["central"], edgecolor="0.3", label="Central F1 (best)",
    )
    ax.bar(
        xs + width / 2, local_mean, width=width, yerr=local_std,
        color=COLOR["local"], edgecolor="0.3", capsize=3,
        label="Local F1 (per-client mean ± std)",
    )

    # Annotate FedBN's gap to highlight the eval-regime flip vs FedAvg.
    if "fedbn" in aggs and "fedavg" in aggs:
        i_bn = aggs.index("fedbn")
        bn_local = local_mean[i_bn]
        i_av = aggs.index("fedavg")
        av_local = local_mean[i_av]
        gap = bn_local - av_local
        ax.annotate(
            f"FedBN local ({bn_local:.3f})\n"
            f" > FedAvg local ({av_local:.3f}): +{gap:.3f}",
            xy=(xs[i_bn] + width / 2, bn_local),
            xytext=(xs[i_bn] + 0.7, bn_local + 0.12),
            fontsize=8,
            arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8),
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFACD",
                      edgecolor="0.5", lw=0.6),
        )

    ax.set_xticks(xs)
    ax.set_xticklabels([AGG_LABEL[a] for a in aggs])
    ax.set_ylabel("F1 macro")
    ax.set_ylim(0, 1.0)
    ax.set_title("PTB-XL binary: eval-regime flip confirmed (Dir α=0.1)")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "fig_phase6_central_vs_local_ptbxl"
    _save(fig, out)
    return out


# ---------------------------------------------------------------------------
# Figure 2: PTB-XL DP privacy-utility (dirichlet_a01)
# ---------------------------------------------------------------------------


def _series_for_dp(
    df: pd.DataFrame, method: str, *, yfield: str,
) -> tuple[list[float], list[float]]:
    rows = df[df["method"] == method].sort_values("target_epsilon")
    xs = [EPS_POS[eps] for eps in rows["target_epsilon"]]
    ys = [float(v) if not pd.isna(v) else float("nan") for v in rows[yfield]]
    return xs, ys


def _err_for_dp(df: pd.DataFrame, method: str) -> list[float]:
    rows = df[df["method"] == method].sort_values("target_epsilon")
    return [float(v) if not pd.isna(v) else 0.0 for v in rows["local_f1_std"]]


def _draw_dp_panel(
    ax: plt.Axes, df: pd.DataFrame, *,
    yfield: str, with_errorbars: bool, ylabel: str,
    title: str, legend: bool,
) -> None:
    # Stringent-DP shading (ε ≤ 2 in log axis).
    ax.axvspan(0.85, 2.0, color="0.85", alpha=0.4, zorder=0)
    ax.text(
        math.sqrt(0.85 * 2.0), 0.97,
        "stringent DP regime", ha="center", va="top",
        fontsize=8, color="0.35", zorder=1,
    )

    for method in ("dp_fedavg_groupnorm", "dp_fedbn"):
        xs, ys = _series_for_dp(df, method, yfield=yfield)
        if not xs:
            continue
        if with_errorbars:
            yerr = _err_for_dp(df, method)
            ax.errorbar(
                xs, ys, yerr=yerr,
                marker=MARKER[method], color=COLOR[method],
                linewidth=2, markersize=7, capsize=3,
                label=DP_METHOD_LABEL[method],
            )
        else:
            ax.plot(
                xs, ys, marker=MARKER[method], color=COLOR[method],
                linewidth=2, markersize=7, label=DP_METHOD_LABEL[method],
            )

    # Single-point dp_fedavg at ε=∞ (refused at finite ε).
    xs, ys = _series_for_dp(df, "dp_fedavg", yfield=yfield)
    if xs:
        if with_errorbars:
            yerr = _err_for_dp(df, "dp_fedavg")
            ax.errorbar(
                xs, ys, yerr=yerr,
                marker=MARKER["dp_fedavg"], color=COLOR["dp_fedavg"],
                linestyle="", markersize=9, capsize=3,
                label=DP_METHOD_LABEL["dp_fedavg"] + " (ε=∞ only)",
            )
        else:
            ax.plot(
                xs, ys, marker=MARKER["dp_fedavg"], color=COLOR["dp_fedavg"],
                linestyle="", markersize=9,
                label=DP_METHOD_LABEL["dp_fedavg"] + " (ε=∞ only)",
            )

    ax.axvline(5.5, color="0.7", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xscale("log")
    ax.set_xticks(EPS_TICKS)
    ax.set_xticklabels(EPS_TICKLABELS)
    ax.set_xlabel("Privacy budget ε")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0.85, 12.0)
    ax.grid(axis="y", alpha=0.3)
    if legend:
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)


def fig_dp_privacy_utility_ptbxl(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    _draw_dp_panel(
        axes[0], df,
        yfield="central_f1_best", with_errorbars=False,
        ylabel="Best central F1 macro",
        title="Central eval", legend=True,
    )
    _draw_dp_panel(
        axes[1], df,
        yfield="local_f1_mean", with_errorbars=True,
        ylabel="Local F1 macro (per-client mean)",
        title="Local eval", legend=False,
    )
    fig.suptitle(
        "PTB-XL binary: DP privacy–utility trade-off (Dir α=0.1)",
        y=1.02, fontsize=12,
    )
    fig.tight_layout()
    out = FIG_DIR / "fig_phase6_dp_privacy_utility_ptbxl"
    _save(fig, out)
    return out


# ---------------------------------------------------------------------------
# Figure 3: cross-task summary (MIT-BIH + PTB-XL non-DP, dirichlet_a01)
# ---------------------------------------------------------------------------


def _draw_central_local_panel(
    ax: plt.Axes, *,
    aggs: list[str],
    central: list[float],
    local_mean: list[float],
    local_std: list[float],
    title: str,
    ylabel: str | None = "F1 macro",
    show_legend: bool,
    highlight_fedbn: bool = True,
) -> None:
    xs = np.arange(len(aggs))
    width = 0.36

    # Highlight FedAvg + FedBN columns to draw the eye to the rank-flip pair.
    if highlight_fedbn and "fedbn" in aggs and "fedavg" in aggs:
        i_av = aggs.index("fedavg")
        i_bn = aggs.index("fedbn")
        for i in (i_av, i_bn):
            ax.axvspan(i - 0.5, i + 0.5, color="#FFF6CC", alpha=0.45, zorder=0)

    ax.bar(
        xs - width / 2, central, width=width,
        color=COLOR["central"], edgecolor="0.3", label="Central F1",
    )
    ax.bar(
        xs + width / 2, local_mean, width=width, yerr=local_std,
        color=COLOR["local"], edgecolor="0.3", capsize=3,
        label="Local F1 (mean ± std)",
    )

    # Annotate the FedBN-vs-FedAvg rank reversal:
    #   central:  FedBN is below FedAvg (gap < 0)
    #   local:    FedBN is above FedAvg (gap > 0)
    if highlight_fedbn and "fedbn" in aggs and "fedavg" in aggs:
        i_av = aggs.index("fedavg")
        i_bn = aggs.index("fedbn")

        # Central gap (FedBN - FedAvg)
        gap_c = central[i_bn] - central[i_av]
        sign_c = "+" if gap_c >= 0 else ""
        # Local gap (FedBN - FedAvg)
        gap_l = local_mean[i_bn] - local_mean[i_av]
        sign_l = "+" if gap_l >= 0 else ""

        # Place a single combined callout above the FedBN column.
        y_top = max(central[i_bn], local_mean[i_bn] + (local_std[i_bn] or 0)) + 0.06
        ax.text(
            i_bn, min(y_top, 1.02),
            f"FedBN − FedAvg\n"
            f"central: {sign_c}{gap_c:.3f}\n"
            f"local:   {sign_l}{gap_l:.3f}",
            ha="center", va="bottom", fontsize=8,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFACD",
                      edgecolor="0.5", lw=0.6),
        )

    ax.set_xticks(xs)
    ax.set_xticklabels([AGG_LABEL[a] for a in aggs])
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.10)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    if show_legend:
        ax.legend(loc="lower right", fontsize=9, framealpha=0.9)


def fig_cross_task_summary(
    mit: pd.DataFrame, ptb: pd.DataFrame,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)

    # MIT-BIH panel — schema: aggregation, central_f1, local_f1_mean, local_f1_std
    aggs_m = [a for a in AGG_ORDER if a in set(mit["aggregation"])]
    cen_m = [float(mit.loc[mit["aggregation"] == a, "central_f1"].iloc[0]) for a in aggs_m]
    lm_m = [float(mit.loc[mit["aggregation"] == a, "local_f1_mean"].iloc[0]) for a in aggs_m]
    ls_m = [float(mit.loc[mit["aggregation"] == a, "local_f1_std"].iloc[0]) for a in aggs_m]
    _draw_central_local_panel(
        axes[0], aggs=aggs_m, central=cen_m, local_mean=lm_m, local_std=ls_m,
        title="MIT-BIH (5-class beat) · Dir α=0.1",
        show_legend=True,
    )

    # PTB-XL panel — schema: aggregation, central_f1_best, local_f1_mean, local_f1_std
    aggs_p = [a for a in AGG_ORDER if a in set(ptb["aggregation"])]
    cen_p = [float(ptb.loc[ptb["aggregation"] == a, "central_f1_best"].iloc[0]) for a in aggs_p]
    lm_p = [float(ptb.loc[ptb["aggregation"] == a, "local_f1_mean"].iloc[0]) for a in aggs_p]
    ls_p = [float(ptb.loc[ptb["aggregation"] == a, "local_f1_std"].iloc[0]) for a in aggs_p]
    _draw_central_local_panel(
        axes[1], aggs=aggs_p, central=cen_p, local_mean=lm_p, local_std=ls_p,
        title="PTB-XL (binary record) · Dir α=0.1",
        show_legend=False, ylabel=None,
    )

    fig.suptitle(
        "Cross-task validation: FedBN ranks below FedAvg on central but above on local "
        "— both datasets (no-DP, Dir α=0.1)",
        y=1.02, fontsize=12,
    )
    fig.tight_layout()
    out = FIG_DIR / "fig_phase6_cross_task_summary"
    _save(fig, out)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    _setup_style()
    paths: list[Path] = []

    ptb_nondp = _load_ptbxl_nondp()
    if ptb_nondp.empty:
        print("ERROR: no dirichlet_a01 rows in ptbxl_summary.csv")
        return 2
    paths.append(fig_central_vs_local_ptbxl(ptb_nondp))

    try:
        ptb_dp = _load_ptbxl_dp()
    except FileNotFoundError:
        ptb_dp = pd.DataFrame()
    if ptb_dp.empty:
        print("WARN: no dirichlet_a01 rows in ptbxl_dp_summary.csv — skipping fig 2.")
    else:
        paths.append(fig_dp_privacy_utility_ptbxl(ptb_dp))

    mit_nondp = _load_mitbih_nondp_dirichlet()
    paths.append(fig_cross_task_summary(mit_nondp, ptb_nondp))

    print("Generated figures:")
    for p in paths:
        print(f"  - {p.relative_to(REPO_ROOT)}.png")
        print(f"  - {p.relative_to(REPO_ROOT)}.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
