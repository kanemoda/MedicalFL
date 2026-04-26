#!/usr/bin/env python
"""Phase 5 figures + LaTeX table.

Inputs
------
- ``results/metrics/dp_summary.csv`` (Phase 5, 14 rows)
- ``results/metrics/central_vs_local_label_skew_c2.csv`` (Phase 4)
- ``results/metrics/central_vs_local_dirichlet_a01.csv`` (Phase 4)

Outputs
-------
- ``results/figures/fig_privacy_utility_central.{png,pdf}``
- ``results/figures/fig_privacy_utility_local.{png,pdf}``
- ``results/figures/fig_privacy_utility_dual.{png,pdf}``  (money figure)
- ``results/figures/fig_central_local_phase5_comparison.{png,pdf}``
- ``results/tables/phase5_summary_table.tex``
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = REPO_ROOT / "results" / "metrics"
FIG_DIR = REPO_ROOT / "results" / "figures"
TABLE_DIR = REPO_ROOT / "results" / "tables"

# Distinct, color-blind safe (Paul Tol's bright palette).
COLOR = {
    "dp_fedavg": "#4477AA",            # blue
    "dp_fedavg_groupnorm": "#EE6677",  # red
    "dp_fedbn": "#228833",             # green
    # Phase 4 no-DP methods
    "fedavg": "#4477AA",
    "fedprox": "#CCBB44",
    "fedbn": "#228833",
    "fedperf": "#AA3377",
}
MARKER = {
    "dp_fedavg": "o",
    "dp_fedavg_groupnorm": "s",
    "dp_fedbn": "D",
}
METHOD_LABEL = {
    "dp_fedavg": "DP-FedAvg (raw BN)",
    "dp_fedavg_groupnorm": "DP-FedAvg + GroupNorm",
    "dp_fedbn": "DP-FedBN (ours)",
}
PARTITION_LABEL = {
    "label_skew_c2": "Label skew (C=2)",
    "dirichlet_a01": "Dirichlet α=0.1",
}
PARTITIONS = ["label_skew_c2", "dirichlet_a01"]

# X-axis layout: log positions for 1 and 3, then a separated rightmost slot for ∞.
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


def _load_phase5() -> pd.DataFrame:
    df = pd.read_csv(METRICS_DIR / "dp_summary.csv")
    # Normalise target_epsilon → float (with inf for "inf" string).
    df["target_epsilon"] = df["target_epsilon"].apply(
        lambda v: math.inf if str(v).strip().lower() in ("inf", "+inf") else float(v)
    )
    return df


def _load_phase4() -> dict[str, pd.DataFrame]:
    return {
        part: pd.read_csv(METRICS_DIR / f"central_vs_local_{part}.csv")
        for part in PARTITIONS
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _series_for(
    df: pd.DataFrame,
    method: str,
    partition: str,
    *,
    yfield: str,
) -> tuple[list[float], list[float]]:
    rows = df[(df["method"] == method) & (df["partition"] == partition)]
    rows = rows.sort_values("target_epsilon")
    xs = [EPS_POS[eps] for eps in rows["target_epsilon"]]
    ys = [float(v) if v == v else float("nan") for v in rows[yfield]]
    return xs, ys


def _err_for(df: pd.DataFrame, method: str, partition: str) -> list[float]:
    rows = df[(df["method"] == method) & (df["partition"] == partition)]
    rows = rows.sort_values("target_epsilon")
    return [float(v) if v == v else 0.0 for v in rows["local_f1_std"]]


def _draw_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    partition: str,
    *,
    yfield: str,
    with_errorbars: bool,
    legend: bool,
    ylabel: str,
    title_prefix: str = "",
) -> None:
    # Stringent-DP shaded region (ε ≤ 2 in log-axis terms): position 0.9 to 2.0.
    ax.axvspan(0.85, 2.0, color="0.85", alpha=0.4, zorder=0)
    ax.text(
        math.sqrt(0.85 * 2.0), ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] > 0 else 0.97,
        "stringent DP regime", ha="center", va="top", fontsize=8,
        color="0.35", zorder=1,
    )

    # Lines for groupnorm + fedbn over (1, 3, ∞)
    for method in ("dp_fedavg_groupnorm", "dp_fedbn"):
        xs, ys = _series_for(df, method, partition, yfield=yfield)
        if not xs:
            continue
        if with_errorbars:
            yerr = _err_for(df, method, partition)
            ax.errorbar(
                xs, ys, yerr=yerr,
                marker=MARKER[method], color=COLOR[method],
                linewidth=2, markersize=7, capsize=3,
                label=METHOD_LABEL[method],
            )
        else:
            ax.plot(
                xs, ys, marker=MARKER[method], color=COLOR[method],
                linewidth=2, markersize=7, label=METHOD_LABEL[method],
            )

    # Single point for naive DP-FedAvg at ε=∞
    xs, ys = _series_for(df, "dp_fedavg", partition, yfield=yfield)
    if xs:
        if with_errorbars:
            yerr = _err_for(df, "dp_fedavg", partition)
            ax.errorbar(
                xs, ys, yerr=yerr,
                marker=MARKER["dp_fedavg"], color=COLOR["dp_fedavg"],
                linestyle="", markersize=9, capsize=3,
                label=METHOD_LABEL["dp_fedavg"] + " (ε=∞ only)",
            )
        else:
            ax.plot(
                xs, ys, marker=MARKER["dp_fedavg"], color=COLOR["dp_fedavg"],
                linestyle="", markersize=9, label=METHOD_LABEL["dp_fedavg"] + " (ε=∞ only)",
            )

    # Vertical separator before the ∞ slot
    ax.axvline(5.5, color="0.7", linewidth=0.8, linestyle="--", zorder=0)

    ax.set_xscale("log")
    ax.set_xticks(EPS_TICKS)
    ax.set_xticklabels(EPS_TICKLABELS)
    ax.set_xlabel("Privacy budget ε")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title_prefix}{PARTITION_LABEL[partition]}")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0.85, 12.0)
    ax.grid(axis="y", alpha=0.3)
    if legend:
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)


# ---------------------------------------------------------------------------
# Figure 1: privacy-utility central
# ---------------------------------------------------------------------------


def fig_privacy_utility_central(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, part in zip(axes, PARTITIONS):
        _draw_panel(
            ax, df, part,
            yfield="central_f1_best",
            with_errorbars=False,
            legend=(part == "label_skew_c2"),
            ylabel="Best central F1 macro",
        )
    fig.suptitle("Phase 5: Privacy–Utility (Central Eval)", y=1.02, fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "fig_privacy_utility_central"
    _save(fig, out)
    return out


# ---------------------------------------------------------------------------
# Figure 2: privacy-utility local
# ---------------------------------------------------------------------------


def fig_privacy_utility_local(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, part in zip(axes, PARTITIONS):
        _draw_panel(
            ax, df, part,
            yfield="local_f1_mean",
            with_errorbars=True,
            legend=(part == "label_skew_c2"),
            ylabel="Local F1 macro (per-client mean)",
        )
    fig.suptitle("Phase 5: Privacy–Utility (Per-Client Local Eval)", y=1.02, fontsize=12)
    fig.tight_layout()
    out = FIG_DIR / "fig_privacy_utility_local"
    _save(fig, out)
    return out


# ---------------------------------------------------------------------------
# Figure 3: privacy-utility dual (money figure)
# ---------------------------------------------------------------------------


def _annotate_advantage(
    ax: plt.Axes, *, x: float, y_top: float, y_bot: float, label: str,
    side: str = "right",
) -> None:
    """Draw a vertical bracket near (x, y_top..y_bot) and label the gap.

    ``side`` chooses whether the bracket and text sit to the right or left of
    ``x`` — flip to "left" when ``x`` is near the panel's right edge so the
    text doesn't get clipped.
    """
    if side == "right":
        bracket_x = x * 1.18
        text_x = bracket_x * 1.1
        ha = "left"
    else:
        bracket_x = x / 1.18
        text_x = bracket_x / 1.1
        ha = "right"
    ax.annotate(
        "",
        xy=(bracket_x, y_top), xytext=(bracket_x, y_bot),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.2),
    )
    ax.text(
        text_x, (y_top + y_bot) / 2.0,
        label,
        ha=ha, va="center", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFACD", edgecolor="0.5", lw=0.6),
    )


def fig_privacy_utility_dual(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)
    rows_titles = [("label_skew_c2", "Label skew (C=2)"),
                   ("dirichlet_a01", "Dirichlet α=0.1")]
    cols_titles = [("central_f1_best", False, "Central"),
                   ("local_f1_mean", True, "Local")]

    for r, (part, _) in enumerate(rows_titles):
        for c, (yfield, with_err, _) in enumerate(cols_titles):
            ax = axes[r][c]
            _draw_panel(
                ax, df, part,
                yfield=yfield,
                with_errorbars=with_err,
                legend=(r == 0 and c == 0),
                ylabel="F1 macro",
                title_prefix=f"{cols_titles[c][2]}: ",
            )

    # Annotations on the label_skew_c2 LOCAL panel (axes[0][1])
    ax_local_ls = axes[0][1]
    # ε=3: DP-FedBN 0.467 vs DP-GroupNorm 0.295
    _annotate_advantage(
        ax_local_ls, x=3.0, y_top=0.467, y_bot=0.295,
        label="+0.17 advantage\n(DP-FedBN @ ε=3)",
    )
    # ε=∞: DP-FedBN 0.937 vs DP-FedAvg 0.829 (point on the right edge → label LEFT)
    _annotate_advantage(
        ax_local_ls, x=10.0, y_top=0.937, y_bot=0.829,
        label="+0.11 (no DP)", side="left",
    )

    fig.suptitle(
        "DP-FedBN preserves local-eval advantage under DP on label-skew, "
        "mixed on Dirichlet α=0.1",
        y=1.00, fontsize=12,
    )
    fig.tight_layout()
    out = FIG_DIR / "fig_privacy_utility_dual"
    _save(fig, out)
    return out


# ---------------------------------------------------------------------------
# Figure 4: Phase 4 + Phase 5 grouped-bar comparison
# ---------------------------------------------------------------------------


def _phase5_lookup(df: pd.DataFrame, method: str, partition: str, eps: float) -> tuple[float, float, float]:
    """Return (central, local_mean, local_std) for a Phase 5 cell.

    Returns NaNs for refused/missing cells (i.e. naive DP-FedAvg at finite ε).
    """
    rows = df[(df["method"] == method) &
              (df["partition"] == partition) &
              (df["target_epsilon"].apply(lambda v: math.isclose(v, eps) if not math.isinf(v) else math.isinf(eps)))]
    if rows.empty:
        return float("nan"), float("nan"), float("nan")
    r = rows.iloc[0]
    return (
        float(r["central_f1_best"]),
        float(r["local_f1_mean"]),
        float(r["local_f1_std"]) if not pd.isna(r["local_f1_std"]) else 0.0,
    )


def fig_central_local_phase5_comparison(
    df: pd.DataFrame, p4: dict[str, pd.DataFrame],
) -> Path:
    """Three side-by-side groupings per panel: no-DP, DP ε=3, DP ε=1."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)

    no_dp_methods = [("fedavg", "FedAvg"), ("fedprox", "FedProx"),
                     ("fedbn", "FedBN"), ("fedperf", "FedPerf")]
    dp_methods = [("dp_fedavg", "DP-FedAvg"),
                  ("dp_fedavg_groupnorm", "DP-FedAvg+GN"),
                  ("dp_fedbn", "DP-FedBN")]
    width = 0.36

    for ax, part in zip(axes, PARTITIONS):
        labels: list[str] = []
        centrals: list[float] = []
        locals_: list[float] = []
        local_errs: list[float] = []
        boundaries: list[float] = []  # x-positions where group separators go

        x = 0
        # Group 1: no-DP (Phase 4)
        p4_df = p4[part]
        for agg, lbl in no_dp_methods:
            row = p4_df[p4_df["aggregation"] == agg]
            if row.empty:
                continue
            r = row.iloc[0]
            labels.append(lbl)
            centrals.append(float(r["central_f1"]))
            locals_.append(float(r["local_f1_mean"]))
            local_errs.append(float(r["local_f1_std"]))
            x += 1
        boundaries.append(x - 0.5)

        # Group 2: DP ε=3 (Phase 5)
        for method, lbl in dp_methods:
            c, lm, ls = _phase5_lookup(df, method, part, 3.0)
            labels.append(lbl)
            centrals.append(c)
            locals_.append(lm)
            local_errs.append(ls)
            x += 1
        boundaries.append(x - 0.5)

        # Group 3: DP ε=1 (Phase 5)
        for method, lbl in dp_methods:
            c, lm, ls = _phase5_lookup(df, method, part, 1.0)
            labels.append(lbl)
            centrals.append(c)
            locals_.append(lm)
            local_errs.append(ls)
            x += 1

        xs = np.arange(len(labels))
        centrals_arr = np.array([0.0 if math.isnan(v) else v for v in centrals])
        locals_arr = np.array([0.0 if math.isnan(v) else v for v in locals_])
        local_err_arr = np.array(local_errs)

        # Mark refused cells with hatch (DP-FedAvg @ finite ε)
        c_hatch = ["//" if math.isnan(v) else "" for v in centrals]
        l_hatch = ["//" if math.isnan(v) else "" for v in locals_]

        bars_c = ax.bar(xs - width / 2, centrals_arr, width=width, color="#4477AA",
                        edgecolor="0.3", label="Central F1")
        bars_l = ax.bar(xs + width / 2, locals_arr, width=width, yerr=local_err_arr,
                        color="#EE6677", edgecolor="0.3", capsize=2, label="Local F1 mean")
        for bar, h in zip(bars_c, c_hatch):
            bar.set_hatch(h)
        for bar, h in zip(bars_l, l_hatch):
            bar.set_hatch(h)

        # REFUSED text on hatched bars
        for i, (c, lm) in enumerate(zip(centrals, locals_)):
            if math.isnan(c):
                ax.text(xs[i], 0.04, "REFUSED", ha="center", va="bottom",
                        fontsize=7, color="0.3", rotation=90)

        # Group separators
        for b in boundaries:
            ax.axvline(b, color="0.55", linewidth=0.8, linestyle="--")

        # Group labels above
        ymax = 1.05
        ax.text((boundaries[0]) / 2, ymax * 0.99, "no-DP (Phase 4)",
                ha="center", va="top", fontsize=10, color="0.2",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#EEEEEE", edgecolor="none"))
        ax.text((boundaries[0] + boundaries[1]) / 2, ymax * 0.99, "DP ε=3",
                ha="center", va="top", fontsize=10, color="0.2",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#EEEEEE", edgecolor="none"))
        ax.text((boundaries[1] + len(labels) - 0.5) / 2, ymax * 0.99, "DP ε=1",
                ha="center", va="top", fontsize=10, color="0.2",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#EEEEEE", edgecolor="none"))

        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, ymax)
        ax.set_ylabel("F1 macro")
        ax.set_title(PARTITION_LABEL[part])
        ax.grid(axis="y", alpha=0.3)
        if part == PARTITIONS[0]:
            ax.legend(loc="lower right", fontsize=9)

    fig.suptitle(
        "Eval-regime flip is more pronounced in no-DP setting; partially survives DP",
        y=1.02, fontsize=12,
    )
    fig.tight_layout()
    out = FIG_DIR / "fig_central_local_phase5_comparison"
    _save(fig, out)
    return out


# ---------------------------------------------------------------------------
# Action 5: LaTeX table
# ---------------------------------------------------------------------------


def _method_tex(method: str) -> str:
    return {
        "dp_fedavg": "DP-FedAvg",
        "dp_fedavg_groupnorm": "DP-FedAvg+GroupNorm",
        "dp_fedbn": r"\textbf{DP-FedBN}",
    }[method]


def _partition_tex(partition: str) -> str:
    return {
        "label_skew_c2": "Label skew (C=2)",
        "dirichlet_a01": r"Dir($\alpha$=0.1)",
    }[partition]


def _eps_tex(eps: float) -> str:
    if math.isinf(eps):
        return r"$\infty$"
    return f"{eps:.0f}"


def _eps_achieved_tex(achieved: object, target: float) -> str:
    if math.isinf(target):
        return "—"
    if achieved is None or (isinstance(achieved, float) and math.isnan(achieved)) or pd.isna(achieved):
        return "REFUSED"
    return f"{float(achieved):.4f}"


def write_latex_table(df: pd.DataFrame) -> Path:
    """Sort: partition → method → ε descending (∞, 3, 1).

    For naive DP-FedAvg at finite ε, write a REFUSED row using the progress CSV.
    """
    # Pull REFUSED rows from dp_sweep_progress.csv (dp_summary only has DONE).
    progress = pd.read_csv(METRICS_DIR / "dp_sweep_progress.csv")
    progress["target_epsilon"] = progress["target_epsilon"].apply(
        lambda v: math.inf if str(v).strip().lower() in ("inf", "+inf") else float(v)
    )

    method_order = ["dp_fedavg", "dp_fedavg_groupnorm", "dp_fedbn"]
    eps_order = [math.inf, 3.0, 1.0]

    rows_out: list[str] = []
    for part in PARTITIONS:
        for method in method_order:
            for eps in eps_order:
                # Find a DONE row first.
                done = df[(df["method"] == method) & (df["partition"] == part) &
                          (df["target_epsilon"].apply(
                              lambda v, e=eps: math.isinf(v) and math.isinf(e) or
                              (not math.isinf(v) and not math.isinf(e) and math.isclose(v, e))
                          ))]
                if not done.empty:
                    r = done.iloc[0]
                    central = f"{float(r['central_f1_best']):.3f}"
                    lm = float(r["local_f1_mean"])
                    ls = float(r["local_f1_std"])
                    local = f"{lm:.3f} $\\pm$ {ls:.3f}"
                    eps_a = _eps_achieved_tex(r.get("achieved_epsilon_mean"), eps)
                else:
                    # Most recent matching progress row
                    p = progress[(progress["method"] == method) &
                                 (progress["partition"] == part) &
                                 (progress["target_epsilon"].apply(
                                     lambda v, e=eps: math.isinf(v) and math.isinf(e) or
                                     (not math.isinf(v) and not math.isinf(e) and math.isclose(v, e))
                                 ))]
                    if p.empty:
                        continue
                    pr = p.iloc[-1]
                    if str(pr.get("status", "")).upper() == "FAILED":
                        central = "REFUSED"
                        local = "REFUSED"
                        eps_a = "REFUSED"
                    else:
                        central = "—"
                        local = "—"
                        eps_a = "—"
                rows_out.append(
                    f"{_method_tex(method)} & {_partition_tex(part)} & "
                    f"{_eps_tex(eps)} & {eps_a} & {central} & {local} \\\\"
                )
        rows_out.append(r"\midrule")
    if rows_out and rows_out[-1] == r"\midrule":
        rows_out.pop()

    header = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{Phase 5: Privacy--Utility benchmark on MIT-BIH "
        "(5 clients, 30 rounds, batch 96, $\\delta = 10^{-5}$). "
        "Naive DP-FedAvg refused by Opacus at finite $\\epsilon$ due to BatchNorm "
        "incompatibility; rows marked REFUSED are first-class empirical findings, "
        "not omissions.}\n"
        "\\label{tab:phase5_summary}\n"
        "\\begin{tabular}{llrrrr}\n"
        "\\toprule\n"
        "Method & Partition & $\\epsilon_{\\text{target}}$ & $\\epsilon_{\\text{achieved}}$ "
        "& Central F1 & Local F1 (mean $\\pm$ std) \\\\\n"
        "\\midrule\n"
    )
    body = "\n".join(rows_out)
    footer = "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    tex = header + body + footer
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLE_DIR / "phase5_summary_table.tex"
    out.write_text(tex)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    _setup_style()
    df5 = _load_phase5()
    df4 = _load_phase4()

    paths = []
    paths.append(fig_privacy_utility_central(df5))
    paths.append(fig_privacy_utility_local(df5))
    paths.append(fig_privacy_utility_dual(df5))
    paths.append(fig_central_local_phase5_comparison(df5, df4))
    tex_path = write_latex_table(df5)

    print("Generated figures:")
    for p in paths:
        print(f"  - {p.relative_to(REPO_ROOT)}.png")
        print(f"  - {p.relative_to(REPO_ROOT)}.pdf")
    print(f"Generated table:\n  - {tex_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
