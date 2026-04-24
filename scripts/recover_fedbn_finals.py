#!/usr/bin/env python
"""Recover the 6 FedBN runs that crashed at the final reporting step.

The six FedBN runs in the Phase 4 sweep completed all 50 training rounds
successfully — their per-round metrics are present in
``results/metrics/{run_id}_rounds.csv`` — but then the reporting step
raised ``KeyError: 'per_class_precision'`` in
``format_classification_report`` because the FedBN client_avg aggregate
dict does not carry per-class keys. The format function has since been
made tolerant; this script reconstructs the missing per-run JSONs from
the rounds CSVs so downstream figure code can treat these runs as
first-class results.

For each FAILED run in ``sweep_progress.csv`` this script writes a
synthetic ``results/metrics/{run_id}.json`` file marked
``status="RECOVERED"`` and updates the progress CSV row to fill in
final_f1 / best_f1 / best_round from the rounds CSV.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = REPO_ROOT / "results" / "metrics"
PROGRESS_CSV = METRICS_DIR / "sweep_progress.csv"


def _read_rounds(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _float_or_none(s: str) -> float | None:
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    return v


def _best_round(rows: list[dict]) -> tuple[int, float]:
    best_r, best_f1 = -1, float("-inf")
    for r in rows:
        f1 = _float_or_none(r.get("central_f1_macro", ""))
        if f1 is None:
            continue
        if f1 > best_f1:
            best_f1 = f1
            best_r = int(r["round"])
    return best_r, best_f1


def _final_round(rows: list[dict]) -> dict:
    return rows[-1] if rows else {}


def recover_run(run_id: str, partition: str, aggregation: str) -> dict:
    rounds_csv = METRICS_DIR / f"{run_id}_rounds.csv"
    if not rounds_csv.exists():
        raise FileNotFoundError(f"rounds CSV missing for {run_id}: {rounds_csv}")
    rows = _read_rounds(rounds_csv)
    if not rows:
        raise ValueError(f"rounds CSV empty for {run_id}")

    best_r, best_f1 = _best_round(rows)
    final = _final_round(rows)
    final_round_num = int(final["round"])
    final_f1 = _float_or_none(final.get("central_f1_macro", ""))
    final_acc = _float_or_none(final.get("central_accuracy", ""))
    final_auc = _float_or_none(final.get("central_auc_macro", ""))

    best = next(r for r in rows if int(r["round"]) == best_r)
    best_acc = _float_or_none(best.get("central_accuracy", ""))
    best_auc = _float_or_none(best.get("central_auc_macro", ""))

    per_client_final = [
        _float_or_none(final.get(f"client{i}_val_f1", "")) for i in range(5)
    ]

    synthetic = {
        "run_id": run_id,
        "status": "RECOVERED",
        "partition_strategy": partition,
        "strategy": aggregation,
        "dataset": "mitbih",
        "rounds": final_round_num,
        "central_mode": final.get("central_mode", "fedbn_client_avg"),
        "best_round": best_r,
        "best_round_central_f1_macro": best_f1,
        "best_round_central_accuracy": best_acc,
        "best_round_central_auc_macro": best_auc,
        "final_round_central_f1_macro": final_f1,
        "final_round_central_accuracy": final_acc,
        "final_round_central_auc_macro": final_auc,
        "per_client_final_val_f1": per_client_final,
        "note": (
            "Recovered from rounds.csv; final checkpoint and per-class "
            "breakdown not saved due to crash at reporting step."
        ),
    }

    # Convenience mirrors so visualization code that reads the DONE-run
    # schema (best_central_f1_macro, final_central_metrics.f1_macro) also
    # picks up the RECOVERED values without special-casing.
    synthetic["best_central_f1_macro"] = best_f1
    synthetic["final_central_metrics"] = {
        "f1_macro": final_f1,
        "accuracy": final_acc,
        "auc_macro": final_auc,
    }
    # Dual-eval block mirrors what the DONE FedBN runs would have written:
    # we only have the client_avg path (that was the primary central metric
    # used every round); representative is unavailable after a crash.
    synthetic["fedbn_dual_eval"] = {
        "client_avg": {
            "f1_macro": final_f1,
            "accuracy": final_acc,
            "auc_macro": final_auc,
        }
    }

    out_path = METRICS_DIR / f"{run_id}.json"
    out_path.write_text(json.dumps(synthetic, indent=2))
    return {
        "run_id": run_id,
        "best_round": best_r,
        "best_f1": best_f1,
        "final_f1": final_f1,
        "json_path": out_path,
    }


def _update_progress_csv(recovered: dict[str, dict]) -> None:
    if not PROGRESS_CSV.exists():
        raise FileNotFoundError(PROGRESS_CSV)
    rows = list(csv.DictReader(PROGRESS_CSV.open()))
    fieldnames = list(rows[0].keys())
    for row in rows:
        rid = row["run_id"]
        if rid in recovered and row["status"] == "FAILED":
            info = recovered[rid]
            row["status"] = "RECOVERED"
            row["final_f1"] = f"{info['final_f1']:.6f}" if info["final_f1"] is not None else ""
            row["best_f1"] = f"{info['best_f1']:.6f}" if info["best_f1"] is not None else ""
            row["best_round"] = str(info["best_round"])
    with PROGRESS_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    if not PROGRESS_CSV.exists():
        print(f"ERROR: {PROGRESS_CSV} not found", file=sys.stderr)
        return 2
    failed = [
        r for r in csv.DictReader(PROGRESS_CSV.open())
        if r["status"] == "FAILED"
    ]
    if not failed:
        print("No FAILED runs to recover.")
        return 0
    print(f"Recovering {len(failed)} FAILED run(s)...\n")
    recovered: dict[str, dict] = {}
    for row in failed:
        info = recover_run(row["run_id"], row["partition"], row["aggregation"])
        print(
            f"  {info['run_id']}\n"
            f"    best_round={info['best_round']}  "
            f"best_f1={info['best_f1']:.4f}  "
            f"final_f1={info['final_f1']:.4f}\n"
            f"    wrote {info['json_path'].relative_to(REPO_ROOT)}"
        )
        recovered[info["run_id"]] = info
    _update_progress_csv(recovered)
    print(f"\nUpdated {PROGRESS_CSV.relative_to(REPO_ROOT)} "
          f"(status FAILED → RECOVERED for {len(recovered)} run(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
