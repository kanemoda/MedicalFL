#!/usr/bin/env python
"""Phase 4 non-IID × aggregation sweep — 6 × 4 = 24 MIT-BIH runs.

Reads `configs/sweep_noniid.yaml`, materializes one run config per
(partition, aggregation) pair, and calls `train_federated` sequentially.
Already-finished runs (identified by `results/metrics/<run_id>.json`) are
skipped so the sweep is resumable after crashes or interruptions.

Writes:
  - results/metrics/sweep_progress.csv    (row appended per run, tailable live)
  - results/metrics/noniid_summary.csv    (one row per run, written after all
                                           runs finish — headline comparison
                                           table)
  - results/logs/<run_id>_crash.log       (traceback for any failed run)

A run is considered FAILED if `train_federated` raises; the sweep continues.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.federated import train_federated  # noqa: E402
from src.utils.config import load_config  # noqa: E402

METRICS_DIR = REPO_ROOT / "results" / "metrics"
LOG_DIR = REPO_ROOT / "results" / "logs"

SUMMARY_COLUMNS = [
    "run_id",
    "partition",
    "aggregation",
    "test_accuracy",
    "test_f1_macro",
    "test_f1_macro_client_avg",
    "test_f1_macro_representative",
    "test_auc_macro",
    "best_round",
    "convergence_round",
    "wall_clock_seconds",
]

PROGRESS_COLUMNS = [
    "run_id", "partition", "aggregation", "status",
    "final_f1", "best_f1", "best_round", "wall_clock_s",
    "start_time", "end_time",
]


# ---------------------------------------------------------------------------
# Config materialization
# ---------------------------------------------------------------------------

def _build_run_config(
    base_cfg: dict, partition: dict, aggregation: str,
) -> tuple[dict, str]:
    """Merge sweep base cfg with a specific (partition, aggregation) selection."""
    run_cfg = copy.deepcopy(base_cfg)
    # Pop the sweep sub-block so the downstream loader doesn't see it.
    run_cfg.pop("sweep", None)
    fed = run_cfg.setdefault("federation", {})
    fed["strategy"] = aggregation
    fed["partition"] = partition["fn"]

    kwargs = partition.get("kwargs") or {}
    if "alpha" in kwargs:
        fed["dirichlet_alpha"] = float(kwargs["alpha"])
    if "beta" in kwargs:
        fed["quantity_beta"] = float(kwargs["beta"])
    if "classes_per_client" in kwargs:
        fed["classes_per_client"] = int(kwargs["classes_per_client"])

    # FedBN → client_avg primary metric (per Phase 3 decision).
    if aggregation == "fedbn":
        fed["fedbn_central_eval"] = str(
            fed.get("fedbn_central_eval", "client_avg")
        )
    # FedProx needs proximal_mu > 0; all other strategies must use 0 so the
    # proximal loss term is not added.
    if aggregation == "fedprox":
        fed["proximal_mu"] = float(fed.get("proximal_mu", 0.01))
    else:
        fed["proximal_mu"] = 0.0

    run_id = f"{aggregation}_{partition['name']}_mitbih_seed{run_cfg['seed']}"
    run_cfg["run_id"] = run_id
    return run_cfg, run_id


# ---------------------------------------------------------------------------
# Progress logging
# ---------------------------------------------------------------------------

def _init_progress_csv(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        csv.writer(f).writerow(PROGRESS_COLUMNS)


def _append_progress(
    path: Path, row: dict[str, Any],
) -> None:
    with path.open("a", newline="") as f:
        csv.writer(f).writerow([row.get(c, "") for c in PROGRESS_COLUMNS])


# ---------------------------------------------------------------------------
# Convergence round — first round where val F1 ≥ 0.95 * final val F1
# ---------------------------------------------------------------------------

def _convergence_round(rounds_csv: Path) -> int | None:
    if not rounds_csv.exists():
        return None
    with rounds_csv.open() as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r.get("val_f1_weighted")]
    if not rows:
        return None
    vals = [float(r["val_f1_weighted"]) for r in rows]
    final = vals[-1]
    if final <= 0:
        return None
    threshold = 0.95 * final
    for r, v in zip(rows, vals):
        if v >= threshold:
            return int(r["round"])
    return None


# ---------------------------------------------------------------------------
# Sweep main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 4 non-IID sweep.")
    parser.add_argument(
        "--config", default="configs/sweep_noniid.yaml",
        help="Path to sweep YAML.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List the run matrix without training.",
    )
    args = parser.parse_args()

    cfg = load_config(REPO_ROOT / args.config if not Path(args.config).is_absolute()
                      else args.config)
    sweep = cfg.get("sweep", {})
    partitions = sweep.get("partitions", [])
    aggregations = sweep.get("aggregations", [])

    if not partitions or not aggregations:
        print("ERROR: sweep config must list partitions and aggregations")
        return 2

    total = len(partitions) * len(aggregations)
    print(
        f"Phase 4 sweep: {len(partitions)} partitions × "
        f"{len(aggregations)} aggregations = {total} runs"
    )
    if args.dry_run:
        for p in partitions:
            for a in aggregations:
                _, run_id = _build_run_config(cfg, p, a)
                print(f"  - {run_id}")
        return 0

    progress_csv = METRICS_DIR / "sweep_progress.csv"
    _init_progress_csv(progress_csv)

    completed = 0
    skipped = 0
    failed: list[str] = []
    t0_sweep = time.time()

    for i, partition in enumerate(partitions):
        for j, aggregation in enumerate(aggregations):
            idx = i * len(aggregations) + j + 1
            run_cfg, run_id = _build_run_config(cfg, partition, aggregation)
            print(f"\n[SWEEP {idx:2d}/{total}] start {run_id}", flush=True)

            result_json = METRICS_DIR / f"{run_id}.json"
            if result_json.exists():
                print(f"[SKIP] {run_id} — metrics JSON already exists")
                _append_progress(progress_csv, {
                    "run_id": run_id,
                    "partition": partition["name"],
                    "aggregation": aggregation,
                    "status": "SKIPPED",
                    "final_f1": "",
                    "best_f1": "",
                    "best_round": "",
                    "wall_clock_s": 0,
                    "start_time": datetime.now().isoformat(),
                    "end_time": datetime.now().isoformat(),
                })
                skipped += 1
                continue

            t0 = time.time()
            start_iso = datetime.now().isoformat()
            try:
                result = train_federated(run_cfg, "mitbih")
                elapsed = time.time() - t0
                fm = result.get("final_central_metrics", {}) or {}
                final_f1 = fm.get("f1_macro", None)
                best_f1 = result.get("best_central_f1_macro", None)
                best_round = result.get("best_round", None)
                print(
                    f"[DONE {idx:2d}/{total}] {run_id} · "
                    f"final_f1={final_f1 if final_f1 is not None else 'NaN'} · "
                    f"best_f1={best_f1:.4f} @ round {best_round} · "
                    f"{elapsed / 60:.1f} min",
                    flush=True,
                )
                _append_progress(progress_csv, {
                    "run_id": run_id,
                    "partition": partition["name"],
                    "aggregation": aggregation,
                    "status": "DONE",
                    "final_f1": f"{final_f1:.6f}" if final_f1 is not None else "",
                    "best_f1": f"{best_f1:.6f}" if best_f1 is not None else "",
                    "best_round": str(best_round) if best_round is not None else "",
                    "wall_clock_s": f"{elapsed:.1f}",
                    "start_time": start_iso,
                    "end_time": datetime.now().isoformat(),
                })
                completed += 1
            except Exception:  # noqa: BLE001 — deliberate catch-all
                elapsed = time.time() - t0
                tb = traceback.format_exc()
                LOG_DIR.mkdir(parents=True, exist_ok=True)
                crash_path = LOG_DIR / f"{run_id}_crash.log"
                crash_path.write_text(tb)
                print(f"[FAIL {idx:2d}/{total}] {run_id} — see {crash_path}", flush=True)
                _append_progress(progress_csv, {
                    "run_id": run_id,
                    "partition": partition["name"],
                    "aggregation": aggregation,
                    "status": "FAILED",
                    "final_f1": "",
                    "best_f1": "",
                    "best_round": "",
                    "wall_clock_s": f"{elapsed:.1f}",
                    "start_time": start_iso,
                    "end_time": datetime.now().isoformat(),
                })
                failed.append(run_id)

    sweep_elapsed = time.time() - t0_sweep
    print(f"\n[SWEEP COMPLETE] {completed} done · {skipped} skipped · "
          f"{len(failed)} failed · total {sweep_elapsed / 3600:.2f} h")
    if failed:
        print("Failed runs:")
        for f in failed:
            print(f"  - {f}")

    # ------------------------------------------------------------------
    # Build noniid_summary.csv — one row per (partition, aggregation)
    # ------------------------------------------------------------------
    summary_path = METRICS_DIR / "noniid_summary.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(SUMMARY_COLUMNS)
        for partition in partitions:
            for aggregation in aggregations:
                _, run_id = _build_run_config(cfg, partition, aggregation)
                result_json = METRICS_DIR / f"{run_id}.json"
                if not result_json.exists():
                    continue
                try:
                    obj = json.loads(result_json.read_text())
                except Exception:
                    continue
                fm = obj.get("final_central_metrics", {}) or {}
                test_f1_ca = ""
                test_f1_rep = ""
                if aggregation == "fedbn":
                    dual = obj.get("fedbn_dual_eval") or {}
                    if "client_avg" in dual:
                        v = dual["client_avg"].get("f1_macro", None)
                        if v is not None:
                            test_f1_ca = f"{float(v):.6f}"
                    if "representative" in dual:
                        v = dual["representative"].get("f1_macro", None)
                        if v is not None:
                            test_f1_rep = f"{float(v):.6f}"
                rounds_csv = METRICS_DIR / f"{run_id}_rounds.csv"
                conv = _convergence_round(rounds_csv)
                w.writerow([
                    run_id,
                    partition["name"],
                    aggregation,
                    f"{fm.get('accuracy', 0.0):.6f}" if fm else "",
                    f"{fm.get('f1_macro', 0.0):.6f}" if fm else "",
                    test_f1_ca,
                    test_f1_rep,
                    f"{fm.get('auc_macro', 0.0):.6f}" if fm else "",
                    obj.get("best_round", ""),
                    conv if conv is not None else "",
                    f"{obj.get('runtime_seconds', 0.0):.1f}",
                ])
    print(f"Summary → {summary_path.relative_to(REPO_ROOT)}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
