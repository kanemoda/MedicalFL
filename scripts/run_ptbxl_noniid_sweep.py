#!/usr/bin/env python
"""Phase 6A — non-DP FL on PTB-XL binary (4 agg × 3 partition = 12 runs).

Mirrors `scripts/run_noniid_sweep.py` but for the PTB-XL binary task. Reads
`configs/sweep_ptbxl_noniid.yaml`, materialises one run config per
(partition, aggregation) pair, and calls `train_federated(..., 'ptbxl')`.

Resumable: a run is skipped iff `results/metrics/<run_id>.json` exists.

Writes:
  - results/metrics/ptbxl_sweep_progress.csv
  - results/metrics/ptbxl_summary.csv
  - results/logs/<run_id>_crash.log on failure
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
DATASET = "ptbxl"

PROGRESS_COLUMNS = [
    "run_id", "partition", "aggregation", "status",
    "central_f1_best", "central_f1_final",
    "local_f1_mean", "local_f1_std",
    "best_round", "wall_clock_s",
    "start_time", "end_time",
]

SUMMARY_COLUMNS = [
    "run_id", "partition", "aggregation",
    "central_f1_best", "central_f1_final",
    "local_f1_mean", "local_f1_std",
    "central_acc", "central_auc",
    "best_round", "wall_clock_s",
]


def _build_run_config(
    base_cfg: dict, partition: dict, aggregation: str,
) -> tuple[dict, str]:
    run_cfg = copy.deepcopy(base_cfg)
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

    if aggregation == "fedbn":
        fed["fedbn_central_eval"] = str(fed.get("fedbn_central_eval", "client_avg"))
    if aggregation == "fedprox":
        fed["proximal_mu"] = float(fed.get("proximal_mu", 0.01))
    else:
        fed["proximal_mu"] = 0.0

    run_id = f"{aggregation}_{partition['name']}_ptbxl_seed{run_cfg['seed']}"
    run_cfg["run_id"] = run_id
    return run_cfg, run_id


def _init_progress_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="") as f:
        csv.writer(f).writerow(PROGRESS_COLUMNS)


def _append_progress(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", newline="") as f:
        csv.writer(f).writerow([row.get(c, "") for c in PROGRESS_COLUMNS])


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 6A PTB-XL non-DP sweep.")
    parser.add_argument(
        "--config", default="configs/sweep_ptbxl_noniid.yaml",
        help="Path to sweep YAML.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List the run matrix without training.",
    )
    args = parser.parse_args()

    cfg_path = (
        Path(args.config) if Path(args.config).is_absolute()
        else REPO_ROOT / args.config
    )
    cfg = load_config(cfg_path)
    sweep = cfg.get("sweep", {})
    partitions = sweep.get("partitions", [])
    aggregations = sweep.get("aggregations", [])
    if not partitions or not aggregations:
        print("ERROR: sweep config needs partitions and aggregations")
        return 2

    total = len(partitions) * len(aggregations)
    print(
        f"Phase 6A sweep: {len(partitions)} partitions × "
        f"{len(aggregations)} aggregations = {total} runs (dataset=ptbxl, binary)"
    )
    if args.dry_run:
        for p in partitions:
            for a in aggregations:
                _, run_id = _build_run_config(cfg, p, a)
                print(f"  - {run_id}")
        return 0

    progress_csv = METRICS_DIR / "ptbxl_sweep_progress.csv"
    _init_progress_csv(progress_csv)

    completed = 0
    skipped = 0
    failed: list[str] = []
    t0_sweep = time.time()

    idx = 0
    for partition in partitions:
        for aggregation in aggregations:
            idx += 1
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
                    "wall_clock_s": 0,
                    "start_time": datetime.now().isoformat(),
                    "end_time": datetime.now().isoformat(),
                })
                skipped += 1
                continue

            t0 = time.time()
            start_iso = datetime.now().isoformat()
            try:
                result = train_federated(run_cfg, DATASET)
                elapsed = time.time() - t0
                fm = result.get("final_central_metrics") or {}
                final_f1 = fm.get("f1_macro")
                best_f1 = result.get("best_central_f1_macro")
                best_round = result.get("best_round")
                lt = result.get("local_test") or {}
                lm = lt.get("local_test_f1_mean")
                ls = lt.get("local_test_f1_std")
                print(
                    f"[DONE {idx:2d}/{total}] {run_id} · "
                    f"best_central_f1={best_f1:.4f} @ round {best_round} · "
                    f"local_f1_mean="
                    f"{lm if isinstance(lm, float) else float('nan'):.4f} · "
                    f"{elapsed / 60:.1f} min",
                    flush=True,
                )
                _append_progress(progress_csv, {
                    "run_id": run_id,
                    "partition": partition["name"],
                    "aggregation": aggregation,
                    "status": "DONE",
                    "central_f1_best": f"{best_f1:.6f}" if best_f1 is not None else "",
                    "central_f1_final": f"{final_f1:.6f}" if final_f1 is not None else "",
                    "local_f1_mean": f"{lm:.6f}" if isinstance(lm, (int, float)) else "",
                    "local_f1_std": f"{ls:.6f}" if isinstance(ls, (int, float)) else "",
                    "best_round": str(best_round) if best_round is not None else "",
                    "wall_clock_s": f"{elapsed:.1f}",
                    "start_time": start_iso,
                    "end_time": datetime.now().isoformat(),
                })
                completed += 1
            except Exception:  # noqa: BLE001
                elapsed = time.time() - t0
                tb = traceback.format_exc()
                LOG_DIR.mkdir(parents=True, exist_ok=True)
                (LOG_DIR / f"{run_id}_crash.log").write_text(tb)
                print(f"[FAIL {idx:2d}/{total}] {run_id} — see results/logs/{run_id}_crash.log",
                      flush=True)
                _append_progress(progress_csv, {
                    "run_id": run_id,
                    "partition": partition["name"],
                    "aggregation": aggregation,
                    "status": "FAILED",
                    "wall_clock_s": f"{elapsed:.1f}",
                    "start_time": start_iso,
                    "end_time": datetime.now().isoformat(),
                })
                failed.append(run_id)

    sweep_elapsed = time.time() - t0_sweep
    print(
        f"\n[SWEEP COMPLETE] {completed} done · {skipped} skipped · "
        f"{len(failed)} failed · total {sweep_elapsed / 3600:.2f} h"
    )
    if failed:
        print("Failed runs:")
        for f in failed:
            print(f"  - {f}")

    # ----- summary CSV -----
    summary_path = METRICS_DIR / "ptbxl_summary.csv"
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
                fm = obj.get("final_central_metrics") or {}
                lt = obj.get("local_test") or {}
                w.writerow([
                    run_id,
                    partition["name"],
                    aggregation,
                    f"{obj.get('best_central_f1_macro', 0.0):.6f}"
                    if isinstance(obj.get("best_central_f1_macro"), (int, float)) else "",
                    f"{fm.get('f1_macro', 0.0):.6f}" if fm else "",
                    f"{lt.get('local_test_f1_mean', 0.0):.6f}"
                    if isinstance(lt.get("local_test_f1_mean"), (int, float)) else "",
                    f"{lt.get('local_test_f1_std', 0.0):.6f}"
                    if isinstance(lt.get("local_test_f1_std"), (int, float)) else "",
                    f"{fm.get('accuracy', 0.0):.6f}" if fm else "",
                    f"{fm.get('auc_macro', 0.0):.6f}" if fm else "",
                    obj.get("best_round", ""),
                    f"{obj.get('runtime_seconds', 0.0):.1f}",
                ])
    print(f"Summary → {summary_path.relative_to(REPO_ROOT)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
