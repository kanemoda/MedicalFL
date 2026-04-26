#!/usr/bin/env python
"""Phase 6B — DP FL on PTB-XL binary (3 method × 3 ε × 1 partition = 9 runs).

Mirrors `scripts/run_dp_sweep.py` but for the PTB-XL binary task on
``label_skew_c1`` (the FL-relevant extreme — each of 5 clients holds one of
the 2 binary classes).

Resumable: a run is skipped iff `results/metrics/<run_id>.json` exists.
``dp_fedavg`` at finite ε is recorded as a FAILED row (Opacus refuses the
naive BN+DP combo) — this is a first-class empirical finding.

Writes:
  - results/metrics/ptbxl_dp_sweep_progress.csv
  - results/metrics/ptbxl_dp_summary.csv
  - results/logs/<run_id>_crash.log on failure
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
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
    "run_id", "partition", "method", "dp_mode", "aggregation",
    "target_epsilon", "achieved_epsilon", "status",
    "central_f1_best", "central_f1_final",
    "local_f1_mean", "local_f1_std",
    "wall_clock_s", "start_time", "end_time",
]

SUMMARY_COLUMNS = [
    "run_id", "partition", "method", "dp_mode", "aggregation",
    "target_epsilon", "achieved_epsilon_mean",
    "central_f1_best", "central_f1_final",
    "local_f1_mean", "local_f1_std",
    "wall_clock_s",
]


def _eps_to_str(eps: Any) -> str:
    if eps is None:
        return "inf"
    if isinstance(eps, str) and eps.lower() in {"inf", "infinity"}:
        return "inf"
    try:
        v = float(eps)
    except (TypeError, ValueError):
        return "inf"
    return "inf" if math.isinf(v) else f"{v:.1f}"


def _parse_eps(eps: Any) -> float | None:
    if eps is None:
        return None
    if isinstance(eps, str):
        if eps.lower() in {"inf", "infinity", "+inf", "∞"}:
            return None
        v = float(eps)
    else:
        v = float(eps)
    return None if math.isinf(v) else v


def _build_run_config(
    base_cfg: dict,
    partition: dict,
    method: str,
    dp_mode: str,
    aggregation: str,
    target_eps: float | None,
    *,
    run_id_override: str | None = None,
    rounds_override: int | None = None,
    local_epochs_override: int | None = None,
    num_clients_override: int | None = None,
) -> tuple[dict, str]:
    run_cfg = copy.deepcopy(base_cfg)
    run_cfg.pop("sweep", None)

    fed = run_cfg.setdefault("federation", {})
    fed["strategy"] = aggregation
    fed["partition"] = partition["fn"]
    if rounds_override is not None:
        fed["rounds"] = int(rounds_override)
    if local_epochs_override is not None:
        fed["local_epochs"] = int(local_epochs_override)
    if num_clients_override is not None:
        fed["num_clients"] = int(num_clients_override)

    kwargs = partition.get("kwargs") or {}
    if "alpha" in kwargs:
        fed["dirichlet_alpha"] = float(kwargs["alpha"])
    if "beta" in kwargs:
        fed["quantity_beta"] = float(kwargs["beta"])
    if "classes_per_client" in kwargs:
        fed["classes_per_client"] = int(kwargs["classes_per_client"])

    if aggregation == "fedbn":
        fed["fedbn_central_eval"] = fed.get("fedbn_central_eval", "client_avg")
    fed["proximal_mu"] = 0.0

    dp = run_cfg.setdefault("dp", {})
    dp["enabled"] = True
    dp["mode"] = dp_mode
    dp["target_epsilon"] = target_eps

    eps_str = _eps_to_str(target_eps)
    run_id = run_id_override or (
        f"{method}_{partition['name']}_eps{eps_str}_ptbxl_seed{run_cfg['seed']}"
    )
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


def _execute(run_cfg: dict, run_id: str) -> dict | None:
    try:
        return train_federated(run_cfg, DATASET)
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        (LOG_DIR / f"{run_id}_crash.log").write_text(tb)
        return None


def _extract_row(
    *, run_id: str, partition_name: str, method: str, dp_mode: str,
    aggregation: str, target_eps: float | None, result: dict | None,
    elapsed: float, start_iso: str, end_iso: str, status: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_id": run_id,
        "partition": partition_name,
        "method": method,
        "dp_mode": dp_mode,
        "aggregation": aggregation,
        "target_epsilon": "inf" if target_eps is None else f"{target_eps:.1f}",
        "status": status,
        "wall_clock_s": f"{elapsed:.1f}",
        "start_time": start_iso,
        "end_time": end_iso,
    }
    if result is None:
        for k in ("achieved_epsilon", "central_f1_best", "central_f1_final",
                  "local_f1_mean", "local_f1_std"):
            row[k] = ""
        return row
    dp = result.get("dp") or {}
    ae = dp.get("achieved_epsilon_mean")
    row["achieved_epsilon"] = f"{ae:.4f}" if isinstance(ae, (int, float)) else ""
    best = result.get("best_central_f1_macro")
    row["central_f1_best"] = f"{best:.6f}" if isinstance(best, (int, float)) else ""
    fm = result.get("final_central_metrics") or {}
    fc = fm.get("f1_macro")
    row["central_f1_final"] = f"{fc:.6f}" if isinstance(fc, (int, float)) else ""
    lt = result.get("local_test") or {}
    lm = lt.get("local_test_f1_mean")
    ls = lt.get("local_test_f1_std")
    row["local_f1_mean"] = f"{lm:.6f}" if isinstance(lm, (int, float)) else ""
    row["local_f1_std"] = f"{ls:.6f}" if isinstance(ls, (int, float)) else ""
    return row


def _run_smoke_test(cfg: dict) -> tuple[bool, str]:
    """Quick smoke: DP-FedBN on dirichlet_a01 at ε=3, 2 clients × 5 rounds × 1 epoch.

    Returns (passed, message). Cleans up its own metric files before returning.
    """
    smoke_partition = {
        "name": "dirichlet_a01",
        "fn": "dirichlet",
        "kwargs": {"alpha": 0.1},
    }
    run_id = "SMOKE_dp_fedbn_dirichlet_a01_eps3.0_ptbxl"
    run_cfg, _ = _build_run_config(
        cfg, smoke_partition,
        method="dp_fedbn", dp_mode="fedbn", aggregation="fedbn",
        target_eps=3.0,
        run_id_override=run_id,
        rounds_override=5,
        local_epochs_override=1,
        num_clients_override=2,
    )
    print(f"\n[SMOKE] dp_fedbn on dirichlet_a01 ε=3 (2 clients × 5 rounds × 1 epoch)", flush=True)
    t0 = time.time()
    try:
        train_federated(run_cfg, DATASET)
        elapsed = time.time() - t0
        msg = f"PASSED in {elapsed:.1f}s ({elapsed/60:.1f} min)"
        print(f"[SMOKE] {msg}", flush=True)
        return True, msg
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        (LOG_DIR / f"{run_id}_crash.log").write_text(tb)
        msg = f"FAILED — {type(e).__name__}: {e}"
        print(f"[SMOKE] {msg}", flush=True)
        print(f"[SMOKE] traceback at results/logs/{run_id}_crash.log", flush=True)
        return False, msg
    finally:
        for suffix in (".json", "_rounds.csv"):
            p = METRICS_DIR / f"{run_id}{suffix}"
            if p.exists():
                p.unlink()


def _write_summary(
    cfg: dict, partitions: list[dict], method_matrix: list[list[Any]],
) -> Path:
    path = METRICS_DIR / "ptbxl_dp_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(SUMMARY_COLUMNS)
        for partition in partitions:
            for entry in method_matrix:
                method, dp_mode, aggregation, eps_raw = entry
                target_eps = _parse_eps(eps_raw)
                _, run_id = _build_run_config(
                    cfg, partition, method, dp_mode, aggregation, target_eps,
                )
                result_json = METRICS_DIR / f"{run_id}.json"
                if not result_json.exists():
                    continue
                try:
                    obj = json.loads(result_json.read_text())
                except Exception:
                    continue
                dp = obj.get("dp") or {}
                best = obj.get("best_central_f1_macro")
                fm = obj.get("final_central_metrics") or {}
                fc = fm.get("f1_macro")
                lt = obj.get("local_test") or {}
                w.writerow([
                    run_id,
                    partition["name"],
                    method,
                    dp_mode,
                    aggregation,
                    "inf" if target_eps is None else f"{target_eps:.1f}",
                    (f"{dp.get('achieved_epsilon_mean'):.4f}"
                     if isinstance(dp.get("achieved_epsilon_mean"), (int, float))
                     else ""),
                    f"{best:.6f}" if isinstance(best, (int, float)) else "",
                    f"{fc:.6f}" if isinstance(fc, (int, float)) else "",
                    (f"{lt.get('local_test_f1_mean'):.6f}"
                     if isinstance(lt.get("local_test_f1_mean"), (int, float)) else ""),
                    (f"{lt.get('local_test_f1_std'):.6f}"
                     if isinstance(lt.get("local_test_f1_std"), (int, float)) else ""),
                    f"{obj.get('runtime_seconds', 0.0):.1f}",
                ])
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 6B PTB-XL DP sweep.")
    parser.add_argument(
        "--config", default="configs/sweep_ptbxl_dp.yaml",
        help="Path to sweep YAML.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List the run matrix without training.",
    )
    parser.add_argument(
        "--smoke-only", action="store_true",
        help="Run only the smoke test (dp_fedbn on dirichlet_a01 ε=3, ~3 min).",
    )
    parser.add_argument(
        "--skip-smoke", action="store_true",
        help="Skip the smoke test before the full sweep.",
    )
    args = parser.parse_args()

    cfg_path = (
        Path(args.config) if Path(args.config).is_absolute()
        else REPO_ROOT / args.config
    )
    cfg = load_config(cfg_path)
    sweep = cfg.get("sweep", {})
    partitions = sweep.get("partitions", [])
    method_matrix = sweep.get("method_eps_matrix", [])
    if not partitions or not method_matrix:
        print("ERROR: sweep config needs partitions and method_eps_matrix")
        return 2

    total = len(partitions) * len(method_matrix)
    print(
        f"Phase 6B sweep: {len(partitions)} partitions × "
        f"{len(method_matrix)} (method, dp_mode, ε) triples = {total} runs "
        f"(dataset=ptbxl, binary)"
    )
    if args.dry_run:
        for p in partitions:
            for entry in method_matrix:
                method, dp_mode, aggregation, eps_raw = entry
                target_eps = _parse_eps(eps_raw)
                _, run_id = _build_run_config(
                    cfg, p, method, dp_mode, aggregation, target_eps,
                )
                print(f"  - {run_id}")
        return 0

    if args.smoke_only:
        ok, _ = _run_smoke_test(cfg)
        return 0 if ok else 2

    if not args.skip_smoke:
        ok, msg = _run_smoke_test(cfg)
        if not ok:
            print(f"\nSMOKE FAILED — aborting full sweep. {msg}")
            return 2

    progress_csv = METRICS_DIR / "ptbxl_dp_sweep_progress.csv"
    _init_progress_csv(progress_csv)

    completed = 0
    skipped = 0
    failed: list[str] = []
    t0_sweep = time.time()

    idx = 0
    for partition in partitions:
        for entry in method_matrix:
            idx += 1
            method, dp_mode, aggregation, eps_raw = entry
            target_eps = _parse_eps(eps_raw)
            run_cfg, run_id = _build_run_config(
                cfg, partition, method, dp_mode, aggregation, target_eps,
            )
            target_eps_str = "inf" if target_eps is None else f"{target_eps:.1f}"
            print(
                f"\n[SWEEP {idx:2d}/{total}] {run_id} "
                f"(method={method} dp_mode={dp_mode} agg={aggregation} ε={target_eps_str})",
                flush=True,
            )

            result_json = METRICS_DIR / f"{run_id}.json"
            if result_json.exists():
                print(f"[SKIP] {run_id} — already completed")
                _append_progress(progress_csv, _extract_row(
                    run_id=run_id, partition_name=partition["name"],
                    method=method, dp_mode=dp_mode, aggregation=aggregation,
                    target_eps=target_eps,
                    result=json.loads(result_json.read_text()),
                    elapsed=0.0,
                    start_iso=datetime.now().isoformat(),
                    end_iso=datetime.now().isoformat(),
                    status="SKIPPED",
                ))
                skipped += 1
                continue

            t0 = time.time()
            start_iso = datetime.now().isoformat()
            result = _execute(run_cfg, run_id)
            elapsed = time.time() - t0
            end_iso = datetime.now().isoformat()

            if result is None:
                status = "FAILED"
                print(
                    f"[FAIL {idx:2d}/{total}] {run_id} — "
                    f"see results/logs/{run_id}_crash.log",
                    flush=True,
                )
                failed.append(run_id)
            else:
                status = "DONE"
                best = result.get("best_central_f1_macro")
                dp = result.get("dp") or {}
                ae = dp.get("achieved_epsilon_mean")
                lt = (result.get("local_test") or {}).get("local_test_f1_mean")
                lt_str = f"{lt:.4f}" if isinstance(lt, (int, float)) else "nan"
                ae_str = f"{ae:.4f}" if isinstance(ae, (int, float)) else "NA"
                print(
                    f"[DONE {idx:2d}/{total}] {run_id} · "
                    f"best_central_f1={best:.4f} · local_f1_mean={lt_str} · "
                    f"achieved_ε={ae_str} · {elapsed / 60:.1f} min",
                    flush=True,
                )
                completed += 1

            _append_progress(progress_csv, _extract_row(
                run_id=run_id, partition_name=partition["name"],
                method=method, dp_mode=dp_mode, aggregation=aggregation,
                target_eps=target_eps,
                result=result, elapsed=elapsed,
                start_iso=start_iso, end_iso=end_iso, status=status,
            ))

    sweep_elapsed = time.time() - t0_sweep
    print(
        f"\n[SWEEP COMPLETE] {completed} done · {skipped} skipped · "
        f"{len(failed)} failed · total {sweep_elapsed / 3600:.2f} h"
    )
    if failed:
        print("Failed runs:")
        for f in failed:
            print(f"  - {f}")

    summary_path = _write_summary(cfg, partitions, method_matrix)
    print(f"Summary → {summary_path.relative_to(REPO_ROOT)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
