#!/usr/bin/env python
"""Phase 5 DP sweep — 3 methods × 3 epsilons × 2 partitions = 18 MIT-BIH runs.

Reads ``configs/sweep_dp.yaml``, materialises one run config per
(partition, method, ε) triple, and calls ``train_federated`` sequentially.
Finished runs (identified by ``results/metrics/<run_id>.json``) are
skipped so the sweep is resumable after crashes.

Before the real sweep, a smoke test (3 tiny runs — one per method on
label-skew-c2 at ε=3) must pass. If any smoke run fails the full sweep
aborts.

Writes:
  - results/metrics/dp_sweep_progress.csv  (row appended per run)
  - results/metrics/dp_summary.csv         (one row per successful run,
                                            written at the end)
  - results/logs/<run_id>_crash.log        (traceback for any failed run)
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


# ---------------------------------------------------------------------------
# Config materialisation
# ---------------------------------------------------------------------------


def _eps_to_str(eps: Any) -> str:
    if eps is None:
        return "inf"
    if isinstance(eps, str) and eps.lower() in {"inf", "infinity"}:
        return "inf"
    try:
        v = float(eps)
    except (TypeError, ValueError):
        return "inf"
    if math.isinf(v):
        return "inf"
    return f"{v:.1f}"


def _parse_eps(eps: Any) -> float | None:
    """Return None for +∞ / 'inf' / None; else float."""
    if eps is None:
        return None
    if isinstance(eps, str):
        if eps.lower() in {"inf", "infinity", "+inf", "∞"}:
            return None
        v = float(eps)
    else:
        v = float(eps)
    if math.isinf(v):
        return None
    return v


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
    """Merge sweep base cfg with a specific (partition, method, ε) triple."""
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

    # Consistent FedBN central-eval choice with Phase 4.
    if aggregation == "fedbn":
        fed["fedbn_central_eval"] = fed.get("fedbn_central_eval", "client_avg")
    fed["proximal_mu"] = 0.0  # never FedProx in Phase 5

    dp = run_cfg.setdefault("dp", {})
    dp["enabled"] = True
    dp["mode"] = dp_mode
    dp["target_epsilon"] = target_eps  # None for ε=∞

    eps_str = _eps_to_str(target_eps)
    run_id = run_id_override or (
        f"{method}_{partition['name']}_eps{eps_str}_mitbih_seed{run_cfg['seed']}"
    )
    run_cfg["run_id"] = run_id
    return run_cfg, run_id


# ---------------------------------------------------------------------------
# Progress CSV
# ---------------------------------------------------------------------------


def _init_progress_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="") as f:
        csv.writer(f).writerow(PROGRESS_COLUMNS)


def _append_progress(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", newline="") as f:
        csv.writer(f).writerow([row.get(c, "") for c in PROGRESS_COLUMNS])


# ---------------------------------------------------------------------------
# Run dispatch
# ---------------------------------------------------------------------------


def _execute(run_cfg: dict, run_id: str) -> dict | None:
    """Execute one run; return the result dict on success, else None."""
    try:
        result = train_federated(run_cfg, "mitbih")
        return result
    except Exception:  # noqa: BLE001 — deliberate catch-all
        tb = traceback.format_exc()
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        (LOG_DIR / f"{run_id}_crash.log").write_text(tb)
        return None


def _extract_row(
    *,
    run_id: str,
    partition_name: str,
    method: str,
    dp_mode: str,
    aggregation: str,
    target_eps: float | None,
    result: dict | None,
    elapsed: float,
    start_iso: str,
    end_iso: str,
    status: str,
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
        row["achieved_epsilon"] = ""
        row["central_f1_best"] = ""
        row["central_f1_final"] = ""
        row["local_f1_mean"] = ""
        row["local_f1_std"] = ""
        return row
    dp = result.get("dp") or {}
    ae = dp.get("achieved_epsilon_mean", None)
    row["achieved_epsilon"] = f"{ae:.4f}" if isinstance(ae, (int, float)) else ""
    best = result.get("best_central_f1_macro", None)
    row["central_f1_best"] = f"{best:.6f}" if isinstance(best, (int, float)) else ""
    fm = (result.get("final_central_metrics") or {})
    fc = fm.get("f1_macro", None)
    row["central_f1_final"] = f"{fc:.6f}" if isinstance(fc, (int, float)) else ""
    lt = result.get("local_test") or {}
    lm = lt.get("local_test_f1_mean", None)
    ls = lt.get("local_test_f1_std", None)
    row["local_f1_mean"] = f"{lm:.6f}" if isinstance(lm, (int, float)) else ""
    row["local_f1_std"] = f"{ls:.6f}" if isinstance(ls, (int, float)) else ""
    return row


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _run_smoke_tests(cfg: dict) -> tuple[bool, list[str]]:
    """Three tiny runs, one per method, on label_skew_c2 at ε=3.

    Returns (all_passed, messages).
    """
    messages: list[str] = []
    smoke_partition = {
        "name": "label_skew_c2",
        "fn": "label_skew",
        "kwargs": {"classes_per_client": 2},
    }
    smoke_methods: list[tuple[str, str, str]] = [
        ("dp_fedavg",           "fedavg",           "fedavg"),
        ("dp_fedavg_groupnorm", "fedavg_groupnorm", "fedavg"),
        ("dp_fedbn",            "fedbn",            "fedbn"),
    ]
    any_fail = False
    for method, dp_mode, aggregation in smoke_methods:
        run_id = f"SMOKE_{method}_label_skew_c2_eps3.0"
        run_cfg, _ = _build_run_config(
            cfg,
            smoke_partition,
            method,
            dp_mode,
            aggregation,
            target_eps=3.0,
            run_id_override=run_id,
            rounds_override=5,
            local_epochs_override=1,
            num_clients_override=2,
        )
        t0 = time.time()
        print(f"\n[SMOKE] {method} (dp_mode={dp_mode}) — starting", flush=True)
        try:
            train_federated(run_cfg, "mitbih")
            elapsed = time.time() - t0
            print(f"[SMOKE] {method} PASSED in {elapsed:.1f}s", flush=True)
            messages.append(f"{method}: PASSED ({elapsed:.1f}s)")
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            (LOG_DIR / f"{run_id}_crash.log").write_text(tb)
            # Tolerate known failure for naive BN+DP (DPModeUnsupportedError).
            lower = str(e).lower()
            if method == "dp_fedavg" and (
                "dpmodeunsupportederror" in type(e).__name__.lower()
                or "batchnorm" in lower
                or "shouldreplacemodule" in lower
            ):
                print(
                    f"[SMOKE] {method} refused by Opacus (expected) — "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )
                messages.append(
                    f"{method}: REFUSED BY OPACUS (expected) — "
                    f"{type(e).__name__}: {e}"
                )
            else:
                print(f"[SMOKE] {method} FAILED — see {run_id}_crash.log",
                      flush=True)
                messages.append(f"{method}: FAILED — {type(e).__name__}: {e}")
                any_fail = True
        finally:
            # Clean up smoke metric files so they don't pollute the real sweep.
            for suffix in (".json", "_rounds.csv"):
                p = METRICS_DIR / f"{run_id}{suffix}"
                if p.exists():
                    p.unlink()

    return (not any_fail), messages


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------


def _write_summary_csv(
    cfg: dict, partitions: list[dict], method_matrix: list[list[Any]]
) -> Path:
    path = METRICS_DIR / "dp_summary.csv"
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
                except Exception:  # noqa: BLE001
                    continue
                dp = obj.get("dp") or {}
                best = obj.get("best_central_f1_macro", None)
                fm = (obj.get("final_central_metrics") or {})
                fc = fm.get("f1_macro", None)
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 5 DP sweep.")
    parser.add_argument(
        "--config", default="configs/sweep_dp.yaml",
        help="Path to sweep YAML.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List the run matrix without training.",
    )
    parser.add_argument(
        "--skip-smoke", action="store_true",
        help="Skip the smoke test (discouraged — use only on resume).",
    )
    args = parser.parse_args()

    cfg_path = args.config if Path(args.config).is_absolute() else REPO_ROOT / args.config
    cfg = load_config(cfg_path)
    sweep = cfg.get("sweep", {})
    partitions = sweep.get("partitions", [])
    method_matrix = sweep.get("method_eps_matrix", [])
    if not partitions or not method_matrix:
        print("ERROR: sweep config needs partitions and method_eps_matrix")
        return 2

    total = len(partitions) * len(method_matrix)
    print(
        f"Phase 5 sweep: {len(partitions)} partitions × "
        f"{len(method_matrix)} (method, dp_mode, ε) triples = {total} runs"
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

    if not args.skip_smoke:
        print("\n=== Phase 5 smoke tests ===", flush=True)
        smoke_ok, smoke_msgs = _run_smoke_tests(cfg)
        print("\nSmoke summary:")
        for m in smoke_msgs:
            print(f"  {m}")
        if not smoke_ok:
            print("\nSMOKE TEST FAILED — aborting sweep.")
            return 2
        print("Smoke test: all methods either passed or failed with expected reason.", flush=True)
    else:
        print("[warn] --skip-smoke active — trusting previous smoke outcome.")

    progress_csv = METRICS_DIR / "dp_sweep_progress.csv"
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
                print(
                    f"[DONE {idx:2d}/{total}] {run_id} · "
                    f"best_central_f1={best:.4f} · "
                    f"local_f1_mean={lt if isinstance(lt, float) else 'nan':.4f} · "
                    f"achieved_ε={ae if isinstance(ae, (int, float)) else 'NA'} · "
                    f"{elapsed / 60:.1f} min",
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

    summary_path = _write_summary_csv(cfg, partitions, method_matrix)
    print(f"Summary → {summary_path.relative_to(REPO_ROOT)}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
