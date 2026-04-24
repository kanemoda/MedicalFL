#!/usr/bin/env python
"""Build results/metrics/noniid_summary.csv from the 24 per-run artifacts.

Each row carries:
  run_id, partition, aggregation, status, final_central_f1,
  best_central_f1, best_round, wall_clock_s

``status`` is taken from ``sweep_progress.csv`` (DONE or RECOVERED).
``final_central_f1`` and ``best_central_f1`` are derived from each run's
``{run_id}_rounds.csv`` — the last row and argmax-of-central_f1_macro
respectively — so the same extraction path covers DONE and RECOVERED.
``best_round`` and ``wall_clock_s`` come from ``sweep_progress.csv``.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = REPO_ROOT / "results" / "metrics"
PROGRESS_CSV = METRICS_DIR / "sweep_progress.csv"
SUMMARY_CSV = METRICS_DIR / "noniid_summary.csv"

COLUMNS = [
    "run_id",
    "partition",
    "aggregation",
    "status",
    "final_central_f1",
    "best_central_f1",
    "best_round",
    "wall_clock_s",
]


def _rounds_stats(rounds_csv: Path) -> tuple[float | None, float | None, int | None]:
    """Return (final_central_f1, best_central_f1, best_round)."""
    if not rounds_csv.exists():
        return None, None, None
    with rounds_csv.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, None, None
    def _f(r, k):
        try:
            v = float(r[k])
            return v if v == v else None
        except (KeyError, ValueError, TypeError):
            return None
    final = _f(rows[-1], "central_f1_macro")
    best_r, best_v = None, float("-inf")
    for r in rows:
        v = _f(r, "central_f1_macro")
        if v is None:
            continue
        if v > best_v:
            best_v = v
            best_r = int(r["round"])
    best = best_v if best_r is not None else None
    return final, best, best_r


def main() -> int:
    if not PROGRESS_CSV.exists():
        print(f"ERROR: {PROGRESS_CSV} not found", file=sys.stderr)
        return 2
    prog_rows = list(csv.DictReader(PROGRESS_CSV.open()))

    out_rows: list[dict] = []
    for row in prog_rows:
        run_id = row["run_id"]
        rounds_csv = METRICS_DIR / f"{run_id}_rounds.csv"
        final, best, best_r = _rounds_stats(rounds_csv)
        out_rows.append({
            "run_id": run_id,
            "partition": row["partition"],
            "aggregation": row["aggregation"],
            "status": row["status"],
            "final_central_f1": f"{final:.6f}" if final is not None else "",
            "best_central_f1": f"{best:.6f}" if best is not None else "",
            "best_round": str(best_r) if best_r is not None else row.get("best_round", ""),
            "wall_clock_s": row.get("wall_clock_s", ""),
        })

    with SUMMARY_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows → {SUMMARY_CSV.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
