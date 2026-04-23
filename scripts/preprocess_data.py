#!/usr/bin/env python
"""CLI entry point for preprocessing MIT-BIH and/or PTB-XL.

Usage:
    python scripts/preprocess_data.py --dataset {mitbih|ptbxl|all}

Reads raw and processed roots from `configs/base.yaml`. Idempotent: if
`X.npy` already exists in the target output directory, the step is skipped
(pass `--force` to re-run regardless).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `src` importable whether invoked via `make data` or directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.seeding import set_all_seeds  # noqa: E402


def _already_done(out_dir: Path) -> bool:
    if not (out_dir / "X.npy").exists():
        return False
    if (out_dir / "y.npy").exists():
        return True
    return (out_dir / "y_5class.npy").exists() and (out_dir / "y_binary.npy").exists()


def run_mitbih(cfg: dict, force: bool, logger) -> None:
    from src.data.mitbih import build_mitbih_dataset

    raw_root = Path(cfg["data"]["raw_root"])
    processed_root = Path(cfg["data"]["processed_root"])
    raw_path = raw_root / cfg["data"]["mitbih_subdir"]
    out_path = processed_root / "mitbih"

    if not force and _already_done(out_path):
        logger.info(f"[mitbih] cached arrays exist at {out_path}; skipping")
        return

    logger.info(f"[mitbih] building dataset from {raw_path} → {out_path}")
    stats = build_mitbih_dataset(raw_path, out_path)
    (out_path / "stats.json").write_text(json.dumps(stats, indent=2))
    logger.info(
        f"[mitbih] done: {stats['total_beats']} beats, "
        f"class_counts={stats['class_counts']}, "
        f"records={len(stats['records_processed'])}"
    )


def run_ptbxl(cfg: dict, force: bool, logger) -> None:
    from src.data.ptbxl import build_ptbxl_dataset

    raw_root = Path(cfg["data"]["raw_root"])
    processed_root = Path(cfg["data"]["processed_root"])
    raw_path = raw_root / cfg["data"]["ptbxl_subdir"]
    out_path = processed_root / "ptbxl"

    if not force and _already_done(out_path):
        logger.info(f"[ptbxl] cached arrays exist at {out_path}; skipping")
        return

    logger.info(f"[ptbxl] building dataset from {raw_path} → {out_path}")
    stats = build_ptbxl_dataset(raw_path, out_path)
    (out_path / "stats.json").write_text(json.dumps(stats, indent=2))
    logger.info(
        f"[ptbxl] done: {stats['total_records']} records, "
        f"class_counts={stats['class_counts']}, "
        f"dropped={stats['dropped_no_class']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess MIT-BIH and/or PTB-XL.")
    parser.add_argument(
        "--dataset",
        choices=["mitbih", "ptbxl", "all"],
        default="all",
        help="Which dataset(s) to preprocess.",
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "base.yaml"))
    parser.add_argument("--force", action="store_true", help="Reprocess even if cached.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_all_seeds(cfg["seed"])
    logger = get_logger("preprocess")

    if args.dataset in ("mitbih", "all"):
        run_mitbih(cfg, args.force, logger)
    if args.dataset in ("ptbxl", "all"):
        run_ptbxl(cfg, args.force, logger)
    return 0


if __name__ == "__main__":
    sys.exit(main())
