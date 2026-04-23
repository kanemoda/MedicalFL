#!/usr/bin/env python
"""Single-experiment CLI entry point.

Loads a YAML config (with `_base_` inheritance), then dispatches to the
centralized or federated trainer. Federated trainer is a later-phase stub.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a single experiment.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--mode",
        choices=["centralized", "federated"],
        required=True,
        help="Which training regime to launch.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    dataset = cfg.get("dataset")
    if dataset not in {"mitbih", "ptbxl"}:
        raise ValueError(f"config 'dataset' must be 'mitbih' or 'ptbxl'; got {dataset!r}")

    if args.mode == "centralized":
        from src.training.centralized import train_centralized
        train_centralized(cfg, dataset)
    else:
        from src.training.federated import train_federated
        train_federated(cfg, dataset)

    return 0


if __name__ == "__main__":
    sys.exit(main())
