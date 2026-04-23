"""Rich-backed logger that writes to both stdout and `results/logs/<run_id>.log`."""
from __future__ import annotations

import logging
from pathlib import Path

from rich.logging import RichHandler

_LOG_DIR = Path(__file__).resolve().parents[2] / "results" / "logs"


def get_logger(run_id: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger named `run_id` with a rich stdout handler + a file handler.

    Reuses the logger if already configured (idempotent). The log file lives at
    `results/logs/<run_id>.log`.
    """
    logger = logging.getLogger(run_id)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(_LOG_DIR / f"{run_id}.log")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-7s  %(name)s  %(message)s")
    )
    logger.addHandler(file_handler)

    stdout_handler = RichHandler(
        rich_tracebacks=True, show_path=False, show_time=True, show_level=True
    )
    stdout_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stdout_handler)
    return logger
