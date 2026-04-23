"""Deterministic seeding across python / numpy / torch / CUDA."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_all_seeds(seed: int, strict: bool = False) -> None:
    """Seed every RNG source we care about and configure cuDNN.

    Called exactly once per run (usually at the top of the entry-point script)
    with the global seed from the config.

    `strict=False` (default) keeps `cudnn.benchmark=True` → identical results
    across runs on the *same* hardware/torch build (good enough for this
    project's reproducibility claim) and ~20× faster for small 1-D convolutions
    on a GTX 1660 Super.

    `strict=True` forces `cudnn.deterministic=True, benchmark=False` — bit-exact
    across any hardware/torch build but much slower. Use only when verifying a
    cross-machine reproduction or chasing non-determinism bugs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(strict)
    torch.backends.cudnn.benchmark = not bool(strict)
