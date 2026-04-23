"""YAML config loader with `_base_` inheritance via deep merge.

A child config may declare a `_base_: <path>` key at its top level. The path
is resolved relative to the child config file; the base is loaded recursively
and deep-merged under the child (child keys win). Multiple inheritance is not
supported — `_base_` must be a single string or absent.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict where `override` is merged into `base` recursively.

    Nested dicts are merged key-by-key; any non-dict value in `override`
    replaces the corresponding entry in `base` outright.
    """
    out: dict = dict(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving a single optional `_base_` inheritance."""
    path = Path(path).resolve()
    with path.open("r") as f:
        raw = yaml.safe_load(f) or {}

    base_ref = raw.pop("_base_", None)
    if base_ref is None:
        return raw

    base_path = (path.parent / base_ref).resolve()
    base = load_config(base_path)
    return _deep_merge(base, raw)
