"""PTB-XL loader — superclass mapping, Lead-II extraction, z-scored caching.

Layout expected under `raw_path`:
    raw_path/
        ptbxl_database.csv
        scp_statements.csv
        records100/00000/00001_lr.dat  (100 Hz, 12 leads, 1000 samples)
        records100/00000/00001_lr.hea
        records100/01000/...
Lead II is column index 1 in the standard PTB-XL 12-lead order.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm

SUPERCLASSES: list[str] = ["NORM", "MI", "STTC", "CD", "HYP"]
SUPERCLASS_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(SUPERCLASSES)}
NORM_INDEX: int = SUPERCLASS_TO_IDX["NORM"]

LEAD_II_INDEX: int = 1
WINDOW_SAMPLES: int = 1000  # 10 s @ 100 Hz


def binarize_ptbxl_labels(y_5class: np.ndarray, norm_index: int = NORM_INDEX) -> np.ndarray:
    """Map 5-class superclass labels to binary Normal (0) / Abnormal (1).

    Any class other than `norm_index` is abnormal.
    """
    return (y_5class != norm_index).astype(np.int64)


def load_ptbxl_metadata(raw_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (ptbxl_database, scp_statements) with the index on scp_statements set to its code."""
    raw_path = Path(raw_path)
    db = pd.read_csv(raw_path / "ptbxl_database.csv", index_col="ecg_id")
    db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)
    scp = pd.read_csv(raw_path / "scp_statements.csv", index_col=0)
    return db, scp


def extract_superclass(scp_codes: dict[str, float], scp_df: pd.DataFrame) -> str | None:
    """Return the diagnostic superclass with the highest confidence, or None.

    A code is eligible iff it is indexed in `scp_df` AND its `diagnostic_class`
    field is non-null. Among eligible codes, we pick the one whose confidence
    value in `scp_codes` is the largest. Ties are broken by the natural order
    of scp_codes dict iteration (Python 3.7+ insertion order).
    """
    best_cls: str | None = None
    best_conf: float = -1.0
    for code, conf in scp_codes.items():
        if code not in scp_df.index:
            continue
        cls = scp_df.at[code, "diagnostic_class"]
        if not isinstance(cls, str) or not cls:
            continue
        try:
            conf_f = float(conf)
        except (TypeError, ValueError):
            continue
        if conf_f > best_conf:
            best_conf = conf_f
            best_cls = cls
    return best_cls


def load_ptbxl_record(record_path: Path, lead_index: int = LEAD_II_INDEX) -> np.ndarray:
    """Load a single PTB-XL record and return the chosen lead as a 1-D float64 array."""
    rec = wfdb.rdrecord(str(record_path))
    signal = rec.p_signal  # (1000, 12)
    if signal.shape[1] <= lead_index:
        raise ValueError(
            f"Record {record_path} has only {signal.shape[1]} leads; "
            f"requested index {lead_index}."
        )
    return signal[:, lead_index].astype(np.float64)


def _zscore(x: np.ndarray) -> np.ndarray:
    mean = x.mean()
    std = x.std()
    return ((x - mean) / std).astype(np.float32) if std > 1e-8 else (x - mean).astype(np.float32)


def build_ptbxl_dataset(raw_path: Path, out_path: Path) -> dict:
    """Filter + load every PTB-XL record, cache X/y/strat_fold/ecg_id under `out_path`."""
    raw_path = Path(raw_path)
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    db, scp = load_ptbxl_metadata(raw_path)

    # First pass: assign superclass per row, drop rows without one.
    superclasses: list[str | None] = [
        extract_superclass(codes, scp) for codes in db["scp_codes"]
    ]
    db = db.assign(_superclass=superclasses)
    total_input = len(db)
    kept = db[db["_superclass"].notna()].copy()
    dropped = total_input - len(kept)

    xs: list[np.ndarray] = []
    ys: list[int] = []
    folds: list[int] = []
    ecg_ids: list[int] = []
    missing_records: list[int] = []

    for ecg_id, row in tqdm(kept.iterrows(), total=len(kept), desc="PTB-XL records"):
        record_path = raw_path / row["filename_lr"]
        if not (record_path.with_suffix(".dat").exists()):
            missing_records.append(int(ecg_id))
            continue
        lead = load_ptbxl_record(record_path)
        if lead.shape[0] != WINDOW_SAMPLES:
            missing_records.append(int(ecg_id))
            continue
        xs.append(_zscore(lead))
        cls_name = row["_superclass"]
        ys.append(SUPERCLASS_TO_IDX[cls_name])
        folds.append(int(row["strat_fold"]))
        ecg_ids.append(int(ecg_id))

    X = np.stack(xs, axis=0) if xs else np.empty((0, WINDOW_SAMPLES), dtype=np.float32)
    y_5class = np.asarray(ys, dtype=np.int64)
    y_binary = binarize_ptbxl_labels(y_5class, NORM_INDEX)
    fold_arr = np.asarray(folds, dtype=np.int8)
    id_arr = np.asarray(ecg_ids, dtype=np.int64)

    np.save(out_path / "X.npy", X)
    np.save(out_path / "y_5class.npy", y_5class)
    np.save(out_path / "y_binary.npy", y_binary)
    np.save(out_path / "strat_fold.npy", fold_arr)
    np.save(out_path / "ecg_id.npy", id_arr)

    class_counts = np.bincount(y_5class, minlength=len(SUPERCLASSES)).tolist()
    binary_counts = np.bincount(y_binary, minlength=2).tolist()
    fold_counts = {int(f): int((fold_arr == f).sum()) for f in sorted(set(folds))}
    return {
        "total_records": int(X.shape[0]),
        "class_counts": class_counts,
        "class_names": SUPERCLASSES,
        "binary_counts": binary_counts,
        "binary_class_names": ["Normal", "Abnormal"],
        "fold_counts": fold_counts,
        "dropped_no_class": int(dropped),
        "missing_records": missing_records,
    }
