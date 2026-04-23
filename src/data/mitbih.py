"""MIT-BIH Arrhythmia Database loader + AAMI 5-class beat extraction.

Layout assumed under `raw_path`:
    raw_path/
        100.dat  100.hea  100.atr
        101.dat  101.hea  101.atr
        ...
Signals are 2-channel at 360 Hz. We always read channel 0 (per pipeline spec)
and z-score each beat window of 250 samples (±125 around the R-peak).
Beats whose symbol does not map to an AAMI class are dropped.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import wfdb
from tqdm import tqdm

AAMI_MAPPING: dict[str, int] = {
    # N (Normal)
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,
    # S (Supraventricular ectopic)
    "A": 1, "a": 1, "J": 1, "S": 1,
    # V (Ventricular ectopic)
    "V": 2, "E": 2,
    # F (Fusion)
    "F": 3,
    # Q (Unknown / paced)
    "/": 4, "f": 4, "Q": 4,
}

HALF_WINDOW: int = 125
WINDOW_SIZE: int = 2 * HALF_WINDOW


def list_record_ids(raw_path: Path) -> list[str]:
    """Return sorted record IDs (strings like '100') by globbing `*.dat` files."""
    return sorted(p.stem for p in Path(raw_path).glob("*.dat"))


def load_mitbih_record(record_path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load a single MIT-BIH record.

    `record_path` is the path *without* extension (e.g. `.../mitbih/100`).
    Returns:
        signal       — np.ndarray, shape (N_samples, 2), float64
        beat_samples — np.ndarray, shape (N_beats,), int, R-peak sample indices
        beat_symbols — list[str], shape (N_beats,), annotation symbols
    """
    rec = wfdb.rdrecord(str(record_path))
    ann = wfdb.rdann(str(record_path), extension="atr")
    return rec.p_signal, np.asarray(ann.sample), list(ann.symbol)


def extract_beats(
    signal: np.ndarray,
    beat_samples: np.ndarray,
    beat_symbols: list[str],
    window: int = WINDOW_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice channel-0 beat windows and z-score each one.

    Returns three aligned arrays:
        X — float32, shape (n_kept, window)
        y — int64,   shape (n_kept,)           AAMI class indices
        keep_mask — bool, shape (n_beats,)     True iff the beat was kept
    A beat is kept iff:
      (a) the window fits inside the signal (peak ∈ [half, len - half]), AND
      (b) its symbol appears in AAMI_MAPPING.
    """
    if window != WINDOW_SIZE:
        raise ValueError(f"extract_beats expects window={WINDOW_SIZE}, got {window}")

    channel0 = signal[:, 0].astype(np.float64)
    n_total_samples = channel0.shape[0]

    xs: list[np.ndarray] = []
    ys: list[int] = []
    keep = np.zeros(len(beat_samples), dtype=bool)

    for i, (sample_idx, sym) in enumerate(zip(beat_samples, beat_symbols)):
        if sample_idx < HALF_WINDOW or sample_idx > n_total_samples - HALF_WINDOW:
            continue
        cls = AAMI_MAPPING.get(sym)
        if cls is None:
            continue
        win = channel0[sample_idx - HALF_WINDOW : sample_idx + HALF_WINDOW]
        mean = win.mean()
        std = win.std()
        win_n = (win - mean) / std if std > 1e-8 else win - mean
        xs.append(win_n.astype(np.float32))
        ys.append(cls)
        keep[i] = True

    if xs:
        X = np.stack(xs, axis=0)
    else:
        X = np.empty((0, window), dtype=np.float32)
    return X, np.asarray(ys, dtype=np.int64), keep


def build_mitbih_dataset(raw_path: Path, out_path: Path) -> dict:
    """Preprocess every record and cache X/y/record_ids under `out_path`.

    Returns a small stats dict (total beats, per-class counts, per-record
    dropped counts) for logging.
    """
    raw_path = Path(raw_path)
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_rec: list[np.ndarray] = []

    per_record_drops: dict[str, int] = {}
    records_processed: list[str] = []

    record_ids = list_record_ids(raw_path)
    if not record_ids:
        raise FileNotFoundError(f"No .dat files found in {raw_path}")

    for rec_id in tqdm(record_ids, desc="MIT-BIH records"):
        signal, beat_samples, beat_symbols = load_mitbih_record(raw_path / rec_id)
        X, y, keep = extract_beats(signal, beat_samples, beat_symbols)
        if X.shape[0] == 0:
            per_record_drops[rec_id] = int(len(beat_samples))
            continue
        all_X.append(X)
        all_y.append(y)
        all_rec.append(np.full(X.shape[0], rec_id, dtype="S10"))
        per_record_drops[rec_id] = int((~keep).sum())
        records_processed.append(rec_id)

    X_full = np.concatenate(all_X, axis=0)
    y_full = np.concatenate(all_y, axis=0)
    rec_full = np.concatenate(all_rec, axis=0)

    np.save(out_path / "X.npy", X_full)
    np.save(out_path / "y.npy", y_full)
    np.save(out_path / "record_ids.npy", rec_full)

    class_counts = np.bincount(y_full, minlength=5).tolist()
    return {
        "total_beats": int(X_full.shape[0]),
        "class_counts": class_counts,
        "records_processed": records_processed,
        "per_record_drops": per_record_drops,
    }
