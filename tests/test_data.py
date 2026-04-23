"""Data-module tests that touch the actual raw data (MIT-BIH + PTB-XL)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.mitbih import (  # noqa: E402
    AAMI_MAPPING,
    WINDOW_SIZE,
    extract_beats,
    list_record_ids,
    load_mitbih_record,
)
from src.data.ptbxl import (  # noqa: E402
    LEAD_II_INDEX,
    NORM_INDEX,
    SUPERCLASS_TO_IDX,
    binarize_ptbxl_labels,
    extract_superclass,
    load_ptbxl_metadata,
    load_ptbxl_record,
)
from src.utils.config import load_config  # noqa: E402

CFG = load_config(REPO_ROOT / "configs" / "base.yaml")
MITBIH_ROOT = Path(CFG["data"]["raw_root"]) / CFG["data"]["mitbih_subdir"]
PTBXL_ROOT = Path(CFG["data"]["raw_root"]) / CFG["data"]["ptbxl_subdir"]


# ---------------------------------------------------------------------------
# MIT-BIH
# ---------------------------------------------------------------------------

def test_mitbih_load_one_record():
    """Load the first available record; signal has 2 channels, beats align with symbols."""
    rec_ids = list_record_ids(MITBIH_ROOT)
    assert rec_ids, "no MIT-BIH records found"
    signal, beat_samples, beat_symbols = load_mitbih_record(MITBIH_ROOT / rec_ids[0])

    assert signal.ndim == 2
    assert signal.shape[1] == 2, f"expected 2 channels, got {signal.shape[1]}"
    assert beat_samples.ndim == 1
    assert len(beat_samples) == len(beat_symbols)
    assert len(beat_samples) > 0


def test_aami_mapping():
    """AAMI symbol-to-class mapping: key codes present; unknown codes map to None."""
    assert AAMI_MAPPING["N"] == 0
    assert AAMI_MAPPING["V"] == 2
    assert AAMI_MAPPING["F"] == 3
    assert AAMI_MAPPING.get("x") is None
    assert AAMI_MAPPING.get("+") is None  # rhythm marker, not a beat


def test_extract_beats_windows_and_normalization():
    """Synthetic input: z-scored windows of length 250, kept ↔ mappable & in-bounds."""
    rng = np.random.default_rng(0)
    signal = rng.standard_normal((10_000, 2))
    beat_samples = np.array([50, 500, 5_000, 9_950])   # first/last should be edge-dropped
    beat_symbols = ["N", "V", "x", "N"]                # 'x' is unmappable

    X, y, keep = extract_beats(signal, beat_samples, beat_symbols)
    # 'N'@50 → too close to left edge; 'x'@5000 → unmappable; 'N'@9950 → too close to right.
    # Only 'V'@500 should survive.
    assert keep.tolist() == [False, True, False, False]
    assert X.shape == (1, WINDOW_SIZE)
    assert y.tolist() == [2]
    # z-score property.
    assert abs(X[0].mean()) < 1e-5
    assert abs(X[0].std() - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# PTB-XL
# ---------------------------------------------------------------------------

def test_ptbxl_load_one_record():
    """ecg_id 1 → shape (1000, 12); Lead II extraction yields (1000,)."""
    db, _scp = load_ptbxl_metadata(PTBXL_ROOT)
    row = db.loc[1]
    record_path = PTBXL_ROOT / row["filename_lr"]
    assert record_path.with_suffix(".dat").exists(), f"missing {record_path}.dat"

    import wfdb
    rec = wfdb.rdrecord(str(record_path))
    assert rec.p_signal.shape == (1000, 12)

    lead = load_ptbxl_record(record_path, lead_index=LEAD_II_INDEX)
    assert lead.shape == (1000,)


def test_ptbxl_superclass_norm_and_mi():
    """A NORM-only record and a clearly MI-coded record both classify correctly."""
    db, scp = load_ptbxl_metadata(PTBXL_ROOT)

    # Row 1: scp_codes = {'NORM': 100.0, ...}
    assert extract_superclass(db.loc[1, "scp_codes"], scp) == "NORM"

    # Find an MI example — any row with a code whose diagnostic_class == 'MI' at positive confidence.
    mi_codes = scp.index[scp["diagnostic_class"] == "MI"].tolist()

    mi_ecg_id: int | None = None
    for ecg_id, codes in db["scp_codes"].items():
        if any(code in mi_codes and float(conf) > 0 for code, conf in codes.items()):
            norm_conf = max((float(codes[c]) for c in codes if c == "NORM"), default=0.0)
            mi_max = max(float(codes[c]) for c in codes if c in mi_codes)
            if mi_max > norm_conf:
                mi_ecg_id = int(ecg_id)
                break
    assert mi_ecg_id is not None, "no MI example found in PTB-XL"
    assert extract_superclass(db.loc[mi_ecg_id, "scp_codes"], scp) == "MI"


def test_ptbxl_superclass_to_idx_covers_five():
    """All 5 superclasses appear in the index map."""
    assert set(SUPERCLASS_TO_IDX.keys()) == {"NORM", "MI", "STTC", "CD", "HYP"}
    assert sorted(SUPERCLASS_TO_IDX.values()) == [0, 1, 2, 3, 4]


def test_binarize_ptbxl_labels():
    """NORM → 0, all other classes → 1."""
    y5 = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    y_bin = binarize_ptbxl_labels(y5, NORM_INDEX)
    assert y_bin.dtype == np.int64
    assert y_bin.tolist() == [0, 1, 1, 1, 1]


def test_ptbxl_binary_loader():
    """y_binary.npy exists, values in {0, 1}, class balance matches expectations."""
    processed = Path(CFG["data"]["processed_root"]) / "ptbxl"
    y_bin_path = processed / "y_binary.npy"
    if not y_bin_path.exists():
        pytest.skip("y_binary.npy not generated yet")
    y_bin = np.load(y_bin_path)
    assert set(np.unique(y_bin).tolist()) == {0, 1}
    counts = np.bincount(y_bin).tolist()
    assert counts == [9246, 12142], f"unexpected binary counts {counts}"
