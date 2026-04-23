# MedicalFL

**DP-FedBN: a BatchNorm-local formulation of differentially private federated learning for ECG classification under non-IID conditions.**

Primary datasets: MIT-BIH Arrhythmia (48 records, 360 Hz) and PTB-XL (~21 k records, 100 Hz, Lead II).
Hardware target: single GTX 1660 Super 6 GB, CUDA 11.8+, Python 3.11, seed = 42.

## Quick start

```bash
make setup     # create .venv + install pinned deps + verify CUDA
make data      # preprocess MIT-BIH and PTB-XL into data/processed/
make sanity    # run data assertions + emit sanity figures
make test      # pytest tests/
```

Raw data is expected to already live at `data/raw/` (MIT-BIH in WFDB format,
PTB-XL in the standard PhysioNet layout). The pipeline never downloads,
deletes, or moves raw files.

See [`federated_ecg_master_plan.md`](./federated_ecg_master_plan.md) for the
full research plan and [`DECISIONS.md`](./DECISIONS.md) for the running log
of non-obvious design choices.
