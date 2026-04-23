# Federated ECG Classification — Unified Implementation Plan

**Working title:** *DP-FedBN: A BatchNorm-Local Formulation of Differentially Private Federated Learning for ECG Classification under Non-IID Conditions*

**Author:** Efe Deniz Bağlar, Ege University (advisor: Prof. Dr. Hasan Bulut)
**Hardware:** Single GTX 1660 Super 6 GB, CUDA 11.8+, Ubuntu, Python 3.11
**Project root:** `/home/kanemoda/MedicalFL/`
**Raw data (pre-downloaded):** `/home/kanemoda/MedicalFL/data/raw/` — contains both MIT-BIH and PTB-XL
**Timeline:** 7 days to advisor-ready package; 2–3 weeks to preprint-submittable paper
**Executor:** Claude Code Opus 4.7 (max effort), guided phase-by-phase

---

## 1. Core claim and novelty

**Primary contribution (the novelty that carries the paper):**

> We formulate **DP-FedBN**, the correct composition of DP-SGD and FedBN. Opacus (and DP-SGD in general) is incompatible with BatchNorm because per-sample gradients are ill-defined under batch statistics. The standard workaround replaces BN with GroupNorm, which destroys the FedBN mechanism. Our formulation applies DP-SGD only to Conv/Linear parameters; BN parameters and running statistics stay local, never cross the federation boundary, and therefore do not require DP protection. This is simple, correct, and to our knowledge underspecified in prior FL+DP literature.

**Supporting contributions (breadth):**

1. **Systematic non-IID benchmark** on ECG: 6 partitioning strategies × 4 aggregation methods (FedAvg, FedProx, FedBN, FedPerf) on MIT-BIH.
2. **Privacy-utility frontier under FL**: ε sweep at {0.5, 1, 2, 5, 10, ∞} × {FedAvg, FedBN}, showing DP-FedBN degrades more gracefully than DP-FedAvg under realistic non-IID.
3. **Cross-dataset validation** on PTB-XL (separate federation, not joint) demonstrating findings generalize.

**Publication target (realistic):** MDPI Sensors / Electronics (fast OA turnaround), ML4H workshop at NeurIPS, IEEE BHI, EMBC. Short-form (8–10 pages) is the right size for a single-seed benchmark + formulation paper.

**What this is NOT:** a "we propose a new aggregation method" paper. FedPerf (softmax-weighted-by-val-F1) was originally considered our method but is well-covered by prior work (FedAdp, FedNova variants). We include it as one baseline among four to honestly contextualize DP-FedBN.

---

## 2. Hard constraints

- **Single GTX 1660 Super 6 GB VRAM.** No remote compute. No cloud.
- **7 days to advisor deliverable.** Code + experiments + memo + slides.
- **Single seed (42) for all runs in the MVP.** Multi-seed variance analysis deferred to post-sprint expansion.
- **CUDA-only.** No MPS considerations — that was a leftover from an earlier planning iteration.
- **Raw data is already on disk** at `/home/kanemoda/MedicalFL/data/raw/`. Do NOT re-download. Do NOT delete or move existing raw files. MIT-BIH format is currently unverified; Phase 0+1 starts with a mandatory inventory step.
- **No scope creep without `DECISIONS.md` entry.** Claude Code must log any deviation.
- **Deterministic everything:** seed all RNGs, fix `cudnn.deterministic = True`, log git SHA + pip freeze per run.

---

## 3. Target directory structure

The project lives at `/home/kanemoda/MedicalFL/` (existing directory, raw data already inside). Do NOT create a nested `federated-ecg/` subdirectory; scaffold directly into `/home/kanemoda/MedicalFL/` and preserve anything already there.

```
/home/kanemoda/MedicalFL/
├── Makefile                          # make setup|data|train-*|sweep-*|figures|paper|clean|all
├── pyproject.toml                    # project metadata + ruff config
├── requirements.txt                  # pinned versions
├── .gitignore                        # Python + data/ + results/ + checkpoints
├── README.md                         # quick start
├── DECISIONS.md                      # running log of non-obvious choices
├── configs/
│   ├── base.yaml                     # shared hyperparams
│   ├── centralized_mitbih.yaml
│   ├── centralized_ptbxl.yaml
│   ├── federated_base.yaml           # shared FL defaults
│   ├── sweep_noniid.yaml             # 6×4 matrix definition
│   ├── sweep_dp.yaml                 # 6×2 matrix definition
│   └── crossdataset_ptbxl.yaml
├── src/
│   ├── data/
│   │   ├── mitbih.py                 # loading, beat extraction, AAMI mapping
│   │   ├── ptbxl.py                  # loading, superclass mapping, lead-II extraction
│   │   └── partition.py              # IID + 5 non-IID strategies
│   ├── models/
│   │   ├── cnn1d.py                  # primary model, BN-heavy for FedBN
│   │   └── resnet1d.py               # optional ablation
│   ├── federation/
│   │   ├── client.py                 # local training client, DP-aware
│   │   ├── server.py                 # round orchestration, central eval
│   │   ├── aggregation.py            # FedAvg, FedProx, FedBN, FedPerf
│   │   └── dp.py                     # Opacus wiring with BN exclusion (DP-FedBN)
│   ├── training/
│   │   ├── centralized.py
│   │   └── federated.py
│   ├── evaluation/
│   │   ├── metrics.py                # accuracy, F1, AUC, per-class breakdown
│   │   └── visualization.py          # all paper figures
│   └── utils/
│       ├── seeding.py
│       ├── logging.py
│       └── config.py                 # YAML loading + inheritance
├── scripts/
│   ├── run_experiment.py             # main entry: single experiment
│   ├── run_noniid_sweep.py           # 24 experiments
│   ├── run_dp_sweep.py               # 12 experiments
│   ├── run_crossdataset.py           # 4 experiments on PTB-XL
│   ├── generate_figures.py           # reproduce all paper plots
│   └── generate_deliverables.py      # advisor memo + slides from results
├── tests/
│   ├── test_data.py
│   ├── test_partition.py
│   ├── test_models.py
│   ├── test_federation.py
│   └── test_dp_fedbn.py              # critical: BN exclusion sanity
├── data/                             # gitignored
│   ├── raw/                          # ALREADY POPULATED with MIT-BIH + PTB-XL; do not touch
│   └── processed/                    # created by preprocessing
├── results/                          # gitignored
│   ├── logs/
│   ├── checkpoints/
│   ├── metrics/                      # per-run JSON + CSV
│   └── figures/
├── paper/                            # LaTeX source
│   ├── main.tex
│   ├── figures/
│   ├── tables/
│   └── references.bib
└── .venv/                            # gitignored
```

---

## 4. Experimental matrix (full)

All runs use **seed = 42**, CUDA device 0, CNN1d model unless noted. One seed for MVP; variance analysis is post-sprint work.

### Track A — MIT-BIH (primary)

#### Phase 2 — Centralized baseline (1 run)
- `centralized_mitbih_cnn1d_seed42` — CNN1d on full dataset, 50 epochs.

#### Phase 3 — Federation sanity (1 run)
- `fedavg_iid_mitbih_seed42` — 5 clients, IID split, FedAvg, 50 rounds, 5 local epochs. Must come within 2–3% of centralized.

#### Phase 4 — Non-IID sweep (24 runs)

| Partition              | × | Aggregation                          | = | Runs |
|------------------------|---|--------------------------------------|---|------|
| IID                    |   |                                      |   |      |
| label_skew (C=2)       |   | FedAvg                               |   |      |
| quantity_skew (β=0.5)  | × | FedProx (μ=0.01)                     | = | 24   |
| dirichlet α=0.1        |   | FedBN                                |   |      |
| dirichlet α=0.5        |   | FedPerf (τ=1.0)                      |   |      |
| dirichlet α=1.0        |   |                                      |   |      |

- 5 clients, 50 rounds, 5 local epochs, batch 64.

#### Phase 5 — DP sweep (12 runs)
On Dirichlet α=0.5 (moderate non-IID):

| ε value           | × | Aggregation          | = | Runs |
|-------------------|---|----------------------|---|------|
| 0.5, 1, 2, 5, 10, ∞ | × | FedAvg, **FedBN**  | = | 12   |

- δ = 1e-5, max_grad_norm = 1.0.
- Runs with ε=∞ reproduce Phase 4 entries for consistency check.

### Track B — PTB-XL (cross-dataset validation, 4 runs)

- `centralized_ptbxl_cnn1d_seed42`
- `fedavg_dir05_ptbxl_seed42`
- `fedbn_dir05_ptbxl_seed42`
- `fedbn_dp2_dir05_ptbxl_seed42` (the key "does DP-FedBN help on another dataset?" run)

**Total: 42 runs.** Sequential runtime estimate on 1660 Super with CNN1d:

- MIT-BIH federated (5 clients, ~20k samples/client, ~50 rounds): **~60–90 min/run**.
- MIT-BIH centralized: **~15 min**.
- MIT-BIH federated with DP: **~90–120 min** (Opacus overhead ~1.5×).
- PTB-XL federated (larger per client): **~60–90 min**.

Aggregate: 24 non-DP runs × 75 min = 30 hrs + 12 DP runs × 105 min = 21 hrs + 4 PTB-XL = 5 hrs + 2 centralized = 30 min ≈ **~57 hours of compute**. Spread over days 4–6 with machine left running, comfortable.

---

## 5. Phase execution plan

| Day | Phase          | Work                                                                  |
|-----|----------------|-----------------------------------------------------------------------|
| 1   | 0 + 1          | Scaffolding, deps, Makefile, data inventory + preprocess, partition   |
| 2   | 2              | Models, centralized baseline (Track A + B), metrics infra             |
| 3   | 3              | FedAvg + FedProx + FedBN + FedPerf, IID sanity run                    |
| 4   | 4 start + 5    | Launch 24-run non-IID sweep overnight; implement DP-FedBN             |
| 5   | 5              | Launch 12-run DP sweep overnight; verify non-IID sweep completed      |
| 6   | 6 + 7 start    | Cross-dataset (4 runs); start figures & analysis                      |
| 7   | 7 + 8          | Finalize figures, generate advisor memo + slides, push to GitHub      |

Phase-wise Claude Code prompts are below. Each prompt is designed to be **copy-pasted as-is** into Claude Code. No editing needed. After each phase, Claude Code returns results; then the next prompt is pasted in.

---

## 6. Communication protocol

After each phase, Claude Code reports:
- What was built (file paths).
- Tests run and passed.
- Anomalies encountered.
- Decisions logged to `DECISIONS.md`.
- A short "ready for next phase?" confirmation.

User reviews. If OK → paste next prompt. If not → debug in the chat.

**Critical decision points:**
- After Phase 2: centralized F1 on MIT-BIH must be ≥0.85 macro. Below that, debug (probably class weighting, LR, or R-peak detection).
- After Phase 3: FedAvg IID must be within 3% of centralized. Below that, debug (sync issues, aggregation bug).
- After Phase 4: Does FedBN beat FedAvg under non-IID? This is the story's core. If no, investigate — BN local should help with feature distribution shift.
- After Phase 5: Is DP-FedBN gap over DP-FedAvg visible at ε≤2? That's the DP claim.

---

## 7. Risk mitigation

| Risk                                                 | Mitigation                                                                        |
|------------------------------------------------------|-----------------------------------------------------------------------------------|
| Opacus+BN refuses to wire (version mismatch)        | Two-optimizer approach (see Phase 5 prompt); pin Opacus 1.5.x; fallback: functional BN trick |
| FedBN produces no gain over FedAvg                  | Verify BN exclusion is actually happening; inspect BN running stats per client; if still flat, write honest negative result — still publishable |
| 1660 Super OOM with DP                               | Opacus BatchMemoryManager; physical batch 16, logical 64                          |
| MIT-BIH class imbalance causes degenerate training  | Weighted cross-entropy with inverse-frequency per client                          |
| Sweep run fails mid-run                              | All runs are independent; `scripts/run_noniid_sweep.py` is resumable (skip existing result dirs) |
| Non-IID partitioning produces empty clients         | Validate partition has ≥50 samples per client per class present; if not, retry split or reduce K |
| PTB-XL preprocessing slow (21k records, wfdb I/O)    | Process records in parallel (joblib) or overnight on day 1; cache result once     |
| MIT-BIH raw format non-standard                      | Phase 0+1 begins with mandatory INVENTORY.md; Claude Code must halt and report if format deviates from wfdb |

---

## 8. References (read before implementing corresponding phase)

- **FedBN:** Li et al., ICLR 2021. <https://arxiv.org/abs/2102.07623>
- **DP-SGD:** Abadi et al., CCS 2016. <https://arxiv.org/abs/1607.00133>
- **Opacus:** <https://opacus.ai/> and <https://github.com/pytorch/opacus>
- **FedProx:** Li et al., MLSys 2020. <https://arxiv.org/abs/1812.06127>
- **PTB-XL:** Wagner et al., Sci. Data 2020. <https://www.nature.com/articles/s41597-020-0495-6>
- **MIT-BIH Arrhythmia:** Moody & Mark, IEEE EMB Mag. 2001. <https://physionet.org/content/mitdb/1.0.0/>
- **Dirichlet partitioning:** Yurochkin et al., ICML 2019 (the standard FL non-IID protocol).

---

# PART II — Claude Code Prompts (Copy-Paste Ready)

Each prompt below is self-contained. Paste it into Claude Code as-is when ready to execute that phase.

---

## PROMPT — Phase 0 + 1 (Scaffolding + Data Pipeline)

```
# TASK: Phase 0 + 1 — Scaffolding + data pipeline (pre-downloaded data)

## Hardcoded context

- Project root: /home/kanemoda/MedicalFL/    ← already exists, work inside it
- Raw data already downloaded at: /home/kanemoda/MedicalFL/data/raw/
- Do NOT re-download anything.
- Do NOT delete or move any existing files in data/raw/.
- Target machine: Ubuntu, GTX 1660 Super 6 GB VRAM, CUDA 11.8+, Python 3.11.
- Deterministic everywhere (seed = 42).
- If a decision is unclear: pick a sensible default, log it to DECISIONS.md
  at repo root, continue. Do NOT block with clarification questions unless
  truly ambiguous.
- Priority: "runs end-to-end" > "elegant architecture".

## STEP 0 — Data inventory (DO THIS FIRST, before writing any code)

Before implementing anything, inspect what is actually in data/raw/:

1. Run: `ls -la /home/kanemoda/MedicalFL/data/raw/`
2. Run: `find /home/kanemoda/MedicalFL/data/raw/ -maxdepth 3 -type d`
3. For each top-level directory, run: `ls | head -30` and `du -sh <dir>`.
4. For MIT-BIH, specifically check:
   - Are there `.dat`, `.hea`, `.atr` files (standard wfdb format)?
   - If yes, what's the directory? (likely `mitdb/` or
     `mit-bih-arrhythmia-database-1.0.0/`).
   - Are there any `.zip` / `.tar.gz` files that need to be extracted first?
   - If MIT-BIH files are NOT in standard wfdb format, STOP and report to me
     what you found — we will adapt.
5. For PTB-XL, confirm the standard PhysioNet structure:
   - `records100/` (100 Hz data)
   - `records500/` (500 Hz data, may or may not exist)
   - `ptbxl_database.csv`
   - `scp_statements.csv`
6. Write a short `data/raw/INVENTORY.md` summarizing what you found:
   file counts, directory structure, sizes, any anomalies.

Only after INVENTORY.md exists should you proceed to writing code.

If MIT-BIH is in standard wfdb format (`.dat`/`.hea`/`.atr` per record):
proceed with implementation as planned. Record the exact path to the
MIT-BIH directory in DECISIONS.md and use it consistently.

If MIT-BIH is NOT in standard wfdb format (e.g., pre-extracted CSVs, HDF5,
different naming): STOP and report what you found with a few example
filenames, and wait for the user to confirm how to proceed.

## Directory structure to create (inside /home/kanemoda/MedicalFL/)

Use /home/kanemoda/MedicalFL/ ITSELF as the project root. Do NOT create
a nested federated-ecg/ subdirectory. If /home/kanemoda/MedicalFL/
already has some structure, PRESERVE it and only create what's missing.

/home/kanemoda/MedicalFL/
├── Makefile
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── README.md
├── DECISIONS.md
├── configs/base.yaml
├── src/{data,models,federation,training,evaluation,utils}/__init__.py
├── scripts/
├── tests/
├── data/
│   ├── raw/              ← ALREADY EXISTS with the data, do not touch
│   └── processed/        ← create, gitignored
├── results/{logs,checkpoints,metrics,figures}/   (gitignored)

All __init__.py files: module-level docstring explaining purpose.

## Dependencies (requirements.txt)

torch>=2.2,<2.5
torchvision
opacus>=1.5,<1.6
wfdb>=4.1
h5py
pandas>=2.1
numpy>=1.26,<2.0
scipy>=1.12
scikit-learn>=1.4
matplotlib>=3.8
seaborn>=0.13
pyyaml>=6.0
tqdm>=4.66
rich>=13.0
pytest>=8.0

DO NOT add tensorflow, lightning, hydra, wandb.

Install into /home/kanemoda/MedicalFL/.venv/. Verify CUDA:
python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

## Makefile targets (use .venv/bin/python)

- make setup         → create .venv, pip install -r requirements.txt
- make data          → python scripts/preprocess_data.py --dataset all
                       (NO download — data is already in data/raw/)
- make sanity        → python scripts/sanity_check_data.py
- make test          → pytest tests/
- make clean         → rm -rf data/processed/ results/
                       (NEVER touch data/raw/)
- make clean-results → rm -rf results/

Stubs for later phases:
- make train-central, train-fed, sweep-noniid, sweep-dp, crossdataset,
  figures, paper, all

## configs/base.yaml

seed: 42
device: "cuda"
num_workers: 4
data:
  raw_root: "/home/kanemoda/MedicalFL/data/raw"
  processed_root: "/home/kanemoda/MedicalFL/data/processed"
  mitbih_subdir: "<TBD — fill in based on INVENTORY.md>"
  ptbxl_subdir:  "<TBD — path to dir containing records100/ and ptbxl_database.csv>"
mitbih:
  sample_rate: 360
  window_samples: 250
  num_classes: 5
  class_names: ["N", "S", "V", "F", "Q"]
ptbxl:
  sample_rate: 100
  window_samples: 1000
  num_classes: 5
  class_names: ["NORM", "MI", "STTC", "CD", "HYP"]
  lead: "II"
training:
  batch_size: 64
  lr: 0.001
  weight_decay: 0.0001
  optimizer: "adamw"
federation:
  num_clients: 5
  rounds: 50
  local_epochs: 5

After Step 0 inventory, fill in mitbih_subdir and ptbxl_subdir with the
actual paths relative to data/raw/.

## MIT-BIH processing (assuming standard wfdb format)

For each record in the MIT-BIH directory:
1. Load via `wfdb.rdrecord(record_path)` → signal shape (N_samples, 2).
2. Load annotations via `wfdb.rdann(record_path, extension='atr')` →
   .sample (R-peak indices), .symbol (beat codes).
3. USE CHANNEL 0 only.
4. For each annotated beat:
   - Skip if beat_sample < 125 or beat_sample > len(signal) - 125.
   - Extract window = signal[beat_sample - 125 : beat_sample + 125, 0].
   - Z-score normalize the window.
   - Map symbol to AAMI class:
     N: N, L, R, e, j
     S: A, a, J, S
     V: V, E
     F: F
     Q: /, f, Q
     DROP beats with any other symbol.
5. Cache as:
   data/processed/mitbih/X.npy         (float32, [n_beats, 250])
   data/processed/mitbih/y.npy         (int64,   [n_beats])
   data/processed/mitbih/record_ids.npy (S10,    [n_beats])
6. Expected total: ~100k beats (varies with drop criteria).

## PTB-XL processing (standard PhysioNet format)

1. Load ptbxl_database.csv and scp_statements.csv with pandas.
2. For each row in ptbxl_database.csv:
   - Parse scp_codes via ast.literal_eval.
   - Look up each code's diagnostic_class in scp_statements.csv.
   - If no code has a non-null diagnostic_class: DROP record.
   - Else: superclass = code with highest probability in scp_codes.
3. Load signal via `wfdb.rdrecord(filename_lr)` (records100, 100Hz).
4. Signal shape (1000, 12). Extract Lead II = column index 1.
5. Z-score normalize.
6. Cache as:
   data/processed/ptbxl/X.npy          (float32, [n_records, 1000])
   data/processed/ptbxl/y.npy          (int64,   [n_records])
   data/processed/ptbxl/strat_fold.npy (int8,    [n_records])
   data/processed/ptbxl/ecg_id.npy     (int64,   [n_records])
7. Expected total: ~21k records after filtering.

## src/data/mitbih.py

- AAMI_MAPPING: dict[str, int] as above.
- load_mitbih_record(record_path: Path) → (signal, beat_samples, beat_symbols).
- extract_beats(signal, beat_samples, beat_symbols, window=250) →
  (X, y, record_tag).
- build_mitbih_dataset(raw_path: Path, out_path: Path) → orchestrates, tqdm.

## src/data/ptbxl.py

- load_ptbxl_metadata(raw_path: Path) → pd.DataFrame.
- extract_superclass(scp_codes_str: str, scp_statements_df) → str | None.
- load_ptbxl_record(record_path: Path, lead_index: int = 1) → np.ndarray[1000].
- build_ptbxl_dataset(raw_path: Path, out_path: Path) → orchestrates.

## src/data/partition.py

6 strategies with signature:
    partition_fn(X, y, num_clients, seed, **kwargs) →
        list[tuple[np.ndarray, np.ndarray, np.ndarray]]   # (X_i, y_i, idx_i)

1. partition_iid(X, y, num_clients, seed): stratified equal split.
2. partition_label_skew(X, y, num_clients, seed, classes_per_client=2).
3. partition_quantity_skew(X, y, num_clients, seed, beta=0.5).
4. partition_dirichlet(X, y, num_clients, seed, alpha): standard protocol
   (Yurochkin 2019). For each class, sample proportions ~ Dir(alpha, K).

Also plot_partition_distribution(partitions, class_names, save_path) →
stacked bar chart.

Handle empty-class edge cases: log warning, don't crash.

## src/utils/seeding.py

def set_all_seeds(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

## src/utils/config.py

def load_config(path: str) → dict:
    YAML loader supporting `_base_` inheritance via deep merge.

## src/utils/logging.py

Basic rich logger to results/logs/{run_id}.log + stdout.

## scripts/preprocess_data.py

CLI: python scripts/preprocess_data.py --dataset {mitbih|ptbxl|all}
Reads raw paths from configs/base.yaml. Idempotent (skip if X.npy exists).
tqdm progress.

## scripts/sanity_check_data.py

Loads processed arrays. Prints AND asserts:
1. MIT-BIH: total beats, class distribution, unique record count.
   Assert no unexpected drops; log which records missing.
2. PTB-XL: total records after filter, superclass distribution, fold 1–10 counts.
3. No NaN/inf in X.
4. No empty classes.
5. results/figures/sanity_class_balance.png (side-by-side bars).
6. results/figures/sanity_samples.png (grid of 5 examples per class per dataset).
Exit nonzero on assertion failure.

## Tests

tests/test_data.py:
- test_mitbih_load_one_record: load first available record, check shapes.
- test_ptbxl_load_one_record: load ecg_id 1, shape (1000,12), Lead II shape (1000,).
- test_aami_mapping: 'N'→0, 'V'→2, 'F'→3, 'x'→None.
- test_ptbxl_superclass: NORM and MI examples classify correctly.

tests/test_partition.py:
- test_iid_preserves_samples.
- test_iid_class_balance (within 5% of global).
- test_label_skew_limited_classes (≤ classes_per_client).
- test_dirichlet_alpha_large_approaches_iid (alpha=100 → within 10%).
- test_dirichlet_alpha_small_is_skewed (alpha=0.1 → max share >0.8 for some class).

## .gitignore

__pycache__/
*.pyc
.venv/
data/raw/
data/processed/
results/
checkpoints/
*.npy
*.npz
*.pt
.DS_Store
.idea/
.vscode/

## pyproject.toml

[project]
name = "medicalfl"
version = "0.1.0"
requires-python = ">=3.11"

[tool.ruff]
line-length = 100
target-version = "py311"

## DECISIONS.md

Initialize with "Phase 0+1 Decisions":
- Exact path found for MIT-BIH in data/raw/.
- Exact path found for PTB-XL in data/raw/.
- Format decisions (e.g., "MIT-BIH in wfdb format at X, using wfdb.rdrecord").
- Storage format for processed data.
- AAMI drop policy for unknown beat codes.

## README.md

Short: project name, one-line description, quick-start
(make setup && make data && make sanity && make test), link to
federated_ecg_master_plan.md.

## Execution order

1. STEP 0: Inventory data/raw/. Write INVENTORY.md. Confirm MIT-BIH is
   standard wfdb format. If not, STOP and report.
2. Create directory skeleton + config/metadata files (preserve existing).
3. Create .venv, install deps, verify CUDA.
4. Implement utils (seeding, config, logging).
5. Update configs/base.yaml with actual paths from INVENTORY.md.
6. Implement src/data/mitbih.py.
7. Implement scripts/preprocess_data.py.
8. Run preprocessing for MIT-BIH first; verify output.
9. Implement src/data/ptbxl.py; extend preprocess_data.py.
10. Run preprocessing for PTB-XL.
11. Implement src/data/partition.py.
12. Implement tests. Run `make test`.
13. Implement scripts/sanity_check_data.py.
14. Run `make sanity`.
15. Report.

## STOP after Phase 0+1. Report:

- Contents of INVENTORY.md (paste).
- Disk usage: data/raw/ (existing), data/processed/ (new).
- MIT-BIH: total beats, class distribution, records processed, drops.
- PTB-XL: total records after filtering, superclass distribution.
- All test outcomes (pass/fail list).
- Sanity figures (describe).
- DECISIONS.md entries added.
- Any anomalies.

Do NOT begin Phase 2.
```

---

## PROMPT — Phase 2 (Models + Centralized Baseline)

```
# TASK: Phase 2 — Implement models and centralized baseline training.

## Context

Phase 0+1 already produced the data pipeline. Data is in data/processed/.
Partition logic is in src/data/partition.py. Config inheritance works.

You are implementing models + centralized training. No federated code yet.

## src/models/cnn1d.py

Primary model for the entire study. MUST have BatchNorm layers throughout
(critical for later FedBN/DP-FedBN experiments).

Architecture (works for both MIT-BIH 250-sample and PTB-XL 1000-sample inputs
via AdaptiveAvgPool):

class CNN1D(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.bn_fc = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Input: (B, C, L). If input is (B, L, C), transpose.
        if x.dim() == 3 and x.size(1) != 1:
            x = x.transpose(1, 2)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.gap(x).squeeze(-1)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def __repr__(self):
        total = sum(p.numel() for p in self.parameters())
        return f"CNN1D(num_classes={...}, params={total:,})"

Expected param count: ~30k. This is intentional — small, fast, BN-heavy.

## src/models/resnet1d.py

Optional deeper model. Implement a 4-block 1D-ResNet with channels
[64, 128, 256, 512], BN after each conv, skip connections, stride-2
downsample between blocks. AdaptiveAvgPool1d(1) → Linear(512, num_classes).
~4M params.

Leave this as a nice-to-have ablation. Don't use for main experiments.

## src/evaluation/metrics.py

def compute_metrics(y_true, y_pred, y_prob, class_names) → dict:
    Returns:
      accuracy, precision_macro, recall_macro, f1_macro,
      per_class_f1: list,
      per_class_precision: list,
      per_class_recall: list,
      auc_macro: float (one-vs-rest),
      confusion_matrix: 2D list.

All via sklearn. Handle edge cases (single-class predictions).

## src/training/centralized.py

def train_centralized(config: dict, dataset_name: str) → dict:
    Steps:
    1. set_all_seeds(config['seed']).
    2. Load processed dataset: X, y.
    3. Stratified train/val/test split 70/15/15 (sklearn, seed=42).
    4. Create TensorDataset + DataLoader (num_workers, pin_memory).
    5. Compute class weights: inverse frequency, normalized to mean=1.
    6. Instantiate model (CNN1D), move to cuda.
    7. AdamW optimizer, CrossEntropyLoss(weight=class_weights), no scheduler.
    8. Train for config['epochs'] epochs:
       - Per epoch: full training pass, validation pass, log metrics.
       - Save best checkpoint by val_f1_macro to
         results/checkpoints/{run_id}/best.pt.
    9. Load best checkpoint, evaluate on test set. Compute full metrics dict.
    10. Save results to results/metrics/{run_id}.json:
        {
          "run_id": str,
          "config": <echoed config>,
          "train_history": list of per-epoch dicts,
          "test_metrics": <full metrics>,
          "runtime_seconds": float,
          "git_sha": str,
          "torch_version": str,
          "cuda_version": str,
        }
    11. Save training curves plot to results/figures/{run_id}_training.png:
        two-panel: loss (train/val) and F1-macro (train/val) over epochs.

CLI: scripts/run_experiment.py --config <path> --mode centralized

## configs/centralized_mitbih.yaml

_base_: "base.yaml"
run_id: "centralized_mitbih_cnn1d_seed42"
dataset: "mitbih"
model: "cnn1d"
training:
  epochs: 50
  batch_size: 64
  lr: 0.001

## configs/centralized_ptbxl.yaml

_base_: "base.yaml"
run_id: "centralized_ptbxl_cnn1d_seed42"
dataset: "ptbxl"
model: "cnn1d"
training:
  epochs: 50
  batch_size: 32           # larger samples, more memory
  lr: 0.001

## tests/test_models.py

- test_cnn1d_forward_mitbih_shape: input (4, 1, 250), output (4, 5).
- test_cnn1d_forward_ptbxl_shape: input (4, 1, 1000), output (4, 5).
- test_cnn1d_input_transpose_handling: input (4, 250, 1) works same as (4, 1, 250).
- test_cnn1d_grad_flow: loss.backward() on random batch runs without error.
- test_cnn1d_param_count: total params < 50k.

## Makefile update

train-central:
    .venv/bin/python scripts/run_experiment.py --config $(CONFIG) --mode centralized

Default CONFIG = configs/centralized_mitbih.yaml.

## Execution

1. Implement everything above.
2. Run `make test` — all new tests must pass.
3. Run centralized training on MIT-BIH: `make train-central`.
4. Run centralized training on PTB-XL:
   `make train-central CONFIG=configs/centralized_ptbxl.yaml`.
5. Report results.

## Expected outcomes (success criteria)

- MIT-BIH centralized: test accuracy ≥0.95, F1 macro ≥0.85, AUC macro ≥0.95.
  Training time ≤15 min on 1660 Super.
- PTB-XL centralized: test F1 macro ≥0.70 (PTB-XL is harder, lead-II-only).
  Training time ≤25 min.
- If F1 macro on MIT-BIH is below 0.80 — STOP and debug before proceeding.
  Likely causes: wrong AAMI mapping, class imbalance not handled, data leakage
  in split.

## STOP after Phase 2. Report:

- Test metrics for both datasets (full classification report: per-class P/R/F1,
  confusion matrix).
- Training curves (describe briefly).
- Runtime.
- DECISIONS.md new entries.
- Anomalies.
```

---

## PROMPT — Phase 3 (Federation Core)

```
# TASK: Phase 3 — Implement federated learning core with 4 aggregation strategies.

## Context

Phase 2 produced working models + centralized training. Now implement
federated learning: client class, server, four aggregation strategies
(FedAvg, FedProx, FedBN, FedPerf). No DP yet — Phase 5.

## src/federation/client.py

class FederatedClient:
    def __init__(self, client_id: int, X, y, X_val, y_val,
                 model_fn: Callable[[], nn.Module], config: dict):
        self.client_id = client_id
        self.train_data = (X, y)
        self.val_data = (X_val, y_val)
        self.model_fn = model_fn
        self.config = config
        self.device = torch.device("cuda")

    def train(self, global_params: dict, extras: dict | None = None) → dict:
        """
        Returns:
            {
                "params": state_dict of updated LOCAL model,
                "num_samples": int,
                "train_loss": float (avg over local epochs),
                "val_metrics": dict (from local val set),
            }
        """
        # 1. Instantiate model, load global_params.
        # 2. Build local DataLoader.
        # 3. Compute class weights from local y.
        # 4. Optimizer: AdamW, loss: CrossEntropyLoss(weight=class_weights).
        # 5. For FedProx: add proximal term to loss if extras['proximal_mu'] > 0.
        #    proximal_term = (mu/2) * sum((p - g)^2 for p, g in zip(local, global))
        # 6. Train for config['federation']['local_epochs'].
        # 7. Compute val_metrics on local val.
        # 8. Return dict.

    def evaluate(self, params: dict) → dict:
        # Load params, evaluate on local val set, return metrics.

## src/federation/aggregation.py

Four strategies. All have signature:
    aggregate(client_updates: list[dict], global_model: nn.Module,
              config: dict) → new_global_state_dict

### FedAvg

Weighted average by num_samples:

    total = sum(u['num_samples'] for u in updates)
    new_state = {}
    for key in updates[0]['params']:
        new_state[key] = sum(
            u['params'][key] * u['num_samples'] / total for u in updates
        )

Apply to ALL parameters and buffers.

### FedProx

Aggregation is identical to FedAvg. The difference is in the client: when
config says 'fedprox', pass extras={'proximal_mu': 0.01, 'global_params':
global_params} to client.train(), which adds the proximal term.

### FedBN

Aggregation = FedAvg, but EXCLUDE BatchNorm parameters AND buffers from both
exchange and aggregation. In practice:
- When building global_params to send to clients, only include non-BN keys.
- When clients return params, they only return non-BN keys.
- Aggregation only operates on non-BN keys.
- Each client retains its local BN weights, biases, running_mean, running_var
  across rounds — they NEVER sync.

To identify BN keys robustly:

def is_bn_key(model: nn.Module, key: str) → bool:
    # Walk named_modules, find the module that owns `key`, check isinstance
    # of BatchNorm1d/BatchNorm2d.
    for module_name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            if key.startswith(module_name + ".") or key == module_name:
                return True
    return False

Cache the set of BN keys once at init time per model architecture.

### FedPerf (included as a benchmark baseline, NOT "our method")

Aggregation weighted by client's reported validation F1:

    f1_scores = np.array([u['val_metrics']['f1_macro'] for u in updates])
    tau = config.get('fedperf_tau', 1.0)
    weights = softmax(f1_scores / tau)
    new_state = {}
    for key in updates[0]['params']:
        new_state[key] = sum(u['params'][key] * w
                             for u, w in zip(updates, weights))

Also log `weights` to the round metrics for inspection.

## src/federation/server.py

class FederationServer:
    def __init__(self, model_fn, aggregation: str, config: dict,
                 central_test_set: tuple):
        # Build initial global model, determine BN keys if fedbn,
        # initialize client list lazily.

    def run_round(self, round_idx: int, clients: list) → dict:
        # 1. Extract current global params (filter BN for fedbn).
        # 2. For each client in clients:
        #      extras = self._build_extras(client_update=None)  # see below
        #      update = client.train(global_params, extras)
        # 3. Call aggregate(...).
        # 4. Load aggregated params into global model.
        # 5. Evaluate global model on central_test_set.
        # 6. Return round_metrics dict with:
        #    round, client_losses, aggregation_weights,
        #    global_test_metrics, per_client_val_metrics, wall_clock_s.

    def _build_extras(self, ...):
        # FedProx: {'proximal_mu': mu, 'global_params': current_global_params}.
        # Others: None.

    def run(self, clients, num_rounds) → list[dict]:
        # Loop rounds, call run_round, log per round.
        # Save per-round metrics to results/metrics/{run_id}/rounds.csv.

Important for FedBN central eval: Since the server doesn't have BN params
(they're all local), the "global model" for central evaluation is
ambiguous. For this study:
- Use ONE representative client's BN stats + the aggregated non-BN params
  to evaluate the central test set. Document this clearly.
- ALSO evaluate each client on the central test set using THAT client's
  full local model, and report the mean ± std as "global_test_metrics" for
  FedBN runs. This is honest and avoids the ambiguity.
- Log BOTH metrics in the round CSV (representative client version AND
  per-client average version). Note which one is the "primary" reported
  metric in DECISIONS.md. Recommendation: per-client-average is primary for
  FedBN, since FedBN's whole point is that each client ends up with its own
  model.

## src/training/federated.py

def train_federated(config: dict, dataset_name: str) → dict:
    Steps:
    1. set_all_seeds.
    2. Load processed dataset.
    3. Hold out central test set: stratified 15% of full dataset.
    4. From the remaining 85%, run partition strategy from config:
       config['partition'] = 'iid' | 'label_skew' | 'quantity_skew' |
                             'dirichlet_0.1' | 'dirichlet_0.5' | 'dirichlet_1.0'
       Within each client's data, 80/20 train/val split stratified where possible.
    5. Build FederatedClient objects.
    6. Build FederationServer with chosen aggregation.
    7. Run for config['federation']['rounds'] rounds.
    8. Save all metrics to results/metrics/{run_id}/.
    9. Save final global model checkpoint.
    10. Save partition visualization to results/figures/{run_id}_partition.png.

CLI: scripts/run_experiment.py --config <path> --mode federated

## configs/federated_base.yaml

_base_: "base.yaml"
training:
  batch_size: 64
  lr: 0.001
federation:
  num_clients: 5
  rounds: 50
  local_epochs: 5
  aggregation: "fedavg"      # overridden per experiment
  partition: "iid"           # overridden per experiment
  fedprox_mu: 0.01
  fedperf_tau: 1.0

## configs/federated_iid.yaml

_base_: "federated_base.yaml"
run_id: "fedavg_iid_mitbih_seed42"
dataset: "mitbih"
model: "cnn1d"
federation:
  partition: "iid"
  aggregation: "fedavg"

## tests/test_federation.py

- test_fedavg_identical_params: 3 clients with same params → aggregate = same.
- test_fedavg_weighted: 2 clients with different params and different
  sample counts → result is correct weighted mean.
- test_fedbn_bn_keys_detected: CNN1d has 4 BatchNorm modules; 8 param keys
  (weight + bias × 4) + 12 buffer keys (running_mean, running_var,
  num_batches_tracked × 4) must be flagged as BN.
- test_fedbn_non_bn_aggregated_only: simulate a round, verify that after
  aggregation, each client's BN params differ from each other but their
  Conv/FC params are identical.
- test_fedperf_softmax_weights: clients with F1 [0.9, 0.5, 0.5] produce
  weights where client 0's weight > client 1's.
- test_fedprox_proximal_term: with mu > 0, a client trained for 1 step
  from identical init has lower weight drift than with mu = 0.
- test_one_round_runs: 2 clients, 1 round, CNN1d on random 100-sample data,
  runs without error and loss is finite.

## Makefile update

train-fed:
    .venv/bin/python scripts/run_experiment.py --config $(CONFIG) --mode federated

## Execution

1. Implement everything above.
2. Run `make test` — all tests pass.
3. Run federated IID FedAvg sanity: `make train-fed CONFIG=configs/federated_iid.yaml`.
4. Report results.

## Expected outcomes

- All tests pass.
- FedAvg IID on MIT-BIH reaches test F1 macro within 3% of centralized baseline.
- Per-round logging clean.
- Runtime per round ≤2 min on 1660 Super (total ~100 min for 50 rounds).

## If FedAvg IID is more than 5% behind centralized

STOP and debug. Likely causes:
- Global eval using wrong state dict format (common).
- Local data not sampled correctly (some clients getting duplicates).
- Sync issue (global_params not actually being loaded into clients).

## STOP after Phase 3. Report:

- All test results.
- FedAvg IID test metrics.
- Global F1 convergence curve (describe).
- Per-client train loss distribution at round 50 (describe).
- Comparison table: centralized vs FedAvg IID.
- DECISIONS.md entries (especially the FedBN central eval choice).
```

---

## PROMPT — Phase 4 (Non-IID Sweep, 24 runs)

```
# TASK: Phase 4 — Run the 6×4 non-IID × aggregation experimental sweep.

## Context

Phases 0-3 complete. You have a working federated pipeline with four
aggregation strategies and six partition strategies.

Now run the systematic sweep and generate the benchmark figures.

## configs/sweep_noniid.yaml

_base_: "federated_base.yaml"
dataset: "mitbih"
model: "cnn1d"
federation:
  rounds: 50
  local_epochs: 5
  num_clients: 5

sweep:
  partitions:
    - "iid"
    - "label_skew"
    - "quantity_skew"
    - "dirichlet_0.1"
    - "dirichlet_0.5"
    - "dirichlet_1.0"
  aggregations:
    - "fedavg"
    - "fedprox"
    - "fedbn"
    - "fedperf"

## scripts/run_noniid_sweep.py

Orchestrator:
1. Load configs/sweep_noniid.yaml.
2. For each (partition, aggregation) pair:
   a. Build a run-specific config by merging.
   b. Set run_id = f"{aggregation}_{partition}_mitbih_seed42".
   c. Check if results/metrics/{run_id}/final.json exists. If yes, skip
      (resumability).
   d. Call train_federated(run_config, "mitbih").
3. After all 24 runs complete, build a summary CSV at
   results/metrics/noniid_summary.csv with columns:
   run_id, partition, aggregation, test_accuracy, test_f1_macro,
   test_auc_macro, convergence_round, wall_clock_seconds

Where convergence_round = first round where val_f1_macro reaches ≥95%
of the final val_f1_macro (or None if never).

## src/evaluation/visualization.py

Generate the following figures after the sweep:

### fig_noniid_heatmap.png + .pdf

6×4 heatmap:
- Rows: partition strategies (in the order listed above).
- Columns: aggregation methods (in the order listed above).
- Cell values: test F1 macro (from summary CSV).
- Color scale: viridis or RdYlGn.
- Annotate each cell with the value (2 decimal places).
- Title, axis labels, colorbar.

### fig_noniid_convergence.png + .pdf

6-panel grid (2 rows × 3 cols), one panel per partition strategy.
Each panel: 4 lines (one per aggregation), x-axis = round, y-axis =
global test F1 macro.
Shared y-axis across panels. Legend in one panel only.

### fig_partition_distributions.png + .pdf

6-panel grid, one panel per partition strategy.
Each panel: stacked bar chart of class distribution per client.
Use consistent class color palette across panels.

### fig_degradation_from_iid.png + .pdf

Bar chart:
- X-axis: partition strategies (excluding IID).
- Y-axis: ΔF1 from IID baseline (negative values).
- Grouped bars per aggregation method.

Save all figures to results/figures/. Use:
- plt.rcParams.update({'font.size': 10, 'figure.dpi': 100})
- Save at 300 DPI for PNG, vector for PDF.
- Use seaborn "colorblind" palette.

## Makefile update

sweep-noniid:
    .venv/bin/python scripts/run_noniid_sweep.py

figures-noniid:
    .venv/bin/python -c "from src.evaluation.visualization import generate_noniid_figures; generate_noniid_figures()"

## Execution

1. Implement run_noniid_sweep.py + visualization functions.
2. Run `make sweep-noniid`. This will take ~25-30 hours of compute.
   Left running overnight on days 4-5 is fine.
3. Once complete, run `make figures-noniid`.
4. Report.

## Monitoring

While the sweep runs, print a progress summary every time a run finishes:
    [SWEEP 7/24] fedbn_dirichlet_0.1_mitbih_seed42 done. F1=0.7234, 78min.

Also write a live-updating results/metrics/sweep_progress.csv so it can
be inspected while running.

## STOP after Phase 4. Report:

- Completion status: X/24 runs successful.
- The summary CSV contents.
- The 4 figures (describe each).
- Key scientific observations:
  * Does non-IID hurt? By how much, per strategy?
  * Does FedBN beat FedAvg under non-IID? Where?
  * Does FedProx help vs. FedAvg?
  * Any surprises?
- DECISIONS.md entries.

## IMPORTANT

If FedBN does NOT show gains over FedAvg on any non-IID partition — do NOT
try to fix this by tuning. Honest negative results on some partitions is
expected and acceptable. Just report it. We need to know which partitions
FedBN helps on for the paper's story.
```

---

## PROMPT — Phase 5 (DP-FedBN + DP Sweep, 12 runs)

```
# TASK: Phase 5 — Implement DP-FedBN (the novel formulation) and run DP sweep.

## Context

Phases 0-4 done. You have: full federated pipeline, 4 aggregators, non-IID
sweep completed, figures generated.

Now implement the PAPER'S NOVEL CONTRIBUTION: DP-FedBN, and run the
privacy-utility sweep.

## The novel formulation (read this carefully)

Standard DP-SGD (Abadi et al. 2016, Opacus) is incompatible with BatchNorm:

- DP-SGD needs per-sample gradients.
- BatchNorm's output for sample i depends on ALL other samples in the batch
  (via batch mean and variance).
- Therefore per-sample gradients under BN are ill-defined in the usual sense.
- Opacus's default "fix" is to replace BN with GroupNorm via
  ModuleValidator.fix(). This works mathematically but destroys the FedBN
  mechanism entirely.

Our formulation (DP-FedBN):

- FedBN already keeps BN parameters (weight γ, bias β) AND running statistics
  (running_mean, running_var) LOCAL to each client. They are never shared.
- If BN params never leave the client, they don't need DP protection in the
  federated threat model.
- Therefore: apply DP-SGD ONLY to non-BN parameters (Conv, Linear weights
  and biases). Train BN parameters with a separate, non-DP optimizer.
- The privacy accountant tracks only the DP-protected parameters, which is
  also exactly what the server sees.

This is simple, correct, and the straightforward composition is
underspecified in prior FL+DP+medical literature. It is the paper's
central technical contribution.

## src/federation/dp.py

Implement the two-optimizer wiring:

def setup_dp_training(
    model: nn.Module,
    train_loader: DataLoader,
    target_epsilon: float,
    target_delta: float,
    max_grad_norm: float,
    epochs: int,
    exclude_bn_from_dp: bool = True,
) → tuple[nn.Module, Optimizer, Optimizer, DataLoader, PrivacyEngine]:
    """
    Returns: (wrapped_model, dp_optimizer, nondp_optimizer, dp_loader,
              privacy_engine)

    If exclude_bn_from_dp is True (DP-FedBN mode):
      - dp_optimizer manages only non-BN params, wrapped by Opacus.
      - nondp_optimizer manages BN params, plain AdamW.
    If False (DP-FedAvg mode for comparison):
      - Replace BN with GroupNorm via ModuleValidator.fix(model).
      - Single dp_optimizer over all params. nondp_optimizer = None.
    """

Implementation outline for DP-FedBN mode:

1. Identify BN parameters:
    bn_params = []
    non_bn_params = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_params.extend(module.parameters(recurse=False))
        else:
            non_bn_params.extend(p for p in module.parameters(recurse=False)
                                 if not any(p is bp for bp in bn_params))
    # De-duplicate while preserving order.

2. Temporarily freeze BN params so Opacus doesn't try to hook them:
    for p in bn_params: p.requires_grad = False

3. Build non-DP optimizer for non-BN params:
    opt_dp = torch.optim.AdamW(non_bn_params, lr=config.lr,
                               weight_decay=config.weight_decay)

4. Wrap with Opacus:
    from opacus import PrivacyEngine
    privacy_engine = PrivacyEngine(accountant="rdp")
    model, opt_dp, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=opt_dp,
        data_loader=train_loader,
        epochs=epochs,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        max_grad_norm=max_grad_norm,
    )

5. Unfreeze BN params, build non-DP optimizer for them:
    for p in bn_params: p.requires_grad = True
    opt_nondp = torch.optim.AdamW(bn_params, lr=config.lr, weight_decay=0.0)

6. Return tuple.

Training loop modification (in FederatedClient.train when DP is enabled):

    for epoch in range(local_epochs):
        with BatchMemoryManager(data_loader=train_loader,
                                 max_physical_batch_size=16,
                                 optimizer=opt_dp) as safe_loader:
            for x, y in safe_loader:
                x, y = x.to(device), y.to(device)
                opt_dp.zero_grad()
                opt_nondp.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                opt_dp.step()
                opt_nondp.step()

After training, record actual ε from privacy_engine.get_epsilon(delta).

## src/federation/client.py update

When config['dp']['enabled'] is True:
- Call setup_dp_training with exclude_bn_from_dp = (aggregation == 'fedbn').
- Track and report achieved ε per round.
- total_epochs passed to setup_dp_training should be rounds × local_epochs
  (so noise is calibrated for the full federated run from the start).

## configs/sweep_dp.yaml

_base_: "federated_base.yaml"
dataset: "mitbih"
model: "cnn1d"
federation:
  partition: "dirichlet_0.5"
  rounds: 50
  local_epochs: 5
  num_clients: 5

dp:
  enabled: true
  target_delta: 1.0e-5
  max_grad_norm: 1.0

sweep:
  epsilon_values: [0.5, 1.0, 2.0, 5.0, 10.0, null]  # null = no DP
  aggregations:
    - "fedavg"
    - "fedbn"

## scripts/run_dp_sweep.py

Orchestrator, similar to non-IID sweep. 12 runs total.
- For ε = null: disable DP entirely (run identical to Phase 4 for that cell).
- For ε = value: enable DP with that target.
- run_id pattern: "{aggregation}_dp_eps{epsilon}_dir05_mitbih_seed42"
  (use "inf" for no-DP).
- Save summary to results/metrics/dp_summary.csv: run_id, aggregation,
  target_epsilon, achieved_epsilon, delta, test_f1_macro,
  test_accuracy, wall_clock_seconds.

## src/evaluation/visualization.py additions

### fig_privacy_utility.png + .pdf

Line plot:
- X-axis: target ε (log scale). "no DP" as rightmost tick labeled ∞.
- Y-axis: test F1 macro.
- Two lines: FedAvg vs FedBN.
- Horizontal dashed reference: centralized baseline F1.
- Shaded region ε ≤ 2.0 labeled "Strong DP regime".
- Legend, grid, axis labels.

### fig_privacy_cost_per_round.png + .pdf

One panel per target ε (4 panels at most, pick 0.5, 1.0, 5.0, 10.0):
- X-axis: round.
- Y-axis: cumulative ε spent (from privacy_engine.get_epsilon per round).
- Two lines: FedAvg vs FedBN.

### tables/dp_summary.tex

LaTeX tabular:
Columns: ε target, ε achieved (FedAvg), F1 (FedAvg), ε achieved (FedBN),
F1 (FedBN), ΔF1 (FedBN - FedAvg).

## tests/test_dp_fedbn.py

CRITICAL tests:

- test_bn_param_identification: CNN1d has exactly 4 BN modules; verify the
  param identification returns 8 BN params (4 × 2 for weight+bias) and the
  rest as non-BN.
- test_opacus_wiring_runs: create CNN1d, call setup_dp_training with dummy
  synthetic data, verify 10 training steps run without error.
- test_bn_updates_under_dp: after a few DP-FedBN training steps, BN
  running_mean and running_var must have changed from init (they're buffer
  updates during forward pass, not gradient-based).
- test_dp_privacy_spend: after N steps with configured noise multiplier,
  privacy_engine.get_epsilon(delta=1e-5) returns a value within 10% of target.
- test_dp_gradient_clipping: non-BN parameter gradients after clipping have
  norm ≤ max_grad_norm.

## Makefile update

sweep-dp:
    .venv/bin/python scripts/run_dp_sweep.py

figures-dp:
    .venv/bin/python -c "from src.evaluation.visualization import generate_dp_figures; generate_dp_figures()"

## Execution

1. Implement dp.py + client updates + sweep script + visualization.
2. Run `make test` — especially test_dp_fedbn.py. All must pass.
3. Quick DP smoke: one 2-client, 5-round, ε=3 run on small data subset
   (add a `--smoke-test` flag to run_dp_sweep.py). Must finish without error.
4. Run `make sweep-dp`. ~21 hours compute.
5. Generate figures with `make figures-dp`.
6. Report.

## Expected outcomes

- Monotonic curve: higher ε → higher F1.
- At ε = ∞, results reproduce Phase 4 dirichlet_0.5 cells (within RNG noise).
- DP-FedBN vs DP-FedAvg: FedBN should degrade more slowly as ε decreases.
  This is the paper's key claim.
- F1 at ε = 1.0 should be at most 15% below ε = ∞. If worse, max_grad_norm
  likely needs tuning; note in DECISIONS.md.
- Achieved ε should match target within 5-10%.

## If DP-FedBN does NOT show an advantage over DP-FedAvg

This would be a surprising result. Before calling it negative:
1. Verify BN exclusion is actually happening (instrument: log
   len(bn_params), len(non_bn_params), confirm they sum to total).
2. Verify privacy accountant is only counting non-BN steps (sanity check
   the noise_multiplier and sample_rate are what you expect).
3. Verify DP-FedAvg uses GroupNorm replacement (otherwise it's also
   "privileged" with BN, which is not a fair comparison).

If after these checks the result is still flat — report it honestly as
negative. "Under the tested conditions, DP-FedBN shows parity with
DP-FedAvg+GroupNorm" is still a valid finding. Our formulation's claim
is "correctness" (simple, no BN replacement) more than "strictly better
utility".

## STOP after Phase 5. Report:

- All tests passed.
- DP smoke test passed.
- 12/12 runs completion.
- Summary CSV + privacy-utility figure.
- Key observations:
  * F1 degradation pattern vs ε for both aggregators.
  * Does DP-FedBN beat DP-FedAvg+GroupNorm? Where?
  * Achieved ε calibration accuracy.
- Any Opacus/BN incompatibility issues encountered.
- DECISIONS.md entries.
```

---

## PROMPT — Phase 6 (Cross-Dataset Validation on PTB-XL)

```
# TASK: Phase 6 — Validate main findings on PTB-XL (secondary dataset).

## Context

Phases 0-5 done. Main findings:
- Non-IID impact characterized on MIT-BIH (Phase 4).
- DP-FedBN formulation validated on MIT-BIH (Phase 5).

Now confirm findings generalize to a different dataset structure:
- PTB-XL: record-level classification, 5 superclasses, 10-second windows.

This is NOT joint federation across datasets. It is a separate, self-contained
federation on PTB-XL that replicates the best conditions from MIT-BIH.

## configs/crossdataset_ptbxl.yaml

_base_: "federated_base.yaml"
dataset: "ptbxl"
model: "cnn1d"
training:
  batch_size: 32
federation:
  num_clients: 5
  rounds: 50
  local_epochs: 5
  partition: "dirichlet_0.5"

## Runs (4 total)

1. centralized_ptbxl_cnn1d_seed42 (ALREADY DONE in Phase 2 — skip if exists)
2. fedavg_dir05_ptbxl_seed42 — FedAvg, no DP, dirichlet α=0.5
3. fedbn_dir05_ptbxl_seed42 — FedBN, no DP, dirichlet α=0.5
4. fedbn_dp2_dir05_ptbxl_seed42 — DP-FedBN, ε=2, dirichlet α=0.5

## scripts/run_crossdataset.py

Orchestrator calling train_federated for each of the 3 new runs with
PTB-XL as dataset.

## Visualization

### fig_crossdataset_comparison.png + .pdf

Grouped bar chart:
- X-axis: method (FedAvg, FedBN, DP-FedBN@ε=2).
- Y-axis: test F1 macro.
- Two bars per method: MIT-BIH vs PTB-XL (side by side).
- Centralized baseline as horizontal reference line for each dataset.

Save to results/figures/.

## Makefile update

crossdataset:
    .venv/bin/python scripts/run_crossdataset.py

## Execution

1. Implement script + figure.
2. Run `make crossdataset`. ~4-5 hours compute.
3. Generate figure.
4. Report.

## Expected outcomes

- PTB-XL centralized F1 macro ≥ 0.70.
- FedBN > FedAvg on PTB-XL under dirichlet α=0.5 (directionally consistent
  with MIT-BIH Phase 4 result).
- DP-FedBN at ε=2 within ~10-15% of no-DP FedBN (similar to MIT-BIH
  Phase 5 result).

If either direction contradicts MIT-BIH findings, report it honestly.
"Findings transfer partially" is an acceptable narrative.

## STOP after Phase 6. Report:

- All 4 run results.
- Cross-dataset comparison figure.
- Whether findings generalize (honest assessment).
- DECISIONS.md entries.
```

---

## PROMPT — Phase 7 (Paper Figures + Analysis)

```
# TASK: Phase 7 — Generate all final paper figures + comprehensive tables.

## Context

All experiments complete. Time to produce publication-quality visuals.

## Figures to finalize (update or regenerate from saved results)

1. **Fig 1: System diagram** — federation topology with 5 clients, server,
   DP-FedBN data flow highlighted. Use matplotlib or save as draw.io SVG.
   Include a visual distinction between BN params (stay local, red) vs
   Conv/FC params (aggregated, blue).

2. **Fig 2: Dataset statistics** — two subplots (MIT-BIH, PTB-XL), each a
   bar chart of per-class sample counts.

3. **Fig 3: Partition distributions** — 6-panel grid (already done in Phase 4,
   polish styling).

4. **Fig 4: Non-IID heatmap** — 6×4 (already from Phase 4, polish).

5. **Fig 5: Convergence curves** — 6-panel grid (from Phase 4).

6. **Fig 6: Privacy-utility trade-off** — (from Phase 5).

7. **Fig 7: Cross-dataset comparison** — (from Phase 6).

8. **Fig 8 (optional, for appendix): Per-client F1 distributions** — box
   plots comparing client-level variance across aggregation methods.

## Style requirements

All figures:
- Font: 10pt, LaTeX-compatible (Computer Modern if available, else serif).
- Single-column width: 3.5 in. Double-column: 7.2 in.
- 300 DPI PNG + vector PDF.
- Colorblind-friendly palette (seaborn "colorblind").
- Axis labels with units where applicable.
- Legend inside axes (or as single global legend for multi-panel figs).
- Light grid (alpha=0.3).

Apply globally:
plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

## Tables (LaTeX .tex files in paper/tables/)

1. **Table 1: Dataset statistics** — rows: MIT-BIH, PTB-XL. Cols: #records,
   #samples, #classes, sample rate, window size, class balance.
2. **Table 2: Model architecture** — CNN1D and ResNet1D (if used), params,
   depth, input sizes supported.
3. **Table 3: Non-IID benchmark main results** — 6×4 table (reshape of heatmap).
4. **Table 4: Privacy-utility summary** — (from Phase 5).
5. **Table 5: Cross-dataset validation** — (from Phase 6).

Each table uses `\begin{tabular}` with `booktabs` rules (`\toprule`,
`\midrule`, `\bottomrule`). Column types appropriate (r for numbers, c for
centered text).

## scripts/generate_figures.py

Single entry point that regenerates ALL figures from saved results.
Accepts `--out-dir` arg (default paper/figures/).

## scripts/generate_tables.py

Single entry point that generates all LaTeX tables from saved results
to paper/tables/.

## Makefile update

figures:
    .venv/bin/python scripts/generate_figures.py --out-dir paper/figures
    .venv/bin/python scripts/generate_tables.py --out-dir paper/tables

## Execution

1. Polish visualization.py to match style requirements.
2. Implement generate_tables.py.
3. Run `make figures`.
4. Inspect outputs. Iterate on styling if needed.
5. Report.

## STOP after Phase 7. Report:

- List of all figures and tables generated.
- Any stylistic concerns or rendering issues.
- The main heatmap (fig 4) and privacy-utility (fig 6) — these are the
  paper's money figures.
```

---

## PROMPT — Phase 8 (Advisor Deliverable + Paper Draft)

```
# TASK: Phase 8 — Generate advisor memo, slides, and skeleton LaTeX paper.

## Context

All experiments and figures done. Now produce human-facing outputs.

## docs/advisor_memo.md (~2 pages, English)

Sections:
1. Title + metadata (author, advisor, date, repo URL).
2. Summary paragraph (5 sentences): what this phase of the thesis produced,
   the headline numbers, the novel formulation, and the path to the
   preprint.
3. Method (3 paragraphs): data setup (MIT-BIH + PTB-XL, task descriptions),
   federation setup (5 clients, partition strategies, 4 aggregators),
   DP-FedBN formulation (the novel piece — explain the BN exclusion
   rationale in plain language).
4. Results (inline Table 3 + Fig 4 + Fig 6 + brief interpretation).
5. Limitations and next steps (honest: single seed, no formal venue
   submission yet, GroupNorm-BN-replacement comparison missing, etc.).
6. Timeline for phase 2 (2 more weeks for preprint).

Use the results/metrics/*.csv files as the source of truth for numbers.
Do NOT make up numbers. If a run hasn't completed, leave a placeholder
"[TBD]" and mention it in limitations.

## docs/advisor_slides.md (5-7 slides, markdown)

Sections convertible to beamer via:
    pandoc -t beamer docs/advisor_slides.md -o docs/advisor_slides.pdf

Slides:
1. Title.
2. Problem + gap (KVKK context, FL+ECG crowded, specific gap: DP-BN
   incompatibility handling).
3. Setup (datasets, federation, models, partitions).
4. Method (DP-FedBN diagram — reuse Fig 1).
5. Results (Fig 4 + Fig 6 side by side, 3 bullet-point observations).
6. Limitations + next steps.
7. (Optional appendix) cross-dataset validation.

## paper/main.tex

IEEE-style two-column paper skeleton. Sections:
- Abstract (200 words, fill in the headline).
- I. Introduction (problem, gap, contributions list, paper outline).
- II. Related Work (ECG DL, FL in healthcare, DP-FL, FedBN — use the
  references already in the proposal + 2-3 new ones from the literature
  review in phases 0+1).
- III. Method (formal DP-FedBN definition — include the "BN stays local,
  DP only on Conv/FC" argument with a short pseudocode block).
- IV. Experimental Setup.
- V. Results.
- VI. Discussion.
- VII. Conclusion.

Use \input{tables/*.tex} and \includegraphics for figures. References
via BibTeX in paper/references.bib.

Don't write full prose — leave "[EDIT]" placeholders where human writing
is needed. Structure, figure/table placement, and caption text should be
filled in.

## scripts/generate_deliverables.py

Single script that:
1. Reads all results CSVs and JSONs.
2. Computes the summary numbers needed.
3. Fills in docs/advisor_memo.md from a template at
   docs/templates/advisor_memo_template.md (use {{placeholder}} syntax).
4. Same for docs/advisor_slides.md.
5. Writes paper/abstract.tex with the headline numbers.

## README.md (final version)

- Project overview (2 paragraphs).
- Quick start: `make setup && make data && make train-all && make figures`.
- Directory structure explanation.
- How to reproduce specific figures.
- Citation (BibTeX placeholder).
- License.

## requirements.txt

Run `pip freeze > requirements.txt` in the .venv to pin final versions.

## Makefile update

deliverables:
    .venv/bin/python scripts/generate_deliverables.py
    pandoc -t beamer docs/advisor_slides.md -o docs/advisor_slides.pdf || echo "pandoc missing, skipping slides PDF"

paper-compile:
    cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex

all: setup data train-central train-fed sweep-noniid sweep-dp crossdataset figures deliverables

## Execution

1. Write generate_deliverables.py + templates.
2. Run `make deliverables`.
3. Review advisor_memo.md + slides.
4. Commit everything, push to GitHub (if configured).
5. Report.

## STOP after Phase 8. Report:

- advisor_memo.md + advisor_slides.md (or PDF if pandoc worked).
- paper/main.tex compiles cleanly (or lists any missing inputs).
- README.md finalized.
- requirements.txt frozen.
- DECISIONS.md total entry count.
- Summary of the whole 7-day sprint: total runs, total compute hours,
  headline numbers.
- One paragraph of honest limitations + next steps that the user can
  communicate to the advisor.
```

---

# END OF UNIFIED PLAN

**Total prompts: 8.** Paste them in order. Wait for completion + report after each before pasting the next.

If something breaks mid-phase, debug in chat first, THEN resume with the next phase prompt.
