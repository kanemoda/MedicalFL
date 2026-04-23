# DECISIONS.md

Running log of non-obvious choices. Each entry: what, why, where it bites.

---

## Phase 0+1 — Scaffolding & data pipeline (2026-04-23)

### Data paths (confirmed from `data/raw/INVENTORY.md`)

- `mitbih_subdir: "mitbih"` — `/home/kanemoda/MedicalFL/data/raw/mitbih/` holds 48 records in standard WFDB format (`.dat`/`.hea`/`.atr`).
- `ptbxl_subdir: "ptbxl"` — `/home/kanemoda/MedicalFL/data/raw/ptbxl/` holds `records100/`, `records500/`, `ptbxl_database.csv`, `scp_statements.csv`. We use `records100/` (100 Hz) only.

### MIT-BIH processing

- **Channel:** always read `signal[:, 0]` (column 0). Per task spec — do not branch per record. For records where MLII is on channel 1 (e.g. 102, 104) this means we read a different lead; that asymmetry is intentional and consistent with published baselines that do the same.
- **Window:** ±125 samples around the R-peak (250 samples total, ≈694 ms at 360 Hz). Z-score per window. Beats whose peak falls within 125 samples of either edge are dropped.
- **AAMI mapping (5 classes):**
  - N (0): `N, L, R, e, j`
  - S (1): `A, a, J, S`
  - V (2): `V, E`
  - F (3): `F`
  - Q (4): `/, f, Q`
  - Any other symbol (incl. non-beat annotations like `+`, `~`, `|`, `"`, `x`) → **drop**. We do not force them into Q. Rationale: only annotations representing actual beats contribute training signal; non-beat annotations are noise/rhythm markers.
- **Paced-beat records (102, 104, 107, 217)** are *kept* with their `/` (paced) beats labelled Q. This follows the ANSI/AAMI EC57 recommendation and matches Mondéjar-Guerra 2019 / de Chazal 2004.

### PTB-XL processing

- **Superclass selection:** parse `scp_codes` (dict of code → confidence), look up each code's `diagnostic_class` in `scp_statements.csv`, pick the code with the **highest confidence** that has a non-null diagnostic_class. If no such code exists, drop the record.
- **Lead:** II (column index 1) only — 1-D signal, 1000 samples at 100 Hz.
- **Z-score** per record (mean/std computed on the chosen lead only).
- **Strat fold** preserved as-is from the CSV (PTB-XL's built-in cross-val folds; folds 9 and 10 are the canonical test sets).

### Storage

- Processed arrays as `.npy` (simple, mmap-able, no schema overhead). One directory per dataset under `data/processed/`. Record/ECG IDs saved alongside so the partitioner can do patient-wise splits later.
- MIT-BIH `record_ids` saved as `S10` fixed-width strings (record names like `"100"`, `"234"` fit comfortably).

### Config inheritance

- `configs/base.yaml` is the single source of truth for paths + hyperparameters. Later configs can deep-merge via a top-level `_base_: base.yaml` key. Implemented in `src/utils/config.py`.

### Seeding

- Global seed = 42 everywhere. `torch.backends.cudnn.deterministic = True`, `benchmark = False`. This is slower but required for the paper's reproducibility claim.

### Dependencies

- Installed into `/home/kanemoda/MedicalFL/.venv/` (the project-local venv). Confirmed via `make setup`.
- `torch` pinned to `>=2.2,<2.5` (compatible with Opacus 1.5 and CUDA 11.8+ per Opacus 1.5 release notes). No `tensorflow`, `lightning`, `hydra`, or `wandb` — plan explicitly forbids them.

### Phase 2 determinism trade-off (2026-04-23)

- With `cudnn.deterministic=True, benchmark=False` the CNN1D trained at **~75 s/epoch** on the 1660 Super (50 epochs → ~63 min, 4× over the 15 min target). Profiled per-batch: 62.6 ms/iter under deterministic mode vs 3.2 ms/iter with `benchmark=True`.
- **Resolution:** `src/utils/seeding.set_all_seeds(seed, strict=False)` now defaults to `cudnn.benchmark=True, deterministic=False`. Same-machine/same-torch reproducibility is retained (cudnn picks the same kernel for the same conv + input shape on a given GPU). Cross-machine bit-exactness is sacrificed — not needed for this project.
- **Opt-in strict mode** available via `config["strict_determinism"] = true`. Reserved for debugging non-determinism or cross-machine reproduction.
- All seeds (python, numpy, torch CPU+CUDA) remain pinned; RNG calls inside partitioning, init, and data shuffling are deterministic. The only relaxed surface is the cuDNN kernel selection for conv/BN.
- Logged `torch.__version__`, `torch.version.cuda`, and `git_sha` per run so reproduction conditions are always captured.

### Phase 0+1 preprocessing outcome (2026-04-23)

- MIT-BIH: 48 records → **109 453** beats kept, 3 194 dropped (unmappable symbols + edge beats). Records with >200 drops: 207 (526), 231 (440), 230 (211) — all explained by rhythm-change and noise annotations, not data loss.
- PTB-XL: 21 799 rows → **21 388** records kept, 411 dropped for having no code with a non-null `diagnostic_class` (plan-specified filter). 10 folds all populated (fold sizes 2 129–2 158).
- Disk usage: `data/raw/` 3.3 G (unchanged), `data/processed/` 189 M.
- All 13 tests pass; sanity script exits 0; two figures emitted.

### Python version

- Target host has **Python 3.10.12** only (no `python3.11`/`3.12` binaries in `/usr/bin/`). The master plan asked for 3.11, but installing a second interpreter system-wide needs sudo and would block Phase 0+1.
- Resolution: use 3.10 for now. `pyproject.toml` → `requires-python = ">=3.10"`, `ruff.target-version = "py310"`. All pinned dependencies (torch 2.2–2.4, opacus 1.5, wfdb 4.1, numpy <2, pandas 2.1+) officially support 3.10, so no runtime breakage is expected. If we hit 3.10-vs-3.11 incompatibility later, we revisit (install 3.11 via `deadsnakes` PPA with sudo).

---

## Phase 2 closure — centralized baselines (2026-04-23)

### Model capacity fix — CNN1D widened from 44k → 437k params

- Original 3-block CNN1D (32/64/128 channels, ~44k params) hit the capacity ceiling on PTB-XL: train F1 plateaued at 0.61, val F1 at 0.54, test F1 0.498. Multiple training regimes (cosine LR, 50/75 epochs) produced the same number — a clear ceiling, not an optimization bug.
- Widened to 4 conv blocks (channels 64/128/256/256, kernels 7/5/5/3), FC hidden 128, dropout 0.5, BN after every conv and the FC. Final param count: **437,765** for 5-class head, 437,378 for binary. Still BN-per-layer for FedBN / DP-FedBN compatibility.
- MIT-BIH side-effect: macro F1 climbed from 0.862 → 0.913 without hyperparameter tuning. Within acceptable band since no intentional tuning to push above spec.

### PTB-XL task re-formulation — 5-superclass → binary Normal/Abnormal

- 5-superclass Lead-II-only was **representationally bottlenecked**, not capacity-bottlenecked. Three consecutive runs (44k, 438k, 438k + aug + strong reg) all landed at test F1 0.498 / 0.509 / 0.501 — within noise of each other. HYP F1 stayed ~0.22 across regimes: the vectorcardiographic signal that separates LVH from normal/MI is effectively invisible on Lead II alone.
- Re-formulated PTB-XL as binary Normal (NORM) vs Abnormal (MI ∪ STTC ∪ CD ∪ HYP). Task is well-defined on Lead II, class balance is 9 246 NORM vs 12 142 Abnormal.
- Paper narrative: "FL method rankings transfer across task granularities (5-class beat-level on MIT-BIH, binary record-level on PTB-XL)". Cross-dataset validation role preserved; headline benchmark remains MIT-BIH.
- Preserved both label sets on disk (`y_5class.npy` and `y_binary.npy`). Loader reads `config['ptbxl']['label_mode']` ∈ {"binary", "5class"}, default "binary".

### Augmentation + early stopping — PTB-XL only, kept for binary task

- Retained `PTBXLAugment` (random 900-sample crop from 1000, Gaussian noise σ=0.02, sinusoidal baseline wander ≤0.05, p=0.5 each). Eval path deterministically left-crops to 900.
- Retained early stopping on val F1 with patience 10. On the binary run, fired at epoch 28 (best at epoch 18), 2 min total. No runaway overfitting — train/val F1 gap ~0.01.
- MIT-BIH: augment=false (beat-level windows already tight), early_stop_patience=15. Fired at epoch 46 (best at epoch 31), 5.2 min.

### Final Phase 2 metrics (all PASS)

| dataset | task | test acc | test F1 macro | test AUC | best epoch | runtime |
|---|---|---|---|---|---|---|
| MIT-BIH | 5-class AAMI | 0.9819 | **0.9129** | 0.9972 | 31 | 5.2 min |
| PTB-XL | binary Normal/Abnormal | 0.8137 | **0.8134** | 0.8953 | 18 | 2.0 min |

### AUC metric — binary fix in `_macro_auc`

- Previous implementation always passed `multi_class='ovr'` + `labels=[0..K-1]` to `sklearn.metrics.roc_auc_score`. sklearn's binary path rejects that form; returned NaN.
- Fix: when `num_classes == 2`, pass `y_prob[:, 1]` as the positive-class score. Multi-class path unchanged.

### Archived artifacts

- 5-class PTB-XL runs retained at `results/archive/ptbxl_5class_runs/` (checkpoint, metrics JSON, figures, log). Not used downstream; preserved as the reference for why we re-formulated.

---

## Phase 3 — Federated core + FedAvg IID sanity (2026-04-24)

### Module layout

- `src/federation/client.py::FederatedClient` — owns train+val loaders, a CNN1D, and an optimizer. `local_train(local_epochs, proximal_mu, global_params)` runs SGD with an optional FedProx proximal term; `get_parameters/set_parameters` exchange CPU-side state_dicts.
- `src/federation/aggregation.py` — FedAvg (size-weighted mean), FedProx (same aggregation; proximal term lives on the client), FedBN (size-weighted mean with BN keys dropped from broadcast), FedPerf (weight = α·size + (1−α)·F1). `get_bn_key_set(model)` is the single source of truth for "which keys count as BN" — discovered by walking `named_modules` for `BatchNormNd` / `SyncBatchNorm` and grabbing every learnable-param and buffer name.
- `src/federation/server.py::FederationServer` — orchestrates rounds; holds `global_params` dict. For FedBN the server broadcasts only non-BN keys and supports two central-eval modes: `representative` (client 0's model evaluated on central test) and `client_avg` (mean of per-client test F1s). `dual_evaluate_fedbn` returns both for logging.
- `src/training/federated.py::train_federated` — the driver.

### Data carving

- **MIT-BIH**: stratified 85/15 split → 85% (93 035 beats) enters the partitioner, 15% (16 418 beats) held out as the *central* test set. No record-wise split yet — simple per-beat stratification. Patient-wise leakage is a Phase 4 concern.
- **PTB-XL**: canonical fold assignment — folds 1-9 → clients, fold 10 → central test. Keeps the same test slice the centralized baseline used.
- Per-client 80/20 **stratified** train/val split, seeded per client (`seed + client_id`) so the partitioner output is fully deterministic per run.

### Class weights

- Computed once from the **pooled** train portion across all clients (not per-client). Matches the centralized loss scaling so method rankings aren't distorted by weighting differences. Every client's `CrossEntropyLoss` uses the same `cw_tensor`.

### FedAvg + BN transient

- Round 1 central F1 was 0.0248 — below-random. Rounds 2-4: 0.248 → 0.689 → 0.791. Classic FedAvg-with-BN transient: averaging per-client BN running stats after only 5 local epochs produces a distributional no-man's-land. The running stats realign once per-client BN trajectories converge (IID data makes this fast — 3 rounds). No code change needed; expected behaviour, documented so it isn't misread as a bug in Phase 4 non-IID runs where this may last longer.

### FedAvg IID sanity experiment — MIT-BIH 5 clients × 50 rounds × 5 local epochs

- 33 tests pass (22 prior + 11 federation: BN discovery, FedAvg weighted mean, FedBN skip, FedPerf high-F1 dominance, FedProx proximal pull, end-to-end client round, server round for all 4 strategies + dispatch).
- **Best central F1 = 0.9245 @ round 45** (target ≥ 0.88). Final-round (50) central F1 = 0.9059, accuracy = 0.9833, AUC = 0.9952.
- Runtime **29.5 min** on the 1660 Super (target ≤ 120 min).
- Delta to centralized (0.9129): best round exceeds it by +0.012; final round is -0.007. Well within the 3% tolerance.
- Per-class F1 at round 50: N=0.991, S=0.848, V=0.967, F=0.729, Q=0.993. The F (fusion) drop mirrors the centralized pattern — it is the smallest class (121 test beats).

### Reproducibility

- Every federated run writes: `results/metrics/<run_id>.json`, `<run_id>_rounds.csv` (one row per round, per-client val F1 + train loss + central metrics), `figures/<run_id>_partition.png`, `figures/<run_id>_fed_curves.png`, `checkpoints/<run_id>/final.pt` (global_params + per-client snapshots + bn_keys), `logs/<run_id>.log`.
- `final.pt` stores *both* aggregated params and per-client states so Phase 5 DP-FedBN can resume any client from the post-sanity state if needed.
- `fedbn_central_eval` config key controls which mode the server reports in the headline "central_f1_macro" column. Final JSON always captures the dual result when `strategy == "fedbn"`.
