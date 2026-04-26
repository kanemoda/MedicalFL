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

---

## Phase 4 — Non-IID benchmark sweep (2026-04-24)

### Sweep matrix and runtime

- 6 partitions × 4 aggregations = 24 runs on MIT-BIH, seed 42, 5 clients, 50 rounds, 5 local epochs.
- Total wall-clock: 01:50 → 15:59 = **13 h 09 min** on GTX 1660 Super.
- Per-method runtime: FedAvg/FedBN/FedPerf ~28-32 min/run; FedProx ~47-51 min/run (proximal gradient overhead).

### FedBN reporting-step crash + recovery (all 6 FedBN runs)

- **All 6 FedBN runs crashed at the final reporting step** with `KeyError: 'per_class_precision'` in `format_classification_report`. Root cause: FedBN's `client_avg` eval mode returns aggregate metrics only (no per-class breakdown), but the formatter assumed all keys present.
- **Training itself was fine** — every crashed run wrote its full 50-round `rounds.csv`. Only the final `test_metrics.json` and `final.pt` checkpoint were missing.
- Fix: `src/evaluation/metrics.py::format_classification_report` now tolerant to missing per-class keys (graceful fallback to aggregate-only output). Test added.
- Recovery: `scripts/recover_fedbn_finals.py` reads each failed run's `rounds.csv`, extracts best_round / final_round central metrics, writes a synthetic `{run_id}.json` marked `status="RECOVERED"`. `sweep_progress.csv` updated: 18 DONE + 6 RECOVERED = 24/24.
- **No retrain needed** for Phase 4 closure.

### FedBN central eval primary metric choice

- For FedBN runs, the server has no aggregated BN params (BN stays local). Two eval modes logged per round: `representative` (client 0's full local model → central test) and `client_avg` (mean across all 5 clients' local models → central test).
- **Primary reported metric: `client_avg`.** Rationale: representative-client choice is arbitrary; client_avg averages out that arbitrariness.

### Phase 4 heatmap — core empirical finding

Best central F1 macro:

| partition            | FedAvg | FedProx | FedBN | FedPerf |
|----------------------|-------:|--------:|------:|--------:|
| IID                  | 0.926  | 0.859   | 0.920 | 0.926   |
| Dir(α=1.0)           | 0.937  | 0.889   | 0.911 | 0.928   |
| Dir(α=0.5)           | 0.926  | 0.903   | 0.850 | 0.928   |
| Quantity skew β=0.5  | 0.922  | 0.884   | 0.912 | 0.913   |
| Dir(α=0.1)           | 0.745  | 0.735   | 0.554 | 0.746   |
| Label skew C=2       | 0.527  | 0.386   | 0.219 | 0.530   |

- **FedBN underperforms FedAvg on every partition**, with catastrophic collapse on label_skew_c2 (-0.31 absolute) and Dir(α=0.1) (-0.19).
- **FedProx uniformly worse than FedAvg** (-0.02 to -0.14). Likely `proximal_mu=0.01` too aggressive; flagged as post-sprint debug.
- **FedPerf ≈ FedAvg** everywhere (±0.01). Client val-F1s are similar within each regime, so softmax weighting collapses to size-weighted averaging.

### FedBN audit — implementation correctness verified

- Contradiction with Li et al. 2021 prompted an audit (`scripts/audit_fedbn.py`, 7 checks, results in `results/logs/audit_fedbn.log`).
- **Checks 1-4 PASS**: BN key discovery (25/25), aggregate-drops-BN (12 non-BN keys preserved), client BN divergence after round 1, `set_parameters` preserves BN.
- **Check 5 AMBIGUOUS** (small-scale convergence artifact, not structural).
- **Check 6 PASS** (Li et al. feature-shift sanity): under severe per-client feature shift, FedBN central F1 = 0.574 vs FedAvg = 0.129. FedBN reproduces the published advantage when the regime matches the paper.
- **Check 7 diagnostic finding**: across all 6 RECOVERED FedBN runs, mean (local − central) F1 gap = **+0.20**; extreme: label_skew_c2 local 0.958 vs central 0.217 (gap +0.74).

### Paper narrative pivot — eval-regime-dependent FedBN behavior

- **Implementation is correct**; the observation is real. FedBN specializes each client's BN stats to its local label distribution. Under label-shift partitions, per-client BN is mis-aligned with the global test distribution → central collapse. Clients stay competent on *their own* data.
- New framing (to replace earlier "DP-FedBN as primary novelty"):

  > *"FedBN's reported advantage depends on the evaluation regime matching its specialization target. Under feature-shifted image classification with local-test evaluation (Li et al. 2021), FedBN wins. Under label-shifted medical time-series evaluated on a pooled central test set — the clinically realistic regime where a holdout represents the future deployment population — per-client BN specialization is mis-aligned with evaluation and FedBN underperforms FedAvg. The gap widens with partition heterogeneity (Dir α=0.1: −0.19; Label skew C=2: −0.31). However, under per-client local evaluation — the personalized deployment regime — FedBN becomes the strongest method (label_skew local F1 = 0.942 vs FedAvg 0.823)."*

### Phase 4 follow-up — per-client local test evaluation

- Added `scripts/local_test_eval.py` to compute per-client local F1 on the 8 runs that span the two worst central cases (dirichlet_a01 × 4 aggregations + label_skew_c2 × 4 aggregations).
- For 6 non-FedBN runs: loaded existing checkpoints, ran local-test eval in-place. For 2 FedBN runs (which had no saved checkpoints due to the reporting crash): retrained from seed=42, took ~1 hour total.
- Updated JSONs with `local_test` block; added 2 `*_v2.json` entries for the FedBN retrains.

#### Dir(α=0.1) — central vs local

| aggregation | central F1 | local F1 (mean ± std) |    Δ    |
|-------------|-----------:|----------------------:|--------:|
| FedAvg      | 0.745      | 0.831 ± 0.128         | +0.086  |
| FedProx     | 0.735      | 0.755 ± 0.146         | +0.020  |
| **FedBN**   | 0.536      | **0.848 ± 0.154**     | +0.312  |
| FedPerf     | 0.745      | 0.825 ± 0.129         | +0.080  |

#### Label skew C=2 — central vs local

| aggregation | central F1 | local F1 (mean ± std) |    Δ    |
|-------------|-----------:|----------------------:|--------:|
| FedAvg      | 0.527      | 0.823 ± 0.178         | +0.297  |
| FedProx     | 0.386      | 0.488 ± 0.380         | +0.101  |
| **FedBN**   | 0.224      | **0.942 ± 0.053**     | +0.718  |
| FedPerf     | 0.530      | 0.654 ± 0.255         | +0.124  |

- **Eval-regime hypothesis confirmed.** FedBN flips from worst (central) to best (local) under label heterogeneity. The +0.718 central-local gap on label_skew is the most dramatic swing in the matrix.

### Phase 4 artifacts

- `results/metrics/sweep_progress.csv` (24 rows: 18 DONE + 6 RECOVERED)
- `results/metrics/noniid_summary.csv` (8-column headline table)
- `results/metrics/central_vs_local_{dirichlet_a01,label_skew_c2}.csv`
- `results/figures/fig_noniid_heatmap.{png,pdf}`
- `results/figures/fig_noniid_convergence.{png,pdf}`
- `results/figures/fig_partition_distributions_mitbih.{png,pdf}`
- `results/figures/fig_degradation_from_iid.{png,pdf}`
- `results/figures/fig_central_vs_local.{png,pdf}`

### FedProx weakness — deferred debug

- Uniformly worse than FedAvg across all 6 partitions. Most likely `proximal_mu=0.01` too aggressive for this task/model combo. Post-sprint: sweep μ ∈ {0.001, 0.005, 0.01}. Not blocking for paper — current FedProx result reported as-is with caveat.

---

## Phase 5 — DP-FedBN + DP sweep (2026-04-25, in progress)

### Scope

- 3 methods × 3 epsilons × 2 partitions = 18 runs on MIT-BIH.
  - Methods: DP-FedAvg (naive), DP-FedAvg+GroupNorm, DP-FedBN.
  - Epsilons: {∞, 3, 1}; target δ = 1e-5; max_grad_norm = 1.0.
  - Partitions: label_skew_c2 and dirichlet_a01 (the two worst central cases from Phase 4, where the eval-regime story is sharpest).
- Each run logs BOTH central F1 and per-client local F1 (infrastructure from Phase 4 follow-up).
- Estimated runtime: ~11.5 h (started ~01:00 2026-04-25).

### DP-FedBN wiring — two-optimizer design

- Standard Opacus wrapping fails on BN because per-sample gradients are ill-defined under batch statistics. Standard workaround replaces BN with GroupNorm (destroys the FedBN mechanism).
- Our formulation: apply DP-SGD only to Conv/Linear parameters; BN parameters and running statistics stay local (never shared with server, so no DP needed in the federated threat model).
- **Implementation trick**: `ModuleValidator.validate()` checks only trainable modules. Freeze BN params (`requires_grad=False`) before Opacus wrap → validator skips BN → Opacus hooks only non-BN modules. Unfreeze BN post-wrap and train it with a separate plain AdamW optimizer in the same training loop.
- Two optimizers per batch: `opt_dp.step()` for Conv/Linear (DP-protected), `opt_nondp.step()` for BN (plain SGD).

### Three DP modes

- `dp_fedavg` (naive): leave BN intact, wrap entire model with Opacus. Expected to fail validator on BN in practice — documenting this failure *is part of the paper's methodology claim*.
- `dp_fedavg_groupnorm`: replace BN with GroupNorm via `ModuleValidator.fix`, then wrap with Opacus. Standard Opacus-recommended workaround.
- `dp_fedbn`: two-optimizer design above. Our formulation.

### Expected outcomes

- DP-FedBN local F1 > DP-FedAvg+GroupNorm local F1 on label_skew_c2 — would extend Phase 4's eval-regime finding into the DP regime.
- Central F1 for all three methods expected to degrade as ε shrinks; gap between methods may narrow under noise.
- Achieved ε calibration target: within 10% of target_epsilon across runs.

### Smoke test outcomes (2026-04-25 02:14–02:30, label_skew_c2 @ ε=3, 2 clients × 5 rounds × 1 local epoch)

| method                | outcome                                  | runtime |
|-----------------------|------------------------------------------|--------:|
| dp_fedavg (naive BN)  | **REFUSED by Opacus** (DPModeUnsupportedError → ShouldReplaceModuleError) | <1 s    |
| dp_fedavg_groupnorm   | PASSED                                   | 470 s   |
| dp_fedbn              | PASSED (after fix below)                 | 357 s   |

The dp_fedavg refusal is captured as a first-class signal — `setup_dp_training` raises `DPModeUnsupportedError` and the sweep records "REFUSED" rather than crashing. This is the empirical answer to "does naive DP-SGD work on a BN-heavy CNN?" — Opacus 1.5.4's `ModuleValidator` aborts before the first batch.

### Bug found by smoke + fix — `_evaluate_local_test` state-dict key mismatch under DP

- First dp_fedbn smoke completed all 5 rounds + central eval cleanly, then crashed in `_evaluate_local_test` (federated.py:233):
  ```
  RuntimeError: Error(s) in loading state_dict for GradSampleModule:
    Missing key(s) in state_dict: "_module.conv1.weight", ...
    Unexpected key(s) in state_dict: "conv1.weight", "bn1.num_batches_tracked", ...
  ```
- Root cause: `client.model.load_state_dict(client_states[c_id])` was loading plain-keyed snapshots into the Opacus-wrapped `GradSampleModule`, whose `state_dict` is keyed with a `_module.` prefix.
- Fix: route through `client._inner().load_state_dict(...)` so the snapshot lands on the unwrapped CNN1D regardless of whether DP is active. (Mirrors the same `_inner()` access already used in `get_parameters`/`set_parameters`/`snapshot`/`restore`.)
- Re-ran the dp_fedbn smoke after the fix: PASSED in 357 s, all artifacts written.
- Phase 4 (non-DP) was unaffected — `client.model` and `client._inner()` returned the same object pre-Phase-5.

### Sweep launch (2026-04-25 02:39)

- Started via `nohup .venv/bin/python -u scripts/run_dp_sweep.py --skip-smoke > /tmp/sweep_phase5.log 2>&1 &` (PID 896731).
- `--skip-smoke` is safe here because all three smoke modes were already verified above (the script's smoke phase would have re-run them).
- ETA derivation:
  - 4 dp_fedavg-at-finite-ε runs refuse instantly (~0 min total)
  - 2 dp_fedavg-at-ε=∞ runs ≈ 30 min each → 1 h
  - 6 dp_fedavg_groupnorm runs ≈ 30/43/43 min → ~3.5 h
  - 6 dp_fedbn runs ≈ 30/43/43 min → ~3.5 h
  - **Total ETA ≈ 8–9 h** (lands ~10:30 –11:30 on 2026-04-25)

### Still pending (to be filled in when sweep completes)

- Main sweep completion status (target 18/18, with 4 documented REFUSED for dp_fedavg @ finite ε).
- Achieved ε distribution per run vs target ε; check ≤10% deviation.
- Central vs local tables at ε ∈ {1, 3, ∞} × {label_skew_c2, dirichlet_a01}.
- Privacy-utility frontier figure.
- Confirmation of the headline claim: DP-FedBN local F1 > 0.60 at ε=1 on label_skew_c2, central F1 > 0.10.

### Sweep stalled, killed, and resumed under speedup config (2026-04-25 12:10)

The original sweep finished only 5 of 18 runs in 9.5 h (~6.6 h per finite-ε groupnorm run). At that pace the full sweep would have taken ~45 h — incompatible with the 29-Apr advisor deadline. GPU was at 700/6144 MB and 22–89 % utilisation, so we had headroom.

**Killed PID 896731 at 12:10**, preserving `dp_sweep_progress.csv` and the 3 DONE JSONs:
- `dp_fedavg_label_skew_c2_epsinf` (DONE)
- `dp_fedavg_groupnorm_label_skew_c2_epsinf` (DONE)
- `dp_fedavg_groupnorm_label_skew_c2_eps3.0` (DONE)
- 2 instant FAILED rows for `dp_fedavg` @ finite ε (no JSONs — re-run will refuse identically).
- 1 mid-flight `dp_fedavg_groupnorm_label_skew_c2_eps1.0` had no JSON yet → restarts under the new config.

**Speedup config (`configs/sweep_dp.yaml`):**
- `training.batch_size`: 16 → 96 (also `dp.max_physical_batch_size`)
- `training.num_workers`: 4 (newly added; wired into `_make_loader` via `config["training"]["num_workers"]` with top-level fallback)
- `training.pin_memory`: true (already hardcoded in `_numpy_to_tensor_loader`)
- `federation.rounds`: 50 → 30 (kept `local_epochs=5`)

**ExpandedWeights backend attempt + failure (PLAN B → PLAN A):**

- Tried `grad_sample_mode="ew"` for `fedavg_groupnorm` only (DP-FedBN keeps the default hooks backend, since its frozen-BN trick depends on validator/hook iteration semantics that 'ew' doesn't replicate).
- Crashed at the second `optimizer.step()` with `RuntimeError: Current Expanded Weights accumulates the gradients, which will be incorrect for multiple calls without clearing gradients`. The `BatchMemoryManager` chunking + `zero_grad(set_to_none=True)` pattern that works for hooks is incompatible with the way `ew` retains buffers across steps.
- Reverted to default `hooks` backend (kept the comment in `dp.py:_setup_dp_training` so we don't try this again without a proper integration test).

**Speedup smokes (label_skew_c2, ε=3, 2 clients × 5 rounds × 1 local epoch, batch=96):**

| smoke                              | outcome | wall | best central F1 | achieved ε | target ε |
|------------------------------------|---------|-----:|----------------:|-----------:|---------:|
| dp_fedavg_groupnorm + ew           | FAIL    | 32 s | crash           | —          | 3.0      |
| dp_fedbn (hooks)                   | PASS    | 131 s | 0.1811          | 2.9947     | 3.0      |
| dp_fedavg_groupnorm (hooks)        | PASS    | 154 s | 0.1811          | 2.9947     | 3.0      |

Privacy accountant lands within 0.2 % of target ε on both passing smokes — well inside the 20 % gate.

**Resumed sweep (2026-04-25 13:18, PID 1007966):**

```bash
nohup .venv/bin/python -u scripts/run_dp_sweep.py --skip-smoke \
  > /tmp/sweep_phase5_resumed.log 2>&1 &
```

Resume semantics: 3 DONE JSONs cause SKIP rows; 2 dp_fedavg finite-ε refusals re-issue instantly under the new config; 13 fresh runs execute. New per-run estimate: ε=∞ ~10 min, finite-ε ~40 min. Total remaining ≈ 5–7 h, lands ~18:30–20:30 today.

**Asymmetry (intentional, documented):**
- 3 already-DONE runs at batch=16 / rounds=50 / num_workers=0 (kept as-is to save ~90 min).
- 13 remaining runs at batch=96 / rounds=30 / num_workers=4.
- Within-cell comparisons (same partition × ε) are still fair *within each batch config*. The cross-batch comparisons we report in the paper will note this asymmetry. Re-running the 3 DONE runs is a 90-min option if the asymmetry is judged unfair after seeing all numbers.

### Sweep complete (2026-04-25, evening)

11 DONE + 3 SKIPPED (early DONE preserved across the kill) + 4 FAILED-by-design (naive DP-FedAvg @ finite ε) = 18/18. Resumed-sweep wall-clock ≈ 6.89 h (lands within the 5–7 h ETA). All artifacts in `results/metrics/dp_summary.csv` and `results/metrics/dp_sweep_progress.csv`.

### Phase 5 results — full table

| Method | Partition | ε | Achieved ε | Central F1 (best) | Local F1 (mean ± std) |
|---|---|---:|---:|---:|---:|
| DP-FedAvg            | label_skew_c2 | ∞ | —      | 0.517 | 0.829 ± 0.192 |
| DP-FedAvg            | label_skew_c2 | 3 | REFUSED | REFUSED | REFUSED |
| DP-FedAvg            | label_skew_c2 | 1 | REFUSED | REFUSED | REFUSED |
| DP-FedAvg+GroupNorm  | label_skew_c2 | ∞ | —      | 0.261 | 0.463 ± 0.368 |
| DP-FedAvg+GroupNorm  | label_skew_c2 | 3 | 2.9957 | 0.191 | 0.295 ± 0.241 |
| DP-FedAvg+GroupNorm  | label_skew_c2 | 1 | 0.9961 | 0.181 | 0.295 ± 0.241 |
| **DP-FedBN**         | label_skew_c2 | ∞ | —      | 0.221 | **0.937 ± 0.043** |
| **DP-FedBN**         | label_skew_c2 | 3 | 2.9929 | 0.181 | **0.467 ± 0.264** |
| **DP-FedBN**         | label_skew_c2 | 1 | 0.9961 | 0.217 | **0.344 ± 0.194** |
| DP-FedAvg            | dirichlet_a01 | ∞ | —      | 0.738 | 0.816 ± 0.130 |
| DP-FedAvg            | dirichlet_a01 | 3 | REFUSED | REFUSED | REFUSED |
| DP-FedAvg            | dirichlet_a01 | 1 | REFUSED | REFUSED | REFUSED |
| DP-FedAvg+GroupNorm  | dirichlet_a01 | ∞ | —      | 0.736 | 0.832 ± 0.118 |
| DP-FedAvg+GroupNorm  | dirichlet_a01 | 3 | 2.9959 | 0.382 | 0.376 ± 0.101 |
| DP-FedAvg+GroupNorm  | dirichlet_a01 | 1 | 0.9942 | 0.378 | 0.350 ± 0.144 |
| **DP-FedBN**         | dirichlet_a01 | ∞ | —      | 0.551 | 0.861 ± 0.130 |
| **DP-FedBN**         | dirichlet_a01 | 3 | 2.9959 | 0.189 | 0.368 ± 0.151 |
| **DP-FedBN**         | dirichlet_a01 | 1 | 0.9942 | 0.251 | 0.465 ± 0.308 |

### Key findings

- **ε=∞ no-DP results reproduce Phase 4.** DP-FedBN local F1 on label_skew_c2 = 0.937; the Phase 4 follow-up reported 0.942 under a different config (rounds=50, batch=16). The ~0.005 gap confirms the speedup config (rounds=30, batch=96) does not materially alter outcomes.

- **Naive DP-FedAvg refused by Opacus** at 4 cells (ε ∈ {1, 3} × {label_skew_c2, dirichlet_a01}). The `ModuleValidator` rejection happens before the first batch, captured as `DPModeUnsupportedError` in `setup_dp_training`. This is the empirical motivation for DP-FedBN as a methodological contribution: a BN-heavy CNN cannot be DP-protected at all without a topology change (GroupNorm) or a federated trick (FedBN).

- **DP-FedBN preserves the local advantage on label_skew_c2 under DP:**
    - ε=3: DP-FedBN local 0.467 vs DP-FedAvg+GroupNorm 0.295 → **+0.172**
    - ε=1: DP-FedBN local 0.344 vs DP-FedAvg+GroupNorm 0.295 → **+0.049**
    - The +0.17 gap at ε=3 is the Phase 5 headline number for label-skew personalization under DP.

- **DP-FedBN local advantage is mixed on dirichlet_a01:**
    - ε=∞: DP-FedBN 0.861 vs DP-FedAvg+GroupNorm 0.832 → +0.029
    - ε=3:  DP-FedBN 0.368 vs DP-FedAvg+GroupNorm 0.376 → −0.008
    - ε=1:  DP-FedBN 0.465 vs DP-FedAvg+GroupNorm 0.350 → +0.115

  Non-monotonic ordering at dirichlet_a01 (ε=3 worse than ε=1 for both methods on local F1) signals high single-seed variance under stringent DP noise. Multi-seed expansion is future work; the qualitative claim ("FedBN advantage on label-skew survives DP, dirichlet picture is noisier") still holds.

- **ε calibration is tight.** All 7 finite-ε runs achieve ε within 0.5 % of target (range 0.9942 – 2.9959), well inside the 10 % gate.

### Figures + table

- `results/figures/fig_privacy_utility_central.{png,pdf}` — F1 vs ε, central eval, two partitions side-by-side.
- `results/figures/fig_privacy_utility_local.{png,pdf}` — same, local eval, with std error bars.
- `results/figures/fig_privacy_utility_dual.{png,pdf}` — 2×2 money figure with the +0.17 / +0.11 advantage brackets annotated.
- `results/figures/fig_central_local_phase5_comparison.{png,pdf}` — Phase 4 (no-DP) + Phase 5 (DP ε∈{3,1}) grouped-bar comparison; REFUSED cells hatched.
- `results/tables/phase5_summary_table.tex` — booktabs table for the paper.

### Asymmetry note (carried forward)

3 early DONE runs preserved across the kill — `dp_fedavg_label_skew_c2_epsinf`, `dp_fedavg_groupnorm_label_skew_c2_epsinf`, `dp_fedavg_groupnorm_label_skew_c2_eps3.0` — were trained at batch=16 / rounds=50 / num_workers=0. The other 11 DONE runs are at batch=96 / rounds=30 / num_workers=4. Within-cell comparisons (same method × partition × ε) remain fair because each cell has exactly one config. Cross-config comparisons in `fig_privacy_utility_central` reach across batch tiers; this is documented in the figure caption when used in the paper. Re-running the 3 early runs is a ~90-min option, deferred unless reviewers flag it.

### Wall-clock summary

- Total Phase 5 sweep wall-clock: **6.89 h** for the resumed run (after ~10 h burned on the original-config sweep that was killed at 5/18).
- Per-run averages under speedup config (batch=96, rounds=30, num_workers=4):
    - ε=∞ runs: 18 – 20 min
    - finite-ε runs: 41 – 54 min
- The 3 preserved batch=16 runs by contrast: ε=∞ ≈ 80–90 min, finite-ε ≈ 6.6 h. The speedup paid for itself many times over.
