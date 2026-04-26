# MedicalFL

**DP-FedBN: a BatchNorm-local formulation of differentially private federated learning for ECG classification under non-IID conditions.**

A reproducible single-seed benchmark of four federated-learning aggregation
strategies (FedAvg, FedProx, FedBN, FedPerf) on two ECG datasets — MIT-BIH
Arrhythmia (5-class beat classification) and PTB-XL (binary record-level
Normal/Abnormal) — under six non-IID partitionings, plus a privacy-utility
sweep over three differential-privacy budgets (ε ∈ {1, 3, ∞}) using a
two-optimiser DP-FedBN construction that keeps BatchNorm parameters and
running statistics local while DP-protecting Conv/Linear gradients.

**Status:** Phases 0–6 complete (preprocessing, centralized baselines,
federated core, 24-run non-IID sweep on MIT-BIH, 18-run DP sweep on MIT-BIH,
12-run non-IID + 9-run DP cross-task validation on PTB-XL). Paper draft
(Phase 7+8) pending.

**Author:** Efe Deniz Bağlar · **Advisor:** Prof. Dr. Hasan Bulut
(Ege University, Türkiye)

---

## Headline empirical findings

1. **Eval-regime is task-agnostic.** Across MIT-BIH 5-class beats and PTB-XL
   binary records under matched Dirichlet α=0.1 partitions, FedBN ranks below
   FedAvg on a *pooled central* test set but above FedAvg on *per-client local*
   test sets. The same model, training, and federation — and the choice of
   evaluation regime moves macro F1 by up to 0.72 (label-skew C=2 on MIT-BIH:
   FedBN central F1 = 0.224 vs local F1 = 0.942).

2. **DP-FedBN's advantage is partition-dependent.** Under DP-SGD on MIT-BIH
   label_skew_c2 (the FL-relevant extreme), DP-FedBN preserves a clear local
   advantage over the GroupNorm workaround (+0.17 at ε=3, +0.05 at ε=1). Under
   Dirichlet α=0.1 on either dataset the advantage vanishes (MIT-BIH: ≈ 0 at
   ε=3) or reverses (PTB-XL: −0.234 at ε=3). The mechanism: FedBN's
   BN-stats-not-shared design amplifies DP noise that never gets averaged
   across clients.

3. **Naive DP-FedAvg is structurally impossible on a BN-heavy CNN.** Opacus's
   `ModuleValidator` aborts at wrap-time with `UnsupportedModuleError` for
   every BatchNorm module. Reported as a first-class empirical finding (the
   `REFUSED` rows in the DP tables), not omitted.

See [`docs/STORY.md`](./docs/STORY.md) for the chronological narrative and
[`DECISIONS.md`](./DECISIONS.md) for the technical decision log.

---

## Repository layout

```
MedicalFL/
├── configs/                  # YAML configs (base, federated, sweep matrices)
├── data/
│   ├── raw/                  # gitignored — MIT-BIH WFDB + PTB-XL CSV/WFDB
│   └── processed/            # gitignored — .npy arrays + manifest
├── docs/
│   └── STORY.md              # chronological project narrative
├── results/
│   ├── checkpoints/          # gitignored
│   ├── figures/              # all paper figures (.png + .pdf)
│   ├── logs/                 # per-run logs + crash logs + audit_fedbn.log
│   ├── metrics/              # per-run JSON, per-round CSV, summary CSVs
│   └── tables/               # LaTeX booktabs tables for the paper
├── scripts/
│   ├── preprocess_data.py    # build data/processed/ from data/raw/
│   ├── sanity_check_data.py  # data assertions + sanity figures
│   ├── run_experiment.py     # single centralised or federated run
│   ├── run_noniid_sweep.py   # Phase 4 — 24-run non-IID sweep (MIT-BIH)
│   ├── run_dp_sweep.py       # Phase 5 — 18-run DP sweep (MIT-BIH)
│   ├── run_ptbxl_noniid_sweep.py  # Phase 6A — 12-run non-IID sweep (PTB-XL)
│   ├── run_ptbxl_dp_sweep.py # Phase 6B/6B-v2 — DP sweep (PTB-XL)
│   ├── audit_fedbn.py        # 7-check FedBN implementation audit
│   ├── recover_fedbn_finals.py    # reconstruct synth JSONs from rounds.csv
│   ├── local_test_eval.py    # per-client local-test eval for existing runs
│   ├── build_phase5_figures.py    # Phase 5 figures + LaTeX table
│   └── build_phase6_figures.py    # Phase 6 figures (cross-task summary)
├── src/
│   ├── data/                 # mitbih.py, ptbxl.py, partition.py
│   ├── evaluation/           # metrics.py, visualization.py
│   ├── federation/           # client.py, server.py, aggregation.py, dp.py
│   ├── models/               # cnn1d.py (~437k params, BN-heavy)
│   ├── training/             # centralized.py, federated.py
│   └── utils/                # config.py (YAML inheritance), seeding.py, logging.py
├── tests/                    # pytest suite (33 tests, including DP-FedBN)
├── DECISIONS.md              # running log of non-obvious design choices
├── federated_ecg_master_plan.md   # original 8-phase plan from Day 0
├── Makefile                  # setup / data / sanity / test / clean
├── pyproject.toml            # project metadata + ruff config
└── requirements.txt          # pinned deps (torch 2.2–2.4, opacus 1.5)
```

---

## Quick start

```bash
make setup      # create .venv, install pinned deps, verify CUDA
make data       # preprocess MIT-BIH + PTB-XL into data/processed/
make sanity     # data assertions + sanity figures
make test       # pytest tests/  (33 tests)
```

Raw data is expected to already live at `data/raw/` (MIT-BIH in WFDB format,
PTB-XL in the standard PhysioNet layout). The pipeline never downloads,
deletes, or moves raw files.

After preprocessing you should see ≈ 109 453 MIT-BIH beats and ≈ 21 388 PTB-XL
records in `data/processed/`, with a 189 MB total footprint.

---

## Environment

- **OS / GPU:** Linux (Ubuntu); single-GPU. Phases 0–5 ran on GTX 1660 Super
  (6 GB VRAM); Phase 6 sweeps ran on RTX 4070. CUDA 11.8+ in both setups.
- **Python:** 3.10.12. (The master plan asked for 3.11; we kept 3.10 because
  no `sudo` was available to install a second interpreter, and every pinned
  dependency officially supports 3.10. See `DECISIONS.md` § Phase 0+1.)
- **Determinism:** seed = 42 everywhere; `torch.backends.cudnn.benchmark =
  True` (deterministic kernels are 20× slower on 1660 Super and were
  prohibitive at Phase 2). Same-machine, same-PyTorch reproducibility is
  retained; cross-machine bit-exactness is not. Strict mode available via
  `config["strict_determinism"] = true`.
- **Single seed.** All reported numbers are single-seed (42). Multi-seed
  variance analysis is future work.

---

## Reproducing the experiments

The four sweep scripts are all resumable: a run is skipped iff its
`results/metrics/<run_id>.json` exists. Crashed runs leave a traceback in
`results/logs/<run_id>_crash.log`.

### Phase 2 — Centralised baselines (≈ 7 min)

```bash
.venv/bin/python scripts/run_experiment.py \
    --config configs/centralized_mitbih.yaml --mode centralized
.venv/bin/python scripts/run_experiment.py \
    --config configs/centralized_ptbxl.yaml --mode centralized
```

Reference numbers: MIT-BIH macro F1 = **0.913**, PTB-XL binary macro F1 =
**0.813**.

### Phase 3 — FedAvg IID sanity (≈ 30 min)

```bash
.venv/bin/python scripts/run_experiment.py \
    --config configs/federated_iid_mitbih.yaml --mode federated
```

Reference: best central F1 = **0.9245** at round 45 (within 3 % of
centralised, target met).

### Phase 4 — Non-IID sweep on MIT-BIH (≈ 13 h)

```bash
.venv/bin/python scripts/run_noniid_sweep.py
```

24 runs (6 partitions × 4 aggregations). Outputs:
`results/metrics/sweep_progress.csv`, `noniid_summary.csv`,
`central_vs_local_{dirichlet_a01,label_skew_c2}.csv`,
`results/figures/fig_noniid_*.{png,pdf}`.

### Phase 4 follow-up — Per-client local-test evaluation

```bash
.venv/bin/python scripts/local_test_eval.py
```

Adds a `local_test` block to existing JSONs and writes the central-vs-local
CSVs that drive the eval-regime claim.

### FedBN implementation audit

```bash
.venv/bin/python scripts/audit_fedbn.py
```

Seven independent correctness checks (BN key discovery, aggregate exclusion,
client divergence, set_parameters preservation, IID parity, Li-et-al-2021
feature-shift sanity, diagnostic). Output goes to
`results/logs/audit_fedbn.log`. All checks PASS or AMBIGUOUS; none FAIL.

### Phase 5 — DP sweep on MIT-BIH (≈ 7 h with the speedup config)

```bash
.venv/bin/python scripts/run_dp_sweep.py
```

18 runs (3 methods × 3 ε × 2 partitions). Methods are `dp_fedavg` (naive,
expected to REFUSE at finite ε), `dp_fedavg_groupnorm` (the standard Opacus
workaround), and **`dp_fedbn`** (our two-optimiser construction). The script
runs a 3-method smoke at ε=3 first (skip with `--skip-smoke` if already
verified). Outputs: `dp_summary.csv`, `dp_sweep_progress.csv`,
`results/figures/fig_privacy_utility_*.{png,pdf}`,
`results/tables/phase5_summary_table.tex`.

```bash
.venv/bin/python scripts/build_phase5_figures.py
```

Regenerates the four Phase 5 figures + LaTeX table from the summary CSVs.

### Phase 6A — Non-IID sweep on PTB-XL (≈ 2.4 h)

```bash
.venv/bin/python scripts/run_ptbxl_noniid_sweep.py
```

12 runs (4 aggregations × 3 partitions: iid, label_skew_c1, dirichlet_a01).

### Phase 6B-v2 — DP sweep on PTB-XL dirichlet_a01 (≈ 1.9 h)

```bash
.venv/bin/python scripts/run_ptbxl_dp_sweep.py
```

9 runs per partition (3 methods × 3 ε), of which 2 REFUSE per partition. The
sweep YAML (`configs/sweep_ptbxl_dp.yaml`) currently iterates over both
`label_skew_c1` (kept for completeness — see § why label_skew_c1 was
abandoned in DECISIONS.md) and `dirichlet_a01`. Smoke flag:
`--smoke-only` runs a single 60-second DP-FedBN smoke; `--skip-smoke` skips
it before the full sweep.

```bash
.venv/bin/python scripts/build_phase6_figures.py
```

Regenerates the three Phase 6 figures, including the cross-task summary
that contrasts MIT-BIH and PTB-XL Dirichlet results side by side.

---

## Where to look for what

| You want to read about… | Look in… |
|---|---|
| The full project narrative (chronological) | [`docs/STORY.md`](./docs/STORY.md) |
| Every non-obvious decision and its rationale | [`DECISIONS.md`](./DECISIONS.md) |
| The Day-0 plan and prompts | [`federated_ecg_master_plan.md`](./federated_ecg_master_plan.md) |
| Final numerical tables | `results/metrics/*.csv` |
| Final figures (paper-ready) | `results/figures/*.{png,pdf}` |
| FedBN implementation correctness evidence | `results/logs/audit_fedbn.log` |
| LaTeX tables for the paper | `results/tables/*.tex` |
| Per-run training logs | `results/logs/<run_id>.log` |
| Per-run training trajectories (round-by-round) | `results/metrics/<run_id>_rounds.csv` |

---

## Key results at a glance

### Phase 4 — Non-IID benchmark on MIT-BIH (best central F1, single seed)

| partition            | FedAvg | FedProx | FedBN | FedPerf |
|----------------------|-------:|--------:|------:|--------:|
| IID                  | 0.926  | 0.859   | 0.920 | 0.926   |
| Dir(α=1.0)           | 0.937  | 0.889   | 0.911 | 0.928   |
| Dir(α=0.5)           | 0.926  | 0.903   | 0.850 | 0.928   |
| Quantity skew β=0.5  | 0.922  | 0.884   | 0.912 | 0.913   |
| Dir(α=0.1)           | 0.745  | 0.735   | 0.554 | 0.746   |
| Label skew C=2       | 0.527  | 0.386   | 0.219 | 0.530   |

### Phase 4 follow-up — Eval-regime flip (label_skew_c2)

| aggregation | central F1 | local F1 (mean ± std) | Δ |
|-------------|-----------:|----------------------:|--------:|
| FedAvg | 0.527 | 0.823 ± 0.178 | +0.297 |
| FedBN  | 0.224 | **0.942 ± 0.053** | **+0.718** |

### Phase 5 — DP-FedBN vs DP-GroupNorm (local F1, label_skew_c2)

| ε | DP-FedBN | DP-FedAvg+GroupNorm | gap |
|---|---|---|---|
| ∞ | 0.937 | 0.463 | +0.474 |
| 3 | 0.467 | 0.295 | **+0.172** |
| 1 | 0.344 | 0.295 | +0.049 |

### Phase 6B-v2 — DP cross-task on PTB-XL (local F1, dirichlet_a01)

| ε | DP-FedBN | DP-FedAvg+GroupNorm | gap |
|---|---|---|---|
| ∞ | 0.604 | 0.443 | +0.161 |
| 3 | 0.435 | 0.669 | **−0.234** |
| 1 | 0.526 | 0.715 | −0.189 |

ε calibration tight in both DP sweeps: every finite-ε run achieved ε within
0.5 % of target.

---

## Limitations

- **Single seed.** All numbers are single-seed (42); variance is unquantified.
- **Single architecture.** A BN-heavy CNN1D (~437 k parameters); ResNet1D and
  Transformer ablations are future work.
- **Five clients.** Production federated medical-data systems would have
  dozens to hundreds; client-count effects are not characterised.
- **Simulated heterogeneity.** Non-IID partitions are constructed on a single
  pre-pooled dataset; true cross-institutional heterogeneity is not evaluated.
- **Multi-label PTB-XL deferred.** The five superclasses are collapsed to
  binary because Lead II alone is representationally bottlenecked on the
  5-class task; multi-label PTB-XL is scripted but not run.
- **FedProx un-tuned.** μ = 0.01 is uniformly worse than FedAvg; a μ ∈
  {0.001, 0.005} sweep would likely recover competitive performance.

---

## Datasets and references

- **MIT-BIH Arrhythmia Database.** Moody & Mark, *IEEE EMB Magazine* 2001.
  <https://physionet.org/content/mitdb/1.0.0/>
- **PTB-XL.** Wagner et al., *Scientific Data* 2020.
  <https://www.nature.com/articles/s41597-020-0495-6>
- **FedBN.** Li et al., *ICLR* 2021. <https://arxiv.org/abs/2102.07623>
- **DP-SGD.** Abadi et al., *CCS* 2016. <https://arxiv.org/abs/1607.00133>
- **Opacus.** <https://github.com/pytorch/opacus>; BN+DP workaround
  documented in Issue #472.
- **FedProx.** Li et al., *MLSys* 2020. <https://arxiv.org/abs/1812.06127>
- **Dirichlet partitioning protocol.** Yurochkin et al., *ICML* 2019.

---

## Citation

A paper draft is in preparation. Until it is on arXiv, please cite the
repository directly:

```bibtex
@misc{baglar2026medicalfl,
  title  = {DP-FedBN: A BatchNorm-Local Formulation of Differentially
            Private Federated Learning for ECG Classification under
            Non-IID Conditions},
  author = {Bağlar, Efe Deniz},
  year   = {2026},
  note   = {Bachelor's project, Ege University. Advisor: Prof. Dr. Hasan Bulut.}
}
```
