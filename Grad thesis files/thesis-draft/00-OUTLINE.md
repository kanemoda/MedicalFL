<!-- PLANNING DOCUMENT — not part of the thesis body. This file fixes the section
     structure, the standardized terminology, the verified canonical numbers, and
     the figure/table inventory so that all chapters stay mutually consistent. The
     human author should NOT paste this into the template; it is scaffolding for the
     draft. -->

# Thesis Draft — Outline, Glossary, and Source-of-Truth Ledger

**Working title:** *When Does Federated BatchNorm Help under Differential Privacy? An Eval-Regime Characterization of DP-FedBN on Medical ECG Classification*

**Author:** Efe Deniz Bağlar — Department of Computer Engineering, Ege University
**Supervisor:** Prof. Dr. Hasan Bulut

**Source paper:** *When Does Federated BatchNorm Help under Differential Privacy? An Eval-Regime Characterization on Medical ECG Classification* (9-page manuscript, repo root). The paper and the codebase are the single source of truth. Where this outline and the paper disagree on any specific, the paper wins.

---

## 1. The non-negotiable contribution statement (reuse VERBATIM)

The following sentence is the calibrated contribution claim. It must appear, **in substance and strength identical**, in the Abstract, Introduction, Methodology, and Conclusion. No version may claim more than this.

> To our knowledge, this is the **first record-level DP-SGD composition of FedBN's BatchNorm-locality with explicit treatment of the Opacus engineering obstacle**, together with the first systematic medical empirical characterization of that composition across heterogeneity regimes and privacy budgets.

Three contributions (lift from the paper's Introduction, two empirical + one methodological):

1. **Methodological** — a record-level DP-SGD construction of DP-FedBN with engineering documentation: a dual-optimizer construction compatible with off-the-shelf Opacus 1.5 that passes Opacus's `ModuleValidator` without sacrificing FedBN's mechanism.
2. **Empirical 1** — the *eval-regime flip* is task-agnostic: FedBN ranks worst on pooled-central evaluation and best on per-client local evaluation, on both datasets; the central-vs-local gap reaches +0.72 on MIT-BIH label-skew (C=2).
3. **Empirical 2** — DP-FedBN's local advantage is *partition-dependent*: it preserves a +0.17 local-F1 edge on MIT-BIH label-skew at ε=3 but loses to GroupNorm under Dirichlet skew on both datasets (−0.008 MIT-BIH, −0.234 PTB-XL, at ε=3).

### ADCOL framing (the calibration that must NOT drift)

- The term "DP-FedBN" already appears in **ADCOL [1] (Li, He & Song, ICML 2023)**, where Appendix B.11 / Table 17 implements it as a **one-paragraph, party-level DP baseline** — clipping and adding **Laplace** noise to *communicated/aggregated model updates* (the client-level DP recipe of Geyer et al. [4]) — evaluated on five digit datasets at ε ∈ {2, 5, 10}.
- The contribution here is **NOT** the invention of the DP-FedBN concept and **NOT** the invention of the dual-optimizer idea as a concept. It is the **record-level** DP-SGD *construction* (per-sample gradient clipping, strictly stronger than party-level Laplace), the documented Opacus engineering, and the medical multi-partition / multi-budget evaluation.
- Three concrete differences from ADCOL (state these, do not embellish): (i) **privacy granularity** — record-level DP-SGD vs party-level Laplace; (ii) **engineering** — documenting the `ModuleValidator`-rejects-`BatchNorm` obstacle and packaging the freeze-then-unfreeze workaround whose discussion lives only in Opacus GitHub issue #472 (2022); (iii) **empirical scope** — two medical tasks, six non-IID regimes, three budgets (~40 configurations) vs a single label-balanced digit setting.
- **Banned words** except for the one precise claim above: "novel," "groundbreaking," "first-ever," "revolutionary." Keep every claim measured and defensible. If a sentence makes the contribution sound bigger than the paper states, revert to the paper's language.

---

## 2. Standardized terminology (use consistently everywhere)

| Concept | Standard term (use this) | Avoid / notes |
|---|---|---|
| Participating node | **client** (technical); **site / institution / hospital** (deployment metaphor) | Pick "client" for algorithm/threat-model text; "site"/"institution" only in clinical-motivation prose. K clients indexed k = 1..K. |
| The method | **DP-FedBN** | not "DP FedBN", not "DPFedBN" |
| DP optimizer | **DP-SGD** | hyphenated, caps |
| Privacy granularity | **record-level DP** vs **party-level DP** (a.k.a. client-level) | the central ADCOL distinction |
| Guarantee | **(ε, δ)-DP** | ε = privacy budget, δ = failure prob |
| GroupNorm baseline | **DP-FedAvg+GroupNorm** (prose); **DP-FedAvg+GN** (tables only) | the standard Opacus workaround |
| Naive baseline | **DP-FedAvg (raw BN)** | the one Opacus refuses at finite ε |
| Aggregation baselines | **FedAvg, FedProx, FedBN, FedPerf** | FedPerf = contextualizing baseline, NOT a contribution |
| Normalization | **BatchNorm (BN)**, **GroupNorm (GN)** | |
| Heterogeneity | **non-IID** | hyphenated |
| The two evaluations | **central (macro) F1** vs **local (macro) F1** | central = pooled holdout; local = per-client held-out slice |
| The finding | **eval-regime flip** | |
| Primary metric | **macro F1** | preferred over accuracy (class imbalance) |
| Partitions | **IID, quantity skew (β=0.5), Dirichlet α∈{1.0,0.5,0.1}, label skew (C=2 / C=1)** | |
| DP knobs | **ε** budget, **δ**=10⁻⁵, **σ** noise multiplier, **C** clip bound (=1.0) | |
| Library internals | **Opacus**, `PrivacyEngine`, `ModuleValidator`, `GradSampleModule`, `BatchMemoryManager`, `make_private_with_epsilon` | monospace for API names |
| Datasets | **MIT-BIH** (5-class beats), **PTB-XL** (binary records) | |
| Standard | **AAMI EC57** | for MIT-BIH 5-class mapping |
| Prior work label | **ADCOL [1]** | the ICML 2023 paper that named DP-FedBN |

Symbols: $\theta^{\overline{\mathrm{BN}}}$ = non-BN (Conv/Linear) parameters that cross the federation boundary; $\theta^{\mathrm{BN}}$ = BN scale–shift $(\gamma,\beta)$ + running buffers $(\mu,\sigma^2)$ that stay client-local. $\mathcal{D}_k$ = client $k$'s local dataset of size $n_k$. $R$ = rounds, $E$ = local epochs.

---

## 3. Verified canonical numbers (ledger — every chapter draws from here)

All numbers below were cross-checked against `results/metrics/*.csv` and match the paper's tables to the printed precision. Single seed (42), δ = 10⁻⁵.

**Centralized upper bounds (Sec. 6.1):** MIT-BIH 5-class **macro F1 = 0.913**; PTB-XL binary **macro F1 = 0.813**.
(Source: `centralized_mitbih_cnn1d_seed42.json` f1_macro 0.9129; `centralized_ptbxl_cnn1d_binary_seed42.json` f1_macro 0.8134.)

**Table 6.1 — MIT-BIH non-IID benchmark (best central macro F1 across 50 rounds):** source `noniid_summary.csv` (best_central_f1).

| Partition | FedAvg | FedProx | FedBN | FedPerf |
|---|---|---|---|---|
| IID | 0.926 | 0.860 | 0.920 | 0.926 |
| Quantity skew (β=0.5) | 0.922 | 0.884 | 0.912 | 0.913 |
| Dirichlet α=1.0 | 0.937 | 0.889 | 0.911 | 0.928 |
| Dirichlet α=0.5 | 0.926 | 0.903 | 0.850 | 0.928 |
| Dirichlet α=0.1 | 0.745 | 0.735 | 0.554 | 0.745 |
| Label skew (C=2) | 0.527 | 0.386 | 0.219 | 0.530 |

**Table 6.2 — Eval-regime flip, MIT-BIH label skew C=2:** source `central_vs_local_label_skew_c2.csv`.

| Aggregation | Central F1 | Local F1 (mean±std) | Δ |
|---|---|---|---|
| FedAvg | 0.527 | 0.823 ± 0.178 | +0.297 |
| FedProx | 0.386 | 0.488 ± 0.380 | +0.101 |
| FedBN | 0.224 | 0.942 ± 0.053 | +0.718 |
| FedPerf | 0.530 | 0.654 ± 0.255 | +0.124 |

Milder Dirichlet α=0.1 (same flip, smaller magnitude): FedBN central 0.536 vs local 0.848 ± 0.154 (+0.31); FedAvg central 0.745 vs local 0.831 ± 0.128 (+0.09). Source `central_vs_local_dirichlet_a01.csv`.

**Table 6.3 — MIT-BIH DP privacy–utility (central_f1_best; achieved ε within 0.5% of target):** source `dp_summary.csv`.

| Method | Partition | ε | Achieved ε | Central F1 | Local F1 (mean±std) |
|---|---|---|---|---|---|
| DP-FedAvg (raw BN) | Label skew C=2 | ∞ | — | 0.517 | 0.829 ± 0.192 |
| DP-FedAvg (raw BN) | Label skew C=2 | 3 | REFUSED | — | — |
| DP-FedAvg (raw BN) | Label skew C=2 | 1 | REFUSED | — | — |
| DP-FedAvg+GN | Label skew C=2 | ∞ | — | 0.261 | 0.463 ± 0.368 |
| DP-FedAvg+GN | Label skew C=2 | 3 | 2.996 | 0.191 | 0.295 ± 0.241 |
| DP-FedAvg+GN | Label skew C=2 | 1 | 0.996 | 0.181 | 0.295 ± 0.241 |
| DP-FedBN | Label skew C=2 | ∞ | — | 0.221 | 0.937 ± 0.043 |
| DP-FedBN | Label skew C=2 | 3 | 2.993 | 0.181 | 0.467 ± 0.264 |
| DP-FedBN | Label skew C=2 | 1 | 0.996 | 0.217 | 0.344 ± 0.194 |
| DP-FedAvg (raw BN) | Dirichlet α=0.1 | ∞ | — | 0.738 | 0.816 ± 0.130 |
| DP-FedAvg (raw BN) | Dirichlet α=0.1 | 3 | REFUSED | — | — |
| DP-FedAvg (raw BN) | Dirichlet α=0.1 | 1 | REFUSED | — | — |
| DP-FedAvg+GN | Dirichlet α=0.1 | ∞ | — | 0.736 | 0.832 ± 0.118 |
| DP-FedAvg+GN | Dirichlet α=0.1 | 3 | 2.996 | 0.382 | 0.376 ± 0.101 |
| DP-FedAvg+GN | Dirichlet α=0.1 | 1 | 0.994 | 0.378 | 0.350 ± 0.144 |
| DP-FedBN | Dirichlet α=0.1 | ∞ | — | 0.551 | 0.861 ± 0.130 |
| DP-FedBN | Dirichlet α=0.1 | 3 | 2.996 | 0.189 | 0.368 ± 0.151 |
| DP-FedBN | Dirichlet α=0.1 | 1 | 0.994 | 0.251 | 0.465 ± 0.308 |

DP edge at ε=3 (local F1): label skew C=2 → DP-FedBN 0.467 vs GN 0.295 = **+0.17**; Dirichlet α=0.1 → DP-FedBN 0.368 vs GN 0.376 = **−0.008**.

**Table 6.4 — PTB-XL binary non-DP cross-task (central_f1_best, local mean±std):** source `ptbxl_summary.csv`.

| Partition | Aggregation | Central F1 | Local F1 (mean±std) |
|---|---|---|---|
| IID | FedAvg | 0.806 | 0.817 ± 0.008 |
| IID | FedProx | 0.811 | 0.800 ± 0.007 |
| IID | FedBN | 0.797 | 0.691 ± 0.131 |
| IID | FedPerf | 0.812 | 0.789 ± 0.008 |
| Dirichlet α=0.1 | FedAvg | 0.804 | 0.464 ± 0.148 |
| Dirichlet α=0.1 | FedProx | 0.817 | 0.661 ± 0.113 |
| Dirichlet α=0.1 | FedBN | 0.688 | 0.588 ± 0.103 |
| Dirichlet α=0.1 | FedPerf | 0.805 | 0.419 ± 0.234 |
| Label skew C=1 † | FedAvg | 0.363 | 0.400 ± 0.490 |
| Label skew C=1 † | FedProx | 0.389 | 0.431 ± 0.363 |
| Label skew C=1 † | FedBN | 0.326 | 1.000 ± 0.000 |
| Label skew C=1 † | FedPerf | 0.397 | 0.462 ± 0.301 |

† degenerate single-class clients → FedBN local F1 trivially 1.000; documented as a partition design flaw, excluded from DP-FedBN claims.

PTB-XL Dirichlet flip: FedBN local 0.588 beats FedAvg local 0.464 by **+0.123** (vs +0.017 on MIT-BIH Dirichlet); central FedBN 0.688 below FedAvg 0.804.

**Table 6.5 — PTB-XL Dirichlet α=0.1 DP cross-task (central_f1_best, local mean±std):** source `ptbxl_dp_summary.csv`.

| Method | ε | Central F1 | Local F1 (mean±std) |
|---|---|---|---|
| DP-FedAvg (raw BN) | ∞ | 0.807 | 0.552 ± 0.162 |
| DP-FedAvg (raw BN) | 3 | REFUSED | — |
| DP-FedAvg (raw BN) | 1 | REFUSED | — |
| DP-FedAvg+GN | ∞ | 0.784 | 0.443 ± 0.142 |
| DP-FedAvg+GN | 3 (ach. 2.994) | 0.774 | 0.669 ± 0.109 |
| DP-FedAvg+GN | 1 (ach. 0.995) | 0.765 | 0.715 ± 0.148 |
| DP-FedBN | ∞ | 0.688 | 0.604 ± 0.125 |
| DP-FedBN | 3 (ach. 2.994) | 0.400 | 0.435 ± 0.054 |
| DP-FedBN | 1 (ach. 0.995) | 0.451 | 0.526 ± 0.146 |

DP edge at ε=3 (local F1): DP-FedBN 0.435 vs GN 0.669 = **−0.234** (and +0.161 at ε=∞).

**Synthesis (Sec. 6 + 7):** DP-FedBN's local advantage is partition-dependent: label skew with ≥3 classes/task (MIT-BIH C=2) → **+0.17 at ε=3, DP-FedBN wins**; Dirichlet α=0.1 → tied or losing (MIT-BIH −0.008, PTB-XL −0.234 at ε=3), **GroupNorm wins**.

**Dataset facts:** MIT-BIH 109,453 beats across 48 records, channel 0, ±125-sample (250-sample, ≈694 ms @ 360 Hz) beat windows, z-scored, AAMI 5-class (N≈83% … F/Q rare). PTB-XL 21,388 records (9,246 NORM / 12,142 Abnormal), Lead II, 1000 samples (10 s @ 100 Hz), z-scored, folds 1–9 → clients / fold 10 → central test. Model: 4-block BN-heavy CNN1D ≈437k params. K=5 clients, R=50 (non-DP) / R=30 (DP), E=5, batch 96, AdamW lr 1e-3 wd 5e-4, clip C=1.0, custom in-process simulator (not Flower). Audit: 7 checks incl. synthetic feature-shift reproduction (+0.45 central F1 for FedBN over FedAvg in the toy BN-shift setup) → implementation confirmed correct.

---

## 4. Chapter-level outline

### 00-abstract.md (250–350 words)
Problem (privacy + per-site personalization tension) → approach (DP-FedBN dual-optimizer) → key results (eval-regime flip task-agnostic; partition-dependent DP advantage; naive DP-FedAvg refused) → contribution (verbatim, §1). 4–6 keywords. Template header block (title/author/degree/supervisor/date placeholder/page placeholder).

### 01-introduction.md (1,800–2,500 words)
- 1.1 Motivation — privacy in medical ML; ECG-based cardiac anomaly detection & clinical relevance; why FL; why DP on top of FL; the BN ⊕ DP-SGD ⊕ FL three-way tension.
- 1.2 Problem statement — keep BN-locality (FedBN's non-IID mechanism) while honoring record-level DP; Opacus refuses BN.
- 1.3 Contributions — verbatim three-point list + ADCOL acknowledgment (§1).
- 1.4 Thesis organization — one paragraph mapping chapters 2–7.

### 02-fundamental-concepts.md (4,000–5,500 words) — pedagogical home of all background
- 2.1 ECG signals and arrhythmia classification (leads, beats, AAMI EC57).
- 2.2 Deep learning for ECG (1-D CNNs, class imbalance, macro F1).
- 2.3 Federated learning (FedAvg; the non-IID problem: covariate/label/quantity skew; FedProx).
- 2.4 Batch normalization (formal definition; train vs eval statistics; why BN is special in FL → FedBN).
- 2.5 Differential privacy (ε–δ definition; Gaussian mechanism; DP-SGD: per-sample clipping + noise; privacy accounting / RDP).
- 2.6 The BatchNorm–DP-SGD conflict (per-sample gradient ill-definedness; why Opacus rejects BN).

### 03-related-work.md (2,200–3,000 words)
- 3.1 Privacy-preserving FL (FedAvg, open problems, threat models).
- 3.2 Differential privacy in healthcare ML (DP-SGD; client- vs record-level; DP in medical FL).
- 3.3 BatchNorm in FL / FedBN literature (FedBN, extensions; GroupNorm/other norm fixes; ResNetFed; FedOnco-Bench).
- 3.4 Deep learning for ECG / arrhythmia classification (MIT-BIH, PTB-XL benchmarks).
- 3.5 ADCOL (ICML 2023) and exact positioning — the three differences; honest "what we do/do not contribute."

### 04-methodology.md (2,800–3,800 words) — load-bearing
- 4.1 System and threat model (honest-but-curious server + malicious peers; what crosses the boundary).
- 4.2 Model architecture (CNN1D, BN placement).
- 4.3 Preliminaries recap (DP-SGD eq.; BN obstacle) — reference Ch. 2, do not re-derive fully.
- 4.4 FedBN aggregation rule (Eq. partition θ).
- 4.5 The record-level DP-SGD construction (dual optimizer; freeze-then-unfreeze; why it is record-level DP) — the core contribution (paper framing).
- 4.6 Handling BN statistics under DP and FL (γ,β via plain opt; μ,σ² via forward EMA; no direct noise; indirect drift).
- 4.7 Privacy accounting (per-client accountant on θ^¬BN; server needs no DP composition).
- 4.8 Algorithm 1 box + implementation pitfalls (state-dict prefix; ExpandedWeights).

### 05-experimental-setup.md (2,200–3,000 words)
- 5.1 Datasets & preprocessing (MIT-BIH; PTB-XL; why binary on PTB-XL).
- 5.2 Non-IID partitioning (six strategies; figure).
- 5.3 Model & training details (rounds, epochs, batch, optimizer, class weights).
- 5.4 DP configuration (ε, δ, σ, C; three DP methods).
- 5.5 Evaluation regimes (central vs local; macro F1) + metrics.
- 5.6 Implementation, reproducibility, hardware (custom simulator not Flower; seed 42; Opacus 1.5; PyTorch 2.4; single-seed disclosure).

### 06-results-and-discussion.md (2,800–3,800 words)
- 6.1 Centralized baselines.
- 6.2 Non-IID FL benchmark on MIT-BIH (Table 6.1; three patterns; FedProx caveat).
- 6.3 The eval-regime flip (Table 6.2; Fig eval-regime; mechanism; 7-check audit).
- 6.4 DP privacy–utility frontier on MIT-BIH (Table 6.3; Fig dual; REFUSED rows; label-skew edge; Dirichlet mixed).
- 6.5 Cross-task replication on PTB-XL (Tables 6.4–6.5; Fig cross-task; label_skew_c1 flaw; DP loss on Dirichlet).
- 6.6 Discussion — why the DP edge is partition-dependent (mechanism); practitioner rubric; eval-regime as a methodological variable.
- 6.7 Limitations.

### 07-conclusion.md (1,000–1,500 words)
Restate contribution (verbatim), three findings (measured), limitations recap, concrete future work (multi-seed; multi-label 12-lead PTB-XL; ResNet/Transformer; more clients; FedProx tuning; stronger threat models on deployed BN).

### 99-references.md
IEEE numeric style. Reuse the paper's 22 references; add verified background references for the expanded Ch. 2–3 (mark uncertain ones `[VERIFY]`).

---

## 5. Figure & table inventory (real repo files only)

**Figures (final numbering at assembly; HTML comment in each chapter notes the source file):**

| Draft ref | Source file (results/figures/) | Used in | Content |
|---|---|---|---|
| Partition distributions | `fig_partition_distributions_mitbih.png` | 5.2 | 6-panel per-client class distribution, MIT-BIH, 5 clients |
| Centralized curves (MIT-BIH) | `centralized_mitbih_cnn1d_seed42_training.png` | 6.1 | train/val loss + macro F1 over epochs |
| Centralized curves (PTB-XL) | `centralized_ptbxl_cnn1d_binary_seed42_training.png` | 6.1 | train/val loss + macro F1 over epochs |
| Non-IID degradation | `fig_degradation_from_iid.png` and/or `fig_noniid_heatmap.png` | 6.2 | central F1 vs partition severity / heatmap |
| Eval-regime flip | `fig_central_vs_local.png` | 6.3 | central vs local F1, Dir α=0.1 + label skew C=2 (paper Fig 1) |
| DP privacy–utility | `fig_privacy_utility_dual.png` | 6.4 | 2×2 (label skew / Dirichlet) × (central / local) (paper Fig 2) |
| Cross-task summary | `fig_phase6_cross_task_summary.png` | 6.5 | MIT-BIH + PTB-XL Dir α=0.1 central vs local (paper Fig 3) |
| (optional) PTB-XL DP frontier | `fig_phase6_dp_privacy_utility_ptbxl.png` | 6.5 | PTB-XL DP central/local vs ε |
| (optional) example fed curves | e.g. `dp_fedbn_label_skew_c2_eps3.0_mitbih_seed42_fed_curves.png` | 6.4 | round-by-round F1 trajectory |

**Tables:** Table 6.1–6.5 as in §3 (all markdown, numbers verified). Also a small Methodology Algorithm 1 box and a DP-configuration table in Ch. 5.

---

## 6. Drafting discipline
- Put each background explanation **once** (Ch. 2). Elsewhere reference it.
- Keep the paper and the template **read-only**.
- Mark anything not in paper/code with `[TODO: …]`; mark uncertain citations `[VERIFY: …]`.
- After each chapter: self-check every number and figure reference against §3 and the paper, then commit.
