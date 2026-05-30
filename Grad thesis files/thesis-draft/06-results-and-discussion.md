# 6. Results and Discussion

This chapter reports the experimental results in five threads — the centralized baselines that pin the upper bound, the non-IID federated benchmark on MIT-BIH, the eval-regime flip, the differential-privacy privacy–utility frontier on MIT-BIH, and the cross-task replication on PTB-XL — and then interprets them: why DP-FedBN's advantage is partition-dependent, what the findings imply for practitioners, and why the choice of evaluation regime is itself a result. The chapter closes with the limitations. Every number reported is a single-seed (seed 42) realization, a point whose consequences are addressed in Section 6.7.

## 6.1 Centralized baselines

A single-machine model trained on the full pooled training data establishes the ceiling that any federated configuration could at best approach. On MIT-BIH five-class beat classification it reaches a test **macro F1 of 0.913**, and on PTB-XL binary record classification a test **macro F1 of 0.813**. These two numbers are the reference points for everything that follows: they pin the "IID-but-federated" performance that the federated runs are measured against, and they confirm that the binary PTB-XL reformulation (Section 5.1.3) reaches a healthy, balanced operating point where the five-super-class attempt had stalled near 0.50.

<!-- FIGURE: results/figures/centralized_mitbih_cnn1d_seed42_training.png and centralized_ptbxl_cnn1d_binary_seed42_training.png — per-epoch training/validation loss and macro-F1 curves for the two centralized baselines. -->

The per-epoch training and validation curves for the two centralized models (Figure 6.1) show stable convergence with early stopping on validation macro F1, and serve as a sanity check that the shared architecture is well-matched to both tasks before any federation or privacy machinery is introduced.

## 6.2 Non-IID federated benchmark on MIT-BIH

Table 6.1 reports the best central macro F1 for each combination of partition and aggregation on MIT-BIH, across the full $6 \times 4$ grid of six partitions and four aggregation strategies.

**Table 6.1 — MIT-BIH five-class non-IID benchmark: best central macro F1 across 50 rounds (single seed). Higher is better.**

| Partition | FedAvg | FedProx | FedBN | FedPerf |
|---|---|---|---|---|
| IID | 0.926 | 0.860 | 0.920 | 0.926 |
| Quantity skew ($\beta=0.5$) | 0.922 | 0.884 | 0.912 | 0.913 |
| Dirichlet $\alpha=1.0$ | 0.937 | 0.889 | 0.911 | 0.928 |
| Dirichlet $\alpha=0.5$ | 0.926 | 0.903 | 0.850 | 0.928 |
| Dirichlet $\alpha=0.1$ | 0.745 | 0.735 | 0.554 | 0.745 |
| Label skew ($C=2$) | 0.527 | 0.386 | **0.219** | 0.530 |

<!-- FIGURE: results/figures/fig_noniid_heatmap.png — heatmap of best-central-round macro F1 across the partition x aggregation grid; FedBN's label-skew cell (0.22) is the visual outlier. -->

Three patterns are visible, and they are reinforced by the heatmap view of the same grid in Figure 6.2.

First, in the IID-and-mild-skew partitions — IID, quantity skew, and Dirichlet $\alpha \ge 0.5$ — every aggregation sits close to the centralized 0.913 ceiling. Federation costs essentially nothing in this benign regime; the choice of aggregation rule barely matters.

Second, severe label heterogeneity — Dirichlet $\alpha = 0.1$ and label skew $C = 2$ — collapses every aggregation, but **FedBN collapses the most**. On label skew $C = 2$, FedBN reaches a central macro F1 of only 0.219, while FedAvg, FedProx, and FedPerf reach 0.527, 0.386, and 0.530 respectively. This is a striking result, because it contradicts the original FedBN paper [9], which reported FedBN *outperforming* FedAvg under non-IID data. The apparent contradiction is the entry point to the central finding of this thesis and is resolved in Section 6.3.

Third, FedProx at its un-tuned proximal coefficient $\mu = 0.01$ is consistently worse than FedAvg across every partition. This is a known sensitivity of the proximal term to its hyperparameter rather than a deficiency of the method, and tuning $\mu$ is left to future work.

## 6.3 The eval-regime flip

The collapse of FedBN's central F1 on label-heterogeneous data was surprising enough to warrant a careful audit of the implementation before any conclusion was drawn (described at the end of this section). That audit's diagnostic check measured, for the FedBN runs, the per-client *local* macro F1 alongside the central macro F1 — and the result is the empirical core of this thesis. Table 6.2 reports both regimes on the most extreme partition, label skew $C = 2$.

**Table 6.2 — MIT-BIH label skew $C=2$: central macro F1 (pooled holdout) versus per-client local macro F1 (mean ± std across 5 clients). FedBN ranks worst centrally and best locally.**

| Aggregation | Central F1 | Local F1 (mean ± std) | $\Delta$ (local − central) |
|---|---|---|---|
| FedAvg | 0.527 | 0.823 ± 0.178 | +0.297 |
| FedProx | 0.386 | 0.488 ± 0.380 | +0.101 |
| FedBN | **0.224** | **0.942 ± 0.053** | **+0.718** |
| FedPerf | 0.530 | 0.654 ± 0.255 | +0.124 |

The numbers tell a single, sharp story. FedBN's central F1 is 0.224 — the lowest of the four aggregations — while its local F1 is 0.942 — the highest of the four. **The same model, with the same training and the same federation, ranks last under one evaluation regime and first under the other.** The central-versus-local gap reaches +0.72, the largest in the table by a wide margin. This is the *eval-regime flip*.

<!-- FIGURE: results/figures/fig_central_vs_local.png — grouped bars of central vs per-client local macro F1 for the four aggregations, on Dirichlet alpha=0.1 and label skew C=2; FedBN is lowest on central, highest on local. -->

Figure 6.3 visualizes the flip across both severe partitions. On label skew $C = 2$, FedBN's central bar is the shortest while its local bar is the tallest; the pattern is the same, with smaller magnitude, on Dirichlet $\alpha = 0.1$.

**Mechanism.** The flip has a clean explanation rooted in what FedBN keeps local: the BatchNorm running statistics. Each client's BN statistics are calibrated to *that client's* local label distribution. Under label skew $C = 2$, a client owns only two of the five classes, so its BN statistics are tuned for the activation distribution of those two classes. When that client's full model — convolutional features shared across the federation, but BN specialized to two classes — is asked to classify the *central* test set, which contains all five classes in MIT-BIH's natural proportions, its BN normalization collapses on the three classes the client never trained on. Pooled-central evaluation therefore punishes FedBN's specialization. Per-client local evaluation, where each client is only ever asked about its own distribution, rewards exactly that specialization, which is why FedBN's local F1 is the highest of the four.

The same pattern holds on the milder Dirichlet $\alpha = 0.1$ partition with smaller magnitude: FedBN's central F1 is 0.536 versus a local F1 of 0.848 ± 0.154, a +0.31 flip, while FedAvg shows a central 0.745 versus local 0.831 ± 0.128, a +0.09 flip. The sign and the ordering are preserved; only the magnitude shrinks as the heterogeneity softens.

**Implementation audit.** Because this result contradicts the published FedBN finding, the collapse was subjected to a seven-check audit before being accepted: verification of BatchNorm-key discovery, of the server-side BatchNorm exclusion, of post-round BatchNorm divergence across clients, of parameter-set integrity, of IID parity (FedBN matching FedAvg when the data is IID), and a synthetic feature-shift sanity test that reproduces the regime FedBN was originally designed for. In that synthetic feature-shift setup, FedBN correctly recovers a **+0.45 central-F1 advantage over FedAvg**, exactly as the original paper would predict — confirming that the implementation is correct and that FedBN does help in the feature-shift regime it was built for. The central-F1 collapse on *label*-heterogeneous medical data is therefore a real property of the regime, not a bug: FedBN's advantage and disadvantage are two faces of the same locality mechanism, and which one is observed depends on whether evaluation is pooled-central or per-client local.

## 6.4 Differential-privacy privacy–utility frontier on MIT-BIH

The three differentially private methods — DP-FedAvg (raw BatchNorm), DP-FedAvg+GroupNorm, and DP-FedBN — were run across $\varepsilon \in \{1, 3, \infty\}$ on the two most challenging partitions, label skew $C = 2$ and Dirichlet $\alpha = 0.1$. Table 6.3 reports central and local F1 for every combination; the achieved $\varepsilon$ values fall within 0.5% of target on all finite-$\varepsilon$ runs.

**Table 6.3 — MIT-BIH DP privacy–utility benchmark (single seed, $\delta = 10^{-5}$). Naive DP-FedAvg is refused by Opacus at finite $\varepsilon$.**

| Method | Partition | $\varepsilon$ | Central F1 | Local F1 (mean ± std) |
|---|---|---|---|---|
| DP-FedAvg (raw BN) | Label skew $C{=}2$ | $\infty$ | 0.517 | 0.829 ± 0.192 |
| DP-FedAvg (raw BN) | Label skew $C{=}2$ | 3 | REFUSED | — |
| DP-FedAvg (raw BN) | Label skew $C{=}2$ | 1 | REFUSED | — |
| DP-FedAvg+GroupNorm | Label skew $C{=}2$ | $\infty$ | 0.261 | 0.463 ± 0.368 |
| DP-FedAvg+GroupNorm | Label skew $C{=}2$ | 3 | 0.191 | 0.295 ± 0.241 |
| DP-FedAvg+GroupNorm | Label skew $C{=}2$ | 1 | 0.181 | 0.295 ± 0.241 |
| **DP-FedBN** | Label skew $C{=}2$ | $\infty$ | 0.221 | **0.937 ± 0.043** |
| **DP-FedBN** | Label skew $C{=}2$ | 3 | 0.181 | **0.467 ± 0.264** |
| **DP-FedBN** | Label skew $C{=}2$ | 1 | 0.217 | **0.344 ± 0.194** |
| DP-FedAvg (raw BN) | Dirichlet $\alpha{=}0.1$ | $\infty$ | 0.738 | 0.816 ± 0.130 |
| DP-FedAvg (raw BN) | Dirichlet $\alpha{=}0.1$ | 3 | REFUSED | — |
| DP-FedAvg (raw BN) | Dirichlet $\alpha{=}0.1$ | 1 | REFUSED | — |
| DP-FedAvg+GroupNorm | Dirichlet $\alpha{=}0.1$ | $\infty$ | 0.736 | 0.832 ± 0.118 |
| DP-FedAvg+GroupNorm | Dirichlet $\alpha{=}0.1$ | 3 | 0.382 | 0.376 ± 0.101 |
| DP-FedAvg+GroupNorm | Dirichlet $\alpha{=}0.1$ | 1 | 0.378 | 0.350 ± 0.144 |
| **DP-FedBN** | Dirichlet $\alpha{=}0.1$ | $\infty$ | 0.551 | 0.861 ± 0.130 |
| **DP-FedBN** | Dirichlet $\alpha{=}0.1$ | 3 | 0.189 | 0.368 ± 0.151 |
| **DP-FedBN** | Dirichlet $\alpha{=}0.1$ | 1 | 0.251 | 0.465 ± 0.308 |

<!-- FIGURE: results/figures/fig_privacy_utility_dual.png — 2x2 grid (rows: label skew C=2, Dirichlet alpha=0.1; columns: central, local) of macro F1 vs epsilon for the three DP methods. -->

Figure 6.4 plots this frontier as a $2 \times 2$ grid of central and local F1 against $\varepsilon$ for the two partitions. Three observations follow.

First, **naive DP-FedAvg is empirically refused.** Four of the eighteen runs — raw BatchNorm at both finite budgets on both partitions — terminate at setup time because Opacus's `ModuleValidator` will not accept a BatchNorm-containing model under unmodified DP-SGD. This is not an omission in the experimental matrix; it is the empirical fact that motivates a non-naive composition such as DP-FedBN or the GroupNorm fix, and it is recorded as a first-class `REFUSED` result rather than quietly dropped.

Second, **on label skew, DP-FedBN preserves a clear local edge.** At $\varepsilon = 3$, DP-FedBN's local F1 is 0.467 versus 0.295 for DP-FedAvg+GroupNorm — a **+0.17 gap that survives differential privacy**. At the stringent $\varepsilon = 1$ the gap shrinks to +0.05 but remains positive, and at $\varepsilon = \infty$ (DP infrastructure but no noise) it is +0.47. FedBN's specialization mechanism transfers under DP on this partition: even noised, a per-client BatchNorm tuned to two classes beats a globally-normalized GroupNorm on each client's own data.

Third, **on Dirichlet, the local edge is mixed.** At $\varepsilon = 3$, DP-FedBN's local F1 of 0.368 is effectively tied with DP-FedAvg+GroupNorm's 0.376 — a −0.008 difference well within single-seed noise. At $\varepsilon = 1$, DP-FedBN wins again (0.465 versus 0.350, a +0.115 gap), and at $\varepsilon = \infty$ it wins by 0.029. The non-monotonic ordering across $\varepsilon$ on this partition is a single-seed signature, and no strong claim is drawn from its precise shape; what is robust is that the large, decisive local edge seen on label skew is absent here.

## 6.5 Cross-task replication on PTB-XL

To test whether these findings are specific to MIT-BIH or hold as task-agnostic federated-learning phenomena, the non-DP and DP sweeps were replicated on PTB-XL binary record classification — a genuinely different task (ten-second records rather than 250-sample beats, 100 Hz rather than 360 Hz, two classes rather than five).

**Non-DP: the eval-regime flip is confirmed.** Table 6.4 reports the twelve non-DP runs across three partitions and four aggregations.

**Table 6.4 — PTB-XL binary, non-DP cross-task replication (best central F1; local mean ± std).**

| Partition | Aggregation | Central F1 | Local F1 (mean ± std) |
|---|---|---|---|
| IID | FedAvg | 0.806 | 0.817 ± 0.008 |
| IID | FedProx | 0.811 | 0.800 ± 0.007 |
| IID | FedBN | 0.797 | 0.691 ± 0.131 |
| IID | FedPerf | 0.812 | 0.789 ± 0.008 |
| Dirichlet $\alpha=0.1$ | FedAvg | 0.804 | 0.464 ± 0.148 |
| Dirichlet $\alpha=0.1$ | FedProx | 0.817 | 0.661 ± 0.113 |
| Dirichlet $\alpha=0.1$ | FedBN | **0.688** | **0.588 ± 0.103** |
| Dirichlet $\alpha=0.1$ | FedPerf | 0.805 | 0.419 ± 0.234 |
| Label skew $C=1$ † | FedAvg | 0.363 | 0.400 ± 0.490 |
| Label skew $C=1$ † | FedProx | 0.389 | 0.431 ± 0.363 |
| Label skew $C=1$ † | FedBN | 0.326 | 1.000 ± 0.000 |
| Label skew $C=1$ † | FedPerf | 0.397 | 0.462 ± 0.301 |

<small>† On a binary task, label skew $C=1$ gives every client exactly one class; FedBN's local F1 is then trivially 1.000 (each client predicts its single class perfectly on its own single-class slice). This is documented as a partition design flaw, not a finding, and is excluded from the DP-FedBN claims.</small>

On Dirichlet $\alpha = 0.1$, FedBN's central F1 of 0.688 is the lowest of the four aggregations, while its local F1 of 0.588 beats FedAvg's local 0.464 by **+0.123** — a *larger* absolute margin than the corresponding comparison on MIT-BIH Dirichlet (+0.017). The eval-regime flip — FedBN below FedAvg on central, above on local — survives a complete change of task, dataset, sampling rate, and class count. Figure 6.5 shows the two datasets side by side, with FedBN ranking below FedAvg on central F1 and above on local F1 in both.

<!-- FIGURE: results/figures/fig_phase6_cross_task_summary.png — MIT-BIH and PTB-XL panels (Dirichlet alpha=0.1, no DP), central vs local bars, annotated with the FedBN-minus-FedAvg gaps (central negative, local positive) on both datasets. -->

The label skew $C = 1$ rows illustrate why that partition is excluded from the substantive claims: with one class per client, FedBN's local F1 is exactly 1.000 with zero variance at every budget, because each client's model perfectly predicts its only class on its own single-class data. The result is invariant to method, partition, and DP noise; it is an artifact of the partition on a binary task, not an algorithmic property.

**DP: DP-FedBN loses on PTB-XL Dirichlet.** Table 6.5 reports the nine-cell DP sweep on the meaningful PTB-XL partition, Dirichlet $\alpha = 0.1$.

**Table 6.5 — PTB-XL Dirichlet $\alpha=0.1$ DP cross-task (best central F1; local mean ± std).**

| Method | $\varepsilon$ | Central F1 | Local F1 (mean ± std) |
|---|---|---|---|
| DP-FedAvg (raw BN) | $\infty$ | 0.807 | 0.552 ± 0.162 |
| DP-FedAvg (raw BN) | 3 | REFUSED | — |
| DP-FedAvg (raw BN) | 1 | REFUSED | — |
| DP-FedAvg+GroupNorm | $\infty$ | 0.784 | 0.443 ± 0.142 |
| DP-FedAvg+GroupNorm | 3 | 0.774 | **0.669 ± 0.109** |
| DP-FedAvg+GroupNorm | 1 | 0.765 | **0.715 ± 0.148** |
| DP-FedBN | $\infty$ | 0.688 | 0.604 ± 0.125 |
| DP-FedBN | 3 | 0.400 | 0.435 ± 0.054 |
| DP-FedBN | 1 | 0.451 | 0.526 ± 0.146 |

At $\varepsilon = \infty$, DP-FedBN's local F1 of 0.604 beats DP-FedAvg+GroupNorm's 0.443 by +0.161, preserving the non-DP rank flip once the DP infrastructure is plumbed in but before any noise is added. At finite $\varepsilon$, however, the ordering **reverses**: DP-FedAvg+GroupNorm's local F1 at $\varepsilon = 3$ is 0.669 versus DP-FedBN's 0.435, a **−0.234** loss for DP-FedBN; at $\varepsilon = 1$ the ordering remains, 0.715 versus 0.526, a −0.189 loss. On PTB-XL Dirichlet, DP noise does not merely erode DP-FedBN's advantage — it inverts it.

**Synthesis.** Combining the MIT-BIH and PTB-XL DP results, **DP-FedBN's local advantage is partition-dependent**, with a consistent sign across both datasets:

- Label skew with at least three classes per task (MIT-BIH $C = 2$): **+0.17 at $\varepsilon = 3$; DP-FedBN wins.**
- Dirichlet $\alpha = 0.1$: tied or losing under DP — MIT-BIH −0.008 and PTB-XL −0.234 at $\varepsilon = 3$, the same sign in a sharper magnitude; **DP-FedAvg+GroupNorm wins.**

## 6.6 Discussion

### 6.6.1 Why DP-FedBN's edge is partition-dependent

FedBN's no-DP local advantage comes from one design choice: keep each client's BatchNorm running statistics local and never average them across clients. In a heterogeneous regime, the per-client BatchNorm is a *better* normalization for that client's data than any globally-averaged BatchNorm could be, because it is matched to the client's own distribution. That is the whole source of FedBN's local-eval strength.

Under differential privacy, this same locality becomes a liability whose severity depends on the partition. As established in Section 4.6, the BatchNorm running statistics and affine parameters receive no DP noise *directly* — the running buffers are forward-pass moving averages, and the affine parameters are updated by the plain optimizer. But the noised non-BatchNorm gradients change the convolutional features, and those features are exactly the activations whose statistics feed the BatchNorm running buffers. DP noise therefore disturbs each client's BatchNorm behavior *indirectly*, and — crucially — there is **no averaging across clients to smooth that disturbance out**, because FedBN deliberately keeps the BatchNorm state local. Each client's BatchNorm consequently drifts in directions driven more by upstream noise than by data structure, and it loses the very specialization that was its advantage. GroupNorm, by contrast, has no running statistics, and the DP noise that reaches its affine parameters *is* averaged across clients during aggregation, which damps it. This asymmetry — FedBN's per-client normalization accumulates DP-driven drift without averaging, GroupNorm's shared normalization averages it away — is the mechanism, and it is consistent with the sign of the result on both datasets.

Why, then, does label skew differ from Dirichlet? The answer is the *amount of specialization available to lose*. On label skew $C = 2$, each client owns only two of five classes, and the specialization that a per-client BatchNorm provides is so much better than a globally-averaged one that even a noised FedBN beats GroupNorm on local evaluation at $\varepsilon = 3$. Specialization wins when the "specialization distance" between a client's distribution and the global average is large. On Dirichlet $\alpha = 0.1$, each client has skewed-but-not-truncated class proportions — it still sees every class, just in unusual ratios — so the specialization distance is small, a globally-averaged BatchNorm is a reasonable normalizer for any client, and the DP noise that FedBN's locality fails to average away costs more than the residual specialization gains. GroupNorm wins. The magnitude of DP-FedBN's loss tracks how little specialization there was to protect.

### 6.6.2 A practitioner rubric for medical federated learning

These findings translate into a small, heterogeneity-typed decision rubric for medical FL practitioners:

1. **Label-truncated heterogeneity** (specialty hospitals, single-condition clinics, where each site sees only a few conditions): DP-FedBN preserves a clear local-F1 edge under DP, and the dual-optimizer overhead is justified.
2. **Proportional heterogeneity** (general hospitals with a mixed but complete case-mix): DP-FedAvg+GroupNorm is simpler and at least as good — often better.
3. **Local-deployment priority** (each site runs its own personalized model): FedBN without DP gives the strongest local F1; if DP is required, DP-FedBN is competitive on label-truncated data and competitive-or-worse on proportional data.
4. **Strict regulatory pooled-central requirements** (a single model evaluated on a pooled benchmark): FedAvg or FedPerf, not FedBN — the eval-regime flip means FedBN's central F1 ranks worst even when its local F1 ranks best.

### 6.6.3 The evaluation regime is a methodological variable

The most transferable lesson is methodological. A reader who reports "central F1 on a held-out test pool" as the primary metric will conclude that FedBN underperforms. A reader evaluating the same models per-client may conclude that FedBN wins. Whether to report central or local F1 is therefore not a stylistic choice — **it changes the conclusion.** When the deployment scenario is itself ambiguous between a pooled, single-model setting and a per-site personalized-model setting, both regimes should be reported. Most federated-learning benchmarking papers, including the original FedBN paper [9], report only one regime; this thesis suggests that the regime is a missing variable whose omission can silently determine which method appears best.

## 6.7 Limitations

The findings should be read with the following limitations in mind, several of which are the natural starting points for future work (Chapter 7).

- **Single seed.** All reported numbers are single-seed realizations (seed 42). The non-monotonicities across $\varepsilon$ on the Dirichlet partition are single-seed signatures, and multi-seed variance quantification is the single most important deferred experiment.
- **One architecture.** A fixed BatchNorm-heavy CNN1D (≈437k parameters) is used throughout; whether the eval-regime flip and the partition-dependent DP behavior persist for ResNet-style or Transformer architectures is untested.
- **Simulated heterogeneity.** Non-IID partitions are constructed on pooled datasets rather than measured across genuinely separate institutions, so true cross-institutional covariate shift is not captured. Relatedly, the MIT-BIH central test set is a beat-level (intra-patient) stratified holdout (Section 5.5).
- **Five clients.** Production federated systems involve dozens to hundreds of participants; client-count effects are not characterized.
- **ECG-only validation.** The findings are demonstrated on two ECG tasks; their reach to other medical modalities is an open question.
- **Un-tuned FedProx.** FedProx uses $\mu = 0.01$ and consistently under-performs FedAvg, likely from insufficient hyperparameter tuning rather than a deficiency of the method.
- **Binary PTB-XL.** Multi-label PTB-XL over all five super-classes using the full twelve leads (with macro-AUC as the metric) is the natural extension and is left as future work.
