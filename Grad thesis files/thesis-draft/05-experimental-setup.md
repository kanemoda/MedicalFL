# 5. Experimental Setup

This chapter specifies the experimental protocol in full: the two ECG datasets and their preprocessing, the six non-IID partition schemes used to simulate heterogeneity, the model and training configuration, the differential-privacy parameters, the two evaluation regimes, and the implementation and reproducibility details. Every value reported here is taken from the source paper and the released code; nothing is reconstructed or assumed.

## 5.1 Datasets and preprocessing

Two datasets are used, chosen so that the central empirical claim can be tested on two deliberately different tasks: beat-level multi-class classification at one sampling rate, and record-level binary classification at another.

### 5.1.1 MIT-BIH Arrhythmia Database

The MIT-BIH Arrhythmia Database [18] is used for beat-level five-class classification under the ANSI/AAMI EC57 standard [19], [20], [21]. The database comprises 48 half-hour two-channel ambulatory ECG recordings sampled at 360 Hz. Each beat is read on channel 0 (column 0 of the WFDB signal). R-peak locations are taken from the standard `.atr` annotations, and each beat is windowed as $\pm125$ samples around its R-peak — 250 samples in total, approximately 694 ms at 360 Hz. Beats whose R-peak falls within 125 samples of either signal edge are dropped because their window would not fit. Each beat window is z-score-normalized.

Beat labels follow the AAMI five-class mapping, which groups the raw annotation symbols as
$$
\mathrm{N} = \{\mathrm{N, L, R, e, j}\}, \;\;
\mathrm{S} = \{\mathrm{A, a, J, S}\}, \;\;
\mathrm{V} = \{\mathrm{V, E}\}, \;\;
\mathrm{F} = \{\mathrm{F}\}, \;\;
\mathrm{Q} = \{/, \mathrm{f, Q}\}.
$$
Non-beat annotation symbols (rhythm markers, signal-quality markers, and the like) are dropped rather than forced into the Q class, and the four paced-record subjects (records 102, 104, 107, 217) are retained with their paced (`/`) beats labelled Q, following EC57 and de Chazal et al. [20]. The final dataset contains **109,453 beats across 48 records**. Its class distribution is severely imbalanced — N $\approx 82.8\%$ (90,595 beats), Q $\approx 7.3\%$ (8,040), V $\approx 6.6\%$ (7,235), S $\approx 2.5\%$ (2,781), and F $\approx 0.7\%$ (802) — which is the reason macro F1, not accuracy, is the primary metric (Section 5.5).

### 5.1.2 PTB-XL

PTB-XL [22] is used for record-level binary classification (Normal versus Abnormal). The 100 Hz `records100/` variant is used. For each record, the diagnostic super-class is taken to be the SCP code with the highest annotated confidence whose `diagnostic_class` field is non-null in `scp_statements.csv`; records for which no such code exists are dropped. The binary task is then NORM versus the union of the four abnormal super-classes $\{\mathrm{MI, STTC, CD, HYP}\}$. Each record uses **Lead II only** (column 1 of the 12-lead matrix), z-score-normalized, 1000 samples — ten seconds at 100 Hz. The final dataset contains **21,388 records**, of which 9,246 are NORM and 12,142 Abnormal, a roughly balanced split. The canonical PTB-XL stratified folds are preserved: folds 1–9 enter the client partitions and fold 10 serves as the central test set.

### 5.1.3 Why a binary task on PTB-XL

The choice of a binary PTB-XL task is deliberate and is documented here because it is a design decision rather than an arbitrary simplification. We initially attempted the PTB-XL five-super-class classification on Lead II. Three successive training regimes — a 44k-parameter model, then a 437k-parameter model with data augmentation and stronger regularization, and a cosine learning-rate schedule — all plateaued at a test macro F1 of approximately 0.50, with the hypertrophy (HYP) class stuck near 0.22. This was a representational bottleneck of single-lead input, not a capacity ceiling: the information needed to separate all five super-classes is not reliably present in Lead II alone. We therefore reformulated PTB-XL as a binary task (Normal versus Abnormal), which is well-defined on Lead II and reaches a balanced central macro F1 of approximately 0.81. Full multi-label PTB-XL classification using all twelve leads is a natural extension and is listed as future work in Chapter 7.

## 5.2 Non-IID partitioning

Because both datasets are pooled corpora rather than natively federated collections, statistical heterogeneity is *simulated* by partitioning each dataset across the clients with controlled non-IID schemes. Six partition strategies are used on MIT-BIH:

- **IID** — a stratified random split: each class is divided into roughly equal chunks across clients, so every client sees the same class proportions.
- **Quantity skew** ($\beta = 0.5$) — client *sizes* follow a power law drawn from a Dirichlet distribution, so the smallest client holds about half the data of the median client, while class proportions remain roughly IID.
- **Dirichlet** $\alpha \in \{1.0, 0.5, 0.1\}$ — each class's allocation across clients is drawn from a Dirichlet prior; smaller $\alpha$ yields greater label heterogeneity. $\alpha = 1.0$ is mild, $\alpha = 0.1$ is severe.
- **Label skew** ($C = 2$) — each client owns exactly two of the five classes; this is the most extreme label heterogeneity available, with each class guaranteed to appear in at least one client.

The Dirichlet scheme is the standard mechanism for synthesizing label heterogeneity [28], [29] and is worth stating concretely. For each class $c$, a proportion vector $(p_{c,1}, \dots, p_{c,K})$ is drawn from a symmetric Dirichlet distribution $\mathrm{Dir}(\alpha, \dots, \alpha)$, and that class's samples are distributed across the $K$ clients in those proportions. A small $\alpha$ produces proportion vectors concentrated on a few clients — strong skew — while a large $\alpha$ pushes the proportions toward uniformity and hence toward IID. Quantity skew applies the same Dirichlet draw to client *sizes* rather than per-class shares.

A consequence of this design that must be stated plainly is that the heterogeneity here is **simulated on a pooled dataset** rather than measured across genuinely separate institutions. This is a deliberate trade-off — it gives precise, reproducible control over the type and severity of non-IID-ness, at the cost of not capturing real cross-institutional covariate shift (different devices, populations, and acquisition protocols). The implication for the results is discussed in the limitations (Chapter 6).

For the binary PTB-XL task only three of these are well-defined: IID, Dirichlet $\alpha = 0.1$, and label skew $C = 1$. The latter is the binary analogue of MIT-BIH's $C = 2$ label skew, but it produces degenerate single-class clients (each client holds exactly one of the two classes); the consequences of this degeneracy are documented in Chapter 6 and that partition is excluded from the substantive DP-FedBN claims. All partitions are seeded deterministically — each client's shard depends on the global seed plus its client index — so a partition is reproducible exactly.

<!-- FIGURE: results/figures/fig_partition_distributions_mitbih.png — six-panel stacked-bar visualization of per-client class distributions for all six MIT-BIH partitions across the five clients. -->

**Figure 5.1** visualizes the six MIT-BIH partitions as per-client class-distribution stacked bars across the five clients. The progression from IID (every client a near-replica of the global distribution) through the Dirichlet schemes (increasingly lopsided class shares) to label skew $C = 2$ (each client a tall bar of just two colours) makes the controlled severity of the heterogeneity visible at a glance.

## 5.3 Model and training configuration

The model is the BatchNorm-heavy 1-D CNN described in Section 4.2 (approximately 437k parameters), used unchanged across all datasets, partitions, and privacy budgets. All federated experiments use $K = 5$ clients.

Each federated round runs $E = 5$ local epochs of mini-batch training per client at a batch size of 96. The total number of rounds is $R = 50$ for the non-DP experiments and $R = 30$ for the DP experiments. The optimizer is AdamW [17] with learning rate $1 \times 10^{-3}$ and weight decay $5 \times 10^{-4}$; the server aggregates after each round. The cross-entropy loss is weighted by inverse class frequency, computed on the pooled training shards so that rare classes are not drowned out — the same weighting scheme the centralized baselines use. A standard 80/20 stratified train/validation split is taken within each client's shard; the validation slice is never used for gradient updates and serves both for convergence monitoring and, at the end of training, as the per-client local test set (Section 5.5).

The four non-DP aggregation strategies are FedAvg, FedProx, FedBN, and FedPerf. FedProx uses a proximal coefficient $\mu = 0.01$. FedBN excludes the BatchNorm parameters and buffers from aggregation per Equation (4.1). FedPerf is a personalization-aware baseline that weights client updates according to their local validation macro F1 (so that better-performing clients contribute more); it is included only to contextualize FedBN and is not claimed as a contribution.

**Centralized baseline.** To establish an upper bound that any federated configuration could at best approach, a single-machine model is trained on the full pooled training data of each dataset. The centralized runs use the same CNN1D architecture and AdamW optimizer, trained for up to 50 epochs with early stopping on validation macro F1 (patience 15 for MIT-BIH, 10 for PTB-XL). For the harder PTB-XL task, light label-preserving signal augmentation is applied during training — a random temporal crop, additive Gaussian noise in normalized units, and a slow sinusoidal baseline wander, each applied stochastically — which is the same augmentation referenced in the five-super-class attempt of Section 5.1.3. These centralized numbers are reported in Chapter 6 as the pooled ceiling; they are not themselves a federated result.

## 5.4 Differential-privacy configuration

The differential-privacy experiments sweep three privacy budgets, $\varepsilon \in \{1, 3, \infty\}$, at a fixed $\delta = 10^{-5}$. The budget $\varepsilon = \infty$ denotes no DP noise and is used to separate the effect of plumbing in the DP infrastructure from the effect of the noise itself. For the finite budgets, the noise multiplier $\sigma$ is chosen per client by Opacus's `make_private_with_epsilon` accountant at batch size 96, so that each client's training meets its target $(\varepsilon, \delta)$ guarantee end-to-end. The per-sample gradient clipping bound is $C = 1.0$. Opacus 1.5 is used with its default hooks-based per-sample-gradient backend (not the ExpandedWeights backend, for the reason given in Section 4.8), and the `BatchMemoryManager` bounds the physical batch size during per-sample gradient computation.

Three DP methods are compared, exactly the three of Section 4.5.1:

| Method | BatchNorm handling | Expected behaviour |
|---|---|---|
| DP-FedAvg (raw BN) | unmodified BatchNorm | refused by Opacus's `ModuleValidator` at finite $\varepsilon$ |
| DP-FedAvg+GroupNorm | BatchNorm replaced by GroupNorm | runs; the standard recommended workaround |
| DP-FedBN | BatchNorm frozen from DP, kept local | runs; the construction of this thesis |

The DP sweeps are run on the two most challenging MIT-BIH partitions (label skew $C = 2$ and Dirichlet $\alpha = 0.1$) and on the meaningful PTB-XL partition (Dirichlet $\alpha = 0.1$), spanning the three methods and three budgets. Because the privacy budget is a *target* that the accountant calibrates the noise to meet, the actually *achieved* $\varepsilon$ is recorded at the end of each finite-$\varepsilon$ run by querying the accountant; across all finite-$\varepsilon$ runs the achieved budget lands within $0.5\%$ of the target (for example, a target of 3 is achieved at approximately 2.99), confirming that the privacy guarantee is calibrated as intended rather than merely nominal.

## 5.5 Evaluation regimes and metrics

A central methodological commitment of this thesis is that **every run is evaluated under two regimes**, because the choice between them is itself one of the findings (Chapter 6):

- **Central (macro) F1** — the final model that is most representative of the federation is evaluated on a pooled, held-out central test set, and macro F1 is reported. For FedAvg, FedProx, and FedPerf, the representative model is the single global model; for FedBN, which has no single global model, both the first client's personalized model and the per-client average are evaluated. The central test set is a 15% stratified holdout for MIT-BIH and the canonical fold 10 for PTB-XL.
- **Local (macro) F1** — each client's final personalized model is evaluated on its own held-out validation slice, drawn from that client's own (non-IID) data distribution; the mean and standard deviation across the five clients are reported.

The distinction is exactly the pooled-versus-per-client distinction foreshadowed in Section 2.3: central F1 measures performance on the global data mixture, local F1 measures per-client performance on each client's own distribution. The primary metric in both regimes is **macro F1**, preferred over accuracy because of the severe class imbalance on MIT-BIH (Section 5.1.1); macro-averaged one-versus-rest AUC is reported as a secondary metric. For the central evaluation, the reported headline number for a run is the best macro F1 achieved across rounds; local F1 is measured at the end of training. Because FedBN has no single global model, its central evaluation is computed in two ways — the first client's personalized model evaluated on the central test set, and the average of all clients' personalized models each evaluated on it — and both are logged; the qualitative finding (Chapter 6) does not depend on which is chosen.

One caveat on the MIT-BIH central test set should be stated for completeness. It is a 15% stratified holdout drawn at the *beat* level, so beats from a given patient may appear in both the client pool and the central test set; this is the intra-patient evaluation setting discussed in Section 3.4, and it is less strict than a patient-disjoint inter-patient split. The PTB-XL central test set, being the canonical fold 10, respects the dataset's patient-aware fold assignment. This caveat is noted again among the limitations.

## 5.6 Implementation, reproducibility, and hardware

All federated experiments are run on a **custom, sequential, in-process simulator** rather than a full federated framework such as Flower. The choice is pragmatic: a minimal in-process simulator avoids the inter-process-communication overhead that would otherwise be incurred across the 100-plus runs in the experimental matrix. The four sweep scripts are resumable — a run is skipped if its results file already exists — which made the multi-day experimental campaign robust to interruption.

Reproducibility is pinned to a single random seed, 42, used everywhere. `torch.backends.cudnn.deterministic` is set to `True`, all random-number generators are seeded, and the git commit hash is logged with every run. Following the source paper, `cudnn.benchmark` is left `True` for convolution-kernel autotuning; this makes kernel selection non-deterministic *across* machines but stable on a given GPU and PyTorch combination, so results are reproducible on the same hardware. The software stack is PyTorch 2.4 and Opacus 1.5. The hardware was an NVIDIA GTX 1660 Super (6 GB) for the initial phases and an NVIDIA RTX 4070 from the PTB-XL cross-task replication onward.

One disclosure is important enough to state here as well as in the limitations. **All reported numbers are single-seed realizations** (seed 42). Quantifying the variance across seeds is the single most important deferred experiment, and it is discussed as a limitation in Chapter 7. The released code, configurations, and per-run logs make every number in this thesis reproducible on matching hardware.
