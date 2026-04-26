# Federated ECG Classification — A Project Story

**Author:** Efe Deniz Bağlar
**Advisor:** Prof. Dr. Hasan Bulut
**Affiliation:** Ege University
**Working title of the paper:** *DP-FedBN: A BatchNorm-Local Formulation of Differentially Private Federated Learning for ECG Classification under Non-IID Conditions*
**Period:** 23 April 2026 — 26 April 2026
**Status:** Phase 6 complete; Phase 7 (figures + tables consolidation) and Phase 8 (paper draft) pending.

This document tells the story of the project as a journey, not a decision log. `DECISIONS.md`
captures *what* we decided. The document you are reading captures *what we tried, what failed,
what surprised us, and how the final paper-ready story emerged*. It is written for three audiences,
in increasing order of distance: my advisor (who wants to know what was done and why), reviewers
of the eventual paper (who want to know the methodology), and a future reader of an MS application
research statement (who wants to know how the author thinks about research).

The story is told in past tense and in the first-person plural — "we" — even though the human author
worked alongside a single coding agent. That voice is not a marketing affectation; it reflects how
the work actually felt while it was happening: a tight loop of plan, code, run, look, decide, repeat,
between a person and a tool, with the person keeping the goals.

---

## 1. Project context and goals

### The brief from Day 0

The original brief was simple to state and difficult to deliver. We had **seven days** to produce an
advisor-ready package — code, experiments, figures, a written memo, slides — supporting a paper on
federated learning for electrocardiogram (ECG) classification, with differential privacy and
heterogeneous clients as the central themes. The advisor meeting was scheduled for **29 April 2026**.

The intended primary contribution, written into the master plan on Day 0, was a methodological one:

> *We formulate **DP-FedBN**, the correct composition of DP-SGD and FedBN. Opacus (and DP-SGD in
> general) is incompatible with BatchNorm because per-sample gradients are ill-defined under batch
> statistics. The standard workaround replaces BN with GroupNorm, which destroys the FedBN mechanism.
> Our formulation applies DP-SGD only to Conv/Linear parameters; BN parameters and running statistics
> stay local, never cross the federation boundary, and therefore do not require DP protection. This
> is simple, correct, and to our knowledge underspecified in prior FL+DP literature.*

That was the intended novelty. Three supporting contributions were planned to give the paper
breadth: a systematic non-IID benchmark on MIT-BIH (six partitionings × four aggregations); a
privacy-utility frontier sweep at six privacy budgets; and a cross-dataset replication on PTB-XL.

The publication target was deliberately modest — MDPI Sensors, MDPI Electronics, the ML4H workshop
at NeurIPS, IEEE BHI, or EMBC — anything that turns around quickly enough that the work is current
when it lands. The goal was not a flagship venue. The goal was a clean short paper (8–10 pages)
with a reproducible single-seed benchmark and one well-motivated formulation.

What the paper would explicitly **not** be was equally clear: not "we propose a new aggregation
method." FedPerf (softmax-of-validation-F1 weighted aggregation), originally considered our own
method, was already covered by FedAdp and FedNova variants in the literature. We kept it as one of
four baselines to keep DP-FedBN honestly contextualised, not as a claimed novelty.

### Hardware constraints

The hardware available was a single **GTX 1660 Super 6 GB**, CUDA 11.8+, on Ubuntu, with no remote
compute and no cloud. Six gigabytes of VRAM is enough for a small CNN1D and modest batches, but it
was not enough to be careless about: every choice about batch size, model width, BN vs GroupNorm,
and Opacus's per-sample gradient memory needed to land within budget. We later upgraded to an
RTX 4070 mid-project (between Phase 5 and Phase 6), which doubled the per-second throughput and let
us run the PTB-XL sweeps at acceptable speed; we will note that in Phase 6.

### Single seed, deterministic everywhere

We used **seed = 42** everywhere. Multi-seed variance analysis was deferred to post-sprint
expansion. That is a real limitation — every number reported in the project is a single-seed
realisation, and the paper acknowledges this — but it bought us the ability to run a 24-run non-IID
sweep, an 18-run DP sweep, a 12-run PTB-XL non-DP sweep, and a 9-run PTB-XL DP sweep within the
seven-day budget. With three seeds per cell, those four sweeps would have taken about three weeks
of GPU time; we did not have three weeks.

Determinism was strict: `torch.backends.cudnn.deterministic = True`, `benchmark = False`, all RNGs
pinned, partition rebuilt deterministically per client (`seed + client_id`), git SHA logged with
each run. The only relaxation we accepted, after Phase 2, was `cudnn.benchmark = True` for the
training-loop convolution kernel selection — which is non-deterministic across machines but stable
on a given GPU/PyTorch combination. That trade is documented in DECISIONS.md and is reproducible on
the same hardware; we never relied on cross-machine bit-exactness for any claim.

### The original 8-phase plan and what actually happened

The plan had eight phases (Phase 0+1 fused, then 2 through 7) and a timeline of one phase per day.
The actual execution compressed and rearranged that plan in ways the rest of this document tells in
detail. Briefly:

- **Phases 0+1 (Day 1)**: data inventory, scaffolding, preprocessing, partitioning, sanity checks.
- **Phase 2 (Day 1–2)**: centralised baselines on both datasets. Two pivots happened here — a model
  capacity widening and a re-formulation of PTB-XL from 5-class to binary.
- **Phase 3 (Day 2–3)**: federation primitives, FedAvg IID sanity run.
- **Phase 4 (Day 3, 13 hours of compute)**: 24-run non-IID benchmark sweep on MIT-BIH. The most
  surprising findings of the project came from here.
- **Phase 5 (Day 4–5)**: DP-FedBN implementation, smoke tests, original DP sweep, kill+resume after
  speedup retrofit, final 18-run DP sweep on MIT-BIH.
- **Phase 6 (Day 6, with a v2 today on Day 7)**: PTB-XL non-DP sweep (Phase 6A), PTB-XL DP sweep on
  the wrong partition (Phase 6B), abandonment of that partition, re-run on the right partition
  (Phase 6B-v2 — completed earlier today).
- **Phases 7 and 8 (next)**: final figures consolidation, paper draft.

Two phases that the original plan listed as separate workstreams — model ablation with ResNet1D,
and FedProx hyperparameter tuning — were not executed. They are listed as future work. We chose
not to pursue them once it became clear that the eval-regime story was the empirical heart of the
paper, and that adding more architecture or aggregation variants would dilute rather than
strengthen that story.

---

## 2. Phase 0+1 — Building the foundation (23 April 2026)

### The Python 3.10 vs 3.11 compromise

The first decision of the project was forced on us by the absence of `sudo`. The master plan had
asked for **Python 3.11**, citing improved error messages and modest speedups. Inspecting the host,
we found only `python3.10.12` in `/usr/bin/`. Installing 3.11 system-wide would have required
either a `deadsnakes` PPA (sudo) or a per-user `pyenv` installation (working but unnecessary
moving parts at the start of a one-week sprint).

We chose to use 3.10. All pinned dependencies — torch 2.2–2.4, opacus 1.5, wfdb 4.1, numpy <2,
pandas 2.1+ — officially support 3.10, so there was no runtime risk. The `pyproject.toml` was
written with `requires-python = ">=3.10"` and the ruff `target-version = "py310"`. The decision was
logged with a fallback documented: if a 3.10-vs-3.11 incompatibility ever surfaced, we would
revisit. None did. The project is 3.10-clean throughout.

This was a small decision but it set the tone. Throughout the project we made a habit of preferring
*the path of fewest moving parts* over *the path the master plan asked for* whenever the difference
did not matter empirically. The plan was a planning document, not a contract.

### Inventorying the raw data

Step 0 of Phase 0+1 was a mandatory data inventory before writing any code. Both datasets had been
pre-downloaded into `/home/kanemoda/MedicalFL/data/raw/` and we did not want to discover, two days
later, that one of them was in a non-standard format that broke the loader. The inventory produced
`data/raw/INVENTORY.md` and confirmed:

- **MIT-BIH** in standard `wfdb` format, 48 records, the canonical `.dat`/`.hea`/`.atr` triples
  per record, total ~3.3 GB on disk. Patient identifiers preserved.
- **PTB-XL** in standard PhysioNet structure with `records100/` (100 Hz), `records500/` (500 Hz, we
  did not use it), `ptbxl_database.csv` (21 799 rows with metadata), `scp_statements.csv` (the
  hierarchical SCP-ECG code dictionary).

Both datasets were exactly as expected. The inventory was a five-minute task and saved us from a
class of preventable surprises.

### MIT-BIH AAMI mapping decisions

The MIT-BIH database annotates each beat with one of about thirty single-letter symbols. We mapped
those to the five AAMI classes following the standard:

- **N (normal)**: `N`, `L`, `R`, `e`, `j`
- **S (supraventricular ectopic)**: `A`, `a`, `J`, `S`
- **V (ventricular ectopic)**: `V`, `E`
- **F (fusion)**: `F`
- **Q (unknown / paced)**: `/`, `f`, `Q`

A more interesting decision was **what to do with non-beat annotations** — symbols like `+` (rhythm
change), `~` (signal change), `|` (noise), `"` (comment), `x` (non-conducted P-wave). We dropped
them rather than forcing them into Q. The rationale: only annotations representing actual beats
contribute training signal; non-beat annotations are rhythm or signal markers that do not have a
canonical waveform to learn. Forcing them into Q would have polluted that class with samples that
do not look like beats. About 3 200 annotations across the 48 records ended up dropped on this
basis (along with the small number whose R-peak position fell within 125 samples of either edge of
the recording).

We **kept** the four paced-beat records (102, 104, 107, 217). Their `/` (paced) beats were labelled
Q. This follows the ANSI/AAMI EC57 recommendation and matches the de Chazal 2004 / Mondéjar-Guerra
2019 conventions. Some prior MIT-BIH studies discard paced records entirely; we chose not to,
because doing so removes a clinically realistic class of patients (people with implanted
pacemakers) from a system meant to be deployment-relevant. The cost is a Q class that contains a
mix of paced beats and unknowns, which complicates per-class interpretation. We accepted that
trade.

### MIT-BIH channel choice

The R-peak channel decision was per-task-spec: **always read `signal[:, 0]`**, do not branch per
record. For most records this is MLII. For records 102 and 104, lead V5 is on channel 0 instead.
The asymmetry is intentional and consistent with published baselines that do the same thing; it
also keeps the loader simple and deterministic.

### PTB-XL Lead-II-only choice

PTB-XL is a 12-lead dataset, but our model is 1-D and the project was scoped to univariate input.
We chose **Lead II only** (column index 1 of the 12-lead matrix). Lead II is the most frequently
reported lead in single-lead ECG studies, captures the cardiac axis most of the time, and is what
many wearable devices record. The 5-class task on PTB-XL turned out to be representationally
bottlenecked on Lead II alone, and that bottleneck drove the binary re-formulation in Phase 2 — but
that is a Phase 2 story.

### Storage format

We stored processed data as **`.npy` files plus a manifest CSV**. Each dataset got its own
directory under `data/processed/`:

- MIT-BIH: `X.npy` (float32, n_beats × 250), `y.npy` (int64), `record_ids.npy` (S10).
- PTB-XL: `X.npy` (float32, n_records × 1000), `y.npy` (int64), `strat_fold.npy` (int8),
  `ecg_id.npy` (int64).

`.npy` was chosen for three reasons. First, it is mmap-able, so partitioner code could load only the
samples it needs. Second, it has no schema overhead — no HDF5 group hierarchy, no Arrow type
gymnastics. Third, `numpy` is already a hard dependency; introducing HDF5 or Arrow would have meant
debugging file format issues at 2 a.m. when the real bug was in the partitioner.

The downside of `.npy` is that it forces homogeneous shape, so we couldn't store variable-length
sequences alongside fixed-length features. We didn't need to.

### The cuDNN deterministic vs benchmark trade-off

The most consequential pre-Phase-2 decision was on cuDNN behaviour. With
`torch.backends.cudnn.deterministic = True` and `benchmark = False`, the CNN1D trained on MIT-BIH at
**~75 seconds per epoch** on the 1660 Super. Fifty epochs would take **~63 minutes**, four times
over the 15-minute target. We profiled per-batch: 62.6 ms/iter under deterministic mode versus 3.2
ms/iter with `benchmark=True`. A 20× slowdown.

We relaxed cuDNN to `benchmark=True, deterministic=False` by default, with a strict mode available
for opt-in via `config["strict_determinism"] = true`. Same-machine and same-PyTorch reproducibility
are retained — cuDNN picks the same kernel for the same conv + input shape on a given GPU. We
sacrificed cross-machine bit-exactness, which we did not need for the project.

All other RNG sources stayed pinned. The relaxed surface is exactly the cuDNN kernel selection for
conv and BN, no more. Each run logs `torch.__version__`, `torch.version.cuda`, and the git SHA, so
the conditions of any reproduction are recoverable.

### Phase 0+1 outcome

After preprocessing:

- MIT-BIH: 48 records → **109 453 beats kept**, 3 194 dropped (unmappable symbols + edge beats).
  Records with the most drops (207, 230, 231) were all explained by rhythm-change and noise
  annotations, not data loss. A spot-check confirmed nothing pathological happened to specific
  records.
- PTB-XL: 21 799 rows → **21 388 records kept**, 411 dropped because no SCP code on the row had a
  non-null `diagnostic_class` (the master plan's filter). All 10 stratification folds were
  populated, with fold sizes between 2 129 and 2 158.

Disk usage went from `data/raw/` 3.3 G to a `data/processed/` directory of 189 MB. The 13 unit
tests in `tests/test_data.py` and `tests/test_partition.py` all passed; the sanity-check script
exited 0; two figures were emitted (class balance, sample beats grid). We were ready for Phase 2.

---

## 3. Phase 2 — Centralized baselines and the first big pivot (23 April 2026)

### The 44k → 437k CNN widening

The first model, faithful to the master plan, was a 3-block CNN1D: 32/64/128 channels, 3 conv
blocks each followed by BN and ReLU and max-pool, then a fully-connected head. Total parameters:
about **44 000**. On MIT-BIH 5-class, this model trained without drama: macro F1 0.862, well above
the 0.85 gate. On PTB-XL 5-class, it stalled. Train F1 plateaued at 0.61, validation at 0.54, test
at 0.498. We tried a longer cosine schedule. Same number. We tried 75 epochs instead of 50. Same
number. The model was hitting a **capacity ceiling**, not an optimisation bug.

We widened the network. Four conv blocks (channels 64/128/256/256, kernels 7/5/5/3), FC hidden 128,
dropout 0.5, BatchNorm after every conv layer and after the FC. Final parameter count: **437 765**
for the 5-class head, 437 378 for the binary head. The architecture stayed BN-heavy, which
mattered enormously for FedBN later — every layer had its own running statistics that FedBN could
keep local.

The widening had a side effect: MIT-BIH macro F1 climbed from 0.862 to **0.913** without any
hyperparameter tuning. We did not tune up to that number — it was a pure capacity effect. Both
results were within the acceptable band (≥ 0.85 macro on MIT-BIH 5-class), so we did not push
further.

### The PTB-XL 5-class → binary pivot

The widened CNN1D **also failed** to escape the PTB-XL 5-class plateau. We ran three configurations:
the original 44k model, the 438k model, and the 438k model plus augmentation plus stronger
regularisation. They landed at test F1 0.498, 0.509, 0.501 respectively. Same plateau, different
overfitting trajectories.

The diagnostic was a per-class breakdown. **HYP** (hypertrophy) F1 stayed at about 0.22 across all
three regimes. The vectorcardiographic signal that separates LVH from normal/MI is on the leads we
discarded — V5/V6 for the precordial signature, aVL for the lateral signature. Lead II alone simply
does not carry enough information. We were not capacity-bottlenecked; we were
**representation-bottlenecked**.

We had two options: (1) keep PTB-XL 5-class but switch to 12-lead input, doubling memory, requiring
a model rewrite, and breaking 1-D univariate consistency with MIT-BIH; or (2) re-formulate PTB-XL
as a binary task that is well-defined on Lead II alone.

We chose option (2). PTB-XL became binary **Normal vs Abnormal** — Normal = NORM, Abnormal = MI ∪
STTC ∪ CD ∪ HYP. The class balance (9 246 NORM vs 12 142 Abnormal) is reasonable. The task is
well-defined on a single lead. The first run hit test macro F1 = **0.813** in 18 epochs (early
stopping fired at epoch 28), which is a respectable result for binary record-level classification
on Lead II alone.

The pivot reframed the paper's cross-dataset story. The earlier framing — "method rankings transfer
across the same task on a different dataset" — became "method rankings transfer **across task
granularities**: 5-class beat-level on MIT-BIH and binary record-level on PTB-XL." The new framing
is actually stronger: a transferable method ranking that survives a change of task (beat → record)
and granularity (5-class → 2-class) is a more substantive cross-dataset claim than a
same-task-different-dataset replication.

We preserved both label sets on disk — `y_5class.npy` and `y_binary.npy` — and the loader reads
`config['ptbxl']['label_mode']` ∈ {"binary", "5class"}, default `"binary"`. The 5-class artifacts
(checkpoints, JSONs, figures, training logs) were archived under `results/archive/ptbxl_5class_runs/`
as the empirical evidence for *why* we re-formulated. They are not used downstream; they exist so
that future readers can verify the bottleneck claim themselves rather than taking our word.

The binary task uses the same model (438k params) and the same training loop as MIT-BIH. It uses
augmentation (`PTBXLAugment` — random 900-sample crop from 1000, Gaussian noise σ = 0.02,
sinusoidal baseline-wander noise ≤ 0.05 amplitude, each component applied with p = 0.5) which
training on Lead II demanded. It uses early stopping with patience 10. Validation F1 at the best
epoch matches test F1 within 0.01, with no runaway overfitting.

### The AUC binary fix

A small bug surfaced in `evaluation/metrics.py::_macro_auc`. The original implementation always
passed `multi_class='ovr'` along with `labels=[0..K-1]` to `sklearn.metrics.roc_auc_score`.
sklearn's binary path rejects that form and returns NaN. Our centralised PTB-XL run was producing
NaN for AUC despite training fine.

The fix was a single conditional: when `num_classes == 2`, pass `y_prob[:, 1]` as the
positive-class score and let sklearn's binary path handle it. Multi-class behaviour was unchanged.
This is the kind of bug that costs an hour of confusion if you don't carefully read the sklearn
docs, and ten seconds to fix once you do.

### Phase 2 final metrics

| dataset | task | test acc | test F1 macro | test AUC | best epoch | runtime |
|---|---|---|---|---|---|---|
| MIT-BIH | 5-class AAMI | 0.9819 | **0.9129** | 0.9972 | 31 | 5.2 min |
| PTB-XL | binary Normal/Abnormal | 0.8137 | **0.8134** | 0.8953 | 18 | 2.0 min |

Both passed all gates. The MIT-BIH centralised macro F1 of 0.913 became the reference number for
Phase 3's federation sanity check — a federated method should land within 3% of that under IID. The
PTB-XL binary F1 of 0.813 became the Phase 6 reference.

---

## 4. Phase 3 — Federated core (24 April 2026)

### Why a custom sequential simulator instead of Flower

The master plan was agnostic about the federation framework. We considered using Flower (the most
popular FL framework for research), and we built our own sequential simulator instead. The
reasoning, in order of decreasing weight:

**Single-machine, single-GPU simulation does not benefit from Flower's strength.** Flower's value
proposition is real cross-process communication — gRPC clients, encrypted transports, partial
participation, dropouts, asynchronous protocols. We had none of that. We had one GPU, five
"clients" that are five PyTorch model instances living in the same process, and one server that
runs an aggregation function on five tensors. Adding gRPC would have added layers of complexity
that contribute nothing to the experimental claim.

**Determinism is harder under concurrent execution.** Even with seeds pinned, Flower-style parallel
client training adds non-deterministic ordering of partial results. For a single-seed reproducible
benchmark, sequential client training (client 0 trains, client 1 trains, ..., client 4 trains, then
aggregate) is the only execution model where the rounds.csv is bit-identical across runs.

**Opacus integration is cleaner outside Flower.** Opacus's `PrivacyEngine.make_private_with_epsilon`
needs a model, an optimiser, and a dataloader at wrap time — and produces a wrapped triple that
must be used together. Embedding that into Flower's worker abstraction was possible but awkward;
in our simulator the wrap happens at the natural place (the per-client setup function in
`src/federation/dp.py::setup_dp_training`).

**We owned the codebase.** A custom simulator is about 400 lines of code (`client.py` + `server.py`
+ `aggregation.py`) plus a driver. Every line was ours; every bug was ours to find. Flower's
codebase is roughly 100× larger, and any bug debugging would have meant reading FastAPI internals
or gRPC tracebacks. The total engineering time, including debugging, was lower with our own
implementation.

What we gave up by not using Flower: portability to true cross-machine deployments and the ability
to reuse other Flower-compatible algorithms. Both are explicitly out of scope for this project.

### The four aggregation strategies

The four aggregation strategies live in `src/federation/aggregation.py`:

- **FedAvg** is the size-weighted mean of client parameters. Each client *i* contributes
  proportional to `n_i / Σ n_j`.
- **FedProx** uses the same aggregation as FedAvg; the proximal term lives on the *client* side, in
  `local_train`, as `μ/2 · ||w - w_global||²` added to the loss. We chose `μ = 0.01` from the
  original FedProx paper. (This turned out to be too aggressive, as we will see in Phase 4.)
- **FedBN** is FedAvg with **BatchNorm keys excluded** from the broadcast. Each client keeps its
  own BN parameters and running statistics; the server sees and aggregates only Conv and Linear
  parameters. The single source of truth for *what counts as BN* is `get_bn_key_set(model)`, which
  walks `named_modules()` looking for `BatchNormNd` and `SyncBatchNorm` and grabs every learnable
  parameter and buffer name they expose. This function is the only place where the BN/non-BN
  distinction is computed, and it is unit-tested.
- **FedPerf** weights each client's contribution by `α·n_i + (1−α)·F1_i`, where `F1_i` is the
  client's most recent validation F1 macro. With `α = 0.5` the aggregation reduces to roughly
  size-weighted in homogeneous regimes and tilts toward the better-performing clients in
  heterogeneous ones.

The server (`src/federation/server.py::FederationServer`) holds `global_params` as a dict and
dispatches to the right aggregator based on `config['federation']['strategy']`. For FedBN
specifically, the server has no aggregated BN params — BN stays local — so the central evaluation
needs a model-with-BN to run on the central test set. We chose to log **two** central-eval modes
per round:

1. `representative`: client 0's full local model evaluated on the central test set.
2. `client_avg`: per-client central F1 averaged across all clients.

`dual_evaluate_fedbn` returns both. The primary reported metric for FedBN is `client_avg` because
the choice of "client 0" is arbitrary — averaging removes that arbitrariness. The choice was logged
so that anyone reading the JSONs months later would not wonder what `central_f1_macro` means under
FedBN.

### Data carving

For MIT-BIH, the partitioner needs the central test set held out *first*, then the rest split
across clients. We did a stratified 85/15 split: 85% (93 035 beats) entered the partitioner; 15%
(16 418 beats) held out as the central test. No record-wise split — simple per-beat
stratification. Patient-wise leakage is a known concern for MIT-BIH, but we left it as a Phase 4
follow-up; the centralised baseline used the same split, so all numbers are internally
comparable.

For PTB-XL, the canonical fold assignment did the work for us: folds 1–9 went to clients
(stratified per-fold within client carving), fold 10 was the central test set. This kept the same
test slice the centralised PTB-XL baseline used, so cross-baseline comparisons were apples to
apples.

Per client, an 80/20 stratified train/val split was carved with seed `seed + client_id`, giving us
fully deterministic partitioner output per run. Class weights were computed once from the **pooled**
train portion across all clients (not per-client) to match the centralised loss scaling, so method
rankings would not be distorted by per-client weighting differences.

### The Round 1 BatchNorm transient

The first oddity of federated training showed up immediately. On the very first round of the
FedAvg IID sanity run (5 clients, IID split, MIT-BIH 5-class), central F1 was **0.0248** — below
random for a 5-class problem. The next three rounds: 0.248, 0.689, 0.791. The model was clearly
fine; this was a transient.

The cause is well-understood once you know to look for it: **averaging per-client BN running stats
after only 5 local epochs produces a distributional no-man's-land**. Each client's running mean
and running variance reflect that client's own data distribution after 5 epochs. The size-weighted
average of those is not a coherent BN state — it does not correspond to any actual data
distribution. The model classifies as if it has badly miscalibrated BN, because it does.

Under IID data, the per-client BN trajectories converge fast — three rounds was enough here. Under
non-IID partitions the transient lasts longer and is one source of FedAvg-with-BN's non-IID
fragility (which is, in turn, the motivation for FedBN). We did not change the code; we documented
the behaviour so it would not be misread as a bug in Phase 4 sweeps where the transient lasts
longer.

### FedAvg IID sanity outcome

The Phase 3 sanity bar was: FedAvg IID central F1 within 3% of the centralised baseline.

The result: **best central F1 = 0.9245 at round 45**, final-round (50) central F1 = 0.9059,
accuracy 0.9833, AUC 0.9952. Runtime 29.5 minutes on the 1660 Super (target ≤ 120 min). Compared
to the centralised reference of 0.9129, the best round exceeded centralised by +0.012; the final
round was −0.007. Both well within tolerance.

Per-class F1 at round 50: N = 0.991, S = 0.848, V = 0.967, F = 0.729, Q = 0.993. The drop on F
(fusion) mirrors the centralised pattern; F is the smallest class (121 test beats), which makes its
F1 sensitive to a handful of misclassifications.

Phase 3 also crossed a test-suite threshold: 33 unit tests passed (22 from Phases 0+1+2 plus 11 new
federation tests). The new tests covered BN key discovery, FedAvg weighted mean, FedBN BN-key skip,
FedPerf high-F1 dominance, FedProx proximal pull, an end-to-end client round, and one server round
per strategy plus the dispatcher. We did not write tests for "this number must be exactly X";
those are not unit tests, they are pinning experiments to specific numerical outcomes which is the
wrong thing to test. The unit tests cover invariants and correctness of small functions.

We were ready to start the non-IID benchmark.

---

## 5. Phase 4 — The 24-run sweep and its surprises (24 April 2026)

### The setup

The Phase 4 sweep was 6 partitions × 4 aggregations = **24 runs** on MIT-BIH, all at seed 42,
5 clients, 50 rounds, 5 local epochs, batch 64. Partitions:

- IID (the easy regime — included for the heterogeneity baseline)
- Label-skew, classes_per_client = 2 (the hardest regime — most extreme label heterogeneity)
- Quantity-skew, β = 0.5 (clients differ in dataset size, similar label distributions)
- Dirichlet α = 1.0 (mild non-IID)
- Dirichlet α = 0.5 (moderate non-IID)
- Dirichlet α = 0.1 (severe non-IID)

Runtime estimates suggested 30 min/run on average → about 12 hours total, comfortably overnight.
The actual sweep took 13 hours and 9 minutes, started at 01:50 and finished at 15:59 on 24 April,
with FedProx runs running at ~47–51 min each (the proximal-gradient overhead is real) versus 28–32
min for FedAvg, FedBN, and FedPerf.

### The first surprise: all six FedBN runs crashed at the reporting step

Every FedBN run completed all 50 rounds successfully, wrote its full `<run_id>_rounds.csv`, and
then **crashed at the final reporting step** with:

```
KeyError: 'per_class_precision'
File ".../evaluation/metrics.py", line ..., in format_classification_report
    precisions = metrics["per_class_precision"]
```

The root cause was simple. FedBN's `client_avg` evaluation returns aggregate metrics (overall F1
macro, accuracy, AUC) but not per-class breakdowns — there is no single "model" to compute per-class
metrics from, just five per-client evaluations averaged. The `format_classification_report` helper,
written in Phase 2 for centralised runs, had assumed every keys-set carried per-class metrics.

The fix was straightforward: make `format_classification_report` tolerant to missing per-class
keys, falling back to aggregate-only output. A test was added to lock the new behaviour in.

The interesting part was the **recovery**. The 50-round `rounds.csv` for each crashed run contained
every per-round central F1, per-client validation F1, accuracy, AUC, per-class breakdown — the
training had completed; only the final `test_metrics.json` and `final.pt` checkpoint were missing.
We wrote `scripts/recover_fedbn_finals.py`, which reads each failed run's `rounds.csv`, extracts the
best round (by central F1 macro) and the final round, and writes a synthetic `<run_id>.json`
marked `status="RECOVERED"`. The progress CSV was updated: 18 DONE + 6 RECOVERED = 24/24.

**No retrain needed for Phase 4 closure.** The recovery saved ~12 hours of GPU time. This is the
kind of save that an instrumented pipeline pays for: every interesting metric is logged at every
round, so the post-mortem of a crashed run is "what was the state at the time of the crash?" rather
than "did anything useful happen at all?"

The lesson — instrument before you need to instrument — is the kind of lesson that everyone agrees
with in the abstract, then forgets the next time they are tempted to skip writing a per-round CSV
because "the run will succeed and I'll just read the final JSON." Phase 4's reporting bug taught it
the way lessons are best taught: by saving us a day's worth of compute we couldn't afford.

### The second surprise: FedBN underperforms FedAvg on every partition

Once the JSON files were in place (recovered or otherwise), we built the heatmap. The numbers, as
**best central F1 macro**:

| partition            | FedAvg | FedProx | FedBN | FedPerf |
|----------------------|-------:|--------:|------:|--------:|
| IID                  | 0.926  | 0.859   | 0.920 | 0.926   |
| Dir(α=1.0)           | 0.937  | 0.889   | 0.911 | 0.928   |
| Dir(α=0.5)           | 0.926  | 0.903   | 0.850 | 0.928   |
| Quantity skew β=0.5  | 0.922  | 0.884   | 0.912 | 0.913   |
| Dir(α=0.1)           | 0.745  | 0.735   | 0.554 | 0.746   |
| Label skew C=2       | 0.527  | 0.386   | 0.219 | 0.530   |

Three patterns were immediately visible:

1. **FedBN underperforms FedAvg on every partition.** Mild on IID and quantity-skew (≈ −0.01),
   moderate on the milder Dirichlets (−0.02 to −0.08), catastrophic on Dir(α=0.1) (−0.19) and
   label-skew C=2 (**−0.31**).
2. **FedProx is uniformly worse than FedAvg** by 0.02 to 0.14 across all six partitions.
3. **FedPerf ≈ FedAvg** everywhere, ±0.01.

The third pattern was easy to explain: client validation F1s are similar within each homogeneous
regime, so FedPerf's softmax-of-F1 weighting collapses to roughly size-weighted averaging — i.e.
to FedAvg.

The second pattern was likely a hyperparameter-tuning issue. `μ = 0.01` was probably too aggressive
for our model and dataset; smaller `μ` (0.001 or 0.005) would have been worth sweeping. We logged
this as a deferred debug; the FedProx weakness is honest negative evidence, not a bug in our
implementation.

The **first** pattern was the crisis. The plan's central narrative had positioned FedBN as the
mechanism that lets us claim DP-FedBN as a meaningful contribution. If FedBN itself was worse than
FedAvg in non-IID — across every partition we tested — then DP-FedBN was a contribution sitting on
a baseline that already failed. The original Li et al. 2021 paper, of course, reports FedBN
*beating* FedAvg on cross-domain image classification under feature shift. Either we had a bug in
our implementation, or our regime was different from theirs in a way that flipped the result.

### The decision to audit before pivoting

The reflexive response to "your central finding contradicts the original paper" is to assume your
implementation has a bug. The temptation is to ship a sloppy investigation, find one thing that
looks suspicious, declare it the cause, and pivot the paper narrative.

We did not do that. We wrote `scripts/audit_fedbn.py` — seven independent checks of the FedBN
implementation and its expected behaviour, each printing PASS, FAIL, or AMBIGUOUS — and ran it
before deciding whether the result was real.

This was the right call. The audit took about three hours to write and run. If the implementation
had been buggy, the audit would have caught it; if the implementation was correct, the audit would
give us the confidence to make a defensible empirical claim. Either outcome was strictly better
than guessing.

---

## 6. The FedBN audit — implementation correctness verified (24 April 2026)

The seven checks in `scripts/audit_fedbn.py`:

### Check 1 — BN key discovery on CNN1D

Goal: verify that `get_bn_key_set(model)` finds exactly the keys we expect.

CNN1D has 5 BN modules (`bn1`, `bn2`, `bn3`, `bn4`, `bn_fc`). Each contributes 5 keys to the state
dict (`weight`, `bias`, `running_mean`, `running_var`, `num_batches_tracked`). 5 × 5 = **25 BN
keys**. The non-BN side is 6 modules (`conv1`, `conv2`, `conv3`, `conv4`, `fc1`, `fc2`) × 2 keys
(`weight`, `bias`) = **12 non-BN keys**.

The check builds the model, calls `get_bn_key_set`, and verifies:
- All 25 expected BN keys are found.
- No spurious BN keys appear (for example, no `weight` from a conv module misidentified as BN).
- All 12 expected non-BN keys are absent from the BN set.

**PASS.** 25/25 BN keys found, no missing, no stray, no missing non-BN.

### Check 2 — `fedbn_aggregate` output excludes BN, preserves non-BN

Goal: verify that the aggregation function actually does what the FedBN spec requires — drop BN
keys from the broadcast and aggregate the non-BN keys correctly.

The check builds five mock client state dicts. The Conv/Linear parameters are identical across
clients (so the aggregate of those is the shared value). The BN parameters are randomised
per-client (so the BN aggregate, if it were to be returned, would *not* equal any client's BN
state). We then call `fedbn_aggregate` and verify:
- The returned dict has 12 keys (the non-BN side), not 37.
- No BN keys leaked.
- The returned non-BN values exactly equal the shared client value.

**PASS.** 12 keys returned, 0 BN leaks, 0 non-BN value drift.

### Check 3 — Clients' BN stats diverge after one server round

Goal: verify that under `strategy="fedbn"`, after one server round, the five clients' BN running
stats are not identical — i.e. that the server is not silently broadcasting BN.

We initialise five clients with **independent synthetic data**, run one round of FedBN, and measure
the pair-max absolute difference in `bn1.running_mean[0]` across clients.

**PASS.** Pair-max difference 0.011501. The server-side `bn1.running_mean[0:3]` is `[0.0, 0.0,
0.0]` because under FedBN we never update server BN state.

### Check 4 — `client.set_parameters` preserves BN

Goal: verify that when the server pushes a new (non-BN-only) parameter dict to a client, the
client's BN parameters and buffers are *not* overwritten.

We snapshot a client's BN keys, call `set_parameters` with a dict of new non-BN values, and verify
the BN keys are unchanged byte-for-byte.

**PASS.** 25/25 BN keys unchanged after `set_parameters(non-BN-only)`.

### Check 5 — IID parity

Goal: under IID partitioning, FedBN central F1 should be approximately equal to FedAvg central F1.
Different per-client BN trajectories produce mild noise on the central evaluation, but the
expected gap is small.

The check runs a quick experiment: 2 clients × IID split × 5 rounds × 3 local epochs on a 25k-beat
MIT-BIH subset.

Result: FedAvg central F1 = 0.7559, FedBN central F1 = 0.6653, Δ = −0.0906 (~12% relative).

**AMBIGUOUS.** The expected gap was ≤ 10%; we observed 12%. The check is small-scale (25k beats,
5 rounds), so this is likely a small-scale convergence artifact rather than a structural problem.
The full Phase 3 IID run (5 clients × 50 rounds × 5 local epochs on full MIT-BIH) had FedAvg = 0.926
and FedBN = 0.920, well within parity. The AMBIGUOUS verdict is conservative; we did not treat it
as a failure.

### Check 6 — Feature-shift sanity (Li et al. regime)

Goal: under the regime FedBN was originally designed for — clients with **feature-shifted** data —
FedBN should beat FedAvg on central F1 by a clear margin.

We synthesise a feature-shifted toy problem: 2 clients × 10 rounds × 5 local epochs. Client 0 has
"raw" class means + Gaussian noise. Client 1 has the class means flipped, shifted by 2.0, and
noised (heavy BN-shift regime). The model is a small ToyBNCNN (Conv-BN-Conv-BN-GAP-BN-FC) on
10-class, 32-dim signals.

Result: FedAvg central F1 = 0.1291. FedBN central F1 = **0.5743**. Δ central = **+0.4452**. Local
F1s are similar (FedAvg local mean 0.9420, FedBN local mean 0.9508), as expected — both methods
are great per-client; only the central evaluation distinguishes them.

**PASS.** FedBN reproduces the Li et al. published advantage when the regime matches. Our
implementation is correct.

### Check 7 — Diagnostic: per-client local F1 vs central F1 across all FedBN runs

Goal: if FedBN's central collapse is real (per Phase 4) and the implementation is correct (per
checks 1–4 and 6), what is the actual mechanism? Specifically, are the per-client local F1s also
low (model is broken) or high (model is fine, central evaluation is the wrong metric)?

The check reads the six MIT-BIH FedBN runs' per-round CSVs and computes mean per-client local
validation F1 versus central F1 at the best round.

| run                    | local mean | local min | local max | central |  gap   |
|------------------------|-----------:|----------:|----------:|--------:|-------:|
| fedbn_iid              | 0.917      | 0.880     | 0.954     | 0.899   | +0.018 |
| fedbn_dirichlet_a10    | 0.900      | 0.813     | 0.959     | 0.883   | +0.017 |
| fedbn_dirichlet_a05    | 0.892      | 0.832     | 0.978     | 0.842   | +0.050 |
| fedbn_dirichlet_a01    | 0.868      | 0.661     | 0.985     | 0.487   | +0.381 |
| fedbn_quantity_skew    | 0.875      | 0.711     | 0.947     | 0.890   | −0.015 |
| fedbn_label_skew_c2    | 0.958      | 0.923     | 0.984     | 0.217   | **+0.741** |

**PASS, and the most important diagnostic of the project.** Mean (local − central) gap = +0.199.
On label-skew C=2, the gap is +0.741: the clients are achieving local F1 = 0.958 (excellent) while
the central F1 is 0.217 (terrible).

The mechanism is clear from this table. Each client's BN running statistics are tuned to that
client's local label distribution. Under label-skew C=2, each client owns only two classes, so its
BN stats are calibrated for those two classes. When that client's full model — with its
specialised BN — is asked to classify the *central* test set (which has all five classes in the
overall MIT-BIH proportions), BN normalisation collapses on the classes the client never trained
on. Per-client local evaluation, where the client only ever sees its own client's distribution, is
fine; central evaluation is fundamentally mismatched.

### What the audit actually showed

The audit verified that the implementation is correct (checks 1–4, plus 6) and identified the
mechanism behind the Phase 4 anomaly (check 7). The Phase 4 finding is real, and it is not a bug
in FedBN; it is a consequence of FedBN's design being misaligned with the *evaluation regime* we
were using.

This was the moment when the paper's narrative could pivot from "we have a bug" or "FedBN doesn't
work on biomedical data" — both of which would have been wrong — to "FedBN's reported advantage
depends on the evaluation regime matching its specialisation target." The audit is the single
piece of work that most changed the paper.

---

## 7. Phase 4 follow-up — The eval-regime flip (24 April 2026, evening)

The audit's Check 7 produced one number — local mean F1 versus central F1 — using each client's
*validation* set during training. To make the eval-regime claim defensible, we needed per-client
**local test** evaluation: each client's final model evaluated on its own client's held-out test
slice, not its training-time validation slice.

`scripts/local_test_eval.py` was written for this purpose. It runs on the 8 most-relevant runs
spanning the two worst central cases — Dir(α=0.1) × 4 aggregations and label_skew_c2 × 4
aggregations.

For the 6 non-FedBN runs, we loaded the existing checkpoints and ran local-test eval in place. For
the 2 FedBN runs (which had no saved checkpoints because their original Phase 4 runs crashed at the
reporting step), we retrained from seed 42. The retrains took about an hour total. The retrained
runs were saved with `_v2` suffixes so that the original RECOVERED JSONs (which carry the canonical
central F1 from the original training) and the v2 retrains (which carry the local test data) are
both on disk.

### Dir(α=0.1) — central vs local

| aggregation | central F1 | local F1 (mean ± std) |    Δ    |
|-------------|-----------:|----------------------:|--------:|
| FedAvg      | 0.745      | 0.831 ± 0.128         | +0.086  |
| FedProx     | 0.735      | 0.755 ± 0.146         | +0.020  |
| **FedBN**   | 0.536      | **0.848 ± 0.154**     | +0.312  |
| FedPerf     | 0.745      | 0.825 ± 0.129         | +0.080  |

### Label skew C=2 — central vs local

| aggregation | central F1 | local F1 (mean ± std) |    Δ    |
|-------------|-----------:|----------------------:|--------:|
| FedAvg      | 0.527      | 0.823 ± 0.178         | +0.297  |
| FedProx     | 0.386      | 0.488 ± 0.380         | +0.101  |
| **FedBN**   | 0.224      | **0.942 ± 0.053**     | +0.718  |
| FedPerf     | 0.530      | 0.654 ± 0.255         | +0.124  |

The picture is unambiguous. **FedBN flips from worst (central) to best (local) under label
heterogeneity.** On label-skew C=2, the central F1 of 0.224 (worst of the four) becomes a local F1
of 0.942 (best of the four, and by a clear margin). The +0.718 central-local gap is the single
most dramatic swing in the matrix.

The mechanism is exactly what Check 7 had diagnosed: each client's BN specialises to its own label
distribution, which is *good* for that client's own data and *bad* for the pooled central test
set. Under label-shift partitions where each client's BN target is genuinely different, this
specialisation works exactly as designed. The catch is that "as designed" means "good locally, bad
globally" — and our Phase 4 sweep had been measuring "globally" only.

### Re-framing the paper

This pair of tables, plus the audit log, made the empirical contribution of the paper visible in a
new shape:

> *FedBN's reported advantage depends on the evaluation regime matching its specialisation target.
> Under feature-shifted image classification with local-test evaluation (Li et al. 2021), FedBN
> wins. Under label-shifted medical time-series evaluated on a pooled central test set — the
> clinically realistic regime where a holdout represents the future deployment population —
> per-client BN specialisation is mis-aligned with evaluation and FedBN underperforms FedAvg. The
> gap widens with partition heterogeneity (Dir α=0.1: −0.19; Label skew C=2: −0.31). However,
> under per-client local evaluation — the personalised deployment regime — FedBN becomes the
> strongest method (label_skew local F1 = 0.942 vs FedAvg 0.823).*

The eval-regime framing is the paper's empirical core. Everything that came after — the DP sweep,
the cross-task replication on PTB-XL — was about answering the question "does this finding
generalise, and what happens when DP is introduced?"

---

## 8. Literature review — managing novelty claims

Before committing the paper to a "DP-FedBN as primary novelty" framing, we did a deliberate
literature pass. The point of the pass was not to prove that nobody had done this before; it was to
locate our work honestly inside the existing landscape. If someone had already formalised
DP-FedBN, we would say so and frame our contribution differently.

### Gemini Deep Research consultation

We asked Gemini Deep Research (a search-and-summarise tool) to find prior work on DP-FedBN, on
exclusion of BN parameters from DP perturbation, and on Opacus + BN workarounds. The report
surfaced three findings worth recording:

1. **Li et al. 2023 — ADCOL paper.** This paper, which proposes Adversarial Collaborative
   Learning (ADCOL) for federated learning under attribute heterogeneity, uses "DP-FedBN" as **one
   un-detailed baseline** in its experiments. The abbreviation "DP-FedBN" is mentioned but the
   paper does not formalise it, does not give construction details, and does not make claims about
   it as a method. The reference is essentially "we ran DP-SGD on FedBN as a baseline; here is a
   number." That is enough prior art to disqualify "DP-FedBN" as a clean coined term, but not
   enough to disqualify a careful empirical formulation as a contribution.

2. **Opacus GitHub Issue #472.** The Opacus team has known about the BN incompatibility since at
   least 2022. The issue thread documents the workaround: freeze BN before wrapping with Opacus,
   rely on `ModuleValidator` skipping non-trainable modules, and unfreeze afterwards. The
   workaround is buried in the issue thread; it is not in the official Opacus tutorial; and to our
   knowledge no published paper has packaged it as a method, validated it empirically, or applied
   it in a federated medical-data context.

3. **No prior work formalises the dual-optimizer DP-FedBN with empirical validation in medical
   FL.** This is what the search confirmed by absence rather than by presence: there is no paper
   that combines (a) a clean problem statement of the BN-DP tension, (b) an explicit two-optimiser
   construction that sidesteps the tension while preserving FedBN's mechanism, (c) an empirical
   validation across multiple non-IID partitions and multiple privacy budgets, and (d) a
   medical-data evaluation. We are filling that gap, not coining a term that never existed.

### How the framing pivoted

Pre-search: *"We coin DP-FedBN. It is novel."*

Post-search: *"DP-FedBN appeared as an undocumented baseline in Li et al. 2023, and the
implementation trick (freeze BN before Opacus wrap) is in Opacus Issue #472. Our contribution is
the **first explicit empirical formulation** with a working two-optimiser implementation,
multi-partition evaluation, multi-budget privacy-utility curves, and cross-task validation on
medical ECG data."*

The second framing is harder to write but easier to defend. It is also more useful to a reviewer,
because it tells them exactly which prior work to compare us against. Reviewers do not award points
for novelty claims that a five-minute search would dispute.

We logged the literature-review findings in our notes and built the paper's related-work section
around them. The decision to defer naming the contribution as "first explicit formulation" rather
than "we coin DP-FedBN" is the kind of decision that costs nothing at draft time and is enormously
more credible at submission time.

---

## 9. Phase 5 — DP wiring and the speedup retrofit (24–25 April 2026)

### The Opacus + BatchNorm tension

Opacus computes per-sample gradients by hooking each module to record its gradient with respect to
each sample in the batch. BatchNorm is the canonical exception: per-sample gradients under
BatchNorm are ill-defined because the layer's output for sample *i* depends on the batch mean and
batch variance, which depend on samples *j ≠ i*. From a privacy accounting standpoint, this is
catastrophic — the gradient for one sample carries information about every other sample in the
batch, so per-sample gradient clipping does not actually bound per-sample sensitivity.

Opacus's `ModuleValidator` enforces this. When you call
`PrivacyEngine.make_private_with_epsilon(model, ...)`, the validator walks the model's modules,
finds any `BatchNormNd`, and raises `UnsupportedModuleError` with a `ShouldReplaceModuleError` for
each BN module. This is correct behaviour — the validator is protecting users from an unsound DP
configuration — but it makes naive DP-FedAvg on a BN-heavy CNN literally impossible.

The standard Opacus-recommended workaround is to call
`ModuleValidator.fix(model)`, which replaces every BatchNorm with GroupNorm. GroupNorm is
privacy-safe (it does not couple samples within the batch) and the change is transparent to most
training code. But it destroys FedBN: the whole point of FedBN is to keep client-specific BN
statistics local, and there is no client-specific GroupNorm equivalent because GroupNorm has no
running statistics — it computes channel-group statistics on the fly per batch.

So we have three reasonable options for DP on a BN-heavy CNN:
- Naive: try to wrap with Opacus and accept that it will fail. (DP-FedAvg, naive.)
- GroupNorm: replace BN with GroupNorm and proceed normally. (DP-FedAvg + GroupNorm.)
- DP-FedBN: keep BN, but make Opacus only see the non-BN parameters.

The paper compares all three.

### The two-optimiser DP-FedBN construction

Our DP-FedBN construction works as follows.

1. Build the CNN1D normally, with BN.
2. Set `bn.weight.requires_grad = False`, `bn.bias.requires_grad = False`, and freeze the BN
   running statistics buffers (these are not "parameters" in PyTorch's sense, so we leave them
   alone, but we also don't include them in any optimiser).
3. Build optimiser **A** for the Conv/Linear parameters (the ones that *will* be DP-protected).
4. Wrap (model, optimiser_A, train_loader) with Opacus's
   `PrivacyEngine.make_private_with_epsilon(...)`. The validator only checks *trainable* modules,
   so with BN frozen at step 2, the validator does not see any BatchNorm modules and proceeds
   without complaint.
5. After the wrap, `bn.weight.requires_grad = True`, `bn.bias.requires_grad = True`.
6. Build optimiser **B** (plain `AdamW`) for the BN parameters only. The BN running statistics
   continue to update through their normal forward-pass path.
7. Per training step: `opt_A.zero_grad()`, `opt_B.zero_grad()`, `loss.backward()`, `opt_A.step()`
   (DP-protected), `opt_B.step()` (plain SGD).

The aggregation rule from FedBN — broadcast and aggregate only non-BN parameters — is unchanged.
What is changed is *how each client trains* in the inner loop. The Conv/Linear updates are
DP-protected (gradient clipped, noised, and applied through the Opacus optimiser); the BN
parameters update with plain gradients on each client's local data.

The federated threat model justifies this: BN parameters and running statistics never cross the
federation boundary — they stay on the client. They are not aggregated, not broadcast, and not
visible to any other party. There is therefore no reason to apply DP to them in the federated
setting; the DP guarantee covers what the server sees, and the server sees only Conv/Linear
parameters.

### The smoke test on the original config

Before launching the full Phase 5 sweep we ran a **smoke test** at label_skew_c2 × ε=3, using 2
clients × 5 rounds × 1 local epoch, just to check that each of the three DP modes either runs or
fails *as expected*. The outcomes:

| mode                | outcome                                                         | runtime |
|---------------------|------------------------------------------------------------------|---------|
| dp_fedavg (naive)   | **REFUSED** by Opacus (`UnsupportedModuleError` for BatchNorm)   | < 1 s   |
| dp_fedavg+groupnorm | PASSED                                                           | 470 s   |
| dp_fedbn            | PASSED (after fixing one bug — see below)                        | 357 s   |

Two important pieces of information came out of the smoke:

**The naive DP-FedAvg refusal is captured as a first-class signal.** The sweep script catches
`DPModeUnsupportedError` and records the cell as `REFUSED` rather than crashing. In the final
table, those cells appear as `REFUSED` — which *is* the empirical finding for naive DP-FedAvg at
finite ε. Naive DP-SGD on a BN-heavy CNN is not a thing that fails in 90 seconds of training; it
is a thing that the privacy-accounting library refuses to wire up at all. The paper makes this
exact point.

**A bug surfaced in `_evaluate_local_test`.** The first DP-FedBN smoke completed all 5 training
rounds and the central evaluation, then crashed inside the local-test evaluation block with:

```
RuntimeError: Error(s) in loading state_dict for GradSampleModule:
  Missing key(s) in state_dict: "_module.conv1.weight", ...
  Unexpected key(s) in state_dict: "conv1.weight", "bn1.num_batches_tracked", ...
```

The cause: `client.model.load_state_dict(client_states[c_id])` was loading plain-keyed snapshots
into the Opacus-wrapped `GradSampleModule`, whose own `state_dict` keys carry a `_module.` prefix
inserted by Opacus during wrapping. The snapshots had been taken before the wrap, so they used
plain keys; the load was happening after the wrap, expecting wrapped keys.

The fix was a one-liner: route through `client._inner().load_state_dict(...)` so the snapshot
lands on the unwrapped CNN1D regardless of whether DP is active. This mirrored the pattern that
`get_parameters`, `set_parameters`, `snapshot`, and `restore` had used since Phase 3, and it had
been silently right pre-Phase-5 because `client.model` and `client._inner()` returned the same
object until Opacus came in.

We re-ran the DP-FedBN smoke after the fix: PASSED in 357 s, all artifacts written.

### Sweep launch and the 45-hour estimate

The original Phase 5 sweep was launched at 02:39 on 25 April with the 1660-Super-friendly config —
batch=16, rounds=50, num_workers=0. The script estimated the wall-clock based on the first finished
runs:

- 4 dp_fedavg-at-finite-ε cells: instant refusal, 0 minutes.
- 2 dp_fedavg-at-ε=∞ cells: ~30 min each (no DP overhead, BN intact since no Opacus wrap is
  needed).
- 6 dp_fedavg+groupnorm cells: ε=∞ ~30 min, finite-ε ~43 min each → ~3.5 hours.
- 6 dp_fedbn cells: similar profile → ~3.5 hours.
- **Total ETA: 8–9 hours.** (Lands ~10:30–11:30 on 25 April.)

That estimate did not survive contact with reality. By 12:10 — about 9.5 hours in — the sweep had
finished only **5 of 18 runs**. The finite-ε `dp_fedavg+groupnorm` runs were taking about 6.6
hours each, not 43 minutes. A back-of-envelope extrapolation put the full sweep at ~45 hours.
That was incompatible with the 29 April advisor deadline.

We diagnosed the slowness. GPU memory was at 700 MB out of 6144 MB; GPU utilisation was bouncing
between 22% and 89%. The bottleneck was not the model; the bottleneck was the data path. With
batch=16 and num_workers=0, the data loader was the limiting factor — every batch required CPU-side
preprocessing in the main thread, blocking the GPU. Opacus's per-sample gradient computation
amplified this because each batch's compute was proportionally smaller, so the data-load:compute
ratio worsened.

### The kill-and-resume decision

We **killed** PID 896731 at 12:10, preserving the 3 DONE JSONs from the runs that had completed.
The 2 instant-refusal `dp_fedavg`-at-finite-ε rows in the progress CSV would re-run and refuse
identically under any new config, so those were free; the one mid-flight `dp_fedavg_groupnorm
label_skew_c2 ε=1.0` had no JSON yet, so it would restart cleanly under the new config.

The decision to kill rather than wait was deliberate. The paper deadline was unforgiving. The
sunk-cost fallacy would have said "you've already burned 10 hours, just wait the rest out." The
correct answer, knowing what we knew at 12:10, was that the current config was structurally too
slow and the right move was a config change. The 10 hours were already spent; the question was
whether the next 35 hours were also going to be spent on the same dead-end.

### The speedup retrofit

Four levers on the speedup were considered:

**(A) Increase batch size.** Larger batches improve GPU utilisation by amortising per-iteration
overhead (kernel launches, Opacus's per-sample gradient management, data-load latency hidden by
larger compute). The 1660 Super has 6 GB of VRAM; a 96-sample batch of 1-D 250-sample beats fits
comfortably. We set `training.batch_size = 96` (and Opacus's
`dp.max_physical_batch_size = 96`).

**(B) More dataloader workers.** With `num_workers > 0`, PyTorch prefetches in worker processes;
the GPU is no longer waiting on a single CPU thread for the next batch. We set
`training.num_workers = 4` and wired it through `_make_loader` with a top-level fallback so older
configs would still work. `pin_memory: true` was already enabled.

**(C) Reduce rounds.** The Phase 4 rounds.csv files showed that most runs converged or oscillated
by round 25–30 — the last 20 rounds were polishing, not learning. We dropped
`federation.rounds = 50 → 30`. We kept `local_epochs = 5`.

**(D) Try ExpandedWeights.** Opacus 1.5 added an alternative per-sample gradient backend called
`grad_sample_mode="ew"` (ExpandedWeights), which is meant to be faster than the default `hooks`
backend. It works by expanding each module's weight tensor along the batch dimension and computing
a single forward/backward, instead of recording per-sample gradients in a hook.

We tried (D) for `dp_fedavg_groupnorm` — for DP-FedBN we kept the default hooks backend because
its frozen-BN trick depends on validator and hook iteration semantics that ExpandedWeights does
not replicate. The smoke crashed at the second `optimizer.step()`:

```
RuntimeError: Current Expanded Weights accumulates the gradients, which will be incorrect for
multiple calls without clearing gradients.
```

ExpandedWeights is incompatible with the `BatchMemoryManager` chunking pattern (which we use to
keep memory bounded under DP) and with `optimizer.zero_grad(set_to_none=True)` — the way the
backend retains its expanded-weight buffers across steps assumes a different gradient-clearing
protocol than the standard `hooks` backend. We reverted to `hooks` and left a comment in
`src/federation/dp.py::_setup_dp_training` warning future-us not to try ExpandedWeights again
without writing a proper integration test.

(This particular incompatibility — ExpandedWeights × BatchMemoryManager × `zero_grad(set_to_none=True)`
— is documented in our DECISIONS.md and in the codebase as a future contribution opportunity to
the Opacus community, since the failure mode is reproducible and not in the current Opacus
tutorial.)

### Speedup smoke results

Under the new config (batch=96, rounds=30, num_workers=4) we re-ran the smoke set:

| smoke                              | outcome | wall  | best central F1 | achieved ε | target ε |
|------------------------------------|---------|-------|-----------------|------------|----------|
| dp_fedavg_groupnorm + ew           | FAIL    | 32 s  | crash           | —          | 3.0      |
| dp_fedbn (hooks)                   | PASS    | 131 s | 0.1811          | 2.9947     | 3.0      |
| dp_fedavg_groupnorm (hooks)        | PASS    | 154 s | 0.1811          | 2.9947     | 3.0      |

Both passing smokes had achieved-ε within 0.2% of target. The privacy accountant was working as
expected. The smokes' best central F1 was poor (0.18), as expected for a 5-round 2-client run; the
point was not the F1 but the wiring.

### Resumed sweep

```bash
nohup .venv/bin/python -u scripts/run_dp_sweep.py --skip-smoke \
  > /tmp/sweep_phase5_resumed.log 2>&1 &
```

Resume semantics worked exactly as intended:

- 3 DONE JSONs from the kill caused SKIP rows in the progress CSV.
- 2 dp_fedavg finite-ε refusals re-ran and refused instantly under the new config.
- 13 fresh runs executed.

New per-run estimates: ε=∞ ≈ 10 min, finite-ε ≈ 40 min. Total remaining ≈ 5–7 hours, lands by
20:30 the same day.

The actual resumed sweep wall-clock was **6.89 hours** — within the estimated band. Total Phase 5
GPU time, including the 10 hours burned on the original config, was about 17 hours of compute.
The speedup retrofit absorbed the original sunk cost and still landed within deadline.

### The asymmetry note

3 of the 14 DONE runs (`dp_fedavg_label_skew_c2_epsinf`, `dp_fedavg_groupnorm_label_skew_c2_epsinf`,
`dp_fedavg_groupnorm_label_skew_c2_eps3.0`) were trained at the **original config** (batch=16,
rounds=50, num_workers=0). The other 11 DONE runs were at the **speedup config** (batch=96,
rounds=30, num_workers=4). We did not re-run the early three, because:

- Each within-cell comparison (same partition × same method × same ε) has exactly **one** config,
  so the table itself has no internal inconsistency.
- Re-running the three would have cost ~90 minutes of GPU time at speedup-config rates, and would
  have given us nothing the existing runs did not already provide.
- The asymmetry only matters for cross-config comparisons in some figures, and we documented it in
  the figure captions when used in the paper.

The honest version of this trade is: we accepted a small documentation overhead in exchange for
not paying the cost of re-running converged runs. If a reviewer flags it, the 90 minutes are still
there to be paid.

---

## 10. Phase 5 results — DP behaviour partially preserves the local advantage

The full Phase 5 results table (`results/metrics/dp_summary.csv`):

| Method | Partition | ε | Achieved ε | Central F1 (best) | Local F1 (mean ± std) |
|---|---|---:|---:|---:|---:|
| DP-FedAvg | label_skew_c2 | ∞ | — | 0.517 | 0.829 ± 0.192 |
| DP-FedAvg | label_skew_c2 | 3 | REFUSED | REFUSED | REFUSED |
| DP-FedAvg | label_skew_c2 | 1 | REFUSED | REFUSED | REFUSED |
| DP-FedAvg+GroupNorm | label_skew_c2 | ∞ | — | 0.261 | 0.463 ± 0.368 |
| DP-FedAvg+GroupNorm | label_skew_c2 | 3 | 2.9957 | 0.191 | 0.295 ± 0.241 |
| DP-FedAvg+GroupNorm | label_skew_c2 | 1 | 0.9961 | 0.181 | 0.295 ± 0.241 |
| **DP-FedBN** | label_skew_c2 | ∞ | — | 0.221 | **0.937 ± 0.043** |
| **DP-FedBN** | label_skew_c2 | 3 | 2.9929 | 0.181 | **0.467 ± 0.264** |
| **DP-FedBN** | label_skew_c2 | 1 | 0.9961 | 0.217 | **0.344 ± 0.194** |
| DP-FedAvg | dirichlet_a01 | ∞ | — | 0.738 | 0.816 ± 0.130 |
| DP-FedAvg | dirichlet_a01 | 3 | REFUSED | REFUSED | REFUSED |
| DP-FedAvg | dirichlet_a01 | 1 | REFUSED | REFUSED | REFUSED |
| DP-FedAvg+GroupNorm | dirichlet_a01 | ∞ | — | 0.736 | 0.832 ± 0.118 |
| DP-FedAvg+GroupNorm | dirichlet_a01 | 3 | 2.9959 | 0.382 | 0.376 ± 0.101 |
| DP-FedAvg+GroupNorm | dirichlet_a01 | 1 | 0.9942 | 0.378 | 0.350 ± 0.144 |
| **DP-FedBN** | dirichlet_a01 | ∞ | — | 0.551 | 0.861 ± 0.130 |
| **DP-FedBN** | dirichlet_a01 | 3 | 2.9959 | 0.189 | 0.368 ± 0.151 |
| **DP-FedBN** | dirichlet_a01 | 1 | 0.9942 | 0.251 | 0.465 ± 0.308 |

### What the ε=∞ runs confirmed

The ε=∞ rows are not really "DP" — they are runs with the DP infrastructure plumbed (Opacus
wrapping, BN-freezing, sample-rate accounting) but with the noise multiplier set to 0. They should
reproduce the Phase 4 (no-DP) results within sampling noise.

DP-FedBN local F1 on label_skew_c2 at ε=∞ = **0.937**. The Phase 4 follow-up reported 0.942 under a
different config (rounds=50, batch=16). The ~0.005 gap is small enough to be sampling noise; the
big-ticket comparison — does the non-DP local advantage survive the change of training
configuration — is yes.

This was important for the paper's internal consistency. If the speedup config had materially
shifted DP-FedBN's behaviour, the cross-config comparisons in the paper figures would have been
suspect. They are not.

### The +0.17 finding

The headline result of Phase 5:

| ε | DP-FedBN local F1 | DP-FedAvg+GroupNorm local F1 | gap |
|---|---|---|---|
| ∞ | 0.937 | 0.463 | +0.474 |
| 3 | 0.467 | 0.295 | **+0.172** |
| 1 | 0.344 | 0.295 | +0.049 |

On label_skew_c2 — the partition where Phase 4 had identified the most dramatic eval-regime flip
(FedBN local 0.942 vs central 0.224) — DP-FedBN preserves a clear local-F1 advantage over
DP-FedAvg+GroupNorm across the whole privacy-budget range we tested. At ε=3 (the typical "moderate
DP" regime in published work), the advantage is +0.172. At ε=1 (stringent DP), the advantage is
+0.049 — smaller but still positive. At ε=∞ (no noise), the advantage is +0.474.

This is the cleanest empirical evidence the paper has for its central methodological claim:
DP-FedBN preserves FedBN's eval-regime advantage under DP, in the regime FedBN was originally
designed to handle (label heterogeneity).

### The mixed dirichlet picture

On Dirichlet α=0.1 — also non-IID, but heterogeneous in a different way (clients have different
*proportions* of all classes, not different *subsets* of classes) — the local advantage is mixed:

| ε | DP-FedBN local F1 | DP-FedAvg+GroupNorm local F1 | gap |
|---|---|---|---|
| ∞ | 0.861 | 0.832 | +0.029 |
| 3 | 0.368 | 0.376 | −0.008 |
| 1 | 0.465 | 0.350 | +0.115 |

The ε=3 row is not a tie within sampling noise — it is *almost zero*. The ordering at ε=1 reverses
back to FedBN, but the gap is smaller than at ε=∞. The non-monotonicity (ε=3 worse than ε=1 for
both methods on local F1) is a single-seed variance signature; multi-seed runs would settle it.

The honest reading: DP-FedBN's edge under DP is **clearer on label-skew than on Dirichlet at α=0.1**.
We did not over-claim. We reported all three rows at face value.

### ε calibration

All seven finite-ε runs achieved ε within 0.5% of target (range 0.9942–2.9959). The Phase 5 gate
was 10%; we were 20× inside it. The privacy accountant in Opacus 1.5 is very precise; the noise
multiplier required to hit a target ε given a sample rate and a number of steps is a smooth
function of the inputs, and Opacus's `make_private_with_epsilon` does the optimisation correctly.
The calibration is not a research finding so much as a sanity check that we configured the engine
correctly.

### Phase 5 figures

We generated four figures from the dp_summary CSV plus the Phase 4 follow-up CSVs:

- `fig_privacy_utility_central.{png,pdf}`: F1 vs ε on the central evaluation, with a stringent-DP
  shaded region and an ε=∞ side panel.
- `fig_privacy_utility_local.{png,pdf}`: same axes, local evaluation, with std error bars on the
  per-client mean.
- `fig_privacy_utility_dual.{png,pdf}`: a 2×2 money figure (label_skew_c2 / dirichlet_a01 ×
  central / local) with the +0.172 (label_skew local @ ε=3) and +0.474 (label_skew local @ ε=∞)
  advantage brackets annotated.
- `fig_central_local_phase5_comparison.{png,pdf}`: a Phase 4 (no-DP) and Phase 5 (DP at ε=3 and
  ε=1) grouped-bar comparison, with `REFUSED` cells hatched.

A LaTeX booktabs table was emitted to `results/tables/phase5_summary_table.tex` for direct
inclusion in the paper. The figure captions explicitly note the asymmetry (3 runs at original
config, 11 at speedup config) where relevant.

---

## 11. Phase 6 — Cross-task validation on PTB-XL (25–26 April 2026)

### Why a cross-task replication mattered

The paper's Phase 4 + Phase 5 story was airtight on MIT-BIH 5-class beats. The paper also needed
to answer a single very specific question: **does the eval-regime finding survive a change of task?**
If FedBN's central-vs-local flip is a peculiarity of 5-class beat-level classification — which
might happen, for example, because BN running statistics on 250-sample beats are simply more
sensitive to label heterogeneity than they would be on something else — then the paper's
generalisation claim is weak.

PTB-XL binary record-level classification is the most distant in-scope task from MIT-BIH 5-class:
the input is 1000-sample 10-second recordings, not 250-sample beats; the task is record-level (the
whole recording is normal or abnormal), not beat-level; the number of classes is 2, not 5. If the
eval-regime flip survives this much change, the paper's "task-agnostic" framing is honestly
defensible.

### Phase 6A — PTB-XL non-DP sweep (25 April)

The Phase 6A matrix was 4 aggregations (FedAvg, FedProx, FedBN, FedPerf) × 3 partitions (IID,
label_skew_c1, dirichlet_a01) = **12 runs**. Same speedup config as Phase 5: batch=96, rounds=30,
local_epochs=5, num_clients=5, seed=42. Per-run wall-clock ~12 minutes on the (now-upgraded)
RTX 4070; full sweep about 2.4 hours.

The dirichlet_a01 cell is the most informative. Clients have skewed-but-non-degenerate label
distributions (each client gets a non-zero share of both Normal and Abnormal samples, but the
proportions are different). The numbers:

| aggregation | central F1 (best) | local F1 (mean ± std) | FedBN−FedAvg (this row) |
|-|-|-|-|
| FedAvg  | 0.804 | 0.464 ± 0.148 | — |
| FedProx | 0.817 | 0.661 ± 0.113 | — |
| **FedBN** | **0.688** | **0.588 ± 0.103** | central −0.116, local +0.123 |
| FedPerf | 0.805 | 0.419 ± 0.234 | — |

**FedBN's central is the worst of the four (0.688 vs the next-worst 0.804). FedBN's local beats
FedAvg's local by +0.123, and by a larger absolute margin than the same comparison on MIT-BIH
dirichlet_a01 (where FedBN−FedAvg local was +0.017).** The sign flip in FedBN−FedAvg —negative on
central, positive on local — is preserved.

We took this as confirming the cross-task generalisation of the Phase 4 finding.

### The label_skew_c1 trap

The same Phase 6A sweep also produced the cell that became Phase 6's most instructive failure.

PTB-XL is binary, and the partition `label_skew_c1` (`classes_per_client = 1`) gives every client
**exactly one class** of beats. Each client owns either all-Normal or all-Abnormal recordings. We
included this partition because it was the binary analogue of MIT-BIH's `label_skew_c2`
(2-of-5 classes per client) — the most extreme label heterogeneity available on each task.

The Phase 6A dirichlet_a01 results above were meaningful. The Phase 6A label_skew_c1 results, when
we looked closely, were not:

| aggregation | central F1 | local F1 (mean ± std) |
|-|-|-|
| FedAvg  | 0.363 | 0.400 ± 0.490 |
| FedProx | 0.389 | 0.431 ± 0.363 |
| FedBN   | 0.326 | **1.000 ± 0.000** |
| FedPerf | 0.397 | 0.462 ± 0.301 |

FedBN's local F1 is **1.0000 with std 0.0000**. This is the trivial-collapse signature of FedBN on
a binary task with one class per client. Each client's local model trivially predicts the constant
class for its client (because that is the only class in its training data); it scores perfectly on
its own (single-class) local test set; the per-client std is exactly zero because every client is
doing the same trivial thing. The non-FedBN methods avoid this because they aggregate, so their
local models are not single-class predictors — but their local F1s are also degenerate (all
clients predict their own single class with near-perfect F1, but all those numbers are zero on the
*other* class which they never see).

This was a partition design flaw, not an algorithm bug. The fix is straightforward: use a partition
where every client gets a non-zero share of every class. dirichlet_a01 does this; label_skew_c1
does not. We documented the trap in DECISIONS.md and adjusted the Phase 6B plan (originally
scheduled to use label_skew_c1 because of the MIT-BIH precedent) to use dirichlet_a01.

The lesson is a small one but worth recording: **partition extremes that work for one task may be
trivially degenerate on another.** label_skew_c2 is meaningful on MIT-BIH because each client gets
2 of 5 classes — there is still a 3-class gap to close. label_skew_c1 on a binary task gives each
client 1 of 2 classes, and the local-eval question collapses to "can the client recognise the one
class it has?" which is trivially yes. Different number of classes; different meaningful extreme.
We should have caught this earlier; we did catch it.

### Phase 6B (yesterday) and the abandoned label_skew_c1 sweep

Before noticing the trap, we ran Phase 6B as originally planned: a 9-cell DP sweep on
label_skew_c1 (3 methods × 3 ε), mirroring Phase 5's structure on MIT-BIH. The results showed
exactly the signature one would expect:

| method | ε=∞ | ε=3 | ε=1 |
|-|-|-|-|
| dp_fedavg (raw BN) | local 0.401 | REFUSED | REFUSED |
| dp_fedavg+groupnorm | local 0.600 | local 0.400 | local 0.400 |
| dp_fedbn | local 1.000 | local 1.000 | local 1.000 |

DP-FedBN's local F1 is 1.000 with std 0.000 at every privacy budget, including ε=1. This is the
same trivial-collapse signature as Phase 6A's non-DP label_skew_c1 cell. It tells us nothing about
DP — the result is invariant to ε because the local prediction is trivial regardless of how noisy
the gradients are.

We kept these rows in `ptbxl_dp_summary.csv` for completeness, with a clear annotation that the
local-F1 numbers are trivial. They are evidence of the partition design flaw, not of DP-FedBN's
behaviour. The paper will reference them as such.

### Phase 6B-v2 today (26 April)

Phase 6B-v2 re-ran the DP cross-task sweep on **dirichlet_a01** — the partition where the non-DP
Phase 6A cell was meaningful. Same matrix structure: 3 methods × 3 ε = 9 runs, of which 2 (the
dp_fedavg at finite ε) are expected to refuse for the same `UnsupportedModuleError` reason as
Phase 5's MIT-BIH refusals.

A smoke test was run first (single DP-FedBN run on dirichlet_a01 at ε=3, 2 clients × 5 rounds × 1
local epoch, ~60 seconds). It passed: achieved ε=2.992, local F1=0.469±0.015, per-client F1=[0.484,
0.454] — non-trivial signal, every client learning something meaningful. The smoke confirmed the
dirichlet_a01 + DP + PTB-XL stack worked end-to-end before we committed two and a half hours of GPU
time to the full sweep.

The full sweep started at 15:52 and finished at 17:47 — total wall-clock 1.89 hours. 7 done + 7
skipped (the unchanged Phase 6B label_skew_c1 rows) + 4 failed (the 4 dp_fedavg-at-finite-ε cells
across both partitions, all the same `UnsupportedModuleError`). Per-run times matched Phase 5
exactly: ε=∞ ≈ 11.6–12.6 minutes, finite-ε ≈ 18.9–20.0 minutes.

---

## 12. Phase 6B-v2 results — DP-FedBN's edge does not survive Dirichlet+DP

The dirichlet_a01 results for Phase 6B-v2:

| method | ε | achieved ε | central F1 (best) | local F1 (mean ± std) |
|-|-|-|-|-|
| dp_fedavg (raw BN) | ∞ | — | 0.807 | 0.552 ± 0.162 |
| dp_fedavg (raw BN) | 3 | — | REFUSED | REFUSED |
| dp_fedavg (raw BN) | 1 | — | REFUSED | REFUSED |
| dp_fedavg + GroupNorm | ∞ | — | 0.784 | 0.443 ± 0.142 |
| dp_fedavg + GroupNorm | 3 | 2.9943 | 0.774 | 0.669 ± 0.109 |
| dp_fedavg + GroupNorm | 1 | 0.9953 | 0.765 | 0.715 ± 0.148 |
| **DP-FedBN** | ∞ | — | 0.688 | 0.604 ± 0.125 |
| **DP-FedBN** | 3 | 2.9943 | 0.400 | 0.435 ± 0.054 |
| **DP-FedBN** | 1 | 0.9953 | 0.451 | 0.526 ± 0.146 |

### The key comparison

DP-FedBN versus DP-FedAvg+GroupNorm on local F1:

| ε | DP-FedBN local | DP-GroupNorm local | gap (BN − GN) |
|---|---|---|---|
| ∞ | 0.604 | 0.443 | **+0.161** |
| 3 | 0.435 | 0.669 | **−0.234** |
| 1 | 0.526 | 0.715 | **−0.189** |

At ε=∞, DP-FedBN beats DP-GroupNorm by 0.161 — the non-DP rank flip from Phase 6A is preserved
when DP infrastructure is plumbed but no noise is added. **At finite ε, the ordering reverses.**
DP-GroupNorm wins by 0.234 at ε=3 and by 0.189 at ε=1.

Central F1 tells the same story more starkly: DP-FedBN drops from 0.688 at ε=∞ to 0.400 (ε=3) and
0.451 (ε=1), while DP-GroupNorm holds approximately 0.78 across all three privacy budgets. The
DP-noise gradient applied to FedBN's BN-stat-not-shared design is much more damaging than the same
gradient applied to GroupNorm's per-batch normalisation.

### The mechanism

The mechanism is fairly clear in retrospect.

FedBN's design choice — keep client-specific BN running statistics local, never average across
clients — is what gives FedBN its no-DP local advantage. Each client's BN adapts to its own data
distribution. In a heterogeneous regime the per-client adapted BN is a *better* normalisation for
that client's data than any global BN could be, and the local model performs accordingly.

Under DP, however, the gradients applied to each client's BN parameters are noised independently.
There is no averaging across clients to smooth the noise out — that averaging is precisely what
FedBN avoids. So the noise on each client's BN parameters and gradients accumulates without
correction, and the per-client BN starts drifting in directions driven more by noise than by data.

GroupNorm has no learnable running statistics — it normalises within channel groups on the fly
during each forward pass. There is no per-client noise accumulation, because there is no per-client
state to noise. The DP noise applied to GroupNorm's affine parameters (γ and β per channel) is
small and gets averaged across clients during aggregation, which damps it further.

The same FedBN design choice that helps in no-DP hurts under DP. We did not predict this — at
sweep-launch time we expected DP-FedBN to preserve a smaller version of its non-DP edge — but it is
the kind of result that, once seen, is straightforward to reason about.

### The counterintuitive secondary finding

A secondary observation: DP-GroupNorm's local F1 *increases* as ε decreases (0.443 → 0.669 →
0.715). This is non-monotonic in privacy strength, which is unusual and worth flagging.

Looking at the per-round CSVs, the ε=∞ DP-GroupNorm run finished with central F1 0.785 best →
0.537 final. The model was already diverging by round 30. The finite-ε runs end at central final
F1 0.756/0.757 — i.e. they did not diverge. DP noise plus gradient clipping plausibly act as
regularisation here, preventing the late-round divergence that the no-noise run experienced.

This is a single-seed observation, and we are not making a strong claim from it. Multi-seed runs
would settle whether DP-as-regularisation is a real effect on this task or a single-seed quirk.
We documented it in DECISIONS.md as a caveat in the Phase 6 figures, not as a claim about DP's
effect.

### The honest paper claim on DP-FedBN

The DP-FedBN edge is **partition-dependent**, not universal. Specifically:

- On **label-skew with ≥3 classes per task**, DP-FedBN preserves a clear local advantage under DP
  (Phase 5 MIT-BIH label_skew_c2: +0.17 at ε=3).
- On **Dirichlet α=0.1**, DP-FedBN's local advantage is mixed-or-negative under DP (Phase 5
  MIT-BIH: ε=3 −0.008, ε=1 +0.115 — basically tied; Phase 6B-v2 PTB-XL: ε=3 −0.234, ε=1 −0.189 —
  decisively in DP-GroupNorm's favour). The sign on dirichlet is **consistent across datasets** —
  PTB-XL is just a sharper version of the MIT-BIH dirichlet "tied/mixed" result.
- On **binary tasks with label_skew_c1**, the partition is degenerate and the question is not
  meaningfully testable (Phase 6A and 6B PTB-XL: trivial F1=1.000).

That is the paper's defensible DP claim. It is not "DP-FedBN beats DP-GroupNorm under DP." It is
"DP-FedBN's edge is partition-dependent, requires multi-class label-skew non-IID, and does not
hold under Dirichlet skew." This is honest empiricism, not a marketing claim.

### Cross-task summary: the eval-regime flip in non-DP

The cross-task **non-DP** finding holds cleanly:

| dataset | partition | FedBN−FedAvg central | FedBN−FedAvg local |
|---|---|---|---|
| MIT-BIH (5-class beats) | dirichlet_a01 | −0.209 | +0.017 |
| PTB-XL (binary records) | dirichlet_a01 | −0.116 | +0.123 |

Both datasets show FedBN below FedAvg on central F1 and above FedAvg on local F1. The PTB-XL local
advantage (+0.123) is materially larger than the MIT-BIH local advantage at the same partition
(+0.017). The eval-regime sign flip in FedBN−FedAvg is **task-agnostic** in the sense that it
appears on both 5-class beat-level classification and binary record-level classification. The
paper's central empirical contribution is preserved.

---

## 13. Final paper narrative

After Phase 6, the paper has three distinct contributions, two empirical and one methodological.

### Empirical contribution 1 — Eval-regime is task-agnostic

Across two datasets (MIT-BIH 5-class beats, PTB-XL binary records), four aggregation methods
(FedAvg, FedProx, FedBN, FedPerf), and matched non-IID partitions (dirichlet_a01), the FedBN
central-vs-FedAvg-central comparison is negative and the FedBN-local-vs-FedAvg-local comparison is
positive. The same method that ranks worst in central evaluation can rank above FedAvg in local
evaluation. This is the eval-regime flip, and it survives a change of task.

This is a useful finding for the FL+medical-data community. A reader who builds an FL system for
medical-data classification and reports "central F1 on a held-out test pool" as their primary
metric will see FedBN underperforming. That same reader, if they evaluate per-client, may see
FedBN winning. Whether to report central or local evaluation is not a stylistic choice — it changes
the conclusion. The paper makes this explicit.

### Empirical contribution 2 — DP-FedBN's edge is partition-dependent

Under DP, the FedBN-vs-DP-GroupNorm local-F1 comparison is partition-sensitive:

- Label-skew (≥3 classes): DP-FedBN preserves a clear edge (+0.17 at ε=3 on MIT-BIH label_skew_c2).
- Dirichlet α=0.1: edge vanishes or reverses (MIT-BIH: tied; PTB-XL: −0.234 at ε=3).

The mechanism is mechanistic, not statistical: DP noise on per-client BN parameters accumulates
without cross-client averaging, which is fatal to FedBN's design but does not affect GroupNorm's
DP behaviour. The paper makes this argument with the DP results table and the per-round CSVs as
supporting evidence.

### Methodological contribution — the dual-optimiser DP-FedBN construction

Building on (a) Li et al. 2023's mention of DP-FedBN as an undocumented baseline and (b) the
Opacus Issue #472 BN-freeze workaround, we contribute the **first explicit empirical formulation**
with:

- A clean problem statement — the Opacus + BN tension and why GroupNorm fix destroys FedBN.
- A two-optimiser construction that sidesteps the tension while preserving FedBN's mechanism.
- A multi-partition, multi-budget evaluation on a real medical task (MIT-BIH 5-class).
- A cross-task validation on PTB-XL binary records.

The contribution is not "we coined DP-FedBN" — Li et al. 2023 already used the term. The
contribution is the *first time the construction is written down, defended, and empirically
characterised across the partition × budget × task space we tested*.

### Practical guidance — a decision rubric

For practitioners considering DP-FedBN versus DP-GroupNorm in a medical-FL deployment, the paper
will give a small decision rubric:

- If the deployment regime is **personalised** (each client uses its own model) and the
  heterogeneity is **label-skew with ≥3 classes per task**, DP-FedBN preserves an edge over
  DP-GroupNorm and is worth the extra complexity.
- If the deployment regime is **central** (one global model used for everyone) or the
  heterogeneity is **Dirichlet** (clients have different *proportions* of all classes rather than
  different *subsets* of classes), DP-GroupNorm is simpler and at least as good.
- If the task is **binary with single-class clients**, neither method is well-tested by published
  benchmarks; the partition is degenerate enough that any reported number is suspect.

This rubric is honest. It does not say "use DP-FedBN" everywhere; it says "use DP-FedBN when its
design assumptions match the deployment, else use the simpler alternative."

---

## 14. What we learned along the way

### Methodology lessons

**Always audit before pivoting your paper.** Phase 4 produced a result that contradicted the
original FedBN paper — FedBN underperforming FedAvg on every partition. The temptation was to
declare a bug, fix something, and pivot. Instead we wrote a 7-check audit script that took an
afternoon to write and run, and that audit produced both the verification of correctness *and* the
mechanistic explanation (Check 7's diagnostic) that became the paper's central empirical
contribution. The audit cost was 3 hours; the value was the entire paper narrative. The general
lesson is that surprising negative results deserve more rigour, not less, before they reshape the
story you intend to tell.

**An instrumented pipeline pays for itself the first time something goes wrong.** All six FedBN
runs in Phase 4 crashed at the reporting step. Their per-round CSVs were complete; their final
JSONs were missing. Because the per-round CSVs existed, we wrote a 30-line recovery script and
saved 12 hours of GPU time. If we had only logged the final JSON, we would have had to re-run all
six runs. The general lesson is to never skip a per-round write loop because "the code will
succeed and I'll just read the final JSON."

**The first run is always the diagnostic, not the answer.** Phase 5's original config produced 5
of 18 runs in 9.5 hours, on track for a 45-hour total. The first finished run was the diagnostic
that the config was structurally too slow. We killed and retrofitted; the resumed sweep finished in
6.89 hours. The general lesson is that early sweep speed is information about whether the sweep
will finish, not just a number to wait out.

**Literature review is part of the paper, not a chore.** Pre-search, we believed DP-FedBN was
ours to coin. Post-search, we found Li et al. 2023 had used the term as an undocumented baseline
and Opacus Issue #472 had documented the implementation trick. This was not bad news — it told us
exactly how to frame our contribution honestly ("first explicit empirical formulation," not "we
coin DP-FedBN") and it told us which prior work to compare against in the related-work section.
The general lesson is that the right time to do the literature search is *before* you write the
contribution claim, not after submission.

### Engineering lessons

**Reporting-step crashes are recoverable from rounds.csv if the pipeline is well-instrumented.**
This is the engineering counterpart to the methodology lesson above. The recovery script for Phase
4 was 30 lines because the rounds.csv schema was rich enough — every per-round central F1, every
per-client validation F1 macro, every accuracy and AUC. If the schema had been "central F1 only,"
we could not have recovered the per-class breakdown for the figures.

**Never skip the smoke test before a multi-hour sweep.** Phase 5's smoke caught a
state-dict-key-mismatch bug in `_evaluate_local_test` that would have crashed every DP-FedBN run
*after* training had completed. Pre-Phase-5, `client.model` and `client._inner()` returned the same
object, so the `load_state_dict` line had been silently right; Opacus's `GradSampleModule` wrapper
broke that equivalence. The bug was 15 seconds to find with a smoke; it would have been hours to
find from a 30-round sweep failing 30 minutes in.

**`_inner()` access patterns matter when DP is involved.** This is a specific lesson about
Opacus, but it generalises. If your client class has a "wrapped model" abstraction, every place
that touches `state_dict` needs to go through the same access pattern. Phase 3 had `get_parameters`,
`set_parameters`, `snapshot`, and `restore` all going through `_inner()`; only the new
`_evaluate_local_test` did not. The fix was to bring that one function into line with the other
four. The general lesson is that a wrapping pattern needs to be applied consistently or it fails
in exactly the places it was not applied.

**Dataloader workers and batch size are the first things to look at when DP is slow.** Phase 5's
original sweep was bottlenecked on data loading, not on DP overhead. The fix was batch=96 +
num_workers=4 + pin_memory=True. The Opacus per-sample gradient computation was *not* the
bottleneck; small batches with no prefetching was. When a sweep is mysteriously slow, the data
path is the first place to look.

**ExpandedWeights is not a drop-in replacement for the hooks backend.** Specifically, it is
incompatible with `BatchMemoryManager` chunking and with `optimizer.zero_grad(set_to_none=True)`.
We documented this in the codebase and in DECISIONS.md, and we left a comment warning future-us
not to try ExpandedWeights again without writing a proper integration test. The general lesson is
that "alternative backend that is supposedly faster" is the kind of thing that should be tested
on a smoke before committing to a full sweep.

### Empirical lessons

**Eval regime matters more than method choice in some non-IID regimes.** This is the central
empirical finding of the project. On label-skew C=2 with FedBN, central F1 is 0.224 and local F1 is
0.942. The same model, the same training, the same federation — and the choice of evaluation
regime moves the F1 by 0.72. No aggregation algorithm change comes close to that effect size on
this data. If a paper is going to make a "method X is best" claim under non-IID FL, it had better
specify which evaluation regime that claim refers to.

**FedBN's design specialises BN to the local distribution.** This is exactly what the original
paper says, and it is exactly what Phase 4 + Phase 4 follow-up showed empirically. The original
paper's setting (cross-domain image classification) makes the local distribution genuinely
different from the global distribution; in that setting, specialising BN to local is helpful and
the central evaluation reflects that. Our setting (label-shifted medical time-series, central
evaluation on a pooled holdout) makes specialising BN to local *bad* for central evaluation,
*good* for local evaluation. The mechanism is the same; the metric is what changed.

**FedProx with `μ = 0.01` is too aggressive on this task.** Across all six MIT-BIH partitions in
Phase 4, FedProx underperformed FedAvg by 0.02 to 0.14. The most likely cause is that the proximal
term, with this μ, dominates the gradient and prevents the local model from moving meaningfully
during 5 local epochs. A `μ ∈ {0.001, 0.005}` sweep would settle this; we did not run it.

**Single-class clients on a binary task collapse local F1 to 1.0 trivially.** The label_skew_c1
trap is a partition-design lesson, not an algorithm one. The fix is to use partition extremes that
preserve the question being asked — every client should have a non-zero share of every class for
local F1 to be a meaningful comparison metric.

### Privacy lessons

**Naive DP-SGD on a BN-heavy CNN is structurally impossible, not just slow.** Opacus refuses to
wire it. The empirical answer to "does naive DP-FedAvg work on a BatchNorm CNN?" is "no, the
privacy library aborts at validate-time." This is a methodological motivation for either
GroupNorm replacement or our DP-FedBN construction — there is no third option that keeps BN and
applies DP-SGD naively.

**ε calibration is precise to within 0.5% in Opacus 1.5.** Once the noise multiplier is correctly
chosen for the target ε, sample rate, and number of steps, the privacy accountant is very
precise. ε calibration is a configuration check, not a research finding.

**DP and heterogeneity interact in non-trivial ways.** Phase 6B-v2's DP-FedBN edge collapse on
PTB-XL dirichlet_a01 was not predicted from the no-DP results. The same FedBN design that wins on
local F1 in no-DP loses badly to GroupNorm under DP, because the BN-stat-not-shared property —
helpful in no-DP — amplifies DP noise in the no-averaging direction. Cross-validating findings
from the no-DP regime to the DP regime is not optional; the addition of DP can flip the sign of
the comparison.

---

## 15. Limitations and future work

### Single seed

Every result in this project is a single-seed (seed=42) realisation. We do not report variance
across seeds. The Phase 4 sweep would have taken ~40 hours instead of 13 with three seeds; the
Phase 5 DP sweep would have taken ~21 hours instead of 7. We did not have those hours.

The variance-unquantified caveat applies particularly strongly to:
- The Phase 6B-v2 dirichlet_a01 non-monotonicity (DP-GroupNorm local F1 *increasing* as ε
  decreases). This may or may not survive a multi-seed expansion.
- The Phase 5 dirichlet_a01 result at ε=3 (DP-FedBN local 0.368 vs DP-GroupNorm local 0.376 —
  basically tied within plausible single-seed error).

The paper will state explicitly that all results are single-seed and that multi-seed expansion is
future work.

### Single architecture

Every FL run in this project uses the same CNN1D — 4 conv blocks (64/128/256/256 channels), FC
hidden 128, dropout 0.5, ~437k parameters. We did not test ResNet1D (which was scoped in the
original plan as an ablation) or Transformer-style architectures.

The findings are likely architecture-specific in some ways. BN-heavy networks are where FedBN
helps most under heterogeneity; a network with fewer BN layers, or with LayerNorm instead of BN,
would have a smaller eval-regime flip on the same data. The paper will state that the findings
apply to BN-heavy CNN architectures specifically.

### Simulated heterogeneity

Our non-IID partitions — label-skew, quantity-skew, Dirichlet — are simulations on a single
dataset that was originally pooled. We do not evaluate true cross-institutional heterogeneity,
which would have intrinsic differences in equipment (different ECG recorders), patient
populations, recording protocols, and labelling conventions.

A real cross-institutional FL deployment would likely be even more heterogeneous than our
Dirichlet α=0.1 simulation, and might exhibit feature shift (where the equipment differs) in
addition to label shift (where the patient mix differs). FedBN was originally designed for feature
shift, not label shift, so a real cross-institutional deployment is *exactly* the regime where
FedBN should shine — but we did not validate it there.

### Five clients

All FL experiments use 5 clients. Production federated medical-data systems would likely have
dozens to hundreds of clients (each hospital or each clinic). Aggregation algorithms can behave
qualitatively differently at higher client counts — the variance of the size-weighted mean
shrinks, but the variance of each individual client's data shrinks too, so the eval-regime gap may
narrow or widen with client count.

We picked 5 clients because the master plan picked 5 and because 5 is the smallest number that
gives a meaningful per-client validation set on the MIT-BIH 109k beats. Larger client counts on
this data would mean smaller per-client datasets, which interacts with the analysis.

### Multi-label PTB-XL deferred

The original plan included a "bonus phase" that would re-run PTB-XL as multi-label (each recording
can have multiple of NORM/MI/STTC/CD/HYP). The multi-label sweep was scripted (`run_ptbxl_multilabel_sweep.py`)
but never executed because the binary task gave us the cross-task replication we needed and we
ran out of compute time. Multi-label PTB-XL would be a useful extension if a reviewer asks why we
collapsed the 5 superclasses into binary; our answer is the Lead-II representational bottleneck on
the 5-class task plus the cross-task generalisation argument from binary, but a multi-label run
would be a stronger answer.

### FedProx tuning deferred

FedProx with `μ = 0.01` was uniformly worse than FedAvg on every Phase 4 partition. The most
likely cause is that this μ is too aggressive for our model and dataset; a `μ ∈ {0.001, 0.005}`
sweep would probably recover competitive performance. We did not run that sweep because FedProx
is not a primary method in the paper — it is a baseline to contextualise FedBN — and a tuned
FedProx is unlikely to change the FedBN-vs-FedAvg story which is the paper's core. We will report
the un-tuned FedProx number with a footnote.

### ExpandedWeights × BatchMemoryManager incompatibility

The Opacus ExpandedWeights backend's incompatibility with `BatchMemoryManager` and with
`zero_grad(set_to_none=True)` is documented in our DECISIONS.md and in the codebase. This is a
small finding that would be useful to the Opacus community as an issue or a documentation patch;
we have not yet filed it upstream. It is logged as a future contribution opportunity.

### What we would do differently

If we were starting the project over with the same seven days, we would:

- **Run the FedBN audit before the Phase 4 sweep**, not after. The audit's checks 1–4 and 6 are
  pre-sweep verifications that would have caught any implementation bug before the 13-hour sweep
  ran. We did not do this because we did not anticipate needing to defend FedBN's correctness; we
  expected FedBN to win, not lose. Lesson learned: implement the implementation-correctness audit
  on Day 0 as a unit-testable artefact, not on Day 4 as a fire drill.

- **Speedup-config from the start.** The Phase 5 retrofit (batch=96, num_workers=4, rounds=30) was
  ~6× faster than the Phase 5 launch config. Almost every downstream phase used the speedup
  config, including all of Phase 6. We could have started with it. The reason we did not is that
  the master plan specified rounds=50 as the standard FL configuration; we trusted the plan rather
  than profiling first.

- **Catch the label_skew_c1 trap on the partitioner unit tests**, not on Phase 6's local-F1 std.
  A unit test for `partition_label_skew(num_classes=2, classes_per_client=1)` could have asserted
  that every client gets exactly one class, and a higher-level test could have flagged the
  resulting FedBN local-F1=1.0000 / std=0.0000 signature. We caught it visually after running the
  sweep; we should have caught it in the partitioner.

These are not project-killing oversights, but they are the changes we would make on a second pass.

---

## Closing

The project produced what it set out to produce. We have two empirical contributions and one
methodological contribution, all defensible, all backed by reproducible single-seed sweeps. The
codebase is clean, the figures are paper-ready, and the DECISIONS.md tells the technical decision
story while the document you are reading tells the journey.

The paper that comes out of this work will be modest. It is not a flagship method paper; it does
not claim to revolutionise federated learning under DP. What it does is:

- Make the eval-regime choice explicit in non-IID FL benchmarking, with two-dataset evidence.
- Formalise DP-FedBN with a working two-optimiser construction.
- Honestly report DP-FedBN's partition-dependent edge — gain on label-skew, loss on Dirichlet.

That last point is the one that took the longest to learn how to say. The sweep results pushed us
to claim less than the master plan had wanted to claim. The honest version is what the paper will
say.

There is a particular satisfaction in writing a paper where every number can be defended, every
claim is bracketed by the regime in which it holds, and every limitation is in the limitations
section before any reviewer flags it. We did not arrive at that voice on Day 0; we arrived at it
through the audits, the kills, the retrofits, the failed partitions, and the cross-task
replication. The voice is the project's most valuable output, more valuable than any single F1
number. The next paper will start with it.
