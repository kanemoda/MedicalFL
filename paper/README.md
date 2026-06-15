# DP-FedBN — corrected conference paper

This folder contains a corrected, compile-ready IEEE-conference LaTeX reconstruction of the manuscript *"When Does Federated BatchNorm Help under Differential Privacy? An Eval-Regime Characterization on Medical ECG Classification."*

```
paper/
├── dp-fedbn.tex          # the paper (IEEEtran, conference)
├── Makefile              # `make` -> dp-fedbn.pdf
├── figures/
│   ├── fig_eval_regime.pdf      # Fig. 1  (= results/figures/fig_central_vs_local.pdf)
│   ├── fig_privacy_utility.pdf  # Fig. 2  (= results/figures/fig_privacy_utility_dual.pdf)
│   └── fig_cross_task.pdf       # Fig. 3  (= results/figures/fig_phase6_cross_task_summary.pdf)
└── README.md
```

## What was fixed relative to the original PDF

All **numeric results are unchanged** and match `results/metrics/*.csv` exactly. The fixes correct factual/description errors that the thesis review surfaced (see `../Grad thesis files/thesis-draft/REVIEW-NOTES.md` §2):

1. **MIT-BIH class-imbalance statistic.** The original motivated macro F1 with "(N: ~83%, Q: ~0.07%)." The processed data give N≈82.8%, Q≈7.3%, and the **rarest class is F≈0.7%** (802/109,453 beats). Section IV-A now lists the full class distribution and Section IV-E cites "N≈83% versus the rarest class F≈0.7%."
2. **FedPerf aggregation rule.** The original described FedPerf as a "softmax of local validation F1 (temperature 1)." The implementation (`src/federation/aggregation.py`) uses an **equal-weight (α=0.5) convex combination of ℓ₁-normalized dataset size and ℓ₁-normalized local validation macro F1**. Section IV-B now matches the code.
3. **Central test set.** The original said "pooled stratified holdout (15% of each dataset)." Section IV-E now states the actual split: **a 15% stratified holdout for MIT-BIH and the canonical fold 10 for PTB-XL** (per `src/training/federated.py`).
4. **Model architecture.** The compact notation now lists the **fifth BatchNorm** (after FC(128)), matching `src/models/cnn1d.py` (five BN modules, ~437k parameters).
5. **Figure captions.** Captions now describe the actual figure files: Fig. 1 is a grouped bar chart (central=blue, local=red) across two partitions; Fig. 2's DP-FedBN curve is **green** (the original caption said "orange"); Fig. 3 reports the FedBN−FedAvg gaps.

The contested "strictly stronger than party-level Laplace" wording (Section I/II) is **kept as the authors' calibrated positioning**; the thesis `REVIEW-NOTES.md` §2(6) discusses why record-level and client-level DP protect different units, should the authors wish to soften it.

## Building

A LaTeX toolchain with the IEEEtran class is required. On Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y texlive-latex-base texlive-latex-recommended \
    texlive-latex-extra texlive-publishers texlive-science texlive-fonts-recommended
```

Then:

```bash
make          # runs pdflatex twice (for cross-references) -> dp-fedbn.pdf
make clean    # remove build artifacts
```

Or compile directly: `pdflatex dp-fedbn.tex && pdflatex dp-fedbn.tex`.

The source also compiles unmodified on Overleaf (select a recent TeX Live; no extra setup needed).
