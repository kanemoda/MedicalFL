<!-- HANDOFF DOCUMENT for the human author/reviewer. Not part of the thesis body. -->

# Review Notes — DP-FedBN Thesis First Draft

This document is the targeted handoff for the human author. It records every placeholder, every place the draft interpreted or extrapolated beyond the source paper, every paper-vs-code/data discrepancy and how it was resolved, the figure/table provenance, and the word/page accounting. Read this before reviewing the chapters.

**What the draft is.** A complete first draft of the thesis *body* (Abstract + Chapters 1–7 + References) as modular markdown, faithful to the source paper (`When Does Federated BatchNorm Help Under DP.pdf`) and the codebase. Front matter other than the abstract (cover, jury page, ethics declaration, ToC, lists, symbols) is intentionally not written — it is filled into the Ege University template at assembly time.

**Ground-truth discipline.** The paper and code are the single source of truth. Every numeric result was cross-checked against `results/metrics/*.csv` and matches the paper's tables to printed precision (see `00-OUTLINE.md` §3, the verified ledger). Nothing was invented; the only placeholders are two admin fields in the abstract.

---

## 1. `[TODO]` / `[VERIFY]` markers in the draft

Only two markers exist in the body, both admin fields the author must fill:

| File | Marker | Action needed |
|---|---|---|
| `00-abstract.md` (header block) | `[TODO: Month Year]` | Submission month/year |
| `00-abstract.md` (header block) | `[TODO: XX] Pages` | Final rendered page count |

(The `[VERIFY]`/`[TODO]` strings in `00-OUTLINE.md` are part of the drafting-discipline instructions, not real markers.)

No `[VERIFY]` markers were needed inside the chapters: every technical claim is grounded in the paper or code. The background references added for Chapters 2–3 are discussed in §5 below.

---

## 2. Paper-vs-data/code discrepancies — resolved, please confirm

These are the most important items to check. In each case the draft followed the rule "paper and code/data are ground truth," and where the *printed paper* appears to conflict with the *actual data/code*, the draft used the data/code value and flags it here.

1. **MIT-BIH rarest-class percentage (likely a slip in the paper).** The paper's Section IV-E motivates macro F1 by citing the imbalance as "(N: ~83%, Q: ~0.07%)." The processed data (`data/processed/mitbih/stats.json`) gives class counts N 90,595 / S 2,781 / V 7,235 / F 802 / Q 8,040 of 109,453, i.e. **N ≈ 82.8%, Q ≈ 7.3%, and the rarest class is F ≈ 0.7%** (802 beats) — not Q at 0.07%. The draft (`05-experimental-setup.md` §5.1.1) reports the **accurate full distribution from the data**. *Please confirm and, if the paper text is to be matched, decide whether to correct the paper or footnote the figure.*

2. **MIT-BIH record count.** The paper states 109,453 beats across **48 records**; the data (`stats.json` lists 48 processed records; `data/raw/INVENTORY.md` confirms 48 raw records) agrees. The draft uses **48 records / 109,453 beats**. (Note: the 4 paced records 102/104/107/217 are *kept* with Q labels, consistent with 48 rather than the alternative "44-record" convention that excludes them.) No conflict; flagged only because the "44 vs 48" distinction is a common ECG-literature ambiguity.

3. **FedPerf aggregation rule — paper and code disagree.** The paper describes FedPerf as weighting client updates by "softmax of their local validation F1 (temperature 1)." The code (`src/federation/aggregation.py`, `fedperf_aggregate`) instead uses a **linear mix** `w_i = α·size_i + (1−α)·F1_i` with α = 0.5 and L1-normalized terms — not a softmax. Because FedPerf is explicitly *not* a contribution and only contextualizes FedBN, the draft (`05-experimental-setup.md` §5.3) describes it at the level both sources agree on ("weights client updates according to their local validation macro F1"). *Please reconcile the exact rule for the final text.*

4. **PTB-XL central-test fraction.** Section IV-E of the paper says the central test is a "pooled stratified holdout (15% of each dataset)," but Section IV-A and the code (`src/training/federated.py`, `_carve_ptbxl`) use the canonical **fold 10** for PTB-XL (≈10% of records), reserving folds 1–9 for clients. The draft reports MIT-BIH as a 15% stratified holdout and **PTB-XL as fold 10** (per code), which is the faithful reading. Minor.

5. **Best-vs-final central F1 (explains small numeric differences).** Table 6.1 (MIT-BIH non-IID) uses *best-across-rounds* central F1 (from `noniid_summary.csv`, `best_central_f1`), whereas the eval-regime numbers (Table 6.2 and the §6.3 Dirichlet prose) use the *final-round* central F1 from the `central_vs_local_*.csv` files. This is why, e.g., FedBN on Dirichlet α=0.1 reads 0.554 in Table 6.1 but 0.536 in §6.3, and FedBN on label-skew reads 0.219 in Table 6.1 but 0.224 in Table 6.2. This matches the source paper exactly (it shows the same two values) — it is a best-vs-final distinction, not an error.

6. **"Strictly stronger" privacy claim — softened with attribution.** The paper says the record-level DP-SGD construction provides "strictly stronger privacy guarantees than party-level Laplace mechanisms." Because record-level and client-level (party-level) DP protect *different units of adjacency* (Section 2.5.1), a blanket "strictly stronger" is technically contestable — client-level DP can be argued to be a stronger *unit* of protection. The draft therefore renders the claim **with attribution to the source paper** ("which the source paper positions as stronger than…", Ch. 3–4) rather than as a standalone theorem. *Decide whether to restore the paper's exact "strictly stronger" wording or keep the attributed hedge.* This is the one place the draft deliberately did not lift the paper's phrasing verbatim, in service of the "factual accuracy first" priority.

7. **Architecture — five BN modules.** The paper's compact architecture notation lists four BatchNorm layers (after the four convs). The code (`src/models/cnn1d.py`) has a **fifth** BatchNorm after the first fully connected layer (`bn_fc`), consistent with "dropout 0.5 on the penultimate FC." The draft (`04-methodology.md` §4.2) describes **five BN modules**, matching the code and the ~437k parameter count.

8. **Figure color of DP-FedBN.** The paper's Figure 2 caption calls DP-FedBN "orange," but the actual figure file (`fig_privacy_utility_dual.png`) renders it in **green**. The draft keeps figure descriptions color-agnostic to avoid the mismatch.

---

## 3. Interpretations / extrapolations beyond the paper

Places where the draft added framing or detail not stated verbatim in the paper, all grounded in the code or in established background:

- **Clinical motivation (Ch. 1, Ch. 2.1).** The regulatory framing (HIPAA/GDPR/KVKK), the clinical meaning of the AAMI classes, and ECG morphology are standard background, added for the pedagogical chapter; not in the paper but uncontroversial. Citations [25]–[27] support these.
- **Background depth in Ch. 2 (DP, RDP, attacks, normalization alternatives).** The ε–δ definition, Gaussian mechanism, RDP accounting, membership-inference/gradient-inversion motivation, and the record-vs-client-level DP distinction are expanded from the paper's terse treatment using standard references ([23], [24], [30], [31], [32]). The DP-SGD and BN-obstacle equations match the paper's exact forms.
- **The intra-patient MIT-BIH central split (Ch. 3.4, 5.5, 6.7).** The draft notes that the MIT-BIH central test is a beat-level (intra-patient) stratified holdout, which is less strict than a patient-disjoint inter-patient split, and lists it among the limitations. The paper does not raise this; it is an honest observation derived from the code (`_carve_mitbih`). *Confirm you want this surfaced — it is a real but author-discretion limitation.*
- **Centralized baseline configuration (Ch. 5.3).** Batch sizes (64 MIT / 32 PTB), 50 epochs with early stopping, and the PTB-XL augmentation (random crop + Gaussian noise + baseline wander) are read from the run JSONs and `src/data/augment.py`. The paper does not detail the centralized config; this is faithful to the code.
- **GroupNorm group-count `gcd(32, k)` (Ch. 4.5.1).** Taken from `src/federation/dp.py` (Opacus's default substitution). Not in the paper; faithful to the code.
- **Practitioner rubric, mechanism, and synthesis numbers (Ch. 6.6).** Lifted closely from the paper's Discussion (Section VI); wording paraphrased, claims unchanged.

No result, number, dataset, hyperparameter, figure, or citation was invented.

---

## 4. Technical points of mild uncertainty

- **Thesis title.** The draft adapts the paper's title to *"When Does Federated BatchNorm Help under Differential Privacy? An Eval-Regime Characterization of DP-FedBN on Medical ECG Classification."* The paper's exact title omits "of DP-FedBN." Use whichever the author/department prefers.
- **FedBN central-eval mode reported.** The code default is `representative` (client 0); `STORY.md` states the *primary* reported metric is `client_avg`. The paper says both are reported. The draft says both are computed and the qualitative finding is independent of the choice (true: FedBN is lowest-central / highest-local under either). No number depends on this in the reproduced tables, which come directly from the canonical CSVs.
- **Algorithm 1 step numbering.** The draft's Algorithm 1 compresses the paper's 20 numbered lines (with explicit per-BN-module loops) into 13 numbered steps with the same content. Semantics are identical; renumber to taste at assembly.

---

## 5. References

- **[1]–[22]** are reproduced **verbatim** from the source paper's verified bibliography. Reference [12] (Tia et al., 2024) is a thin citation in the paper itself (no venue); reproduced as-is — confirm at final formatting if desired.
- **[23]–[33]** are background works **added** for the expanded Chapters 2–3. All are well-established and I am confident they exist; their venues and years are correct. **Exact page numbers/volumes should be confirmed at final formatting** (e.g., [31] Zhu et al., "Deep Leakage from Gradients," NeurIPS 2019 is given without page numbers, as NeurIPS pagination is inconsistent). None is fabricated; none is marked `[VERIFY]` because I am confident of their existence, but page-level precision is the author's to finalize.
- Citation style is IEEE numeric throughout, matching the paper. All 33 references are cited in the body; the highest index used is 33.

---

## 6. Figures and tables — provenance (real repo files only)

Every figure references an existing file in `results/figures/`; an HTML comment at each figure notes the source. Final figure/chart/table numbering is to be assigned at assembly.

| Draft label | Source file | Verified? |
|---|---|---|
| Fig. 5.1 — six MIT-BIH partitions | `fig_partition_distributions_mitbih.png` | viewed |
| Fig. 6.1 — centralized training curves | `centralized_mitbih_cnn1d_seed42_training.png`, `centralized_ptbxl_cnn1d_binary_seed42_training.png` | described from run JSONs |
| Fig. 6.2 — non-IID heatmap | `fig_noniid_heatmap.png` | viewed (values match Table 6.1) |
| Fig. 6.3 — eval-regime flip | `fig_central_vs_local.png` | viewed (paper Fig. 1) |
| Fig. 6.4 — DP privacy–utility | `fig_privacy_utility_dual.png` | viewed (paper Fig. 2) |
| Fig. 6.5 — cross-task summary | `fig_phase6_cross_task_summary.png` | viewed (paper Fig. 3) |

Optional additional figures available if more visuals are wanted: `fig_degradation_from_iid.png`, `fig_phase6_dp_privacy_utility_ptbxl.png`, and per-run `*_fed_curves.png` trajectories.

Tables 6.1–6.5 are markdown; all values verified against `noniid_summary.csv`, `central_vs_local_*.csv`, `dp_summary.csv`, `ptbxl_summary.csv`, `ptbxl_dp_summary.csv`.

---

## 7. Consistency / anti-drift measures taken

- The **contribution statement** appears in substantively identical wording in the Abstract, Introduction (§1.3), Methodology (§4 preamble), and Conclusion (§7.1). It is the paper's calibrated claim — "first record-level DP-SGD composition of FedBN's BatchNorm-locality with explicit treatment of the Opacus engineering obstacle" — and is never strengthened.
- The **ADCOL acknowledgment** (party-level Laplace baseline, ICML 2023, ref [1]) appears in the Abstract, Introduction, Related Work (§3.5), Methodology (§4 preamble), and Conclusion, always with the same three concrete differences (granularity / engineering / scope).
- Standardized terminology (client, DP-SGD, record-level DP, central vs local F1, DP-FedAvg+GroupNorm, eval-regime flip, etc.) is fixed in `00-OUTLINE.md` §2 and used consistently.
- Each background concept is explained once (Ch. 2) and cross-referenced elsewhere rather than re-explained.
- "Banned" inflationary words (novel, groundbreaking, first-ever, revolutionary) are avoided; the only "first" claim is the precise paper-sanctioned one.

---

## 8. Word counts and page estimate

| File | Words | Target | Status |
|---|---|---|---|
| 00-abstract.md (body ≈ 300) | 398 | 250–350 (body) | ✓ |
| 01-introduction.md | 1,840 | 1,800–2,500 | ✓ |
| 02-fundamental-concepts.md | 4,039 | 4,000–5,500 | ✓ |
| 03-related-work.md | 2,202 | 2,200–3,000 | ✓ |
| 04-methodology.md | 2,945 | 2,800–3,800 | ✓ |
| 05-experimental-setup.md | 2,449 | 2,200–3,000 | ✓ |
| 06-results-and-discussion.md | 3,658 | 2,800–3,800 | ✓ |
| 07-conclusion.md | 1,126 | 1,000–1,500 | ✓ |
| 99-references.md | 1,082 | (list) | — |
| **Body (Ch. 1–7)** | **≈ 18,260** | 18,000–23,000 | ✓ |
| **Body + abstract** | **≈ 18,560** | — | — |

**Estimated rendered pages.** At a typical Ege University thesis format (12 pt, 1.5 line spacing, ≈ 300–350 words/page of text), ≈ 18,560 words of body text is roughly **53–62 pages**, before adding the ~6 figures (~3–4 pages), 5 tables (~2 pages), ~25 display equations (~2 pages), the references (~2 pages), and the unwritten front matter. The body therefore comfortably clears the **50+ rendered-page** target; expect the full assembled document (with front matter) to land around **60–70 pages**.

---

## 9. Assembly checklist for the author

1. Fill the two abstract admin placeholders (date, page count).
2. Decide items in §2 (esp. #1 the MIT-BIH imbalance figure, #3 FedPerf rule, #6 the "strictly stronger" wording).
3. Confirm or remove the intra-patient-split limitation (§3) if out of scope.
4. Finalize reference page/volume details for [23]–[33] (§5).
5. Pour the chapters into the Ege template (`THEADING1` for chapter titles, `TBody` for prose); the template renders chapter headings in numbered all-caps.
6. Place the six figures, assign final figure/table/chart numbers, and build the List of Figures / List of Charts from them.
7. Build Table of Contents and Symbols/Abbreviations from the standardized-terms list in `00-OUTLINE.md` §2.
8. Add front matter (cover, jury, ethics declaration, acknowledgments) — personal/admin, not drafted here.
9. Do a final read for the eval-regime flip framing, which is the thesis's empirical heart and must read cleanly.
