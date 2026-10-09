# Closing block — collagen IV, non-lesion tissue, composition-only clock

2026-10-09. `analysis/closing_block_data.py`, `analysis/closing_block.py`, results in
`results.json`.

**Common setup.**
- **Units and models:** RR (n = 33) and chronic (n = 34) are analysed separately. The animal is
  the unit, folds are leave-one-animal-out, and the model is the unchanged ElasticNetCV clock.
- **Target:** `day_of_sacrifice`, severity-residualised inside each training fold.
- **Statistics:** 2000-resample animal bootstraps; paired-difference bootstraps whenever two ρ
  share animals.
- **No `Global_niche` anywhere.**

**Coefficient selection, stated up front.**
- **Task A:** Col4a1/Col4a2 were **selected from the fitted clock's coefficients**. A4 gives
  their rank among all expressed genes, the selection-aware null. Of the A5 basement-membrane
  genes, only Col4a1/Col4a2 appear in either clock coefficient list; the rest are
  annotation-defined.
- **Tasks B and C:** features are not coefficient-selected (full panel; cell-type proportions).

---

## Task A — collagen IV as a standalone finding

**Question.** Do Col4a1/Col4a2 decline within endothelial cells and VSMCs with disease duration,
independently of severity?

**Verdict.** Partly.
- **Within endothelial cells:** Col4a1/2 fall with duration at matched severity in both
  cohorts, and rank in the top 2–5% of all expressed genes for that association (top 1% in
  chronic, which was not used to pick the genes). But the chronic result depends on the
  severity adjustment and does not survive without it; see the follow-up section at the end.
  Treat it as RR-only.
- **But the severity association is not weaker:** Col4a1/2 *rise* with current severity about
  as strongly.
- **In chronic,** the duration association appears only after severity is removed (raw ρ ≈ −0.28).
- **Across the basement membrane,** coherence is partial.

**A1.** No animal has fewer than 50 cells of either type (endothelial 537–3,181 per animal; VSMC
144–790), so excluding versus including low-n animals is identical.

**A2/A3 — combined Col4a1+Col4a2 mean (single genes are within ±0.05 of these):**

| cohort / cell type | n | day, raw | day \| severity | severity, raw | severity \| day |
|---|---|---|---|---|---|
| RR endothelial | 33 | −0.49 [−0.72, −0.19] | −0.66 [−0.81, −0.38] | +0.50 [0.14, 0.76] | +0.66 |
| RR VSMC | 33 | −0.36 [−0.62, −0.04] | −0.49 [−0.69, −0.16] | +0.47 [0.09, 0.73] | +0.56 |
| chronic endothelial | 34 | −0.28 [−0.60, +0.09] | −0.74 [−0.85, −0.55] | +0.45 [0.13, 0.69] | +0.78 |
| chronic VSMC | 34 | +0.04 [−0.37, +0.41] | −0.43 [−0.67, −0.10] | +0.56 [0.25, 0.76] | +0.66 |

**A4 — rank of |ρ(gene, day | severity)| among all genes expressed in > 10% of that type's cells:**

| cohort / cell type | genes in null | Col4a1 percentile (p) | Col4a2 percentile (p) |
|---|---|---|---|
| RR endothelial | 2,020 | 95.5 (0.045) | 97.8 (0.022) |
| RR VSMC | 1,990 | 93.9 (0.061) | 93.6 (0.064) |
| chronic endothelial | 2,020 | **99.0 (0.010)** | **98.8 (0.012)** |
| chronic VSMC | 1,990 | 91.4 (0.086) | 85.0 (0.150) |

Raw (unadjusted) ranks are lower: chronic endothelial falls to the 52nd–55th percentile. The
standout is specific to the severity-adjusted statistic.

**A5 — basement membrane within endothelial cells, ρ(day | severity):**

| gene | % cells positive | RR | chronic |
|---|---|---|---|
| Col4a1 * | 55 | −0.62 [−0.78, −0.33] | −0.75 [−0.86, −0.55] |
| Col4a2 * | 47 | −0.68 [−0.83, −0.41] | −0.73 [−0.85, −0.52] |
| Lamb1 | 7 | **−0.49 [−0.70, −0.17]** | **−0.40 [−0.61, −0.05]** |
| Hspg2 | 53 | −0.27 [−0.55, +0.04] | **−0.65 [−0.86, −0.31]** |
| Lamc1 | 50 | −0.15 [−0.48, +0.23] | −0.32 [−0.58, +0.01] |
| Nid1 | 32 | −0.24 [−0.57, +0.18] | +0.10 [−0.22, +0.45] |
| Lama4 | 23 | −0.16 [−0.46, +0.21] | −0.18 [−0.45, +0.16] |
| Nid2 | 15 | +0.23 [−0.16, +0.60] | +0.01 [−0.34, +0.37] |
| Col4a3 | 1 | +0.21 [−0.19, +0.55] | +0.34 [−0.06, +0.67] |
| Col4a5 | 5 | **+0.52 [+0.15, +0.75]** | **+0.45 [+0.12, +0.71]** |

\* coefficient-selected. Col4a4 and Col4a6 are not on the panel.
- **Same direction:** Lamb1 (both cohorts) and Hspg2 (chronic) decline with Col4a1/2.
- **No clear trend:** Lama4, Lamc1, Nid1 and Nid2.
- **Opposite direction:** Col4a5, detected in only 5% of cells.

**A6 — technical controls** (ρ with day):
- **Endothelial (both cohorts) and chronic VSMC:** no trend in cell count, median counts per cell
  or median genes per cell.
- **RR VSMC:** cell count rises with day (+0.40 [0.03, 0.67]). Repeating A2 with median counts
  per cell as an extra covariate leaves the result unchanged: Col4a1+2 −0.55 [−0.74, −0.22].

---

## Task B — non-lesion tissue control

**Question.** Does non-lesion tissue date the animal as well as lesion tissue does?

**Verdict.**
- **RR:** yes. Non-lesion tissue dates the animal as well as lesion tissue (paired difference
  +0.02 [−0.10, +0.14]), and the result does not depend on the margin.
- **Chronic:** the result **moves with the margin.** Non-lesion ρ falls from 0.82 at 0 µm to
  0.62 at 150 µm. With the segmentation's false-positive rate (2.4 lesions per control
  section), the compartments are not cleanly separable in chronic, or the chronic signal is
  partly lesion-proximal. This test cannot tell those apart.

Fold clocks are frozen: each animal is scored by the LOAO clock that never saw it. Section
predictions are averaged per animal. A section enters a compartment only with ≥ 200 cells
there.

| | RR | chronic |
|---|---|---|
| cells / animal: lesion (median, min) | 5,845, 53 | 2,287, 10 |
| cells / animal: non-lesion 50 µm | 18,494, 12,638 | 10,824, 4,608 |
| animals usable: lesion / non-lesion | 30 / 33 | 24 / 34 |
| **B2** non-lesion (50 µm) ρ | 0.82 [0.61, 0.92] | 0.76 [0.47, 0.91] |
| lesion compartment ρ | 0.83 [0.64, 0.92] | 0.93 [0.83, 0.97] |
| **B3** non-lesion − lesion, same animals | +0.02 [−0.10, +0.14], P(≤0) 0.34, n = 30 | +0.00 [−0.13, +0.11], P(≤0) 0.44, n = 24 |
| **B5** margin 0 µm | 0.83 [0.64, 0.92] | 0.82 [0.58, 0.93] |
| **B5** margin 150 µm | 0.83 [0.62, 0.93] | **0.62 [0.27, 0.84]** |
| 0 − 50 µm, paired | +0.01 [−0.04, +0.08] | +0.06 [+0.00, +0.15] |
| 150 − 50 µm, paired | +0.01 [−0.04, +0.07] | **−0.13 [−0.28, −0.03]** |
| **B4** de novo non-lesion clock ρ (zero-feature folds) | 0.83 [0.65, 0.92] (0/33) | 0.88 [0.72, 0.95] (0/34) |
| B4: non-zero genes / overlap with the published 135 / same sign | 111 / 58 / 58 | 118 / 18 / 14 |

- **Zero-feature folds:** B2/B3/B5 use the frozen whole-tissue fold clocks (0/33 and 0/34
  zero-feature folds).
- **Chronic B3 caveat:** on the 24 chronic animals with a usable lesion compartment, non-lesion
  ρ is 0.93 too. The lower full-cohort value comes from the 10 animals without lesion tissue.
- **B4 overlap:** the published 135-gene list is the RR clock, so low chronic overlap is
  expected.

---

## Task C — composition-only clock

**Question.** Does the clock read cell-type proportions or cell-intrinsic state?

**Verdict.**
- **Mainly cell state.**
  - Coarse composition alone predicts much less than expression in RR. Its gap in chronic is
    not distinguishable from zero.
  - Composition adds nothing on top of expression.
  - Expression keeps a substantial signal after composition is removed (ρ ≈ 0.5, CI excluding
    0), though it loses about a third.
- **Finer composition (`Anno_L2`)** gets closer to expression. Finer labels are partly state
  labels (Reactive_AST, DAO, MDM, EAE-associated fibroblasts), so the composition/state line
  blurs at that level.

| | RR | chronic |
|---|---|---|
| expression clock (reference) | 0.83 [0.62, 0.93] (0/33) | 0.85 [0.65, 0.94] (0/34) |
| **C1** composition, `Anno_L1_curated` (17 types) | 0.47 [0.14, 0.72] (0/33) | 0.66 [0.40, 0.83] (0/34) |
| **C2** L1 − expression, paired | −0.36 [−0.64, −0.12], P(≤0) 1.00 | −0.19 [−0.48, +0.09], P(≤0) 0.92 |
| **C1** composition, `Anno_L2` (36 types) | 0.70 [0.44, 0.84] (0/33) | 0.75 [0.52, 0.88] (0/34) |
| **C2** L2 − expression, paired | −0.13 [−0.27, −0.03], P(≤0) 1.00 | −0.10 [−0.33, +0.08], P(≤0) 0.85 |
| **C3** expression + L1 composition | 0.81 [0.60, 0.92] (0/33) | 0.85 [0.64, 0.94] (0/34) |
| C3 − expression, paired | −0.02 [−0.05, +0.00] | −0.00 [−0.03, +0.03] |
| **C4** expression residualised on composition | 0.50 [0.16, 0.74] (0/33) | 0.52 [0.18, 0.74] (0/34) |
| C4 − expression, paired | −0.33 [−0.64, −0.08], P(≤0) 1.00 | −0.34 [−0.61, −0.13], P(≤0) 1.00 |

- **Features and labels.**
  - L1 features exclude the non-cell-type labels Doublet and T_B_doublet. L2 features exclude
    ARTIFACT, Doublet, T_B_doublet, Mixed and the two LowQuality labels. All cells stay in the
    proportion denominators.
- **How C3 and C4 were fit.**
  - C3 exempts the composition columns from the top-variance filter, which would otherwise drop
    them because of their tiny variance.
  - C4 residualises genes on severity plus the in-fold PCA of log-ratio L1 proportions to 80%
    variance (3 components in every fold). Using all 17 proportions on ~32 training animals
    would nearly saturate the regression.
- **C5 — non-zero composition coefficients (full-data fit; positive = associated with later
  day | severity):**
  - L1, RR: Fibroblast +3.67, Astrocyte +2.68, Schwann cell +1.07; T cell −1.69, Epithelial
    −1.64, Endothelial −1.46, Neutrophil −1.22, Oligodendrocyte −0.80, Myeloid −0.74, DC −0.52.
  - L1, chronic: Fibroblast +4.41, Astrocyte +3.51; Epithelial −2.98, Endothelial −2.29,
    Neuron −1.86, Oligodendrocyte −1.82, T cell −1.54, VSMC −1.15.
  - The same direction in both cohorts: fibroblasts and astrocytes up with time; endothelium,
    T cells and epithelium down.

## Flagged as uninterpretable here

- **Task B, chronic margin dependence:** not separable from segmentation false positives
  without an independent lesion reference.
- **Task C at the L2 level:** composition and state are not distinct.
- **Task A, severity-positive / duration-negative pattern:** either reading (vascular response
  to acute inflammation that recedes with duration, or a duration-linked decline) fits; this
  analysis cannot separate them.

---

## Task A follow-up — is the duration signal created by the severity adjustment?

`analysis/severity_adjustment.py`, `severity_adjustment/results.json`. Endothelial Col4a1,
Col4a2 and their mean. Coefficient-selected genes (see A4 for the rank null).

### 1. Unadjusted block

| | RR (n = 33) | chronic (n = 34) |
|---|---|---|
| day vs severity | +0.14 [−0.23, +0.47] | +0.58 [+0.26, +0.79] |
| Col4a1 vs day | −0.48 [−0.71, −0.17] | −0.29 [−0.60, +0.05] |
| Col4a2 vs day | −0.51 [−0.74, −0.22] | −0.27 [−0.60, +0.09] |
| mean vs day | −0.49 [−0.72, −0.19] | −0.28 [−0.60, +0.09] |
| mean vs severity | +0.50 [+0.14, +0.76] | +0.45 [+0.13, +0.69] |

### 2. Stratified, no regression (Col4a1+Col4a2 mean vs day within severity bands)

| scheme | band (score range) | RR n | RR ρ | chronic n | chronic ρ |
|---|---|---|---|---|---|
| tertiles | lowest (RR 0–0.75; chr 0–0.5) | 14 | −0.67 [−0.89, −0.21] | 15 | +0.25 [−0.28, +0.79] |
| tertiles | middle (RR 1–2.25; chr 1–1.5) | 10 | −0.85 [−0.97, −0.46] | 8 | −0.57 [−0.96, +0.31] |
| tertiles | highest (RR 2.5–3.25; chr 2.25–3.25) | 9 | −0.59 [−0.99, +0.16] | 11 | −0.91 [−0.99, −0.61] |
| tertiles | **combined** | 33 | **−0.72 [−0.92, −0.46]** | 34 | −0.46 [−0.76, +0.05] |
| fixed | score < 1 | 14 | −0.67 [−0.89, −0.21] | 15 | +0.25 [−0.28, +0.79] |
| fixed | 1 ≤ score < 2 | 7 (too small) | −0.95 [−1.00, −0.69] | 8 | −0.57 [−0.96, +0.31] |
| fixed | score ≥ 2 | 12 | −0.54 [−0.93, +0.13] | 11 | −0.91 [−0.99, −0.61] |
| fixed | **combined** | 33 | **−0.72 [−0.92, −0.47]** | 34 | −0.46 [−0.76, +0.05] |

- **Band sizes:** 7–15 animals per band. Single-band CIs are wide, so only the combined
  estimates are informative.
- **Chronic lowest band:** mostly non-symptomatic and onset animals, so "low severity" there
  is partly "pre-disease".

**Severity-matched pairs** (|Δscore| ≤ 0.25). The fraction where the longer-duration animal
has lower Col4:
- **RR:** 0.78 of 111 pairs (32 animals); sign test p = 1×10⁻⁹; animal-bootstrap CI 0.62–0.92.
- **chronic:** 0.63 of 92 pairs (34 animals); sign test p = 0.016; animal-bootstrap CI
  0.41–0.86.

Pairs share animals, so the sign tests are optimistic; the animal-bootstrap CI is the honest
interval.

### 3. Which "severity"?

ρ with `day_of_sacrifice`: terminal score RR +0.14 / chronic +0.58; cumulative +0.73 / +0.90;
peak +0.45 / +0.74; days since onset +0.98 / +0.94 (n = 28 / 23). Terminal vs cumulative
+0.55 / +0.73; terminal vs peak +0.81 / +0.91. Full matrix in `results.json`.

Col4 mean vs day, adjusted for each in turn:

| adjustment | RR | chronic |
|---|---|---|
| terminal score | −0.66 [−0.81, −0.38] | −0.74 [−0.85, −0.55] |
| cumulative score | −0.55 [−0.75, −0.16] | −0.45 [−0.63, −0.20] |
| peak score | −0.61 [−0.78, −0.34] | −0.66 [−0.80, −0.46] |
| days since onset | −0.13 [−0.52, +0.36] (n 28) | −0.30 [−0.61, +0.15] (n 23) |

The result is stable across the three severity measures. It disappears only when "days since
onset" is the covariate. That variable is a duration measure (ρ 0.94–0.98 with day), not a
severity measure, so that adjustment removes the quantity being tested.

### 4. Project-wide: how much does the clock depend on the adjustment?

| clock adjustment | RR ρ (zero-feature folds) | chronic ρ |
|---|---|---|
| none | 0.80 [0.59, 0.90] (0/33) | 0.86 [0.69, 0.93] (0/34) |
| terminal score (current) | 0.83 [0.62, 0.93] (0/33) | 0.85 [0.65, 0.94] (0/34) |
| cumulative score | 0.78 [0.56, 0.90] (0/33) | **0.64 [0.39, 0.82]** (0/34) |
| none − terminal, paired | −0.03 [−0.12, +0.06] | +0.01 [−0.10, +0.13] |
| cumulative − terminal, paired | −0.05 [−0.28, +0.18] | **−0.21 [−0.41, −0.03]** |

Each ρ is computed against its own target. Selected-gene overlap (full-data fits):

| | none & terminal | none & cumulative | terminal & cumulative | all three |
|---|---|---|---|---|
| RR (84 / 99 / 95 genes) | 57 | 20 | 16 | 12 |
| chronic (117 / 129 / 108 genes) | 72 | 71 | 59 | 48 |

**Reading.**
- **RR:** the clock's accuracy does not depend on the adjustment choice, but its gene list
  does (12 genes shared by all three versions).
- **Chronic:** adjusting for cumulative severity lowers accuracy significantly. Cumulative score
  is itself almost a duration measure in chronic (ρ 0.90 with day).

### 5. Pre-specification for IHC

`IHC_prespecification.md`, committed in `dac8bbf` before any staining.

### Does the duration signal survive without regression adjustment?

**In RR, yes.**
- **Without adjustment:** endothelial collagen IV falls with day (ρ −0.49 [−0.72, −0.19]).
- **Within every severity tertile:** it falls (combined −0.72 [−0.92, −0.46]).
- **Severity-matched pairs:** the longer-duration animal has lower collagen IV in 78% (CI
  62–92%).
- **Why adjustment isn't driving it:** severity and duration are nearly independent in RR, so
  adjustment cannot be creating the signal.

**In chronic, no.**
- **Without adjustment:** the association is not distinguishable from zero (−0.28 [−0.60,
  +0.09]).
- **Stratified:** the combined estimate includes zero (−0.46 [−0.76, +0.05]); only the
  highest-severity band shows a clear decline, and the lowest band points the other way.
- **Severity-matched pairs:** not different from chance (63%, CI 41–86%).
- **Why:** in chronic, severity rises with duration (ρ 0.58), so the adjusted result there
  depends on conditioning on a variable downstream of duration. It should not be reported as
  an independent replication.
