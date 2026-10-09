# Task 4 — the baseline ladder: write-up

2026-10-08/09. Numbers from `runs/baseline_ladder/report.txt` (full tables, every arm),
`results.json`, and the diagnostics listed below. Nothing here is committed.

## Bottom line

**Inconclusive.** At n = 28–33 RR and 23–29 chronic animals, the attention-MIL model
(arm 6) does not measurably exceed animal-level pseudobulk (arm 2) on any target, and
pseudobulk does not measurably exceed arm 6 either. That is a statement about the power of
this test, not evidence that cell-level structure adds nothing.

- On `day_of_sacrifice`, arm 6 has the **higher point estimate and the narrowest CI on the
  board**: ρ 0.885 [0.75, 0.94] (width 0.20) vs arm 2 0.828 [0.62, 0.93] (width 0.30).
  Fisher z p = 0.40.
- On every other target arm 6 is at or below arm 2, also with overlapping CIs
  (smallest p = 0.19, `days_since_last_peak`).
- Pre-registered rule: overlapping CIs = INCONCLUSIVE. All comparisons are inconclusive.

**The comparison is not symmetric.** Arm 2 is a tuned linear model: ElasticNetCV picks its
regularisation inside every training fold. Arm 6 is a deep model with no inner tuning
(fixed 30 epochs, learning rate, width), 3 seeds, ~27 training animals per fold. A deep
model losing in that setting is the expected outcome whether or not the architecture has
anything to offer, so this test is weak. That weakness is itself the finding.

## Main table (ρ, 95% animal-bootstrap CI, CI width)

| target (cohort, n) | resid. on | arm 2 pseudobulk | arm 2r + run | arm 6 attention-MIL | arm 6r + run | 6 vs 2 p |
|---|---|---|---|---|---|---|
| `day_of_sacrifice` (RR, 33) | score | 0.83 [0.62, 0.93] 0.30 | 0.79 [0.58, 0.89] 0.31 | **0.89 [0.75, 0.94] 0.20** | — | 0.40 |
| `days_since_last_peak` (RR, 28) | score+day | 0.70 [0.40, 0.88] 0.47 | 0.45 [0.11, 0.73] 0.62 | 0.46 [0.05, 0.76] 0.71 | 0.35 [0.01, 0.62] 0.62 | 0.19 |
| `onset_day` (RR, 28) | — | 0.62 [0.33, 0.80] 0.47 | 0.14 [−0.25, 0.49] 0.73 | 0.43 [0.08, 0.70] 0.62 | 0.04 [−0.34, 0.40] 0.74 | 0.37 |
| `cumulative_score` (chronic, 29) | score+day | 0.73 [0.51, 0.87] 0.36 | 0.75 [0.48, 0.89] 0.42 | 0.66 [0.35, 0.86] 0.51 | — | 0.62 |
| `slope_sign` (RR, 30) | score | 0.32 [−0.02, 0.62] 0.64 | 0.17 [−0.16, 0.47] 0.64 | 0.16 [−0.23, 0.55] 0.78 | — | 0.54 |
| `onset_day` (chronic, 23) | — | −0.07 [−0.48, 0.32] 0.80 | 0.23 [−0.25, 0.60] 0.86 | 0.23 [−0.21, 0.64] 0.86 | — | 0.34 |

Arm 6 = 3-seed ensemble. Run-adjusted comparison (6r vs 2r): `days_since_last_peak`
p = 0.65, `onset_day` RR p = 0.73. Arms 1, 3, 4, 5 are in `report.txt`; arm 5 (pseudobulk +
niche) never improves on arm 2; arms 3 and 4 (composition only) are below it everywhere.
Fisher z treats the two ρ as independent; they share animals, so the p-values are
conservative. Permutation null for arm 2 on `day_of_sacrifice`: 1000 shuffles, null mean
−0.06, max 0.70, **p = 0.001** (the floor).

**Null targets, reported without interpretation:** `slope_sign` (RR) and `onset_day`
(chronic). On chronic `onset_day`, arms 3 and 4 show ρ = −0.99 / −1.00: in 19/23 and 22/23
folds no feature survived, so the prediction is the leave-one-out mean, which is
anti-correlated with the held-out value by construction. There is **no −1 sentinel in the
data**: the 11 chronic animals without an onset (5 controls, 6 never symptomatic) are NaN
and excluded; the 23 values used run from day 6 to day 20.

## Run sensitivity (Task 1b consequence)

Two RR targets differ by Xenium run (onset median 13 vs 12 days; `days_since_last_peak`
median 1 vs 7) and must be read from the run-adjusted arms 2r/6r:
`onset_day` RR collapses for both models (2r 0.14, 6r 0.04); `days_since_last_peak` falls
to 0.45 / 0.35. `day_of_sacrifice` and chronic `cumulative_score` are run-robust (2r 0.79,
0.75). Run residualisation is fit on each training fold's animals only, for both arms.

## Diagnostics requested at review

1. **Arm 5 = arm 2?** Not by construction. The top-variance prefilter is applied to genes
   only; all 27 niche fractions reach the scaler and elastic net in every fold and get
   non-zero coefficients in every fold for 4/6 targets (median 2–6 fractions; most often
   `Lesion_Lipid`, `Vasculature`, `Lesion_myeloid III`). Arm 5 and arm 2 predictions differ
   (max per-animal difference 0.7–6.7 target units); the ρ coincide on two targets only.
   Reading: pseudobulk already carries the composition signal. No rerun of arms 3–5.
   (`arm5_coefficient_check.{txt,csv}`)
2. **Arm 6 run-adjusted** — table above; residualisation fit in-fold only.
3. **Seed variance on `day_of_sacrifice`:** per-seed ρ 0.887 / 0.902 / 0.875 (range
   0.027). The arm 6 − arm 2 gap (0.057) is about twice the seed range, so seed noise does
   not explain it; animal sampling (CI widths 0.20–0.30) does. Other targets: range
   0.07–0.17. (`arm6_seed_spread.csv`)
4. **Undertrained on `days_since_last_peak`?** No. By epoch 30 the model fits its training
   animals (training ρ 0.93, IQR 0.91–0.94; 0.98 by epoch 60–120). Held-out ρ stays at
   0.37–0.45 at every checkpoint to epoch 120 and held-out MSE stays ≥ 1.05 (no better than
   predicting the training mean). Control target `day_of_sacrifice`: held-out ρ 0.85–0.89,
   MSE ≈ 0.22 throughout. So the gap is generalisation from ~27 training animals, not
   training length: "the architecture does not help here at this n", not "undertrained".
   The held-out curve is test-informed and was not used to choose epochs.
   (`arm6_training_diagnostic.{txt,csv,json}`)
5. **Task 1b chronic anomaly (ρ 0.85 → 0.91 with run).** (a) Run residualisation is fit on
   training animals only; the leak-free fold-wise reference moves the same way (0.853 →
   0.892). (b) R² of `day_of_sacrifice` on run in chronic = 0.0003; R² of gene expression
   on run: median 0.07, 386 genes > 0.5. (c) The two ρ are against different targets
   (day|score vs day|score+run, correlation 0.974). On the **same** target the gain is
   0.852 → 0.880; the rest is the target change. Mechanism: run is unrelated to day but
   strongly related to the predictors, so removing it removes nuisance variance from X —
   batch correction, not leakage. (`runs/clock_run_robustness/chronic_anomaly_check.txt`)

## Pre-registered dissociation — FAILED

Prediction: the ratchet program (Gpnmb, Igf2, Fmod, Fcrls, Plin4) tracks
`days_since_last_peak` and the acute program (Hal, Arg1, Chil3) tracks `slope_sign`, not the
reverse. Result: ratchet vs `days_since_last_peak` ρ +0.22 [−0.15, +0.56]; acute vs
`slope_sign` ρ −0.02 [−0.37, +0.31]. Neither holds. The ratchet program instead tracks
`day_of_sacrifice` (ρ +0.68 [+0.41, +0.83]).

Consequence: `runs/accrual_axis` is **downgraded**. The PEAK1 vs PEAK3 cumulative finding
(p = 0.016, n = 4 vs 5) must no longer be cited as two separable programs; PEAK3 animals
were sacrificed on days 38–49 vs 14–18 for PEAK1, and the simpler reading is a single
dominant time axis. Downgrade notes added (dated 2026-10-08) to `runs/accrual_axis/report.txt`,
`ROADMAP.md`, `docs/rrmap2-duration-clock.md`, `docs/rrmap2-relapse-atlas.md`,
`docs/project-overview.md`, `docs/relapse-phase-model-design.md`. Task 0 step 7 points the
same way: the accrual program's long memory timescale collapses once elapsed time is
partialled out in RR (ρ 0.68 → 0.12), and the acute program's 1-day timescale is current
severity (partial on `score_sacrifice` −0.10).

## Arm 6b — mean-pooling control (supplementary)

Same model, encoder, folds, epochs and 3 seeds as arm 6, with attention switched off
(every cell weighted equally). It separates "deep vs linear at this n" from "does attending
to individual cells help".

| target | arm 2 pseudobulk | arm 6 attention | arm 6b mean-pool | 6 vs 6b p |
|---|---|---|---|---|
| `day_of_sacrifice` (RR, 33) | 0.83 [0.62, 0.93] | **0.89 [0.75, 0.94]** | 0.84 [0.66, 0.92] | 0.46 |
| `days_since_last_peak` (RR, 28) | **0.70 [0.40, 0.88]** | 0.46 [0.05, 0.76] | 0.55 [0.18, 0.79] | 0.66 |
| `cumulative_score` (chronic, 29) | **0.73 [0.51, 0.87]** | 0.66 [0.35, 0.86] | 0.71 [0.47, 0.87] | 0.70 |
| `onset_day` (RR, 28) | **0.62 [0.33, 0.80]** | 0.43 [0.08, 0.70] | 0.47 [0.12, 0.72] | 0.87 |
| `slope_sign` (RR, 30) | 0.32 [−0.02, 0.62] | 0.16 [−0.23, 0.55] | 0.18 [−0.19, 0.52] | — |
| `onset_day` (chronic, 23) | −0.07 [−0.48, 0.32] | 0.23 [−0.21, 0.64] | 0.32 [−0.16, 0.66] | — |

Reading (all comparisons inconclusive, as above):
- **Mean pooling ≈ pseudobulk.** The same deep pipeline, fed the cell average, lands at
  roughly arm 2's level (e.g. `day_of_sacrifice` 0.84 vs 0.83, `cumulative_score` 0.71 vs
  0.73). So the deep pipeline per se is not what loses at this n.
- **Attention adds nothing visible over mean pooling on 5/6 targets** (mean-pool is equal
  or higher). The one exception is `day_of_sacrifice`, where attention is ahead
  (0.89 vs 0.84, p = 0.46) — the only hint that weighting individual cells helps, and it is
  well inside the CIs.
- Taken with the training diagnostic: the attention model fits training animals well but
  generalises worse than its own mean-pooled version on the harder targets, consistent
  with the extra attention parameters overfitting ~27 training animals.

## Learning curve on animals (exploratory, beyond the work order)

`learning_curve.{txt,csv,json}`. Same LOAO folds; training animals subsampled to 50% and
75% (3 draws each, encoder refit on the subsample, one arm 6 seed per draw).

| target | fraction (~n train) | arm 2 ρ | arm 6 ρ (single seed) |
|---|---|---|---|
| `day_of_sacrifice` | 0.50 (16) / 0.75 (24) / 1.00 (32) | 0.71 / 0.80 / 0.83 | 0.76 / 0.86 / 0.89 |
| `days_since_last_peak` | 0.50 (13) / 0.75 (20) / 1.00 (27) | 0.48 / 0.59 / 0.70 | 0.05 / 0.32 / 0.42 |

On the harder target arm 6 gains +0.37 from half to all animals vs +0.22 for arm 2: it is
still on the steep part of its curve, which leans toward a data-scale limit. On
`day_of_sacrifice` the curves are parallel with arm 6 ~0.05 ahead at every size. Three
draws per point — indicative only.

## Task 5

The fork has triggered (arm 6 does not beat arm 2 at p < 0.05 on any target). Stage 2 is
not built. Deliverable: `docs/negative-result-scaling.md`.
