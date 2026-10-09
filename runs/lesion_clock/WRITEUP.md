# Lesion clock — new lesions or growing lesions?

2026-10-09. `analysis/lesion_segment.py` (segmentation), `analysis/lesion_clock.py` (dating).
Every test uses the animal as the unit; animal counts are given next to every statistic.

## 3b / 3c — the question

**Lesion-age spread does not widen with disease duration in either cohort. The whole
distribution shifts together. RR and chronic do not differ in spread at matched duration.**
But a random-gene-set clock produces the same picture, so this is not evidence specific to the
duration clock. It cannot separate "new lesions" from "growing lesions".

| | RR | chronic |
|---|---|---|
| animals (EAE, ≥ 3 lesions) | 30 | 27 |
| within-animal SD of lesion age vs day | ρ −0.01 [−0.40, +0.37] | ρ −0.18 [−0.52, +0.21] |
| mean lesion age vs day | ρ 0.81 [0.57, 0.92] | ρ 0.87 [0.74, 0.92] |
| median within-animal SD (days) | 9.6 | 7.9 |
| median measurement-noise SD (days, transcript sampling only) | 1.1 | 1.0 |
| SD vs number of lesions | ρ −0.49 | ρ −0.03 |

- **3c, RR − chronic spread adjusted for day:** +0.9 days, 95% CI [−0.8, +2.5] (30 vs 27 animals).
  Restricted to the overlapping day range (11–49; 30 vs 22 animals), Mann–Whitney p = 0.044.
  That unadjusted test does not hold up once day is modelled.
- **Random-gene null (4a; 100 sets of 99 genes, the clock's non-zero size):** the observed
  spread-vs-day ρ sits at the 70th percentile (RR) and the 84th (chronic) of the null. The
  RR − chronic difference sits at the 44th. None of the 3b/3c statistics is specific to the
  clock genes.

**Prediction supported:** the "growing / shared state" prediction (spread flat, distribution
shifts) fits better than "new lesions accrue" (spread widens). But the null shows that any
gene set scored this way gives that shape.

## Why within-animal lesion ages cannot be read as lesion age

- **Lesion means track the animal (2c below), so the clock does resolve to lesions through the
  animal-wide signal they share.**
- **The within-animal spread is noise of unknown size.** It is ~9× the transcript-sampling noise
  floor, but that floor ignores differences in cell composition between lesions (the dominant
  source). It is not identified as age.
- **Lesion size barely relates to predicted age within animals:** RR median ρ −0.01 (33
  animals), chronic +0.27 (31 animals; 68% positive).
- **Immune fraction and cell density don't drive it (4b):** within-animal ρ with predicted age
  ranges from −0.32 (RR immune fraction) to +0.18, all below the |ρ| > 0.5 trigger, so the
  residualised rerun was not needed.

## 2c — validation (passed)

Rule fixed before scoring: lesion dating is valid if the per-animal mean of lesion predictions
correlates with the residualised day target at ρ ≥ 0.5 with a CI excluding 0. Lesions were
scored with the leave-one-animal-out clock that never saw the animal; the existing pipeline was
not refit or changed.

| cohort | animals | lesion-mean ρ | section clock ρ (same animals) | lesion-mean ρ, size residualised |
|---|---|---|---|---|
| RR | 33 | 0.81 [0.58, 0.91] | 0.83 [0.63, 0.92] | 0.81 [0.58, 0.91] |
| chronic | 34 | 0.84 [0.66, 0.93] | 0.85 [0.66, 0.95] | 0.86 [0.67, 0.94] |

## 1 — lesions (descriptive)

- **Segmentation:** lesionSegmenter ported to Xenium.
  - Nuclei → cell centroids.
  - Pu.1⁺ → `Anno_L1_curated` Myeloid or DC (the user's choice). A rule needing at least 3 of
    Spi1/Csf1r/Aif1/Cd68/Itgam is available as a sensitivity run but was not run.
  - Everything else uses lesionSegmenter's own functions and config: 10 µm grid, σ 40 µm,
    robust z ≥ 2.5, myeloid gates 0.15 / 0.2, ≥ 5,000 µm², rim 50 / peri 150 µm.
  - Lesion unit = mask + 50 µm perilesional margin.
- **Counts:** 989 lesions in 157 sections (median 6 per section, range 1–16). Area quartiles:
  9,800 / 20,300 / 56,700 µm².
- **False positives:** control animals carry 40 lesions (2.4 per section). Some of these are
  meningeal or perivascular myeloid clusters the gates do not remove; treat lesion counts as
  upper bounds.
- **Lesions per animal (EAE animals):**
  - RR median 21 (30 animals), chronic median 10 (29 animals), Mann–Whitney p = 6×10⁻⁹.
  - Count vs day: RR ρ −0.10 [−0.59, +0.33], chronic +0.18 [−0.22, +0.53].
  - Lesion-area fraction vs day: RR +0.36 [−0.05, +0.66], chronic **+0.59 [+0.22, +0.83]**.

## What this cannot determine

- **Whether any individual lesion is older than another.** Within-animal differences are not
  validated as age, and their noise floor is unknown.
- **New vs growing lesions.** The spread test lacks specificity (random-gene null).
- **Myeloid-label segmentation calibration.** It has not been compared against IHC lesion
  outlines. Only 5 animals overlap with lesionSegmenter (Task 0 gate), all from different cuts.

Files: `lesions.csv`, `sections.csv`, `lesions_per_animal.csv`, `lesion_predictions.csv`,
`per_animal_lesion_age_distribution.csv`, `results.json`, `figures/lesion_age_spread.png`.
