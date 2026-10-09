# Pre-specification: vascular collagen IV protein vs severity and duration (RR only)

**Version 2, written 2026-10-09, before any staining or image scoring.** It supersedes version 1
of the same date (commit `dac8bbf`), which covered both cohorts. To be committed (timestamped)
before images are scored; no edits after scoring begins, deviations in a dated addendum.

## Why RR only

In the chronic cohort, severity and duration are not separable:
- `day_of_sacrifice` correlates with terminal severity at +0.58 and cumulative severity at +0.90.
- Within chronic, the transcript-level collagen IV duration signal appears only after severity
  adjustment and is not distinguishable from zero without it.

So chronic is **uninformative** for this question, not a failed replication. In RR, severity and
duration are nearly independent (ρ +0.14), and the transcript signal holds without any
adjustment (`runs/closing_block/WRITEUP.md`, Task A and follow-up).

## Hypothesis being tested

Within RR endothelial cells (Xenium), Col4a1/Col4a2 transcripts:
- **fall with duration at matched severity:** ρ(day | severity) −0.66 [−0.81, −0.38];
  unadjusted −0.49 [−0.72, −0.19]; within every severity tertile;
- **rise with severity at matched duration:** ρ(severity | day) +0.66.

Among genes with the same severity coupling they rank at the 97th–98th percentile, so this is
not generic severity behaviour.

**The protein prediction is a two-way grid:**
- **at matched severity,** longer duration → **weaker or patchier** vascular collagen IV;
- **at matched duration,** higher severity → **stronger** vascular collagen IV.

| | short duration | long duration |
|---|---|---|
| **high severity** | strongest | intermediate |
| **low severity** | intermediate | **weakest / patchiest** |

## Measurement

- **Stain:** collagen IV (pan) immunofluorescence with CD31 (Pecam1) as vessel mask; DAPI.
- **Vessels:** CD31⁺ objects ≥ 20 µm² in white and grey matter. Meningeal vessels are excluded
  (outer 50 µm of tissue).
- **Primary measure — coverage fraction:** collagen IV⁺ area within the CD31 mask dilated by
  2 µm ÷ that dilated area. Lower coverage = "weaker or patchier".
- **Secondary measure:** mean collagen IV intensity in the same ring, background-subtracted
  against an annulus 5–10 µm out.
- **Per animal:** median over vessels, ≥ 30 vessels per section, one section per region (C, T,
  L) where available, regions averaged. The animal is the unit.
- **Lesion status:** vessels inside segmented lesions are reported separately. The primary
  analysis uses non-lesion vessels.
- **Blinding:** images scored blind to animal, stage, score and day. Thresholds are fixed on two
  PLP-CFA control animals (RR_CFA_1, RR_CFA_2) before any EAE image is opened.

## Animals (RR EAE)

Split at the median terminal score (1.125) and median `day_of_sacrifice` (31.5). \* = an IHC
section of the same animal already exists in lesionSegmenter (different cut), possibly easiest
to restain.

| | short duration (≤ 31.5 d) | long duration (> 31.5 d) |
|---|---|---|
| **low severity (≤ 1.125)** | RR_OS1_1, RR_OS1_2\*, RR_OS2_1, RR_OS2_2, RR_R1_1, RR_R1_2\*, RR_R1_3\*, RR_R1_4, RR_R1_6 (n = 9) | RR_MP_1, RR_MP_2, RR_MP_3, RR_MP_4, RR_R2LONG_1, RR_R2_5 (n = 6) |
| **high severity (> 1.125)** | RR_P1_1, RR_P1_2, RR_P1_3, RR_P1_5, RR_P2M_1, RR_P2M_2 (n = 6) | RR_P2_3\*, RR_P2_4, RR_P3_1\*, RR_P3_2, RR_P3_3, RR_P3_4, RR_P3_5, RR_R2_3, RR_R2_4 (n = 9) |

**Minimum to stain:** at least 4 animals per cell. With fewer, report descriptively only. Which
animals still have usable tissue is not recorded here and must be confirmed.

## Analysis (fixed)

- **Primary test:** within each severity stratum, Mann–Whitney (long vs short) on per-animal
  coverage fraction, one-sided (long < short). The two strata are combined by Stouffer's method.
- **Secondary test:** within each duration stratum, Mann–Whitney (high > low severity),
  one-sided.
- **Supportive:** Spearman ρ(coverage, day) unadjusted, and ρ(coverage, day | terminal score),
  with 2000-resample animal bootstrap CIs.

## What falsifies the hypothesis

- **Falsified at protein level:** at matched severity, long-duration animals show **equal or
  higher** vascular collagen IV coverage than short-duration animals. That means the combined
  one-sided p > 0.2 with point estimates in the opposite direction in both strata.
- **Discordance, not support:** a clear increase with duration, matching published collagen IV
  protein accumulation in MS lesions (van Horssen 2006; Ghorbani & Yong 2021). It would mean
  the transcript decline does not translate into less basement-membrane protein.
- **Severity prediction only:** if only the secondary prediction holds, the transcript duration
  signal is unsupported at protein level.
