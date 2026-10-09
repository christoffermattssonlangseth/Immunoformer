# Pre-specification: vascular collagen IV protein vs severity and duration

**Written 2026-10-09, before any staining or image scoring.** To be committed (timestamped)
before images are scored. Do not edit after scoring begins. Deviations go in a dated addendum.

## Hypothesis being tested

From Xenium, within endothelial cells:
- **Duration:** Col4a1/Col4a2 transcripts are lower in longer-duration animals at matched
  clinical severity (ρ(day | severity) −0.66 RR, −0.74 chronic).
- **Severity:** they are higher in more severe animals at matched duration (ρ(severity | day)
  +0.66 RR, +0.78 chronic).

The protein-level prediction, if transcription sets protein in the vessel basement membrane:

| | short duration | long duration |
|---|---|---|
| **high severity** | highest collagen IV | intermediate |
| **low severity** | intermediate | **lowest** collagen IV |

**Primary prediction:** within each severity stratum, long-duration animals show **lower**
vessel-associated collagen IV than short-duration animals.

**Secondary prediction:** within each duration stratum, high-severity animals show higher
collagen IV than low-severity animals.

## Measurement

- **Stain:** collagen IV (pan) immunofluorescence with CD31 (Pecam1) as vessel mask; DAPI.
- **Vessel mask:** CD31⁺ objects ≥ 20 µm² in white and grey matter. Meningeal vessels are
  excluded (outer 50 µm of tissue).
- **Per vessel:**
  - **primary — coverage fraction:** collagen IV⁺ area within the CD31 mask dilated by
    2 µm ÷ that dilated area;
  - **secondary:** mean collagen IV intensity in the same ring, background-subtracted against
    an annulus 5–10 µm out.
- **Per animal:** median over vessels, ≥ 30 vessels per section, one section per region (C, T,
  L) where available, regions averaged. The animal is the unit.
- **Lesion status:** vessels inside segmented lesions (lesionSegmenter, Pu.1/myeloid density)
  are reported separately. The primary analysis uses non-lesion vessels.
- **Blinding:** images scored blind to animal ID, stage, score and day. Thresholds are fixed on
  2 control animals before any EAE image is opened.

## Animals

Split at the cohort's median terminal score and median `day_of_sacrifice` among EAE animals
(RR: score 1.125, day 31.5; chronic: score 1.0, day 18). \* = an IHC section already exists in
lesionSegmenter (adjacent cut).

**RR**
- **low severity / short:** RR_OS1_1, RR_OS1_2\*, RR_OS2_1, RR_OS2_2, RR_R1_1, RR_R1_2\*,
  RR_R1_3\*, RR_R1_4, RR_R1_6
- **low severity / long:** RR_MP_1, RR_MP_2, RR_MP_3, RR_MP_4, RR_R2LONG_1, RR_R2_5
- **high severity / short:** RR_P1_1, RR_P1_2, RR_P1_3, RR_P1_5, RR_P2M_1, RR_P2M_2
- **high severity / long:** RR_P2_3\*, RR_P2_4, RR_P3_1\*, RR_P3_2, RR_P3_3, RR_P3_4, RR_P3_5,
  RR_R2_3, RR_R2_4

**chronic**
- **low severity / short:** C_NS_1–5, C_OS1_1–5. Caveat: non-symptomatic or onset animals, so
  this cell mixes "early" with "mild".
- **low severity / long:** C_L_1–5
- **high severity / short:** C_P1_1–6
- **high severity / long:** C_M16_1–3, C_S16_1–3, C_S30_1–2

**Minimum to stain:** at least 4 animals per cell per cohort. With fewer, report descriptively
only. Which animals still have usable tissue is unknown and must be confirmed.

## Analysis (fixed)

- **Primary test:** within each severity stratum, Mann–Whitney (long vs short) on per-animal
  coverage fraction, one-sided in the predicted direction. The two strata are combined by
  Stouffer's method, per cohort. The cohorts are reported separately and never pooled.
- **Supportive:** Spearman ρ(coverage, day | terminal score), with 2000-resample animal
  bootstrap CIs.

## What would falsify it

- **Falsified at protein level:** in both cohorts, long-duration animals show equal or *higher*
  coverage than short-duration animals within severity strata (combined one-sided p > 0.2 and
  point estimates in the opposite direction).
- **Discordance, not support:** an increase with duration, matching the published
  accumulation of collagen IV protein in MS lesions (van Horssen 2006; Ghorbani & Yong 2021).
  It would mean the transcript decline does not translate into less basement-membrane protein.
- **Severity prediction:** if only the secondary prediction holds, the transcript duration
  signal is unsupported at protein level.
