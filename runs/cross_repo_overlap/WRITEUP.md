# Task 0 (gate) — do the three repos share animals?

2026-10-09. Animal identity checked from each repo's own files, not its README.

## Answer

**5 animals overlap across all three repos. All are relapsing-remitting, from one Xenium
run (Jun 2025). The gate does not pass: 5 is near-zero for Task 4, which is skipped.**

## Table

| repo | data | animal key | animals |
|---|---|---|---|
| Immunoformer | RRMAP2 Xenium atlas | `sample_name` | 67 (34 chronic, 33 RR) |
| BeyondBoundaries | segmentation-kit stains, same RRMAP2 atlas file | `sample_name` (`config.yaml`: `animal: sample_name`) | 67 — every per-animal results table lists all 67 |
| lesionSegmenter | IHC whole-slide scans (DVP), different cuts | section prefix, e.g. `P2_3_T` | 6 named + 4 unassigned sections |

| intersection | n | animals |
|---|---|---|
| Immunoformer ∩ BeyondBoundaries | 67 | all; same atlas, same key |
| Immunoformer ∩ lesionSegmenter | **5** | RR_OS1_2, RR_P2_3, RR_P3_1, RR_R1_2, RR_R1_3 |
| all three | **5** | same five |
| lesionSegmenter only | 1 | `CFA_L2` (no RRMAP2 animal of that name; RR controls are RR_CFA_1–3, chronic C_CFA_1–6) |

## The five matched animals

| IHC name | RRMAP2 animal | stage | day of sacrifice | score | Xenium run | Xenium regions | IHC regions |
|---|---|---|---|---|---|---|---|
| OS1_2 | RR_OS1_2 | ONSET1 | 15 | 0.25 | Jun 2025 | C, L, T | C, L, + 1 unlabelled |
| P2_3 | RR_P2_3 | PEAK2 | 33 | 2.75 | Jun 2025 | C, L, T | C, L, T |
| P3_1 | RR_P3_1 | PEAK3 | 48 | 2.5 | Jun 2025 | C, L, T | C, L, T |
| R1_2 | RR_R1_2 | REMISSION1 | 25 | 0.75 | Jun 2025 | L, T | C, L, T |
| R1_3 | RR_R1_3 | REMISSION1 | 21 | 0.5 | Jun 2025 | C, L, T | C, L, T |

Matching is by name: the IHC section names are the RRMAP2 RR names without the `RR_`
prefix. The user confirmed (2026-10-09) that the IHC tissue comes from the same animals,
as different cuts. lesionSegmenter's own docs do not record provenance; the DVP folder
holds only the raw `.czi` scans. The 6 IHC animals appear twice (two scan layouts:
`CML_1` + `CML_2_rescan`, and `CML_metal` scenes 0/1), 17 sections each.

## Why Task 4 is skipped

- **n = 5 is near-zero for every Task 4 question.** A leave-one-animal-out model trained on
  4 animals (4b) is not a model. A Spearman correlation on 5 points (4c) reaches two-sided
  p < 0.05 only with a perfect ranking, and its bootstrap CI spans most of [−1, 1].
- **No variation to separate.** All 5 are RR, from one run. Two are REMISSION1, so the
  stage spread is ONSET1, REMISSION1 ×2, PEAK2, PEAK3.
- Per the work order, nothing is substituted.

What would reopen it: IHC on adjacent cuts from more RRMAP2 animals. At ~20 RR animals,
4c becomes a real test. Identifying `CFA_L2` would only add a control.
