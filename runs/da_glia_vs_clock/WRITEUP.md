# DA-glia (Kukanja et al., Cell 2024) vs the duration clock

2026-10-09. `analysis/da_glia.py`, `analysis/da_glia_step3_control.py`. RR n = 33, chronic
n = 34 animals; 95% animal-bootstrap CIs throughout. No zero-feature folds in any clock fit.

## Step 3 — is the clock the DA response? (lead)

**Neither pre-set outcome holds cleanly. Residualising the five DA-glia scores inside each fold
cuts the clock, but no more than residualising five random scores of the same shape.**

| covariates residualised in-fold (RR) | ρ, baseline target (day \| severity) | ρ, own target |
|---|---|---|
| severity only (baseline) | 0.83 [0.62, 0.93] | — |
| severity + 5 DA-glia scores | 0.37 | 0.67 |
| severity + 5 random pseudo-DA scores (20 replicates) | mean 0.51 (5th pct 0.18, min 0.14) | mean 0.71 (5th pct 0.51) |

- **Paired against the baseline,** the DA-residualised clock loses 0.45 [−0.80, −0.14]
  (Wilcoxon p = 0.005, 33 animals). Taken alone, that would read as "the clock collapses".
- **The random control** uses random marker sets of the same sizes, in the same cell types,
  scored the same way. The real DA drop sits at the 15th percentile of that null (baseline
  target) and the 30th (own target).
- **What follows:** the drop is the generic cost of residualising five expression-derived
  covariates on 33 animals. Any such score absorbs part of the time signal, which runs through
  most of the transcriptome (the gene-set nulls in `runs/clock_composition`). It is not
  DA-specific.
- **Not shown:** that the clock *is* the DA-glia programme, so there is no amendment to the
  Cell paper's resolution claim. Also not shown: a second, independent slower layer. The test
  cannot separate them at this n.

## Step 2 — severity or duration?

**The Cell paper's prediction holds:** DA scores track current severity strongly and carry no
positive duration signal once severity is removed. In RR they are *lower* in later animals at
matched severity, which is resolution.

| cohort | DA state | (a) ρ with score | (b) ρ with day, raw | (c) ρ with day \| severity |
|---|---|---|---|---|
| RR | DA-MOL2 | +0.84 [0.65, 0.94] | +0.02 [−0.33, 0.35] | −0.18 [−0.50, 0.18] |
| RR | DA-MOL5/6 | +0.81 [0.61, 0.93] | −0.12 [−0.44, 0.22] | **−0.39 [−0.63, −0.08]** |
| RR | DA-OPC/COP | +0.71 [0.44, 0.88] | −0.25 [−0.55, 0.07] | **−0.50 [−0.73, −0.17]** |
| RR | DA-Astro | +0.84 [0.66, 0.95] | −0.11 [−0.43, 0.23] | **−0.42 [−0.64, −0.10]** |
| RR | DA-MiGL | +0.57 [0.25, 0.81] | −0.09 [−0.45, 0.25] | −0.21 [−0.59, 0.17] |
| chronic | DA-MOL2 | +0.94 [0.88, 0.97] | +0.54 [0.21, 0.76] | −0.03 [−0.35, 0.25] |
| chronic | DA-MOL5/6 | +0.94 [0.86, 0.97] | +0.54 [0.18, 0.77] | −0.03 [−0.37, 0.31] |
| chronic | DA-OPC/COP | +0.90 [0.77, 0.95] | +0.49 [0.13, 0.72] | −0.11 [−0.41, 0.24] |
| chronic | DA-Astro | +0.94 [0.86, 0.97] | +0.51 [0.15, 0.74] | −0.14 [−0.46, 0.24] |
| chronic | DA-MiGL | +0.92 [0.82, 0.96] | +0.50 [0.15, 0.72] | −0.12 [−0.39, 0.21] |

The raw day correlation in chronic (+0.5) is severity rising with day in that cohort. It
disappears after partialling.

## Step 5 — time course

**No DA score rises monotonically without resolving.**
- **Chronic:** every DA score rises and then falls, with the fitted peak at days 28–30.
- **RR:** scores are flat against day (ρ −0.21 to +0.11). The quadratic fits peak at days 20–34
  for DA-MOL2, DA-OPC/COP and DA-MiGL.

So none of the DA states is a candidate substrate for an accumulating clock.
(`figures/da_scores_vs_day.png`)

## Step 4 — overlap with the matrix set

- **One shared gene:** Col20a1. The DA-glia and matrix results are not one finding.
- **The 15 DA marker genes as a clock alone:** ρ 0.72 [0.48, 0.84]. Without Col20a1: 0.70.
  That is in the range random gene sets of similar size reach, so it is not evidence that DA
  genes are special time markers.
- **Matrix result:** its standing is set by `runs/clock_composition/circularity`. The clean
  matrix subset falls into its random null, so that result is not cited here.

## Markers used

Kukanja et al. 2024, **Figure 4B** ("selected DA-glia marker genes"), read from the published
figure. The full marker lists are in the paper's supplementary tables, which were not available.
For each state, the genes high in the DA row relative to its homeostatic counterpart:

| state | markers | cell type used |
|---|---|---|
| DA-MOL2 | Apod, Klk6, Serpina3n, (Serpina3h), C4b | Oligodendrocyte |
| DA-MOL5/6 | Il12rb1, Irgm1, Igtp, C4b | Oligodendrocyte |
| DA-OPC/COP | Col20a1, Serpina3n, Irgm1, Igtp, C4b, B2m, H2-D1 | OPC |
| DA-Astro | Mt1, Serping1, Serpina3n, (Serpina3h), C4b, Irgm1, Igtp | Astrocyte |
| DA-MiGL | B2m, H2-D1, Cd74, H2-Aa, Fcgr2b | Myeloid |

Serpina3h is not on the Xenium panel. Scores follow scanpy `score_genes` logic (25
expression bins, 50 controls per marker), computed per cell and averaged per animal and cell
type, exactly, via per-cell log-expression sums.

## What this cannot determine

- **Whether the clock and the DA programme share mechanism.** The step-3 test has no
  specificity at n = 33.
- **Full DA state definitions.** Only Figure 4B markers were used.
- **Mapping to cell types.** DA states were mapped to `Anno_L1_curated` types, not to the
  paper's own DA/homeostatic clusters.
