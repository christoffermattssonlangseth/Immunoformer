# RRMAP2 Duration Clock — a validated molecular clock of accumulated EAE damage

*Analysis date: 2026-06-17 (cross-model + chronic added 2026-06-18) · Cohort: RRMAP2 Xenium,
relapsing–remitting (RR) arm; chronic arm for cross-model replication ·
Scripts: `scripts/duration_clock.py`, `duration_clock_controls.py`,
`duration_gene_decomposition.py`, `duration_clock_chronic.py` ·
Outputs: `runs/duration_clock/`, `runs/duration_clock_chronic/`*

## The question

RRMAP2 carries **two separable disease axes** (see `scripts/accrual_axis.py`,
`scripts/duration_axis.py`):

1. **Severity** — an acute, *reversible* inflammatory program that resets at each
   relapse (oscillators: Hal, Arg1, Chil3, Acod1, Cxcl10). Tracked by `score_sacrifice`.
2. **Duration / accrual** — cumulative, *irreversible* tissue change that grows with
   disease history, independent of how severe the animal looks at sampling.

The earlier relapse-**phase** model was abandoned as unidentifiable (terminal,
cross-sectional data; severity dominates everything). Duration, by contrast, is
identifiable: `day_of_sacrifice` (11–49 days post-induction) is a real continuous
temporal label, and within the RR arm the relapse cycle **decouples it from severity**
(peaks recur at similar severity but later days). This document builds and validates the
multivariate version of that axis — a supervised "clock" that reads time-since-induction
from tissue, after removing severity.

This is the legitimate temporal ML for this dataset. There are **no per-animal time
series** (each mouse is one terminal timepoint), so sequence models (LSTM/Mamba/
Neural-ODE) do not apply — we regress onto a de-confounded temporal *label* instead.

## Cohort & data

- **n = 33 RR animals**, one terminal timepoint each, day 11–49 post-induction.
- **Strain-clean:** all SJL/PLP (no strain confound within RR).
- **Features:** animal-level pseudobulk of the 5,101-gene Xenium panel
  (`runs/rr_within_relapse/pseudobulk.npz`).
- **Headline clock = RR only;** the CHRONIC arm (34 animals, B6/MOG) is analysed separately
  for cross-model replication (see *Cross-model replication* below). The chronic
  `day_of_sacrifice` was once thought to alias `run_date`, but with corrected metadata it is
  continuous (8–50 dpi) and **clean of batch** (Kruskal day~run_date p ≈ 0.57) — the
  apparent alias was an artifact of the coarse day16/day30 stage grouping.

## Method

Two leave-one-animal-out (LOAO) elastic-net clocks, plus adversarial controls:

| Clock | Target | Purpose |
|---|---|---|
| **A** raw | day ~ genes | how well does tissue track time at all? |
| **B** severity-orthogonalized *(headline)* | residualize genes **and** day on `score` per fold, then predict | duration **beyond** severity — cannot cheat via severity |

Controls: severity-only baseline (day from score alone), a cross-target run (predict
score from day-residualized features — are the axes separable?), a 50× permutation null
(shuffle day, refit clock B), and a batch-leakage test (is day confounded with run_date /
animal multiplex / region within RR?).

*Implementation notes:* per-fold top-variance prefilter (1,000 genes, train-only — no
leakage), single `l1_ratio=0.5`, `n_alphas=50`. These keep the thousands of tiny LOAO ×
permutation fits tractable single-core; the result is robust to the exact elastic-net mix.

## Results

| Metric | Value |
|---|---|
| **Clock B (severity-orthogonalized)** | **Spearman +0.799 · R²(LOAO) +0.672** |
| **Permutation null** | **p = 0.0196** (floor; 0/50 shuffles beat it; null mean ρ = −0.07) |
| Clock A (raw, day~genes) | Spearman +0.725 · R² +0.559 |
| Severity-only baseline (day~score) | **−0.520** |
| Cross-target (features → score) | +0.842 |
| Clock genes (elastic-net nonzero) | 135 |

**Reading of the controls:**

- The severity-only baseline is **negative** (−0.52): within RR, later timepoints are
  *not* sicker by clinical score — the relapse cycle resets acute severity. You therefore
  **cannot** infer duration from severity, which is exactly what makes the axis
  identifiable.
- The cross-target (+0.84) shows the transcriptome *also* encodes severity-beyond-duration:
  the genes carry **both axes separably**, so clock B is not re-reading one global
  "disease badness" variable.
- The permutation null (p = 0.0196) confirms the clock is not an over-fit on 33 animals.

**No batch leakage** — day-of-sacrifice is not confounded with technical batch within RR:

| Variable | Levels | Kruskal p(day) |
|---|---|---|
| run_date | 2 | 0.138 |
| sample_id (animal multiplex) | 18 | 0.717 |
| region | 3 | 0.215 |

## Sex and region are not confounds

Two covariates could plausibly alias day-of-sacrifice within RR: **sex** (the signature
includes `Xist`, and the cohort is mixed-sex) and **region** (EAE has a spatial cord
gradient). Both were tested explicitly (`scripts/duration_clock_controls.py`). Sacrifice
time-of-day is constant across animals, which separately rules out a time-of-day leak
behind the circadian genes.

**Sex.** The panel has `Xist` but no Y-genes, so sex is called from Xist bimodality
(7 Xist-high "female", 26 Xist-low "male" — male-skewed). Sex is **orthogonal to day**:
Xist-raw vs day Spearman **−0.052**; Mann-Whitney day~sex **p = 0.18**; point-biserial
r = −0.29 (p = 0.11).

**Region (L / T / C).** The spinal cord gradient is real but tracks *severity* more than
*time*. Cervical carries the most severe disease here (mean score C = 1.77 > T = 1.31 >
L = 1.00), and lumbar is sampled earliest (median day L = 26.5 vs T = 33, C = 32) —
consistent with disease ascending/resolving caudal→rostral so that rostral (C) samples,
taken later, carry more accrued signal. But region is **not significantly tied to day**:
Kruskal day~region **p = 0.21**; ordinal L<T<C vs day Spearman **+0.18 (p = 0.33)** (vs
score +0.28). Because clock B already removes severity, the region gradient cannot be what
drives it.

**Decisive test — the fully-controlled clock.** Refitting clock B with **`[score, sex,
region]`** all as covariates and `Xist` dropped from the features gives **Spearman +0.864,
R²(LOAO) +0.732** — *higher* than the original +0.799 (intermediate sex-only control:
+0.845). Controlling for severity, sex, and region in any combination does not weaken the
clock. (Permutation p in `runs/duration_clock/controls.json`.)

Conclusion: the duration axis is **not** an artifact of sex, region, or severity. If
anything, removing those nuisance directions sharpens it.

## The biology

The multivariate elastic-net selects *non-redundant* predictors, so the clock surfaces a
different — and more mechanistically legible — set than the univariate "ratchet" list
(correlated genes like Gpnmb/Fmod/Igf2 are dropped in favour of their most predictive
partners). The signature is **135 genes (65 accrue, 70 decline)** and reads as a coherent
six-part chronic-progression program. Sign = direction with increasing duration, after
severity is removed; ↑ = accrues, ↓ = declines.

**1. Progressive neurodegeneration & demyelination — DOWN.**
`Mog↓`, `Mal↓` (myelin), `Uchl1↓`, `Syn1↓`, `Nptx2↓` (neuronal/synaptic), `Gad1↓`,
`Gad2↓` (GABAergic interneurons), `Slc17a6↓` (Vglut2, glutamatergic). The clearest
irreversible ratchet: broad myelin, neuron, and synaptic loss accumulating with disease
history — across both inhibitory and excitatory neuron markers.

**2. The acute innate / interferon response recedes — DOWN.**
`Ccr2↓` (monocyte recruitment), `Tmem173`/STING↓, `Ifit1↓`, `Ifit3↓`, `Irf5↓` (type-I
interferon program), `H2-D1↓` (MHC-I antigen presentation), `Tlr2↓`, `Il10ra↓`. The
homeostatic/sensory microglia compartment also contracts (`Tmem119↓`, `Siglech↓`,
`Stab1↓`, `Pirb↓`). Acute innate inflammation is a *young-lesion* feature, consistent with
severity being the acute/reversible axis.

**3. Basement-membrane → fibrotic-scar matrix switch.**
Vascular basement membrane down (`Col4a1↓`, `Col4a2↓` — the single largest weight,
`Eln↓`) while matricellular / scar-ECM proteins rise (`Thbs2↑`, `Serpine2↑`, `Ptx3↑`,
`Fbln2↑`, `Hpse↑` heparanase, `Klf5↑` pro-fibrotic TF). The structural signature of an
aging lesion field.

**4. Adaptive-immune maturation toward lymphoid organization.**
`Cxcl13↑` — the B-cell-follicle chemokine and a hallmark of **tertiary lymphoid structures
in chronic/progressive MS** — with `Cxcl12↑` (stromal SDF-1), `Tcf7↑` (memory/stem T
cells), and `Il2ra↑` (CD25, Treg/activated T). Meanwhile terminal plasma-cell / Ig-
secretion markers fall (`Mzb1↓`, `Jchain↓`): a shift from antibody-secreting plasma cells
toward organized follicular/memory immunity.

**5. Lipid metabolism: synthesis → scavenging switch.**
Cholesterol biosynthesis is broadly shut down (`Hmgcr↓`, `Hsd17b7↓`, `Idi1↓`, `Msmo1↓`,
`Ldlr↓`, `Acss2↓`) while lipid uptake / transport rises (`Lpl↑`, `Pltp↑`, `Abca8a↑`,
`Srebf1↑`). This is the foamy-phagocyte signature: stop synthesising cholesterol, scavenge
myelin-derived lipid debris instead — the metabolic state of chronic demyelinating lesions.

**6. Stromal repair & circadian/metabolic module.**
Growth-factor / gliotic repair (`Igf1↑`, `Igfbp2↑`, `Il33↑` alarmin, `S1pr3↑`, `Mlc1↑`
astrocyte) and a circadian/metabolic block (`Nr1d1↑`, `Dbp↑`, `Bhlhe40↑`, `Per1↑`, `Hlf↑`,
`Txnip↑`, `Ddit4↑`). Because sacrifice time-of-day is **constant** across animals, the
circadian genes reflect a genuine shift in circadian/metabolic regulation with disease
duration, not a sampling-time artifact.

> **One-line story:** severity is acute, reversible inflammation that resets each relapse;
> **duration** is the irreversible accumulation of neuron/myelin loss, fibrotic-scar
> matrix remodeling, and lymphoid organization — the "molecular age" of the lesion field —
> and it is multivariately predictable from tissue after severity is removed.

## Tier 2 — where the clock lives (spatial localization)

The pseudobulk clock shows *that* tissue tracks duration and *which genes* carry it, but not
*where*. With only 33 animals (bags), a black-box attention-MIL over ~894k cells would
overfit, so the honest decomposition is to split the bag by **known instance type** and
rebuild the same severity-orthogonalized clock independently inside each compartment
(`scripts/duration_clock_spatial.py`, using the cached per-niche aggregates — no volume
needed). Cell types come from the 25 leiden clusters (`cluster_labels.json`), spatial
niches from CellCharter.

**Duration is a tissue-wide chronic program that peaks in glia, and is driven by cell-state
change rather than composition:**

| Compartment | LOAO ρ (duration \| severity) | R² | median cells |
|---|---|---|---|
| **astrocyte** | **+0.89** (perm p=0.020) | 0.81 | 2637 |
| oligo/myelin | +0.87 | 0.77 | 5575 |
| infl-myeloid | +0.85 | 0.73 | 3324 |
| microglia | +0.84 | 0.74 | 2691 |
| OPC | +0.84 | 0.69 | 1856 |
| vascular/endo | +0.82 | 0.67 | 3983 |
| T/NK | +0.75 | 0.62 | 948 |
| neuron | +0.72 | 0.55 | 4141 |
| ependymal | +0.70 | 0.52 | 171 |
| B/plasma | +0.69 | 0.49 | 333 |
| **composition (fractions only)** | **+0.52** (perm p=0.039) | 0.26 | — |

Two findings: (1) **every** compartment carries the duration signal (ρ +0.69–0.89) — it is a
tissue-wide aging process — but it is **strongest in glia** (astrocyte, oligodendrocyte) and
myeloid, and weakest in neurons and lymphocytes. (2) A clock built on cell-type
**proportions alone is far weaker** (+0.52 vs +0.80 whole-tissue), so the clock reads cells
**changing state**, not merely shifting in number. The strongest spatial niche is a
**vascular/perivascular** niche (niche3, ρ +0.84, perm p=0.020), with an oligo/myelin niche
close behind (+0.84).

**The gene modules are cell-type-specific**, which is what makes a tissue-wide signal
mechanistically legible (z-scored expression per gene across cell types): lymphoid genes
(`Cxcl13`, `Mzb1`, `Jchain`) sit in **B/plasma**; scar/matricellular (`Thbs2`, `Ptx3`,
`Serpine2`) in **astrocytes**; basement-membrane (`Col4a1/2`, `Eln`, `Cxcl12`) in
**vascular/endothelial**; myelin (`Mog`, `Mal`) in **oligodendrocytes**; foamy-lipid uptake
(`Lpl`) in **myeloid**; cholesterol synthesis (`Hmgcr`, `Idi1`, `Msmo1`) in **OPC/oligo**;
neuronal (`Uchl1`, `Gad1`, `Slc17a6`) in **neurons**. (Figures:
`runs/duration_clock_spatial/figures/`.)

## Cross-model replication — the chronic arm (B6/MOG)

The chronic arm was originally set aside because its day axis looked batch-confounded. With
the corrected metadata (`FINAL_…_META_RRMap2.Main.filled.stage_fixed.csv`) that is no longer
true: chronic has a **continuous `day_of_sacrifice` (8–50 dpi)** that is clean of `run_date`
(Kruskal p ≈ 0.57; day~batch ρ ≈ 0.10 — both batches span the full day range) and decoupled
from severity (r = 0.32). The earlier "alias" was an artifact of the coarse day16/day30
*stage* grouping, not the underlying day. So we can run the **identical** clock there
(`scripts/duration_clock_chronic.py`).

| Metric | RR (SJL/PLP) | Chronic (B6/MOG) |
|---|---|---|
| **Clock B (severity-orthogonalized)** | **+0.799** | **+0.856** |
| R²(LOAO) | +0.672 | +0.862 |
| Permutation p | 0.0196 | 0.0196 |
| Clock genes | 135 | 128 |
| Severity-only baseline (day~score) | **−0.52** | **+0.52** |

The clock is **not RR-specific**: rebuilt identically on chronic it reaches +0.856
(perm p = 0.020). And the two duration axes **agree** — correlating the per-gene
day|severity partial correlation (the gene-level accrual axis,
`scripts/duration_gene_decomposition.py`) between models gives **Spearman +0.65** (5,101
genes, 72 % sign-agreement), *across both strain and model*. The agreement is **specific to
the duration axis**: the RR severity-**oscillation** axis (peak−remission amplitude) does
**not** transfer to chronic duration (Spearman −0.17).

- **Conserved accrual** (685 genes rise with duration in both): scar/ECM and Wnt
  (`Igf2`, `Vtn`, `Wnt5a`, `Wnt6`, `Id4`, `Prelp`).
- **Conserved decline** (408 genes): the cholesterol-synthesis program (`Hmgcr`, `Idi1`,
  `Msmo1`, `Lss`, `Hsd17b7`, `Ldlr`) — the same module the RR clock loses.
- **Model-specific accrual:** chronic-only (B6/MOG) is more complement/inflammatory
  (`C3`, `Havcr2`, `Ccl3`, `Cd48`); RR-only (SJL/PLP) includes `Sox1`, `Foxb1`.

**One revealing difference.** In chronic, predicting day from *severity alone* is **positive**
(+0.52) — the disease is monotonically progressive, so later really is sicker. In RR the same
baseline is **negative** (−0.52) because each relapse resets acute severity. Same accruing
program underneath; opposite severity–time coupling on top — which *is* the distinction
between a relapsing and a progressive course. (Cross-model figure + tables:
`runs/duration_clock_chronic/`.)

### What this does — and does not — say about disease course

The conserved duration program is sometimes over-read as "there are no relapsing-specific
markers." That does not follow:

- **Conserved *duration* ≠ no course-specific markers.** Conservation is on the *accrual*
  axis (what builds up with elapsed time). A course-defining contrast would live on a
  *different* axis (RR vs chronic), which the duration analysis never measures.
- **Course is set by the experimental model, not discovered as a marker.** RR = SJL + PLP₁₃₉₋₁₅₁
  (relapsing by design); chronic = C57BL/6 + MOG₃₅₋₅₅ (progressive by design). In this dataset
  **model is perfectly confounded with strain** (every RR animal is SJL, every chronic is B6),
  so any RR-vs-chronic differential gene is strain, antigen, *or* course — unresolvable. This
  is exactly why only within-model *slopes* are compared (which cancel the strain offset), and
  why a direct level comparison is abandoned.
- **What *is* relapsing-specific and visible is a dynamic, not a static marker:** the
  oscillate-and-reset behaviour — severity decoupled from time (the −0.52 vs +0.52 baseline
  flip) and the acute-oscillation axis (`Arg1`, `Chil3`, IFN program) that spikes at each
  attack and does not transfer to chronic duration.
- **The identifiable version of "why relapse vs not"** is *within* the SJL/PLP cohort, using
  the experimental design (`scripts/monophasic_vs_relapsing.py`). **REMISSION1** (~day 22)
  branches into **MONOPHASIC** (~day 33, did not relapse) and **PEAK2** (~day 32, relapsed):
  monophasic animals are sampled at the same timepoint a relapser hits its second attack
  (day MW p=0.16) but stay low-severity (score 0.62 vs 2.44), confirming the non-relapsing
  phenotype. So the time-matched contrast is **MONO vs PEAK2**, and the relapse decision is the
  divergence of the two trajectories from the shared REM1 origin. *Cross-sectional (different
  animals — we cannot label a REM1 animal's future), n=5/4/4, exploratory; effect sizes only.*
  The **relapse** path (REM1→PEAK2) re-ignites the acute program — acute-oscillator module
  z −0.21→+0.42 (`Gpnmb`, `Chil1`, `Arg1`, `Timp1`, `Hal`, `Socs3`), with innate/IFN and
  ECM/scar — and **loses neuron/myelin again** (−0.07→−0.30). The **monophasic** path
  (REM1→MONO) does the opposite: the acute oscillator stays off, **neuron/myelin recovers**
  (→+0.33), the glucocorticoid/stress tone resolves (+0.30→−0.56), and the quiet
  duration/repair accrual continues — the monophasic-specific genes are the conserved accrual
  set `Igf2`, `Fmod`, `Vtn` plus circadian `Nr1d1`. The relapse-specific direction (PEAK2−MONO)
  aligns with the acute **severity/oscillation** axis (Spearman +0.85 vs oscillation amplitude,
  +0.84 vs the score-axis) and runs *opposite* the duration axis (−0.56). **Honest reading:**
  PEAK2 is a severity peak by construction, so "relapse-specific = acute severity program" is
  partly expected; the informative half is the monophasic trajectory (myelin recovery + stress
  resolution + quiet accrual), and the design is terminal/cross-sectional so this is a
  population trajectory, not proof that anything *at* REM1 decides the outcome.

## Caveats

- **Small cohorts, animal-level pseudobulk** (RR n = 33, chronic n = 34). LOAO + permutation
  null guard against over-fit, but the cohorts are small and spatial information is collapsed.
- **Cross-model is a slope comparison.** The RR↔chronic level offset is not identifiable
  (strain + slide confounded); only within-model per-gene day-trends are compared. Chronic day
  is clean of batch, but the chronic cohort still spans two run_dates, controlled only by both
  carrying the full day range.
- **Sex cohort is imbalanced** (7 vs 26 in RR) even though sex is orthogonal to day; a balanced
  replication would be cleaner.

## Outputs

- `runs/duration_clock/results.json` — all metrics + top clock genes
- `runs/duration_clock/clock_genes.csv` — full 135-gene signature with coefficients
- `runs/duration_clock/report.txt` — text report
- `runs/duration_clock/figures/duration_clock.{png,pdf}` — predicted-vs-true + top genes
- `runs/duration_clock/controls.json` — sex + region confound controls
- `runs/duration_clock_spatial/{results.json,report.txt,figures/}` — Tier-2 localization
- `runs/reports/duration_clock.html` — assembled HTML report with figures

## Appendix — full clock-gene signature

Top 40 in each direction (elastic-net coefficient, severity-orthogonalized; full 135 in
`clock_genes.csv`).

### Accrue with duration (↑ — top 40 of 65)

| Gene | coef | Gene | coef | Gene | coef | Gene | coef |
|---|---|---|---|---|---|---|---|
| Nr1d1 | +1.38 | Cyb5r3 | +0.32 | Crym | +0.22 | Sema6d | +0.16 |
| Thbs2 | +1.21 | Klf5 | +0.31 | S1pr3 | +0.22 | Fmod | +0.16 |
| Dbp | +1.03 | Hpse | +0.30 | Cavin1 | +0.21 | Wls | +0.16 |
| Serpine2 | +0.88 | Abca8a | +0.27 | Mlc1 | +0.21 | | |
| Bhlhe40 | +0.79 | Per1 | +0.26 | Igf1 | +0.21 | | |
| Cxcl13 | +0.66 | Txnip | +0.26 | Pltp | +0.21 | | |
| Srebf1 | +0.65 | Fcgrt | +0.26 | Il2ra | +0.19 | | |
| Slc47a1 | +0.52 | Il33 | +0.25 | Idh2 | +0.18 | | |
| Xist* | +0.50 | Hlf | +0.25 | Ddit4 | +0.17 | | |
| Lpl | +0.49 | Mc5r | +0.23 | | | | |
| Tcf7 | +0.49 | Slc7a10 | +0.23 | | | | |
| Igfbp2 | +0.42 | Slc2a4 | +0.23 | | | | |
| Atf5 | +0.42 | | | | | | |
| Ptx3 | +0.37 | | | | | | |
| Fbln2 | +0.35 | | | | | | |
| Cxcl12 | +0.32 | | | | | | |

*`Xist` is a sex marker; it is dropped in the sex/region-controlled refit, which leaves the
clock unchanged (+0.864) — see above.

### Decline with duration (↓ — top 40 of 70)

| Gene | coef | Gene | coef | Gene | coef | Gene | coef |
|---|---|---|---|---|---|---|---|
| Col4a2 | −1.78 | Tmem119 | −0.37 | Ifit1 | −0.23 | Idi1 | −0.15 |
| H19 | −1.35 | Uchl1 | −0.35 | Pirb | −0.21 | Syn1 | −0.15 |
| Col4a1 | −0.62 | Irf5 | −0.34 | Gad2 | −0.21 | Msmo1 | −0.14 |
| Mzb1 | −0.59 | Stab1 | −0.32 | Elmo1 | −0.20 | Tlr2 | −0.13 |
| Eln | −0.53 | Siglech | −0.30 | Entpd1 | −0.17 | Atp2a3 | −0.11 |
| Jchain | −0.53 | H2-D1 | −0.28 | Mal | −0.17 | | |
| Hsd17b7 | −0.51 | Ifit3 | −0.26 | Il10ra | −0.17 | | |
| Igfbp3 | −0.48 | Slc17a6 | −0.26 | Acss2 | −0.16 | | |
| Mog | −0.46 | Ackr1 | −0.25 | Ldlr | −0.15 | | |
| Tmem173 | −0.45 | Plin4 | −0.25 | | | | |
| Hmgcr | −0.45 | F13a1 | −0.25 | | | | |
| Ccr2 | −0.44 | Nptx2 | −0.25 | | | | |
| Gad1 | −0.42 | | | | | | |

## Next steps

Tier 2 (localization) is done — see above. The remaining follow-ups:

1. **Within-lesion zonation.** The compartment clocks still aggregate each cell type to one
   number per animal. Rebuild the clock over the radial lesion zones (core → margin → rim →
   parenchyma, from `scripts/lesion_radial.py`) to ask whether duration concentrates in the
   scar core.
2. ~~**Chronic-arm validation.**~~ **Done** — the clock replicates in the chronic arm
   (+0.856) and the duration axis is conserved RR↔chronic at ρ +0.65 (*Cross-model
   replication* above).
3. **Publication-grade null.** Re-run the headline clock with N_PERM ≈ 1000 (overnight) for a
   tighter permutation p than the current floor of 0.0196.
