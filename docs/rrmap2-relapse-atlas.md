# A spatial molecular atlas of the relapsing–remitting EAE cycle (RRMAP2)

*Draft writeup, 2026-06-15. Consolidates the within-RR analyses
(`scripts/rr_within_relapse.py`, `rr_cycle_oscillation.py`, `rr_phase_niche.py`)
and the `Hal` literature work ([`hal-histidine-finding.md`](hal-histidine-finding.md),
[`hal-histidine-deep-research.md`](hal-histidine-deep-research.md)). Figures:
[`relapse_cycle_oscillation.html`](../runs/reports/relapse_cycle_oscillation.html).*

## One-line summary

Across the relapsing–remitting EAE cycle, spinal-cord gene expression is organized
along a single dominant **disease-activity (severity) axis**, and the genes on it
split cleanly into two kinds: an **acute, reversible** program that flares at each
relapse and resets in remission, and an **irreversible, cumulative** program whose
floor ratchets up cycle after cycle. `Hal` is the most reproducible marker of the
acute program and localizes to an inflammatory antigen-presenting myeloid lesion
state.

## Framing: disease activity is the backbone, not a confound

This is a descriptive atlas. The organizing variable is **disease activity** —
indexed by the clinical sacrifice score (`score_sacrifice`, 0–3.25) and by the
relapse-cycle stage, which move together (peaks ≈ 2.6, remissions ≈ 0.8). We treat
that severity axis as the **expected backbone** the molecular changes hang on, and
we map what rides with it. (We separately asked whether anything carries *beyond*
severity — e.g. trajectory direction at matched severity — and found nothing
identifiable in this design; that negative is documented in
[`relapse-phase-model-design.md`](relapse-phase-model-design.md) §7b and is not the
subject here.)

## Dataset and design

- **RRMAP2 Xenium spatial transcriptomics**, mouse spinal cord, full 5101-gene
  panel; single-cell resolution with spatial coordinates and CellCharter niches.
- **Relapse-remitting cohort:** 33 animals (`sample_name`), ~894k cells. Analyses
  are **animal-level** (the unit of independence); within-RR contrasts are
  batch-clean (each stage spread across 9–12 slides).
- **Terminal / cross-sectional:** one stage per animal. The cycle below is a
  population pseudo-trajectory across different mice, **not** a within-animal time
  course — dynamics are *reconstructed*, not observed.
- **Cycle stages (n animals on the cycle, 29; + 4 monophasic):**
  PLP·CFA 3 → ONSET1 2 → ONSET2 2 → PEAK1 4 → REMISSION1 5 → PEAK2 4 →
  REMISSION2 4 → PEAK3 5.

## The two-axis architecture of the cycle

Decomposing each gene's profile along the cycle (peaks vs remissions amplitude;
remission-floor drift REM2−REM1; `scripts/rr_cycle_oscillation.py`) splits the
disease-associated genes into two behaviours:

### Axis 1 — acute, reversible disease activity (oscillators)

Spike at every peak, fall back at every remission, and the remission floor does
**not** rise — they reset each cycle. A coherent inflammatory / M2-repair /
interferon myeloid program:

| gene | peak−remission amplitude | floor drift (REM2−REM1) |
|---|---|---|
| `Arg1` | +1.45 | −0.72 |
| `Chil1` | +1.06 | −0.60 |
| `Chil3` | +1.01 | −0.45 |
| `Timp1` | +0.98 | −0.42 |
| `Acod1` (Irg1) | +0.83 | −0.47 |
| `Gbp2` | +0.74 | −0.46 |
| `Cxcl10` | +0.69 | −0.32 |
| **`Hal`** | **+0.68** | **−0.21** |

Read as the **acute relapse-activity** signature: it indexes *how active the
disease is right now* and is fully reversible.

### Axis 2 — irreversible cumulative accrual (ratchet)

Remission floor **rises** cycle over cycle — expression that persists after the
animal has clinically recovered. Lipid/foamy, ECM/scar, and homeostatic-microglia
repopulation:

| gene | floor drift (REM2−REM1) | note |
|---|---|---|
| `Igf2` | +0.65 | growth/repair |
| `Ptgds` | +0.65 | lipocalin/prostaglandin |
| `Fmod` | +0.59 | ECM / fibrosis |
| `Plin4` | +0.59 | lipid droplet / foamy |
| `Pmp22` | +0.43 | myelin-associated |
| `Gpnmb` | (rising; +1.3 peak-drift) | DAM / foamy macrophage |
| `Fcrls` | +0.37 | homeostatic microglia repopulation |

Read as **accumulated damage / remodelling** — the disease leaves a mark that each
remission does not erase.

### The continuous trajectory (severity-tracking backbone)

Along the ordered cycle (`scripts/rr_within_relapse.py`), **138 genes** move at
BH-q < 0.05. **Up:** inflammation / microglia / complement / humoral
(`C6`, `Fcrls`, `Igf1`, `Igkc`, `Cd109`). **Down:** cholesterol / sterol
biosynthesis, i.e. myelination (`Hmgcr`, `Msmo1`, `Idi1`, `Ldlr`, `Lss`). After
partialling out severity, the inflammation/complement rise is *cycle-specific* and
even strengthens (`Fcrls` +0.84, `Igf1` +0.82, `C6` +0.81 partial ρ), with `Cd47`
falling (−0.75, licensing myelin phagocytosis); the cholesterol/myelin decline is
partly a severity correlate (core sterol genes survive; rate-limiting `Hmgcr`/`Lss`
drop below threshold).

## `Hal` vignette — the standout marker of acute activity

- **Most reproducible disease-associated gene** in RRMAP2: top increasing gene over
  the relapse cycle, the only gene clearing FDR (q ≈ 4e-9) in the RR-vs-chronic
  differential-dynamics interaction, and the one gene robust to every analysis cut.
- **A clean acute oscillator:** spikes at PEAK1/2/3 (≈1.1–1.3 log CP10k) and resets
  at remissions (0.4–0.6); the oscillation is a faithful readout of acute disease
  severity.
- **Cell-type source (this data):** `Hal` is ~10× concentrated in one population
  (leiden_1 cluster 5: `Hal` 2.34 vs 0.22 median across clusters) — an
  **inflammatory antigen-presenting myeloid/macrophage lesion state**
  (`Arg1` 3.18, `Chil3` 3.00, `Cd14` 2.89, `Fn1` 3.99, `Cd74`/MHC-II, `Gpnmb`),
  with homeostatic-microglia markers (`P2ry12`, `Tmem119`, `Fcrls`) low. So `Hal`
  rides with **infiltrating/activated myeloid cells, not resident microglia or
  neutrophils** — sitting inside the same `Arg1`/`Chil3` program it co-oscillates
  with.
- **Biochemistry & caution:** `Hal` (histidine ammonia-lyase) commits histidine to
  catabolism (→ urocanate + ammonia). A tempting bridge to the human "histidine ↓
  in MS" metabolomics literature exists, but a dedicated deep-research pass found it
  weak (HAL absent from CNS parenchyma in healthy data; the histamine/HDC fork, not
  HAL, is the documented neuroinflammation route; serum/CSF discordance). For the
  atlas, `Hal` is best presented as a **robust molecular marker of the acute
  myeloid relapse state**, with the metabolic interpretation flagged as hypothesis.
  See [`hal-histidine-deep-research.md`](hal-histidine-deep-research.md).

## The spatial axis — L→T→C region gradient (within-animal)

Region is a **within-animal** axis: 32/33 RR animals span all of lumbar / thoracic /
cervical (~9k cells each). Centering each gene by its animal mean removes every
animal-level confound (severity, batch, strain), so this is the cleaner,
better-powered backbone. **772 genes** move at BH-q < 0.05 along the within-animal
region rank (`scripts/rr_region_gradient.py`).

- **The raw region axis is dominated by anatomy, not disease.** The top movers are
  **Hox positional-identity** genes — posterior `Hoxa9`/`Hoxb9`/`Hoxa10`/`Hoxc8`
  high in lumbar, anterior `Hoxa5`/`Hoxb5` high in cervical — plus ECM/collagen.
  This is developmental position, and any disease reading must be made against it.
- **The disease component is lumbar-biased.** Inflammation, complement and
  demyelination peak in **lumbar** cord and decline rostrally; neurons and intact
  oligodendrocyte/myelin increase toward **cervical** (cluster composition: neuron
  clusters 4/6 and oligo clusters 1/2/12 gain toward C; inflammatory-myeloid
  cluster 5, T/NK cluster 13 and B/plasma cluster 21 are lumbar/thoracic-biased).
  So cervical is relatively **spared**; lumbar is the **epicenter**.

## Lesion architecture (radial organization)

Segmenting inflammatory cells into lesion objects and profiling expression against
**signed distance to the lesion edge** (`scripts/lesion_radial.py`, 570 lesions)
recovers a textbook EAE/MS lesion structure de novo:

- **Core** (inside): inflammatory-myeloid and demyelinated — `myeloid_core` program
  +0.33 in the core falling to −0.16 outside; `myelin` program −0.45 in core →
  +0.09 in parenchyma (demyelinated centre, intact myelin outside).
- **Margin** (at the edge, −15 to −5 µm): lymphocytes (+0.28) and complement
  (+0.43) peak right at the boundary.
- **Rim** (just outside, +20–40 µm): a reactive-**astrocyte** wall — astrocyte
  program −0.15 in core → +0.28 in the rim (glial border).
- **Parenchyma** (>50 µm): spared myelin/neurons.

So the lesion is a concentric structure: myeloid/demyelinated core → lymphoid+
complement margin → astrocyte rim → spared parenchyma. Figure:
`runs/lesion_radial/figures/radial_profile.png`.

## Myeloid states — a recruitment → activation → resolution axis

The inflammatory-myeloid compartment is three distinct states
(`scripts/myeloid_states.py`), and they organize both temporally and spatially:

- **c18 = recruitment / antigen-presenting / IFN** (`Ccr2` 1.5, `Plac8` 2.4,
  `Ciita` 2.8, `Cd74` 6.5, `Cxcl10`). Sits most **peripherally** — mean −11 µm
  from the lesion edge (margin), where monocytes and T cells enter.
- **c5 = acute glycolytic / M2** (`Arg1` 4.6, `Chil3` 3.8, `Acod1` 3.0, `Hal` 2.9,
  the `Hal`-high state). Deepest in the **core** (−25 µm) and the clean acute
  **oscillator** (amplitude 0.59; 97% acute-oscillating genes).
- **c3 = repair / resident-like** (`Mrc1`, `Igf1`, `C6`). The persistent baseline.

The **c5↔c3 swing** drives the relapse cycle: within the myeloid compartment c5 is
38% at peak → 9% at remission (p=0.009) while c3 is 44% → 70% (p=0.014); c18 stays
flat (~19–21%). Radially, c18 is +12.6 µm more peripheral than c5 (p=8e-7) and
reactive astrocytes (c9) sit +44 µm outside the core (the rim). So **recruitment is
peripheral, activation is in the core**, and resolution is an *in-place temporal
hand-off* (c5 acute surge subsiding into the c3 baseline at the same location, not a
spatial migration). Ordering is inferred from a cross-sectional snapshot. Figures:
`runs/myeloid_states/figures/{state_share_vs_cycle,radial_position}.png`.

## Lesion signaling wiring (spatial ligand-receptor)

Spatial neighborhood-enrichment + ligand-receptor analysis on a *per-section* graph
(`scripts/lesion_signaling.py`; the precomputed global graph had 14.6% spurious
cross-section edges and was rebuilt block-diagonal) maps the lesion's wiring, and it
matches the radial architecture:

- **Adjacency:** T/NK (c13) are embedded in the myeloid core (z ≈ +100 to +146);
  B/plasma (c21) form a tight self-aggregate (z +567) adjacent to T cells
  (follicle-like); reactive astrocytes **avoid** the myeloid core (z −28 to −125) —
  i.e. the astrocyte wall is at the rim, spatially segregated from the core, exactly
  as the radial profile predicts; homeostatic microglia are excluded from the active
  core.
- **Signaling axes (significant, among spatially-adjacent lesion cell types):**
  **complement dominates** — `C3 → C3ar1`/`C5ar1`, with reactive **astrocytes** (not
  myeloid) as the top `C3` source signaling onto microglia and myeloid (notable
  given `C1q` is off-panel); an astrocyte→microglia **`Csf1`/`Il34 → Csf1r`**
  maintenance axis; chemokine **recruitment** (`Ccl2 → Ccr2`, `Cxcl10`/`Cxcl9 →
  Cxcr3`); **lymphoid organization** (`Cxcl13 → Cxcr5`, `Ccl19`/`Ccl21 → Ccr7`);
  and **costimulation** (`Cd80`/`Cd86 → Ctla4`/`Cd28`) wiring the myeloid-core /
  T-cell / B-aggregate triangle. Figures:
  `runs/lesion_signaling/figures/{nhood_enrichment,wiring_axes}.png`.

*(All of the above describes the signaling structure of a severity-driven state; it
is wiring, not a signal beyond severity.)*

## Space × time concordance (the key cross-axis result)

The per-gene region slope (L→T→C) and the relapse-cycle trend correlate
**Spearman −0.57** (over 1206 moving genes; −0.59 genome-wide;
`runs/rr_region_gradient/region_vs_cycle.csv`). The sign is the point: the program
that **rises over disease time** is the same program **concentrated in lumbar
space**. Space and time are two readouts of one disease trajectory — the spatial
epicenter (lumbar) is the temporally most-advanced state, and progression runs
caudal→rostral. This is the atlas's central structural claim and it is well powered
(within-animal).

## Cellular composition

The 25 leiden_1 niches label (by lineage signature) to oligo/myelin ×5, neuron ×4,
vascular/endo ×4, inflammatory-myeloid ×3, microglia ×2, astrocyte ×2, OPC ×2, and
one each of T/NK, B/plasma, ependymal. Along the **temporal** cycle the strongest
expanding niche is a reactive **astrocyte** state (cluster 9, ρ ≈ +0.87); the acute
relapse program centres on the **inflammatory-myeloid** niche (cluster 5, the
`Hal`-high state). Along the **spatial** axis, neuron/oligo niches expand toward
cervical while inflammatory/lymphoid niches concentrate caudally.

## Cross-model conservation — the program is strain-invariant

The atlas above is built on the RR (SJL/PLP) cohort. The **chronic** EAE cohort
(B6/MOG, different strain + antigen + disease course; `scripts/chronic_trajectory.py`)
has its own severity-graded trajectory (control → OS1 → MILD → SEVERE → PEAK1;
3,587 genes at BH-q < 0.05), and it is **batch-robust** (the two chronic run-dates'
independent severity slopes correlate +0.92).

Correlating the two models' *within-model* slopes cancels the strain offset that
makes a direct RR-vs-chronic comparison uninterpretable. The result:

> **Chronic severity trend vs RR relapse-cycle trend: Spearman +0.78** (5101 genes,
> p ≈ 0; +0.79 vs the RR peak−remission amplitude).

So the same disease program runs in both models. The **strain-invariant core**:
complement / DAM **up** (`C6`, `Gpnmb`, `C4b`, `Abca1`, `Stab1`, `Ctss`, `C3ar1`,
`Csf1r`), cholesterol-biosynthesis / myelin **down** (`Msmo1`, `Idi1`, `Hmgcr`,
`Lss`, `Mal`, `Plp1`, `Mog`). Composition agrees: microglia, T/NK, inflammatory-
myeloid and B/plasma niches expand with chronic severity; neurons, astrocytes and
oligodendrocytes contract — the same cellular shift as the RR cycle. Figure:
`runs/chronic_trajectory/figures/conservation_scatter.png`.

*Unique to chronic, but currently blocked:* a severity × **disease-duration** grid
(day-16 vs day-30 sacrifice at matched clinical score). In this cohort the day
suffix is perfectly confounded with run_date, so the "active inflammation vs chronic
accrual at equal severity" question is not yet answerable — it needs matched-batch
sampling. Flagged as the key wet-lab follow-up.

## Same clinical score, different molecular state — the accrual axis

The clinical score is a 1-D readout of *current* disease activity. It captures the
acute, reversible program but is largely blind to *accumulated, irreversible* damage,
which grows with disease **history** (number of relapse cycles). Comparing
matched-severity peaks (`scripts/accrual_axis.py`) shows this directly.

**The clean, strain-matched test is within RR** — PEAK1 (score 2.75) vs PEAK3
(score 2.65), near-identical clinical scores, same SJL/PLP strain:

- **Cumulative program rises** (z −0.97 → +0.13, Mann-Whitney **p = 0.016**):
  `Gpnmb` 1.9 → 2.7 → 3.2, `Fcrls` 0.2 → 0.8 → 1.0, `Igf2` 1.7 → 2.6 → 2.7,
  `Fmod` 0.5 → 1.1 → 1.2 across PEAK1→2→3.
- **Acute program does not accrue** — `Hal` is flat (1.32 / 1.10 / 1.19); the acute
  module is, if anything, highest at the *first* attack (`Arg1`, `Chil3`) and does
  not climb with cycle number (p = 0.19).

So **reaching the same clinical score at a later relapse means carrying more
accumulated damage on a comparable acute flare** — the same score is a molecularly
different state depending on history. This is the strain-clean demonstration that
severity is a lossy projection: it indexes the reversible axis, not the cumulative one.

**Chronic PEAK1 (score 2.96) added for context, but strain-confounded.** Its
absolute position mixes disease history *and* genetic background (B6/MOG vs SJL/PLP),
so it cannot be cleanly compared to the RR peaks. One striking confounded difference:
`Hal` — the headline RR acute marker — is essentially **absent at the chronic peak**
(0.03 vs ~1.2–1.3 in RR peaks), consistent with `Hal` being the one gene that cleared
FDR in the severity×model interaction; whether that is relapse biology or strain
cannot be resolved here. Figure: `runs/accrual_axis/figures/accrual_axis.png`.

### Duration axis — accrual tracks actual time, not just cycle number

`day_of_sacrifice` (continuous, ~11–49 days post-induction) is a direct measure of
disease duration. Within RR the relapse cycle decouples it from severity (rank
collinearity only +0.14: peaks recur at later days at the same score), so a partial
Spearman | `score_sacrifice` isolates the time effect cleanly and strain-matched
(`scripts/duration_axis.py`):

- **Cumulative program vs day | severity: partial ρ = +0.70 (p≈0)** — `Fmod` +0.81,
  `Igf2` +0.73, `Fcrls` +0.65, `Gpnmb` +0.55; 844 genes accrue with time after
  severity adjustment.
- **Acute program vs day | severity: partial ρ = −0.59 (p=0.0004)** — `Hal` flat
  (−0.05), `Arg1` −0.58. At matched severity, *later* disease carries more
  accumulated damage but a *weaker* acute flare.

**Chronic corroborates** (cumulative vs day | severity = +0.72, same direction and
magnitude), and its day-16 vs day-30 matched-severity pairs are the discrete version
— the day-30 "late" animals sit higher in foamy/scar genes (`Gpnmb`, `Cd68`, `Fmod`)
at the same clinical grade. But chronic's time axis is **confounded with run_date**
(day16 and day30 are different batches) and more collinear with severity (+0.58), so
the chronic duration effect is corroborating, not independently clean. Net: the
accrual axis is a genuine **duration** axis — damage scales with real
time-since-induction beyond current severity — shown cleanly in RR. Figure:
`runs/duration_axis/figures/duration_axis.png`.

## Beyond severity — tested to exhaustion, consistently null

A recurring question was whether anything carries information *beyond* current
disease severity (e.g. trajectory direction, or lesion architecture). It does not,
at any layer tested: bulk expression, 10 spatial niches, 25 cell types, cell-type
composition (`relapse-phase-model-design.md`), **whole-lesion geometry**
(`scripts/lesion_morphometry.py`: 888 lesions; count/size/confluence/fragmentation
track severity but 0/22 peak-vs-remission and 0/22 region tests survive
severity-residualization), and finally **lesion internal organization**
(`scripts/lesion_radial.py`: core-rim polarization of myeloid / astrocyte /
complement programs, residualized on severity + lesion size, is null peak-vs-remission
p = 0.25–0.88 and region p = 0.41–0.90). Severity is the organizing axis; the
molecular and spatial state — down to the radial architecture of individual lesions —
is a faithful readout of it, not an independent signal. This is a clean,
design-bounded negative (terminal cross-sectional, ~5 animals/stage); separating
"direction of travel" or "lesion age" from severity would require longitudinal
sampling, not a different analysis.

## Scope and honest limits

- **Cross-sectional:** the cycle is a cross-animal reconstruction; no within-animal
  trajectory, no causal/temporal claims.
- **Small per-stage n** (2–5 animals); programs are strong and reproducible across
  independent analyses, but individual stage means are noisy — figures use
  animal-level pseudobulk with this caveat.
- **Severity is the organizing axis by design.** There is no separately
  identifiable "direction of travel" (escalating vs recovering) beyond severity in
  this dataset — tested and null in bulk, 10 spatial niches, 25 cell types, and
  composition. This bounds the atlas to *what tracks disease activity*, which is the
  intended scope.

## Suggested figures

1. Relapse-cycle temporal profiles with peaks shaded (have:
   `runs/reports/relapse_cycle_oscillation.html`, `figures/cycle_profile.pdf`).
2. Amplitude vs floor-drift scatter separating the two axes
   (`figures/amplitude_vs_floordrift.pdf`).
3. `Hal` cell-type localization — per-cluster `Hal` with the cluster-5 myeloid
   marker panel (*to generate as a figure*).
4. Continuous-trajectory heatmap (top up/down genes × ordered stages;
   `runs/rr_within_relapse/figures/trajectory_heatmap.pdf`).
5. Spatial map of the `Hal`-high niche along the L→T→C gradient (*to generate*).
