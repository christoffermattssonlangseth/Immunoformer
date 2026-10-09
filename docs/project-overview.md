# Immunoformer — project overview

*Synthesis written 2026-06-18. A map of the repo: the main idea, how it was done, and the
main findings. Compiled from the docs and the committed `runs/*/report.txt` + `results.json`
outputs (not by re-running anything — numbers are as the committed outputs state them).
Primary writeups: [`rrmap2-duration-clock.md`](rrmap2-duration-clock.md),
[`rrmap2-relapse-atlas.md`](rrmap2-relapse-atlas.md); assembled report:
[`../runs/reports/duration_clock.html`](../runs/reports/duration_clock.html).*

## The main idea

The repo began as **ImmunoTransformer**: a model to read disease state from spinal-cord
**Xenium spatial transcriptomics** of EAE (the mouse model of MS), in the **RRMAP2** dataset
(1.38M cells, 5101-gene panel, relapsing–remitting + chronic arms). The design — from the
Le Quesne self-supervised-pathology framework
([`le-quesne-framework-applications.md`](le-quesne-framework-applications.md)) — is
cell-as-token → gated attention-MIL → ordinal (CORAL) head, with attention as the
interpretable readout.

The center of gravity then shifted from the model to a **disciplined statistical dissection
of EAE disease biology**, organized around one question: *spinal-cord tissue obviously
reflects how sick an animal is right now — is there anything else in it?* The recurring
answer, and the spine of the whole repo, is:

> **EAE tissue carries two separable molecular axes — `severity` (acute, reversible, resets
> at every relapse) and `duration` (cumulative, irreversible accumulation) — and most of the
> work is about telling them apart from each other and from confounds (strain, batch, region,
> sex, cell composition).**

## How it was done

- **Unit of analysis = the animal.** Animal-level pseudobulk and animal-split
  cross-validation throughout, because sections within an animal are not independent (the
  `meta_sample_id` vs `sample_name` vs `sample_id` hierarchy is a documented trap — see
  [`dataset-hierarchy.md`](dataset-hierarchy.md)).
- **Severity-orthogonalization.** The signature move: residualize both genes and the target
  on clinical `score_sacrifice` within each CV fold, so a result cannot cheat through
  "sicker = more signal."
- **Confound-first.** Nearly every analysis carries a "beyond-severity" test, a permutation
  null, and batch / strain / region / sex checks — and reports the null when there is one.
- **The model.** A gated attention-MIL + CORAL ordinal head on PCA cell features predicting
  `stage`; runs end-to-end (after resolving an OpenMP duplicate-runtime segfault, see
  [`../HANDOFF.md`](../HANDOFF.md)), reaching **Spearman 0.753** on held-out animals.

## Main findings

**1. The two-axis framework (the headline).** Across the relapse cycle, genes split into an
**acute / oscillating** program (`Hal`, `Arg1`, `Chil3`, interferon — spikes at each peak,
resets in remission) and a **ratchet / cumulative** program (remission floor rises cycle over
cycle). Severity is the backbone; duration is the part that survives clinical recovery.
(`scripts/rr_cycle_oscillation.py`, `rrmap2-relapse-atlas.md`.)

> **Downgraded 2026-10-08 (WORKORDER Task 4).** The pre-registered dissociation test failed. Across all RR animals the ratchet program (Gpnmb, Igf2, Fmod, Fcrls, Plin4) does not track `days_since_last_peak` (ρ +0.22 [−0.15, +0.56]) and the acute program (Hal, Arg1, Chil3) does not track slope direction (ρ −0.02 [−0.37, +0.31]); the ratchet program does track `day_of_sacrifice` (ρ +0.68 [+0.41, +0.83]). The PEAK1 vs PEAK3 contrast (p = 0.016, n = 4 vs 5) must no longer be cited as two separable programs: PEAK3 animals are also sacrificed weeks later, and the simpler reading is a single dominant time axis. See `runs/baseline_ladder/report.txt`.

**2. The duration clock.** A supervised elastic-net clock reads time-since-induction *after
severity is removed* — **RR Spearman +0.80** (leave-one-animal-out, permutation p = 0.02),
and it is not a sex / region / batch artifact (it strengthens to +0.86 when those are
controlled). It reads as a coherent "aging lesion field": neuron/myelin loss, a
basement-membrane→fibrotic-scar switch, lymphoid organization, and a
cholesterol-synthesis→scavenging switch. Spatially it is **tissue-wide but glia-led**
(astrocyte +0.89) and driven by cell **state**, not composition (a fractions-only clock is
much weaker, +0.52). (`scripts/duration_clock.py`, `duration_clock_spatial.py`,
`rrmap2-duration-clock.md`.)

**3. Accrual is cell-intrinsic, not "more cells."** A shift-share decomposition finds **~90 %
of the duration trend is per-cell upregulation**; only ~2 of 825 strongly-accruing genes are
composition-driven. The foamy-macrophage program intensifies *per cell* even as the myeloid
fraction falls. (`scripts/duration_composition_intrinsic.py`.)

> **Update 2026-10-09.** This overstates it. At the clock level, residualising cell-type
> composition out inside each fold lowers the duration clock from 0.83 to 0.50 (RR) and 0.85 to
> 0.52 (chronic). The signal is mainly but not purely cell-intrinsic; composition carries a
> substantial minority. The per-gene shift-share result above is kept as originally reported.
> See `runs/closing_block/WRITEUP.md`, Task C.

**4. The duration program is conserved across model and strain.** Rebuilt identically on the
chronic arm (B6/MOG) the clock reaches **+0.856**, and the RR↔chronic duration axes agree at
**Spearman +0.65** — while the relapse *oscillation* axis specifically does **not** transfer
(−0.17). One revealing difference: severity predicts day **negatively** in RR (relapses reset
acute severity) but **positively** in chronic (monotonic progression). (`scripts/duration_clock_chronic.py`.)

**5. `Hal`** is the single most reproducible disease-associated gene — top relapse-cycle
trend, the only gene clearing FDR in the RR-vs-chronic severity×model interaction (q ≈ 4e-9),
and the top onset→peak difference-in-differences gene — an acute antigen-presenting / myeloid
relapse marker. The originally-proposed histidine-depletion bridge to MS was honestly walked
back after deep-research verification. ([`hal-histidine-finding.md`](hal-histidine-finding.md),
[`hal-histidine-deep-research.md`](hal-histidine-deep-research.md).)

**6. Myeloid states and lesion architecture.** Three myeloid states resolve as a
recruitment → activation → resolution axis (`scripts/myeloid_states.py`), and lesions show a
core→rim radial organization (`scripts/lesion_radial.py`).

**7. Honest negatives (a real strength).** Relapse *phase / direction* beyond severity is
**unidentifiable** in this terminal cross-sectional design (`rr_phase_feasibility`: residualized
AUC 0.51, perm p = 0.32). Lesion morphometry and radial organization are **fully explained by
severity** (null beyond-severity tests). The direct RR-vs-chronic level comparison is **dead**
(100 % strain + slide confounded) — addressed instead via score×model interaction and
phase-anchored difference-in-differences, where `Hal` is essentially the one robust signal.

**8. Disease course (exploratory).** "What makes an animal relapse vs not" is not answerable
across RR-vs-chronic (strain-confounded). The identifiable cut is *within* SJL/PLP, using the
design: **REMISSION1** (~day 22) branches into **MONOPHASIC** (~day 33, no relapse) and
**PEAK2** (~day 32, relapsed) — monophasic animals sampled at the same time a relapser hits
its second attack but staying low-severity. The **relapse** trajectory (REM1→PEAK2) re-ignites
the acute oscillator/IFN/myeloid program (`Gpnmb`, `Chil1`, `Arg1`, `Hal`) and loses
neuron/myelin again; the **monophasic** trajectory (REM1→MONO) instead recovers neuron/myelin,
resolves a glucocorticoid/stress tone, and continues the quiet duration/repair accrual (`Igf2`,
`Fmod`, `Vtn`, `Nr1d1`). The relapse-specific direction aligns with the severity/oscillation
axis (+0.85), opposite duration (−0.56). Exploratory (n=5/4/4, cross-sectional, effect sizes
only); PEAK2 is severe by construction, so the informative half is the monophasic trajectory.
(`scripts/monophasic_vs_relapsing.py`.)

## The throughline

The repo's real contribution is not a single model but a **rigorously de-confounded account of
what spinal-cord tissue remembers about disease history**: severity is acute and resets,
duration accumulates irreversibly as a glia-led, cell-intrinsic program conserved across two
EAE models, and the analyses are unusually careful to report what *isn't* identifiable.

> **Update 2026-10-09.** Three parts of this sentence are superseded:
> - **"Glia-led":** no single cell type's clock beats the all-cells clock in paired tests
>   (astrocytes +0.07 [−0.02, +0.20]); the signal is tissue-wide.
> - **"Cell-intrinsic":** mainly, not purely; composition carries about a third (see above).
> - **"Conserved across two EAE models":** the chronic clock and chronic gene-level findings
>   depend on the severity-adjustment choice, because severity and duration are coupled in
>   chronic. It is a parallel analysis, not an independent replication.
>
> Sources: `runs/clock_composition/`, `runs/closing_block/WRITEUP.md`.

## Where things live

| Area | Scripts | Outputs / writeup |
|---|---|---|
| Stage-1 model (attention-MIL + CORAL) | `immunotransformer/`, `configs/rrmap2_stage.yaml` | `runs/rrmap2_stage/`, `HANDOFF.md` |
| Within-RR relapse biology / atlas | `rr_within_relapse.py`, `rr_cycle_oscillation.py`, `rr_phase_niche.py` | `rrmap2-relapse-atlas.md`, `runs/reports/relapse_cycle_oscillation.html` |
| Duration clock + decompositions | `duration_clock.py`, `duration_clock_controls.py`, `duration_gene_decomposition.py`, `duration_composition_intrinsic.py`, `duration_clock_spatial.py` | `rrmap2-duration-clock.md`, `runs/reports/duration_clock.html` |
| Cross-model + course | `duration_clock_chronic.py`, `chronic_trajectory.py`, `monophasic_vs_relapsing.py`, `model_difference.py`, `onset_peak_did.py`, `rr_vs_chronic.py` | `runs/duration_clock_chronic/`, `runs/monophasic_vs_relapsing/` |
| `Hal` | `make_hal_celltype_fig.py` | `hal-histidine-finding.md`, `hal-histidine-deep-research.md` |
| Lesions / myeloid | `lesion_morphometry.py`, `lesion_radial.py`, `lesion_signaling.py`, `myeloid_states.py` | `runs/lesion_*`, `runs/myeloid_states/` |
| Negative results | `rr_phase_feasibility.py`, `relapse-phase-model-design.md` | `runs/rr_phase_feasibility/` |
