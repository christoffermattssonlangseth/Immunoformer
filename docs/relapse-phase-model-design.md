# Design: a relapse-**phase** model for RRMAP2 — severity vs direction vs accrual

*Drafted 2026-06-15. Builds on the within-RR oscillation result
([`relapse_cycle_oscillation.html`](../runs/reports/relapse_cycle_oscillation.html),
`scripts/rr_cycle_oscillation.py`) and the `Hal` thread
([`hal-histidine-finding.md`](hal-histidine-finding.md)). No training done yet —
this is the plan.*

## 1. The problem with the current target

`immunotransformer` (Stage 1) predicts the EAE stage with a **monotonic ordinal
CORAL head** (`losses.py:CoralHead`, `label_order` in `config.py`). CORAL assumes
`control < … < peak < late` is a *line*. But the RR course is a **loop**:

```
        PEAK1            PEAK2            PEAK3        <- apex (severe)
       /     \          /     \          /
  ONSET1     REM1 -- (ONSET2)  REM2 ----            <- onset rises, remission falls
   /
 CFA                                                  <- baseline
```

REMISSION1 is a *down* state wedged between two *up* states. Forcing that onto a
monotonic axis is why `Hal` — a clean oscillator (PK1 1.32, REM1 0.61, PK2 1.10,
REM2 0.41, PK3 1.19) — only showed a weak monotonic `rho` (+0.52). **The model is
mis-specified for the disease it is modelling.** Fixing that is the project.

## 2. The reframing: three latent coordinates, not one ordinal axis

Decompose a section's disease state into three coordinates that the oscillation
analysis already separates empirically:

| coordinate | meaning | gene class that encodes it (from `rr_cycle_oscillation`) | data support |
|---|---|---|---|
| **severity `r`** | how bad right now (magnitude) | acute-amplitude genes; `score_sacrifice` | strong (continuous, animal-level) |
| **phase `θ`** (cyclic) | where in the attack→recovery loop; peaks share an angle, remissions share an angle; the sign of motion = ascending vs descending | **acute oscillators** — `Hal`, `Arg1`, `Chil3`, `Acod1`, `Cxcl10`, `Ccl2` | medium; **onset (ascending) is thin** |
| **accrual `c`** (monotonic) | how many relapses have elapsed — irreversible residue | **ratchet genes** — `Fcrls`, `Gpnmb`, `Plin4`, `Igf2`, `Pmp22` | medium (REM2>REM1 floor rise) |

This is the conceptual payload: **`Hal` and the oscillators live on `θ`; the
DAM/foamy/microglial ratchet genes live on `c`; clinical severity is `r`.** The
current single ordinal head collapses all three into one number.

### The killer question this unlocks

Two sections at the **same severity** `r ≈ 1.0` — a REMISSION animal (recovering,
descending `θ`) and an ONSET animal (deteriorating, ascending `θ`) — are
clinically indistinguishable. **Can the molecular state call the direction of
travel?** "From one snapshot, is this tissue getting worse or recovering?" is
novel, clinically meaningful, and — uniquely — answerable from cross-sectional
data *because* we have onset and remission animals at overlapping `r`.

## 3. Honest constraints (read before building)

- **Cross-sectional = reconstructed, not observed dynamics.** One stage per animal
  (67 animals, terminal). The model *infers* phase from static snapshots; it can
  never prove temporal causation. The defensible claim is "molecular phase is
  decodable," validated on held-out animals — not "we watched it evolve."
- **The ascending limb is the binding constraint.** ONSET1 = 2 animals, ONSET2 = 2.
  So the entire "deteriorating" class is **n = 4 animals**. Peaks (apex, n = 13)
  and remissions (descending, n = 9) are fine; onset is not. The direction test is
  therefore **proof-of-concept-grade**, and a key deliverable is "this justifies
  collecting more onset sections."
- **Severity confound is the whole point and the whole risk.** Peaks are severe,
  remissions mild, so anything correlating with `r` looks like `θ`. Phase must be
  established *within* matched severity, which — see above — is only testable at
  the low-severity end (ONSET2 ~1.0 vs REM1/REM2 ~0.7–1.06), exactly where n is
  smallest.
- **Within-RR is batch-clean** (each stage spread across 9–12 slides; slides mix
  stages) — this is what makes the model defensible where the RR-vs-chronic
  classifier (100% batch/strain-confounded) was not. Stay strictly within the RR
  cohort.
- **Prior art exists** — supervised pseudotime (`psupertime`), RNA-velocity latent
  time, cyclic cell-cycle phase inference. Differentiation here = *cyclic +
  directional + spatial single-cell + attention-attributed*, on a relapse cohort,
  with the explicit severity/phase/accrual disentanglement. Not "we did pseudotime."

## 4. Concrete changes to `immunotransformer`

Minimal, surgical — reuse the `GatedAttentionMIL` trunk and attention; swap the
head and target.

**`config.py` — `ObsSchema`/`DataConfig`:**
- Keep `stage` for deriving targets; add `severity_col = "score_sacrifice"`.
- Add a `cycle_spec`: maps each stage → `(phase_angle θ, limb ∈ {asc,apex,desc,base}, cycle_index c)`.
  Peaks all → `θ = π/2`; remissions all → `θ = 3π/2`; onsets → `θ ≈ 0…π/2`
  ascending; CFA → baseline (masked from phase loss). `c` = 1/2/3 from the stage's
  cycle number.

**`data.py` — `build_bags`:** replace the single `label:int` on `Bag` with a small
target struct `(severity:float, sin_theta:float, cos_theta:float, limb:int,
cycle:int, phase_mask:bool)`. `phase_mask=False` for CFA/baseline (no defined
phase). Splits stay animal-level (already correct).

**`losses.py` — new `PhaseHead` replacing `CoralHead`** on the pooled `z [1,P]`:
- `severity`: `Linear(P,1)` → MSE on `score_sacrifice` (z-scored).
- `phase`: `Linear(P,2)` → unit-normalize → predict `(sinθ,cosθ)`; loss
  `1 - cos(θ̂ - θ)` (circular), applied only where `phase_mask`.
- `direction`: `Linear(P,1)` → BCE on ascending(1)/descending(0), applied only to
  onset∪remission bags (the matched-direction subset).
- `accrual`: `Linear(P,1)` → ordinal/MSE on `cycle` (optional; this is the ratchet
  axis, keep it to test the decomposition).
- Total = weighted sum; weights in config.

**`model.py`:** `GatedAttentionMIL.head = PhaseHead(...)`; `forward` returns the
dict of predictions + `attn` (unchanged attention path — the interpretable readout
we want).

**`train.py` — `evaluate`:** report, on held-out animals:
- severity Spearman (sanity, should be high);
- **phase angular error** (deg) and **peak/remission phase separation**;
- **the headline metric — direction accuracy / AUC at matched severity**
  (onset vs remission, restricted to overlapping `r`);
- accrual Spearman (does `c` track REM1→REM2 floor genes?).
- Keep the attention dump; add a per-bag breakdown of attention mass on `Hal`+ and
  oscillator+ cells.

## 5. Validation & falsification

The model is only interesting if it survives these:

1. **Held-out animals, grouped split** (already enforced). Phase/direction must
   hold on animals never seen — not sections.
2. **Direction-at-matched-severity** (the killer test). Restrict to bags with
   `r` in the onset/remission overlap band; can the model beat chance at
   ascending-vs-descending? *Pre-register that n=4 onset makes this POC-grade.*
3. **Attention attribution.** Do the cells driving the phase prediction express
   `Hal` and the co-oscillators (`Acp5`, `Il7r`, `Cxcr4`, `Arg1`, `Chil3`)? Do the
   cells driving accrual express `Fcrls`/`Gpnmb`? If the decomposition is real, the
   two axes should attribute to the two gene classes — a strong internal check.
4. **Spatial coherence** (if/when niche tokens added). Phase should vary smoothly
   across the lesion, not scatter cell-to-cell.
5. **Negative controls.** (a) Shuffle stage labels within animal → phase signal
   must collapse. (b) Predict slide/`run_date` from `z` → must NOT be decodable
   (batch leakage check; within-RR should pass). (c) Severity-only baseline → the
   phase/direction head must add information beyond `r` alone, else there is no
   story.

**What refutes the hypothesis:** direction is not separable beyond severity once
batch and `r` are controlled; or attention attributes phase to the same cells as
accrual (decomposition is illusory); or the signal vanishes on held-out animals.
Any of these and the honest conclusion is "the relapse cycle is severity, full
stop" — which is itself worth knowing.

## 6. Phased plan (go/no-go gates)

- **Phase 0 — feasibility, pseudobulk, ~hours.** Reuse
  `runs/rr_within_relapse/pseudobulk.npz`. (a) Logistic onset-vs-remission at
  matched severity (LOAO-CV) — is direction decodable *at all*? (b) Fit a circular
  coordinate to the pseudobulk and check peaks/remissions separate in phase
  independent of `r`. **Gate:** if direction AUC ≤ severity-only baseline, stop and
  report "it's severity" — do not build the deep model.
- **Phase 1 — head swap, single-cell.** Implement `PhaseHead`, train within RR with
  animal-level splits, run the §5 validations. Severity + phase first; add
  direction once the trunk trains.
- **Phase 2 — spatial niche tokens.** Bring in the Xenium coordinates the
  pseudobulk discarded: tokenize CellCharter/`leiden_1` niches, let phase vary in
  space, attribute phase to niches. (This is also the standalone "spatial dynamics"
  framing, folded in as a layer.)
- **Phase 3 — attribution write-up.** Cross-reference attention-driver cells/niches
  against the `Hal`/oscillator vs `Fcrls`/ratchet gene classes; produce the figure.

## 7. The novel claim, stated plainly

> In relapsing EAE, a tissue snapshot encodes not just disease *severity* but its
> *direction of travel* — whether the lesion is escalating into a relapse or
> resolving — and these are molecularly separable. An attention model over spatial
> single-cell data reconstructs a cyclic relapse **phase** (carried by an acute
> oscillator program including `Hal`) distinct from cumulative accrual (carried by
> a DAM/foamy ratchet program), and can read direction at matched clinical
> severity.

If Phase 0 passes, that is a real and testable contribution. If it fails, the
disentanglement is the finding's epitaph and we say so. Either way the mis-specified
monotonic head gets fixed.

## 7b. Phase 0 RESULT (2026-06-15) — the gate said STOP

Ran `scripts/rr_phase_feasibility.py` on the animal-level pseudobulk (33 RR animals,
LOAO CV, severity-residualized label-blind PCs, permutation nulls B=2000):

- **TEST A (peak vs remission, well powered): FAILED the gate.** Severity-only
  AUC = 1.0 (trivial — peaks are severe). After removing the linear severity
  effect, residualized molecular AUC = **0.513, perm p = 0.32** — a clean null.
  *Once you take severity out, peak and remission are molecularly indistinguishable
  at the pseudobulk level.* This is not a power problem; it sits exactly at chance.
- **`Hal` is severity, not phase.** Severity-residualized `Hal` peak−rem = **−0.09**
  (≈0). `Hal`'s whole oscillation is explained by how sick the animal is; it carries
  no extra phase/direction information in bulk.
- **TEST B (onset vs remission) does NOT rescue it.** Two tells: (1) severity-only
  AUC = 0.0 means severity *perfectly* separates the groups (onset is systematically
  *lower* severity than remission) — so the "matched severity" premise was wrong,
  these are severity-separable. (2) The residualized AUC = 1.0 on n = 4 onset with
  8 PCs is textbook separable-by-anything (batch/region/overfit), not evidence of
  decoded direction. Do **not** report this as a positive.
- **Decomposition double-dissociation** held only on its main diagonal (ratchet
  accrues: REM2−REM1 = +1.45; oscillators track peak−rem = +1.14) but the off-
  diagonals were non-zero, and TEST A shows the oscillator "phase" term is mostly
  severity anyway. So even the illustration reduces to severity + accrual.

**Verdict: per the pre-registered gate, do NOT build the cyclic-phase deep model.**
At animal-level pseudobulk resolution the relapse cycle is a **severity axis plus a
cumulative-accrual axis** — there is no severity-independent "direction of travel"
signal to learn. The honest reduced finding: `Hal` is a clean molecular marker of
acute disease activity (severity) that fully resets between attacks; the DAM/foamy
ratchet program is the cumulative axis.

**The one thing pseudobulk cannot rule out:** phase that lives in a *specific cell
type or spatial niche* and is averaged away in bulk. That — a per-cell-type /
per-niche repeat of TEST A — is the only remaining shot before abandoning the model
angle, and its prior is now lower. Everything else in §6 (Phases 1–3) is on hold.

### Niche-resolved follow-up (2026-06-15) — also NULL, at every resolution

Ran `scripts/rr_phase_niche.py` on the real h5ad (894k RR cells), repeating TEST A
*inside* each population (per-animal × niche pseudobulk, severity-residualized,
LOAO, permutation nulls, Bonferroni over niches):

- **10 spatial niches (`CellCharter_10`):** every niche below chance (max AUC 0.24);
  none screened in. Composition (niche fractions) AUC 0.03, perm p = 0.99.
- **25 cell types (`leiden_1`):** every cluster below chance (max AUC 0.385); none
  screened in. Composition AUC 0.02, perm p = 0.999.

So phase-beyond-severity is null in **bulk, in 10 spatial niches, in 25 cell types,
and in cell-type composition.** There is no resolution at which "direction of
travel" is molecularly separable from severity in this dataset. (Caveat: the
consistently-below-0.5 AUCs reflect LOAO small-n / class-imbalance bias, not inverse
signal — the inferential point is simply that nothing clears the null.)

**Final verdict: the relapse-phase / direction model is abandoned.** Not a tuning or
architecture problem — a *design* limit: terminal cross-sectional sampling (one
stage/animal) plus severity dominating every axis means trajectory direction is not
identifiable here. The fix is longitudinal sampling or many more onset sections, not
a cleverer model. What survives is the descriptive decomposition: **severity (acute,
reversible — `Hal` + oscillators) + cumulative accrual (irreversible — `Fcrls`/
`Gpnmb` ratchet).**

## 8. Open decisions for next session

- Exact `θ` assignment for onsets (linear ramp vs learned) and whether ONSET2 maps
  to the same ascending angle as ONSET1 (per the corrected order, ONSET2 follows
  ONSET1 rather than opening cycle 2 — revisit how that interacts with `c`).
- Whether accrual `c` is worth a head or is better left as a validation target.
- Severity source: `score_sacrifice` (animal-level clinical) vs a molecular
  severity latent — using the clinical score avoids circularity in the
  direction-at-matched-severity test, so prefer it.
