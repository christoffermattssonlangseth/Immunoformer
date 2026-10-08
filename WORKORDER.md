# Immunoformer — work order
For Claude Code, run from the repo root on the machine with the data volumes mounted.
Written 2026-10-08 against commit `3b24af0`.

## How to use this file

Six tasks, ordered. **Do them one at a time and stop after each for review.** Do not
start task 5 until the results of task 4 have been looked at by a human — task 5's
value depends on what task 4 shows.

Tasks 1, 2 and 3 are independent of each other and of 4–5.

Standing rules for every task:
- Import `immunotransformer` before numpy/torch in any new script (the OpenMP guard
  in `immunotransformer/__init__.py` is load-order sensitive — see HANDOFF.md).
- Any cross-validation splits on **animal** (`sample_name`), never on `sample_id`
  (that is a physical slide holding 2–3 animals) and never on `meta_sample_id`.
- Every fitted transform — residualisation, feature selection, scaling, PCA — goes
  **inside** the training fold. No exceptions, including for "unsupervised" steps.
- Write results to `runs/<task_name>/{results.json, report.txt, figures/}` following
  the existing convention. Print the report to stdout as well.
- Do not commit anything. Leave changes in the working tree for review.

Data paths currently hard-coded across scripts (do not change them in these tasks,
task 3 handles that):
```
RRMAP2 : /Volumes/moldiassd/RRMAP2_xenium_adata/kmeans_separated/
         RRMAP2_xenium_all_samples.cellcharter.companion.ready.with_metadata.rerun.h5ad
mtDSB  : /Volumes/processing2/oligo-mtDSB/data/
         mtDNA_DSB_5k_clustered_annotation_with_rbd_2_cytetype_brain.h5ad
```

---

## Task 0 — Trajectory features (PREREQUISITE for tasks 4 and 6)
`analysis/trajectory_features.py` → `runs/trajectory_features/`

**Why.** Every target currently used in the repo is a terminal scalar:
`score_sacrifice` (clinical score on the day of sacrifice) and `day_of_sacrifice`
(days post-induction). The daily scoring and weight spreadsheets contain the full
course for all 68 animals, and the repo does not yet use them.

Two specific consequences of this, both of which this task fixes:

1. **`day_of_sacrifice` is the wrong clock target.** It measures time since
   *injection*, which includes a latent pre-onset period ranging from day 6 to day 20
   in the chronic cohort. The biologically meaningful clock is **disease duration**,
   `day_of_sacrifice − onset_day`. Severity residualisation does not remove latency,
   because latency is not a severity artefact — and if latency predicts eventual
   course, it is a confounder rather than noise.
2. **The accrual-axis result argues against conditioning on terminal severity.** Its
   finding is that tissue at matched sacrifice score differs according to the history
   that produced that score. Using `score_sacrifice` as the sole covariate therefore
   controls for the wrong thing: it removes the current state but leaves the
   accumulated burden, which is the quantity of interest.

**Do.**
1. Load the two spreadsheets (daily score and daily weight, `Fixed_RRMap2_FinalSamples_*`)
   and join to the atlas on `sample_name`. Apply the known alias map: atlas
   `C_L_1..C_L_5` = spreadsheet `C_M30_1..C_M30_5`, verified by section ID. Report the
   join rate and list any unmatched animals. Expect 62 direct + 5 aliased of 67;
   `C_CFA_5` exists in the spreadsheets but not the atlas.
2. Compute per-animal trajectory features, writing a tidy
   `runs/trajectory_features/animal_trajectory.csv`:

   | feature | definition |
   |---|---|
   | `onset_day` | first day with score ≥ 0.5 sustained for ≥ 2 consecutive days |
   | `disease_duration` | `day_of_sacrifice − onset_day` (NaN for never-symptomatic) |
   | `peak_score`, `peak_day` | maximum score and the day it occurred |
   | `cumulative_score` | trapezoidal area under the daily score curve to sacrifice |
   | `cumulative_since_onset` | same, from `onset_day` only |
   | `n_relapses` | see 3 below |
   | `days_since_last_peak` | sacrifice day minus the day of the last local maximum |
   | `slope_at_sacrifice` | OLS slope of score over the final 3 days (sign = ascending / plateau / descending) |
   | `max_weight_loss` | minimum weight as a fraction of each animal's own baseline |
   | `score_coverage` | fraction of days from d0 to sacrifice with a recorded score |

   Keep `day_of_sacrifice` and `score_sacrifice` in the table for comparison.
3. **Relapse count needs a definition agreed with the person who scored the animals.**
   Implement the first-pass rule already noted in `docs/` (a drop of ≥ 1 point from a
   peak, then a rise of ≥ 1) and additionally a stricter variant (≥ 0.5 / ≥ 0.5 with a
   2-day sustain). Report both, plus their agreement with the stage labels — the rule
   should recover two relapses in the PEAK3 animals. Flag that it also finds relapses
   in 3 chronic MILD30 animals. Do not pick a definition unilaterally; surface the
   disagreement.
4. Resolve the known data issue, or report it as unresolved: animal `C_M16_2` has a
   section (`C2_G3_Mid_1`, region T) whose score curve disagrees with the animal's
   other two sections on days 11–19. Scores are an animal-level property, so one curve
   is wrong. Report which sections disagree and by how much; do not silently average.
5. Correlation matrix of all trajectory features against each other and against
   `score_sacrifice` and `day_of_sacrifice`, per cohort. The purpose is to show which
   features are genuinely new information rather than restatements of terminal severity.

**Acceptance.** `animal_trajectory.csv` exists for all joined animals;
`report.txt` states the join rate, the two relapse counts and their disagreement,
the `C_M16_2` status, and — in one sentence — how much of `cumulative_score` is
*not* explained by `score_sacrifice`. If that last number is small, say so, because
then the trajectory reframing buys less than expected and tasks 4 and 6 should know.

---

## Task 1 — Batch identifiability audit
`scripts/batch_identifiability.py` → `runs/batch_identifiability/`

**Why.** Existing batch checks (`duration_clock.py`) test whether the *label* is
associated with batch via Kruskal–Wallis. That is not the same question as whether
batch is *recoverable from the data*. If run identity is perfectly predictable from
the transcriptome, any model trained across runs can use it as a label proxy — and
`run_date` is aliased with `model` in two of the three Xenium runs.

**Do.**
1. Build animal-level pseudobulk from RRMAP2 (reuse `runs/rr_within_relapse/pseudobulk.npz`
   if it covers all 67 animals; otherwise compute for all 67 and cache alongside it).
2. For each of the targets `run_date`, `sample_id` (slide), `region`, and `model`:
   train a `HistGradientBoostingClassifier` on the top 50 PCs (PCA fit inside the
   fold) with `GroupKFold` on `sample_name`, and report **balanced accuracy against
   the chance rate** `1 / n_classes` plus a label-permutation null (≥ 200 shuffles,
   permuting labels at the animal level).
3. Repeat restricted to the **May 2026 run only** (the one containing both EAE
   models). Report how many animals and which stages that stratum contains.
4. Cross-tabulate `run_date × model`, `run_date × stage`, and `sample_id × model`,
   and write the tables into the report.

**Acceptance.** `report.txt` states, in one sentence each: whether run identity is
recoverable above chance, whether model identity remains recoverable *within* a
single run, and how many animals per model survive in the May 2026 stratum.

**Consequence to record in the report, not to act on yet.** If run is recoverable
and model is not recoverable within a run, then `runs/rr_vs_chronic` (AUC 0.998) is
partly a batch classifier and needs re-running on the clean stratum.

---

## Task 2 — Accrual program in the mtDNA-DSB model
`scripts/accrual_in_mtdsb.py` → `runs/accrual_in_mtdsb/`

**Why.** The accrual/ratchet program is the best biological finding in the repo.
Its origin is unknown: immune-instructed, or a cell-intrinsic damage response. The
mtDNA-DSB dataset is oligodendrocyte-intrinsic damage with essentially no lymphoid
compartment, so it separates those two hypotheses directly. No model required.

**Do.**
1. Define the program from existing outputs, not by hand: take the top accrual genes
   by coefficient from `runs/duration_clock/clock_genes.csv` and the ratchet genes
   named in `runs/accrual_axis/report.txt` (Gpnmb, Plin4, Fcrls, Igf2, Fmod, Pmp22,
   Ptgds), and the acute set (Hal, Arg1, Chil3) as the contrast. Intersect with the
   mtDSB `var_names` and report how many of each survive the panel intersection.
2. In the mtDSB object, compute per-cell program scores with `scanpy.tl.score_genes`
   using a matched control gene set, restricted to oligodendrocyte-lineage cells
   (`cell_class_updated`: oligodendrocytes, immature oligodendrocytes,
   disease-associated oligodendrocytes) and separately for microglia.
3. Aggregate to **animal level** (`sample_id`, n = 12) and test condition
   (control vs mtDSB) with a Mann–Whitney U, stratified by `age` (21 / 60).
   n = 3 per cell — report exact p-values and effect sizes, and say plainly that
   this is a 12-animal design.
4. **Confound note to include:** age is perfectly aliased with slide in this dataset
   (all age-60 on one slide, all age-21 on the other). The condition contrast is
   within-slide and therefore clean; any age contrast is not. State this in the report.
5. Figure: per-animal program score, accrual vs acute, split by condition and age.

**Acceptance.** `report.txt` answers in one sentence: does the accrual program rise
in mtDSB oligodendrocytes without an immune trigger, and does the acute program do
the same (it should not, if the two axes are genuinely separable).

---

## Task 3 — Repository restructure
No new results. Mechanical, but do it in its own session and its own commit series.

**Do.**
1. Create `analysis/` and move the ~30 gene-level study scripts there from `scripts/`.
   Keep in `scripts/` only things that drive `immunotransformer` itself
   (`train`-adjacent, `transfer_experiment.py`, `inspect_labels.py`, `make_synthetic.py`).
2. Create `immunotransformer/paths.py` reading from environment variables with
   sensible defaults, e.g. `RRMAP2_H5AD`, `MTDSB_H5AD`, `OPTIC_H5AD`. Replace every
   hard-coded `/Volumes/...` string with an import from it. Add `.env.example`.
3. Write a `Makefile` declaring the real DAG. At minimum encode that
   `analysis/rr_within_relapse.py` produces `runs/rr_within_relapse/pseudobulk.npz`,
   which `analysis/duration_clock.py` consumes — that dependency is currently
   discoverable only by reading a module-level constant.
4. `.gitignore` `runs/**` except `report.txt`, `results.json`, and `*.csv` summary
   tables. `git rm --cached` the committed `.pt`, `.npz`, `.parquet`, `.pdf`, `.png`
   and `.html` artefacts. Note in the commit message that figures are regenerable
   from the Makefile.
5. Pin the environment in `pyproject.toml` (or add `environment.yml`), and document
   the OpenMP constraint and why `immunotransformer` must be imported first.
6. Delete `.ipynb_checkpoints/`.

**Acceptance.** `make -n all` prints a plausible build order; a fresh clone with
`.env` set and no mounted `runs/` can regenerate `runs/duration_clock/report.txt`.

---

## Task 4 — The baseline ladder
`analysis/baseline_ladder.py` → `runs/baseline_ladder/`

**Why.** The duration clock is animal-level pseudobulk: it discards every cell
boundary and every spatial coordinate and reaches ρ = 0.799. Immunoformer's premise
is that cell- and niche-level structure carry information beyond that. Untested.

**Setup, held identical across all arms.**
- Cohort: **RR only** (n = 33, SJL/PLP, strain-clean, day⊥score ρ = +0.14).
- Splits: leave-one-animal-out on `sample_name`, the same fold assignment for every arm.
- Metrics: Spearman ρ and R² of LOAO predictions, plus bag-level bootstrap CI from
  `immunotransformer/stats.py` and pairwise Fisher-z comparison between arms.
- Reuse `loao_clock()` from `analysis/duration_clock.py` for the fold mechanics
  rather than reimplementing residualisation.

**Targets — REVISED 2026-10-08 after Task 0 reported.**

Task 0 found that the *magnitude* features collapse once severity and day are
accounted for, while the *shape and timing* features survive:

- `score_sacrifice` explains only 12–13 % of `cumulative_score`, but severity **and
  day together** explain 91 % (RR) / 78 % (chronic).
- `disease_duration` is 99 % (RR) / 97 % (chronic) explained by severity + day,
  because onset day barely varies in RR (days 11–15, SD 1). **The latency correction
  that motivated this target does not apply to the RR cohort.**
- Features retaining independent variance: `onset_day`, `days_since_last_peak`,
  `slope_at_sacrifice`, and in RR also `peak_day` and `max_weight_loss`.

Conclusion driving the revision: course is not *how much*, it is *when and in which
direction*. Magnitude is severity integrated over time and is therefore collinear
with both. Run the ladder against these targets instead.

| target | residualise out, per fold | cohort | what it asks |
|---|---|---|---|
| `days_since_last_peak` | `score_sacrifice` + `day_of_sacrifice` | RR | **primary.** Time since the last insult at matched severity and matched day — the accrual axis as a supervised target |
| `slope_at_sacrifice` (sign) | `score_sacrifice` | RR | direction of travel at matched severity — the phase question, labelled from the curve rather than the stage string |
| `onset_day` | nothing | RR + chronic | latency as a constitutive host property |
| `cumulative_score` | `score_sacrifice` + `day_of_sacrifice` | **chronic** | accumulated burden; 22 % independent variance in chronic vs 9 % in RR, so it moves cohort |
| `day_of_sacrifice` | `score_sacrifice` | RR | incumbent comparator, ρ = 0.799 — keep so the effect of every change is visible |

**`disease_duration` is dropped.** It is `day_of_sacrifice` minus a near-constant in
this cohort.

Two targets, two programs, one dissociation to test. If accrual is real, the
cumulative/ratchet program (Gpnmb, Igf2, Fmod, Fcrls, Plin4) should track
`days_since_last_peak` and the acute program (Hal, Arg1, Chil3) should track
`slope_at_sacrifice`, not the reverse. Report the cross-target matrix — each program
score against each target — not just the headline arm comparison. This is a stronger
version of the matched-group contrast in `runs/accrual_axis`.

**Power warning, to be honored in the write-up.** These residualised targets retain
9–22 % of their variance on n = 33 animals. Bootstrap CIs will be wide. Decide in
advance that "arm 6 does not beat arm 2" may mean **inconclusive** rather than
negative, report CI widths alongside every ρ, and do not let a wide-CI null be
written up as a demonstration that cell-level structure adds nothing.

**On `cumulative_score` in chronic.** The chronic cohort has day~score collinearity
ρ = +0.58, which is why `duration_clock.py` excluded it. Residualising on a collinear
pair is unstable, not invalid. Run it, report the variance inflation, and caveat it.

**On `onset_day`.** Onset precedes the tissue snapshot by weeks, so this is only
interpretable if onset day reflects a constitutive host property the tissue still
carries. Frame it that way in the report; do not describe it as prediction.

The `slope_at_sacrifice` target is worth the run because `runs/rr_phase_feasibility`
found onset-versus-remission separable at matched severity (AUC 1.0, perm p = 0.004)
on n = 4 animals labelled by stage string. Deriving ascending/descending from the
daily curve labels every animal, not four, and converts a proof-of-concept into a test.

**Arms, in increasing order of information used.**

| arm | features | source |
|---|---|---|
| 1 | `score_sacrifice` alone | floor; already ρ = −0.52 |
| 2 | animal pseudobulk, 5101 genes | the incumbent clock, ρ = 0.799 |
| 3 | `Global_niche` composition (27 fractions) | spatial, no genes |
| 4 | `Anno_L1_curated` composition (19 fractions) | cellular, no spatial |
| 5 | arm 2 + arm 3 concatenated | does spatial add to bulk, linearly? |
| 6 | attention-MIL over cells (Immunoformer, regression head) | the model |

Arms 1–5 use `ElasticNetCV` with the existing `ENET_KW` settings. Arm 6 needs a
continuous head — add a `regression` option to the existing CORAL model rather than
writing a second model, and train it under the same LOAO folds.

**Also fix while here.**
- Raise `N_PERM` from 50 to ≥ 1000 in the permutation null. With 50, `p = 0.0196`
  is the floor of what the test can report, not a measurement.
- `loao_target()` currently residualises `day` on severity using all 33 animals
  including the held-out one. Add a fold-wise evaluation reference alongside the
  existing full-data one and report both. If they agree, say so explicitly.

**Acceptance.** One table per target in `report.txt`: arm, ρ, R², bootstrap CI, and
the Fisher-z p-value against arm 2. Plus a figure with the arms on the x-axis, ρ with
CI on the y-axis, and one panel per target.

**STOP HERE for human review.**

---

## Task 5 — Conditional on task 4
Only if arm 6 beats arm 2 with a Fisher-z p < 0.05 **on at least one of the
five revised Task 4 targets**: extend to spatial positional
encoding (Stage 2) on the same folds and the same target, as arm 7.

If arm 6 does **not** beat arm 2, do not build Stage 2. Instead write
`docs/negative-result-scaling.md` recording the ladder as a negative result in the
DenAdel frame — a clinically grounded temporal target on 1.38M spatially resolved
cells where cell-level modelling does not beat animal pseudobulk — and stop.

---

## Task 6 — Retarget the trained model
`configs/rrmap2_*.yaml`

Replace `stage` (17 levels, mixes both strains, ties the relapse peaks, and encodes
only the terminal state) as the training target. Using Task 0's
`animal_trajectory.csv`, create configs for whichever targets task 4 showed the model
can actually learn:

- `days_since_last_peak` — regression head, RR only; the primary course target
- `slope_at_sacrifice` — binary ascending/descending at matched severity
- `cumulative_score` — regression head, chronic cohort
- `score_sacrifice` — clean clinical ordinal, keep the CORAL head, as the state
  comparator against the three course targets above

Train within a single EAE model, never pooled across strains, unless the explicit
question is the model × severity interaction.

## Things not to do

- Do not build the Stage 2 transformer before task 4 reports.
- Do not treat `runs/monophasic_vs_relapsing` as anything but exploratory — n = 4
  versus 4. Keep it, caveat it, do not build on it.
- Do not reinterpret `runs/rr_vs_chronic` (AUC 0.998) until task 1 reports. SJL and
  C57BL/6 differ genetically and sit in different Xenium runs; 2,363 genes show a
  model main effect that the existing report correctly declines to interpret.
- Do not fill in `encoders.FrozenFMEncoder` yet. Foundation-model encoders belong on
  the ladder as a later arm, after the simple arms are on the board.
