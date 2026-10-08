# Immunoformer — where things stand and what to do next
Drafted 2026-10-08, after reading the repo at commit `3b24af0`.

## 1. Where you actually are

Two bodies of work have grown in the same repo, and they are at very different
stages.

**(A) The biology — mature.** ~30 analysis scripts producing real, controlled
results. The strongest:

| result | number | control quality |
|---|---|---|
| Duration clock (RR), severity-orthogonalised | ρ = 0.799, LOAO R² = 0.672 | per-fold residualisation + feature selection; perm null; batch checks |
| Duration clock (chronic) | ρ = 0.856, R² = 0.862 | same |
| Cross-model conservation of duration axis | ρ = 0.653, 72 % sign agreement | independent cohorts |
| Accrual vs acute decomposition (within-RR, matched score) | cumulative p = 0.016, acute p = 0.19 | strain-clean, severity-matched |
| Phase-beyond-severity | AUC 0.513, p = 0.32 → **stopped** | correctly called as a negative |

**(B) The model — behind.** `immunotransformer` is a gated attention-MIL + CORAL
head on PCA features. It runs (RRMAP2 stage: MAE 3.41, ρ = 0.753) and it has been
used twice (`rrmap2_stage`, `rr_vs_chronic`). It has never been compared against
the pseudobulk baselines in (A) on the same target, and it is not what produced
any of the headline findings.

**This is the thing to fix, and it is also the grant's whole premise.** The
duration clock is an animal-level pseudobulk ElasticNet. It throws away every cell
boundary and every spatial coordinate, and it reaches ρ = 0.80. Immunoformer
exists to claim that cell-level and niche-level structure carry information that
pseudobulk does not. That claim is currently untested.

---

## 2. The next experiment, and it is one experiment

**Run the duration clock as a baseline against Immunoformer on the identical
target, identical animals, identical LOAO folds.**

- Target: `day_of_sacrifice`, severity-residualised per fold (exactly as in
  `scripts/duration_clock.py`). RR cohort, n = 33, strain-clean, day⊥score.
- Arms, in increasing order of information used:
  1. severity only (`score_sacrifice`) — the floor, already ρ = −0.52
  2. **animal pseudobulk → ElasticNet** — the incumbent, ρ = 0.799
  3. niche composition (`Global_niche`, 27 fractions) → ElasticNet
  4. cell-type composition (`Anno_L1_curated`, 19) → ElasticNet
  5. **attention-MIL over cells → regression head** (Immunoformer)
  6. MIL + spatial positional encoding (Stage 2, only if 5 beats 2)
- Report: LOAO ρ and R² per arm, with the bag-level bootstrap CI and Fisher-z
  comparison already in `immunotransformer/stats.py`.

Both outcomes are publishable. If arm 5 beats arm 2, you have demonstrated that
spatial single-cell structure carries temporal information beyond bulk — which is
the thesis of the grant. If it does not, you have reproduced DenAdel's finding in
a new regime with a clinically grounded target, and you have saved the
consultancy 700 hours of building something the data does not support.

**Do this before the consultants arrive.** It determines what you ask them to
build.

---

## 3. Fix the training target

`configs/rrmap2_stage.yaml` trains on `stage` — 17 levels, mixing the two models,
tying the relapse peaks, ordered by median clinical score. It is the weakest
target in the repo and it confounds strain with disease course.

Replace with targets the repo has already validated:

| target | type | why |
|---|---|---|
| `day_of_sacrifice` \| severity | continuous | the clock target; de-confounded; strain-clean within RR |
| `score_sacrifice` | ordinal, CORAL | clean clinical ordinal, no stage ties |
| cumulative score | continuous | course, not state — from the daily curves |
| cumulative-program score | continuous | the accrual axis; biologically interpretable |

Train within-model (RR or chronic), never pooled across strains, unless the
explicit question is the interaction.

---

## 4. Two cheap experiments that protect everything else

**4a. Can a classifier recover run identity from the transcriptome?**
You check batch leakage with Kruskal tests on `day ~ run_date`. That tests
association with the label, not identifiability of the batch. Train a grouped
classifier to predict `run_date` from animal pseudobulk PCs and report balanced
accuracy against chance. If run is perfectly identifiable (as it was from images
in BeyondBoundaries), then any model trained across runs can use run as a label
proxy, and the RR-vs-chronic comparison — where run and model are aliased in two
of three batches — needs restricting to the May 2026 run, which contains both.

**4b. Is the accrual program immune-instructed or intrinsic?**
This is your best biological finding and the mtDNA-DSB data is the perfect test,
and it needs no model at all. Score the cumulative/ratchet program
(Gpnmb, Igf2, Fmod, Fcrls, Plin4, Pmp22, Ptgds) in mtDSB oligodendrocytes versus
controls, at both ages. There is essentially no lymphoid compartment in that
dataset. If the program appears anyway, it is a cell-intrinsic damage-accrual
response; if it does not, it requires immune instruction. Either way it connects
directly to Falcão 2018 and Jäkel 2019 and it is a one-afternoon analysis.

---

## 5. Repository hygiene before a consultancy

A 700-hour engagement starts with someone else reading this repo. Right now:
one commit, ~30 top-level scripts with no declared execution order, `runs/`
outputs committed to git, no environment pin, no tests, hard-coded
`/Volumes/moldiassd` paths.

Minimum before handover:

- Split `immunotransformer/` (the model, installable) from `analysis/` (the
  gene-level studies). They have different lifecycles.
- One `paths.py` or `.env` for data locations; no absolute volume paths in scripts.
- Declare the DAG. A `Makefile` or `snakemake` file showing that
  `rr_within_relapse` produces the pseudobulk that `duration_clock` consumes.
  Right now that dependency is discoverable only by reading a module constant.
- Stop committing `runs/**` binaries and figures; keep `report.txt` and
  `results.json`, gitignore the rest.
- Pin the environment, including the OpenMP workaround, in `pyproject.toml` /
  `environment.yml` so the segfault story never repeats on a new machine.
- Commit in logical units from here on. One commit for the whole history means
  no one can see what changed when a number changed.

---

## 6. Two small methodological catches

- **`N_PERM = 50`** in `duration_clock.py` makes `p = 0.0196` the *floor* of what
  that test can report — it means no permutation exceeded the observed value, not
  that p is precisely 0.02. For a figure, raise to 1,000+ and report `p < 0.001`,
  or state it as `p ≤ 0.02 (50 permutations)`.
- **`loao_target()`** builds the evaluation reference by residualising `day` on
  severity using *all* animals, including the held-out one. With n = 33 and one
  covariate the optimism is small, but it is a real fold boundary crossing.
  Report both the full-data reference and a fold-wise one; if they agree, say so
  and the objection is closed.

---

## 7. What not to do yet

- **Stage 2 (cell-as-token transformer, spatial PE, dual masking).** Blocked on
  §2. If attention-MIL over cells does not beat pseudobulk, a bigger architecture
  will not either.
- **The monophasic-vs-relapsing contrast as a headline.** n = 4 vs 4, and your
  own report labels it EXPLORATORY. It is a good figure for a grant and a bad
  figure for a paper. Keep it, caveat it, do not build on it.
- **Foundation-model encoders.** Worth doing — but as arm 6/7 of §2, after the
  simple arms are on the board, not instead of them.

---

## 8. Suggested order

| | what | effort |
|---|---|---|
| 1 | Run-identifiability audit (4a) | half a day |
| 2 | Accrual program in mtDNA-DSB (4b) | one day |
| 3 | Repo split, DAG, path config, gitignore (§5) | two days |
| 4 | Baseline ladder arms 1–4 on the clock target (§2) | two days |
| 5 | Immunoformer arm 5, same folds (§2) | three days |
| 6 | Decide Stage 2 on the evidence from 5 | — |

Steps 1, 2 and 4 are independent and can run in parallel. Step 5 depends on 4
only for the comparison, not for the code.
