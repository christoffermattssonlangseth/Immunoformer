# Negative result: cell-level modelling vs animal pseudobulk at 67 animals

*Written 2026-10-09 as the Task 5 deliverable (WORKORDER). Stage 2 (spatial positional
encoding) was not built, by the pre-registered rule.*

## The result

On RRMAP2 (1.38M Xenium cells, 67 animals: 33 relapsing-remitting, 34 chronic), an
attention-MIL model over individual cells (Immunoformer, arm 6) **does not measurably
exceed** a linear model on animal-level pseudobulk (arm 2) on any clinically grounded
temporal target. Pseudobulk does not measurably exceed the attention model either.

| target | pseudobulk ρ [95% CI] | attention-MIL ρ [95% CI] | p (6 vs 2) |
|---|---|---|---|
| time since induction, beyond severity (RR, n = 33) | 0.83 [0.62, 0.93] | 0.89 [0.75, 0.94] | 0.40 |
| days since last flare, beyond severity and day (RR, 28) | 0.70 [0.40, 0.88] | 0.46 [0.05, 0.76] | 0.19 |
| cumulative burden, beyond severity and day (chronic, 29) | 0.73 [0.51, 0.87] | 0.66 [0.35, 0.86] | 0.62 |

The pre-registered threshold for building Stage 2 was arm 6 beating arm 2 at p < 0.05 on at
least one target. It was not met. Full tables: `runs/baseline_ladder/report.txt` and
`WRITEUP.md`.

This sits in the DenAdel frame: a clinically grounded temporal target, on a large spatially
resolved dataset, where cell-level modelling does not beat the animal average.

## Why this is "not shown", not "shown not to help"

- **Too few animals to tell.** The unit of evidence is the animal, not the cell. With 28–33
  animals per target, the 95% intervals are 0.2–0.7 wide, so differences of 0.1–0.2 in ρ
  cannot be resolved.
- **Not a fair fight.** Pseudobulk is a linear model tuned inside every fold. The attention
  model had no tuning, fixed settings and 3 seeds. A deep model is expected to lose here
  whether or not its architecture has something to offer.
- **On the main target, attention-MIL is ahead on the point estimate, with the narrowest
  interval of any arm** (width 0.20 vs 0.30). That is a hint, not a result.

## The open question: data-scale limit or architecture limit?

What we have that bears on it:

| observation | points to |
|---|---|
| The model fits its training animals almost perfectly (training ρ 0.93 by epoch 30) but transfers poorly on the harder target (held-out ρ ≈ 0.4–0.45, no better than the mean in squared error), and more epochs do not help | **data scale**: generalising from ~27 animals |
| On the target that works (time since induction), it transfers well (held-out ρ ≈ 0.89) | the architecture can learn the dominant signal |
| Adding niche composition to pseudobulk changes predictions but not accuracy | the animal average already carries most composition information |
| Same model with attention switched off (mean pooling) does about as well as pseudobulk, and as well as or better than attention on 5 of 6 targets | the deep pipeline is not the handicap; **attention over individual cells adds nothing measurable at this n** — the extra parameters look like they overfit ~27 training animals |
| Learning curve (exploratory): refit on 50 / 75 / 100% of training animals. On `days_since_last_peak` the attention model climbs from ρ 0.05 to 0.42 (single seed) while pseudobulk goes 0.48 → 0.70 — the attention model gains faster and the gap narrows (0.43 → 0.27 → 0.28). On `day_of_sacrifice` both rise in parallel (0.71 → 0.83 vs 0.76 → 0.89) with the attention model ~0.05 ahead at every size | **data scale** on the harder target: the cell-level model is still on the steep part of its curve at ~27 animals; on the main target no sign of a different slope. Indicative only (3 draws per point) |

## What would separate the two

1. **Learning curve on animals** — done, exploratory (`runs/baseline_ladder/learning_curve.txt`):
   on the harder target the attention model gains faster than pseudobulk as animals are
   added, which leans toward a data-scale limit; on the main target the curves are parallel.
   Three subsamples per point, so indicative, not a result.
2. **More animals.** The only direct fix for a data-scale limit. Even ~2× the animals would
   roughly halve the CI widths.
3. **A fair fight.** Tune arm 6 inside each fold (epochs, weight decay) on an inner split of
   the training animals, as the elastic net already is. Expect noise at this n.
4. **Attention vs mean pooling** — done (arm 6b): no measurable benefit from attention at
   this n. Worth repeating if more animals become available, since that is exactly where
   a data-scale limit would show.

## What not to conclude

- Not that cell-level or spatial structure carries no information.
- Not that Immunoformer is worse than pseudobulk: no comparison is significant in either
  direction.
- Not that Stage 2 would fail. It was not built because this stage did not clear its bar,
  not because it was shown to be pointless.
