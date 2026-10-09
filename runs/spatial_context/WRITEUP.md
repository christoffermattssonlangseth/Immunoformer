# Task 2 — spatial context without predefined niches

**Spatial context adds nothing measurable to the duration clock beyond knowing cell type.**
Every spatial feature set lands within 0.05 ρ of either pseudobulk or the per-cell-type
expression reference, and every confidence interval overlaps the baseline (all Fisher
z p ≥ 0.35).

## Method

- **No niche labels, no clustering.** For each RR cell, its k nearest neighbours in the
  same section (k = 10, 30; self excluded) give a neighbourhood vector, the mean log-CP10k
  over all 5,101 genes.
- **Two animal-level matrices per k,** averaged within each `Anno_L1_curated` type:
  - neighbourhood mean;
  - mean absolute difference |own − neighbourhood|.

  The signed difference is a linear combination of the other two and could add nothing.
- **Cell types kept:** 12 with ≥ 20 cells in every RR animal, so nothing is imputed. Dropped:
  B cell, Epithelial, Muscle cell, NK/DC, Neutrophil, T cell, T_B_doublet.
- **Clock unchanged:** RR `day_of_sacrifice`, severity-residualised, LOAO, ElasticNetCV with
  the top-1000-variance filter; zero-feature folds audited.

## Results (RR, n = 33; baseline repeated)

| features | features in | ρ | 95% CI | Fisher z p vs baseline | zero-feature folds |
|---|---|---|---|---|---|
| **baseline pseudobulk** | 5,101 | **0.828** | [0.62, 0.93] | — | 0/33 |
| own expression per cell type (no spatial info; reference) | 61,212 | 0.872 | [0.73, 0.93] | 0.54 | 0/33 |
| k=10 neighbourhood mean | 61,212 | 0.828 | [0.68, 0.91] | 1.00 | 0/33 |
| k=10 pseudobulk + neighbourhood mean | 66,313 | 0.848 | [0.69, 0.92] | 0.80 | 0/33 |
| k=10 \|own − neighbourhood\| | 61,212 | 0.876 | [0.74, 0.94] | 0.49 | 0/33 |
| k=10 pseudobulk + \|own − neighbourhood\| | 66,313 | 0.890 | [0.74, 0.95] | 0.35 | 0/33 |
| k=30 neighbourhood mean | 61,212 | 0.836 | [0.68, 0.91] | 0.92 | 0/33 |
| k=30 pseudobulk + neighbourhood mean | 66,313 | 0.822 | [0.62, 0.91] | 0.94 | 0/33 |
| k=30 \|own − neighbourhood\| | 61,212 | 0.877 | [0.74, 0.94] | 0.48 | 0/33 |
| k=30 pseudobulk + \|own − neighbourhood\| | 66,313 | 0.877 | [0.74, 0.94] | 0.48 | 0/33 |

## Reading

- **Neighbourhood means** reproduce pseudobulk (0.83–0.85). Averaging over neighbours is
  close to averaging over the section.
- **The |own − neighbourhood| matrices** (0.88) match the per-cell-type own-expression
  reference (0.87). Their small edge over pseudobulk comes from splitting by cell type, not
  from spatial context.
- **k = 10 vs k = 30:** no difference, so no spatial scale stands out.
- **Circadian genes:** they are still in all of these feature sets, and Task 1 shows they
  alone predict sacrifice day (ρ 0.86). None of these numbers should be read as "disease
  time" until Task 1's time-of-day question is resolved.
- **Zero-feature folds:** none in any fit.

Cache (gitignored): `context_features.npz`. Machine tables: `report.md`, `results.json`.
