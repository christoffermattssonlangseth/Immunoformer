# Random gene-set null (same pipeline, same folds)

500 sets per null. Matched = same mean-expression decile x detection quintile per gene.

| block | genes | observed rho | null | mean | SD | 5th | 50th | 95th | observed percentile | p | sets with zero-feature folds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| circadian | 17 | +0.855 | random | +0.633 | 0.160 | +0.352 | +0.666 | +0.817 | 98.6 | 0.016 | 12 |
| circadian | 17 | +0.855 | expression_matched | +0.664 | 0.139 | +0.390 | +0.690 | +0.828 | 98.4 | 0.018 | 3 |
| matrix | 58 | +0.918 | random | +0.706 | 0.084 | +0.542 | +0.716 | +0.823 | 100.0 | 0.002 | 0 |
| matrix | 58 | +0.918 | expression_matched | +0.735 | 0.076 | +0.594 | +0.743 | +0.845 | 100.0 | 0.002 | 0 |
| sex | 1 | -0.145 | random | +0.090 | 0.315 | -0.318 | -0.046 | +0.631 | 24.6 | 0.754 | 303 |
| sex | 1 | -0.145 | expression_matched | +0.112 | 0.306 | -0.253 | +0.029 | +0.624 | 19.6 | 0.804 | 307 |

PRE-SET READING (circadian): in the extreme tail of the expression-matched null -> the circadian block IS special; the timing question stays open.
