# Positional confound — machine tables

Positional genes on panel (38): ['Cdx1', 'Cdx2', 'Hoxa1', 'Hoxa10', 'Hoxa11', 'Hoxa13', 'Hoxa2', 'Hoxa3', 'Hoxa4', 'Hoxa5', 'Hoxa7', 'Hoxa9', 'Hoxb1', 'Hoxb13', 'Hoxb2', 'Hoxb3', 'Hoxb4', 'Hoxb5', 'Hoxb6', 'Hoxb7', 'Hoxb8', 'Hoxb9', 'Hoxc10', 'Hoxc11', 'Hoxc12', 'Hoxc5', 'Hoxc6', 'Hoxc8', 'Hoxd1', 'Hoxd10', 'Hoxd11', 'Hoxd12', 'Hoxd13', 'Hoxd3', 'Hoxd4', 'Hoxd9', 'Meis1', 'Meis2']

2a all_cells: positional |coef| share 0.0% of 99 non-zero genes; non-zero positional: none
2a astrocyte: positional |coef| share 5.6% of 90 non-zero genes; non-zero positional: Hoxa10 (+1.05), Hoxa9 (+0.13)

2b Hox PC1 (26% of Hox variance): median by region {'C': -3.554, 'L': 2.653, 'T': -0.037}; Spearman with C<T<L order +0.79; correctly ordered pairs {'C<T': 0.8702651515151515, 'T<L': 0.8953168044077136, 'C<L': 0.9554924242424242}

| comparison | n | rho A | rho B | mean diff | 95% CI | P(diff<=0) | Wilcoxon p | zero-feature folds |
|---|---|---|---|---|---|---|---|---|
| 2c all cells: + position covariate vs baseline | 33 | +0.743 | +0.828 | -0.084 | [-0.234, +0.076] | 0.85 | 0.171 | 0/33 |
| 2d all cells: positional genes excluded vs baseline | 33 | +0.829 | +0.828 | -0.000 | [-0.027, +0.029] | 0.56 | 0.458 | 0/33 |
| 2c astrocyte: + position covariate vs baseline | 33 | +0.793 | +0.895 | -0.103 | [-0.266, +0.042] | 0.93 | 0.135 | 0/33 |
| 2d astrocyte: positional genes excluded vs baseline | 33 | +0.866 | +0.895 | -0.030 | [-0.090, +0.003] | 0.96 | 0.130 | 0/33 |

2e: {"animal_position_vs_day": -0.21307670913762036, "animal_position_vs_score": 0.5880461332290247, "animal_position_kruskal_stage_p": 0.10090684444649355, "within_C_position_vs_day": -0.3330321664448699, "within_C_n": 32, "within_T_position_vs_day": -0.14350749968732476, "within_T_n": 33, "within_L_position_vs_day": -0.17677973029398789, "within_L_n": 33, "animal_position_vs_day_ci": [-0.5562183716263358, 0.15848534603909836]}
2e by stage: {"ONSET1": {"n": 2, "position_median": -1.165, "day": 13.0}, "ONSET2": {"n": 2, "position_median": -0.932, "day": 13.0}, "PEAK1": {"n": 4, "position_median": 1.899, "day": 17.5}, "REMISSION1": {"n": 5, "position_median": 0.394, "day": 22.0}, "PEAK2_MILD": {"n": 2, "position_median": -0.459, "day": 31.0}, "PEAK2": {"n": 2, "position_median": 1.157, "day": 32.5}, "MONOPHASIC": {"n": 4, "position_median": -1.058, "day": 33.0}, "PLP CFA": {"n": 3, "position_median": -2.085, "day": 33.0}, "REMISSION2": {"n": 3, "position_median": -0.627, "day": 44.0}, "PEAK3": {"n": 5, "position_median": 0.347, "day": 48.0}, "REMISSION2_LONG": {"n": 1, "position_median": -1.668, "day": 48.0}}
