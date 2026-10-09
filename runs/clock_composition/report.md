# Task 1 — report (machine-generated tables; see WRITEUP.md for the reading)

Audit loop reproduces loao_clock exactly: True

| condition | n | genes in | rho | 95% CI | Fisher z p vs baseline | zero-feature folds |
|---|---|---|---|---|---|---|
| baseline (all genes) | 33 | 5101 | +0.828 | [+0.62, +0.93] | — | 0/33 |
| 1b without circadian genes | 33 | 5084 | +0.799 | [+0.57, +0.90] | 0.737 | 0/33 |
| 1c circadian genes only | 33 | 17 | +0.855 | [+0.72, +0.92] | 0.722 | 0/33 |
| chronic cumulative_score, all genes (= ladder arm 2) | 29 | 5101 | +0.730 | [+0.51, +0.87] | — | 0/29 |
| 1c chronic cumulative_score, circadian only | 29 | 17 | +0.480 | [+0.16, +0.72] | 0.144 | 0/29 |
| 1d(i) sex genes dropped | 33 | 5100 | +0.825 | [+0.62, +0.92] | 0.967 | 0/33 |
| 1d(ii) sex residualised (covariate) | 33 | 5101 | +0.844 | [+0.64, +0.93] | 0.843 | 0/33 |
| 1e circadian AND sex genes excluded | 33 | 5083 | +0.802 | [+0.59, +0.90] | 0.756 | 0/33 |

circadian score (mean z of ['Nr1d1', 'Dbp', 'Bhlhe40', 'Per1', 'Hlf']) vs day: rho +0.49; vs score_sacrifice: -0.52

sex: {"source": "metadata field `sex` (sheet + atlas); cross-checked against Xist pseudobulk", "sex_genes_on_panel": ["Xist"], "xist_agrees_with_metadata": true (Xist F 1.72-2.24 vs M 0.004-0.12), "sex_x_day_bin": {"F": {"<=20": 3, "21-35": 4, ">35": 0}, "M": {"<=20": 6, "21-35": 11, ">35": 9}}, "sex_x_stage": {"MONOPHASIC": {"F": 1, "M": 3}, "ONSET1": {"F": 1, "M": 1}, "ONSET2": {"F": 1, "M": 1}, "PEAK1": {"F": 1, "M": 3}, "PEAK2": {"F": 1, "M": 1}, "PEAK2_MILD": {"F": 0, "M": 2}, "PEAK3": {"F": 0, "M": 5}, "PLP CFA": {"F": 1, "M": 2}, "REMISSION1": {"F": 1, "M": 4}, "REMISSION2": {"F": 0, "M": 3}, "REMISSION2_LONG": {"F": 0, "M": 1}}, "rho_sex_day": 0.2426668692772672}

1e: {"n_nonzero": 104, "block_mass_clean": {"matrix/scar": 0.23425602873431045, "lipid": 0.08085744038947804, "microglial identity": 0.04275539916214731, "lymphoid": 0.023053955090804653, "tissue loss": 0.005432045931885363, "circadian": 0.0, "sex": 0.0}, "block_mass_baseline_refit": {"matrix/scar": 0.22126106687839964, "circadian": 0.11262508984913702, "lipid": 0.07948194424441304, "microglial identity": 0.0380035257385751, "lymphoid": 0.0373510677816648, "tissue loss": 0.01028722481020811, "sex": 0.0}, "block_mass_published_135": {"matrix/scar": 0.17228912181496686, "circadian": 0.10840402854226196, "lymphoid": 0.06617870041174892, "lipid": 0.06133799231509967, "microglial identity": 0.03656506506316598, "tissue loss": 0.025639095432942308, "sex": 0.014578900374305984}, "first_matrix_gene_rank": 1, "overlap_with_published_135": 80, "same_sign": 80, "baseline_refit_n_nonzero": 99, "overlap_baseline_refit_with_published": 85}
1e top 30: Col4a2 (-1.38), Thbs2 (+1.19), Igfbp2 (+0.79), H19 (-0.62), Eln (-0.53), Srebf1 (+0.48), Tmem173 (-0.48), Cxcl13 (+0.46), Hsd17b7 (-0.46), Col4a1 (-0.43), Cxcl12 (+0.41), Lpl (+0.38), Serpine2 (+0.38), Fmod (+0.37), Ifit1 (-0.37), S1pr3 (+0.37), Wnt6 (+0.36), 9630013A20Rik (-0.35), Fcgrt (+0.35), Ccr2 (-0.35), Ifit3 (-0.34), Usp18 (-0.32), Hmgcr (-0.30), Crym (+0.30), Cavin1 (+0.30), Fcrls (+0.29), Cyb5r3 (+0.29), Col6a1 (+0.27), Atf5 (+0.27), Klf5 (+0.24)

## 1f per cell type (ranked)

| cell type | animals | rho | 95% CI | all cells, same animals | p vs all cells | zero-feature folds | top genes |
|---|---|---|---|---|---|---|---|
| Astrocyte | 33 | +0.895 | [+0.76, +0.95] | +0.828 | 0.305 | 0/33 | Hoxa10 (+1.05), Nr1d1 (+0.92), Hr (-0.73), Daam2 (-0.72), Bhlhe40 (+0.71), Myo7a (+0.65), Amy1 (+0.64), Ciart (+0.51), Ifit1 (-0.48), Pink1 (+0.48) |
| Fibroblast | 33 | +0.892 | [+0.76, +0.94] | +0.828 | 0.339 | 0/33 | Dbp (+1.28), Eln (-0.97), Msln (-0.88), Bhlhe40 (+0.88), Per1 (+0.87), Aspa (+0.87), Pink1 (+0.77), Ccl8 (+0.76), Grem1 (+0.76), Stat2 (-0.72) |
| B cell | 26 | +0.855 | [+0.66, +0.94] | +0.758 | 0.338 | 0/26 | Tagln (-1.20), Atp6v0d2 (+0.72), Txnip (+0.63), Fbxo2 (+0.60), A2m (+0.59), Col4a2 (-0.54), Gja1 (+0.50), Efemp1 (+0.50), Cd163 (-0.49), Appl2 (+0.48) |
| Myeloid | 33 | +0.850 | [+0.68, +0.92] | +0.828 | 0.783 | 0/33 | Itga4 (-0.91), Pink1 (+0.76), Jchain (-0.74), Nr1d1 (+0.70), Svil (-0.68), Ccl5 (-0.67), Plau (+0.63), Nefh (-0.62), Fcrls (+0.61), Icosl (-0.59) |
| VSMC | 33 | +0.845 | [+0.66, +0.93] | +0.828 | 0.836 | 0/33 | Rgcc (+1.02), Pdgfra (-0.89), Hoxa9 (+0.82), F13a1 (-0.80), Wnt7a (+0.79), Jag2 (+0.61), Adgre1 (-0.56), Ciart (+0.55), Ifih1 (-0.53), Myd88 (-0.52) |
| T cell | 30 | +0.843 | [+0.64, +0.93] | +0.871 | 0.699 | 0/30 | Cd4 (-0.99), Thbs2 (+0.96), F13a1 (-0.83), Fcrls (+0.81), Col4a2 (-0.79), Crabp2 (+0.71), Cd79a (-0.63), Slc6a11 (-0.63), Gap43 (+0.57), Cd9 (+0.56) |
| Oligodendrocyte | 33 | +0.827 | [+0.65, +0.91] | +0.828 | 0.989 | 0/33 | Nr1d1 (+1.45), Thbs2 (+1.37), Tmem173 (-0.81), Fam83d (-0.72), Pkd2l1 (+0.72), Kndc1 (-0.68), Lamb2 (+0.61), Trp53 (+0.55), Nod1 (-0.51), Gap43 (+0.50) |
| OPC | 33 | +0.817 | [+0.65, +0.90] | +0.828 | 0.887 | 0/33 | Fcrls (+0.80), Hsd17b7 (-0.78), Thbs2 (+0.75), Ndst1 (+0.75), Dbp (+0.67), Lpl (+0.60), Shmt1 (+0.59), Wnt7b (+0.54), S1pr3 (+0.53), Fam20c (+0.53) |
| DC | 32 | +0.816 | [+0.64, +0.91] | +0.823 | 0.939 | 0/32 | Fcrls (+0.96), Cxcl13 (+0.90), Chil3 (-0.84), Cd209d (-0.83), Cd79a (-0.76), Pink1 (+0.75), Nr2f1 (+0.65), Pltp (+0.59), Igf1 (+0.58), Osmr (+0.50) |
| Schwann cell | 33 | +0.778 | [+0.59, +0.89] | +0.828 | 0.583 | 0/33 | Apc (-1.08), Nr1d1 (+0.96), Mki67 (-0.74), Dock2 (-0.72), Cd24a (-0.72), Pde8a (+0.71), Pkp4 (-0.70), Nrcam (-0.67), Efemp1 (+0.64), Ptpru (+0.64) |
| Endothelial | 33 | +0.766 | [+0.57, +0.88] | +0.828 | 0.502 | 0/33 | Gfra1 (+1.34), Itgb4 (+1.26), Tie1 (+0.89), Lfng (+0.75), Pink1 (+0.59), Dixdc1 (+0.56), A2m (+0.52), Cd79a (-0.46), Cxcl13 (+0.46), Igfbp2 (+0.41) |
| Neuron | 33 | +0.713 | [+0.46, +0.86] | +0.828 | 0.262 | 0/33 | Msmo1 (-0.70), Ccl5 (-0.64), Hsd17b7 (-0.46), Irgm2 (-0.45), Vat1 (+0.45), Irgm1 (-0.43), Idi1 (-0.43), Stat1 (-0.36), Sst (+0.35), Cldn11 (-0.34) |
| Ependymal cell | 33 | +0.643 | [+0.37, +0.81] | +0.828 | 0.104 | 0/33 | Nkx6-2 (+0.81), Pnpla2 (-0.74), Ptprn2 (+0.72), Hoxa10 (+0.72), Slc6a5 (-0.61), Tssc4 (+0.59), Ldlr (-0.58), Nr1d1 (+0.54), Got2 (+0.54), Fbxl13 (+0.52) |
| Doublet | 17 | skipped | | | | | |
| Epithelial | 0 | skipped | | | | | |
| Muscle cell | 1 | skipped | | | | | |
| NK/DC | 20 | skipped | | | | | |
| Neutrophil | 0 | skipped | | | | | |
| T_B_doublet | 9 | skipped | | | | | |
