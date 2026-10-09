# Paired comparisons on the same animals (decisive test)

Bootstrap over animals (2000): both rho recomputed on each resample; difference = A - B.
Wilcoxon signed-rank on per-animal |prediction - residualised day|.

| comparison | n | rho A | rho B | mean diff | 95% CI of diff | P(diff <= 0) | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| B3 nested spatial vs pseudobulk | 33 | +0.858 | +0.828 | +0.029 | [-0.073, +0.162] | 0.32 | 0.339 |
| A2 nested cell type vs pseudobulk | 33 | +0.781 | +0.828 | -0.049 | [-0.140, +0.022] | 0.89 | 0.502 |
| 1f Astrocyte vs all cells (same 33 animals) | 33 | +0.895 | +0.828 | +0.068 | [-0.018, +0.202] | 0.08 | 0.406 |
| 1f B cell vs all cells (same 26 animals) | 26 | +0.855 | +0.758 | +0.101 | [-0.004, +0.251] | 0.03 | 0.803 |
| 1f DC vs all cells (same 32 animals) | 32 | +0.816 | +0.823 | -0.005 | [-0.102, +0.088] | 0.55 | 0.890 |
| 1f Endothelial vs all cells (same 33 animals) | 33 | +0.766 | +0.828 | -0.064 | [-0.214, +0.082] | 0.82 | 0.074 |
| 1f Ependymal cell vs all cells (same 33 animals) | 33 | +0.643 | +0.828 | -0.185 | [-0.437, +0.013] | 0.97 | 0.007 |
| 1f Fibroblast vs all cells (same 33 animals) | 33 | +0.892 | +0.828 | +0.065 | [-0.029, +0.204] | 0.11 | 0.126 |
| 1f Myeloid vs all cells (same 33 animals) | 33 | +0.850 | +0.828 | +0.021 | [-0.044, +0.109] | 0.30 | 0.778 |
| 1f Neuron vs all cells (same 33 animals) | 33 | +0.713 | +0.828 | -0.116 | [-0.282, +0.035] | 0.94 | 0.010 |
| 1f OPC vs all cells (same 33 animals) | 33 | +0.817 | +0.828 | -0.010 | [-0.123, +0.112] | 0.59 | 0.659 |
| 1f Oligodendrocyte vs all cells (same 33 animals) | 33 | +0.827 | +0.828 | -0.001 | [-0.092, +0.103] | 0.53 | 0.888 |
| 1f Schwann cell vs all cells (same 33 animals) | 33 | +0.778 | +0.828 | -0.051 | [-0.192, +0.101] | 0.78 | 0.280 |
| 1f T cell vs all cells (same 30 animals) | 30 | +0.843 | +0.871 | -0.027 | [-0.106, +0.044] | 0.78 | 0.299 |
| 1f VSMC vs all cells (same 33 animals) | 33 | +0.845 | +0.828 | +0.016 | [-0.099, +0.150] | 0.41 | 0.832 |

PRE-REGISTERED READING (B3): difference CI INCLUDES zero -> learned spatial context does not beat pseudobulk; the spage2vec direction is shelved.
