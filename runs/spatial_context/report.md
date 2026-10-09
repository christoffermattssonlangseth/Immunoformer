# Task 2 — spatial context (machine table; see WRITEUP.md)

Cell types kept (>= 20 cells in every RR animal): ['Astrocyte', 'DC', 'Doublet', 'Endothelial', 'Ependymal cell', 'Fibroblast', 'Myeloid', 'Neuron', 'OPC', 'Oligodendrocyte', 'Schwann cell', 'VSMC']
Dropped: ['B cell', 'Epithelial', 'Muscle cell', 'NK/DC', 'Neutrophil', 'T cell', 'T_B_doublet']

| condition | n | features in | rho | 95% CI | Fisher z p vs baseline | zero-feature folds |
|---|---|---|---|---|---|---|
| baseline pseudobulk (all genes) | 33 | 5101 | +0.828 | [+0.62, +0.93] | — | 0/33 |
| own expression per cell type (reference) | 33 | 61212 | +0.872 | [+0.73, +0.93] | 0.537 | 0/33 |
| k=10 neighbourhood mean per cell type | 33 | 61212 | +0.828 | [+0.68, +0.91] | 0.998 | 0/33 |
| k=10 pseudobulk + neighbourhood mean | 33 | 66313 | +0.848 | [+0.69, +0.92] | 0.797 | 0/33 |
| k=10 |own - neighbourhood| per cell type | 33 | 61212 | +0.876 | [+0.74, +0.94] | 0.493 | 0/33 |
| k=10 pseudobulk + |own - neighbourhood| | 33 | 66313 | +0.890 | [+0.74, +0.95] | 0.352 | 0/33 |
| k=30 neighbourhood mean per cell type | 33 | 61212 | +0.836 | [+0.68, +0.91] | 0.923 | 0/33 |
| k=30 pseudobulk + neighbourhood mean | 33 | 66313 | +0.822 | [+0.62, +0.91] | 0.935 | 0/33 |
| k=30 |own - neighbourhood| per cell type | 33 | 61212 | +0.877 | [+0.74, +0.94] | 0.484 | 0/33 |
| k=30 pseudobulk + |own - neighbourhood| | 33 | 66313 | +0.877 | [+0.74, +0.94] | 0.482 | 0/33 |
