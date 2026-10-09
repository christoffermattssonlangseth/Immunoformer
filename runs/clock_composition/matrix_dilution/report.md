# Matrix sign check — machine tables

Focus genes on panel: ['Col4a1', 'Col4a2', 'Lama1', 'Lama2', 'Lama4', 'Lama5', 'Fn1', 'Hspg2', 'Thbs2', 'Serpine2', 'Eln', 'Fbln2', 'Ptx3', 'Fmod']; not on panel: []

## (a) whole-tissue pseudobulk vs day (RR, n = 33)

| gene | rho | 95% CI | rho partial on score | 95% CI |
|---|---|---|---|---|
| Col4a1 | -0.32 | [-0.56, -0.01] | -0.46 | [-0.70, -0.08] |
| Col4a2 | -0.30 | [-0.54, -0.00] | -0.53 | [-0.73, -0.14] |
| Eln | -0.43 | [-0.68, -0.11] | -0.48 | [-0.70, -0.18] |
| Fbln2 | +0.22 | [-0.10, +0.49] | +0.17 | [-0.09, +0.42] |
| Fmod | +0.72 | [+0.44, +0.87] | +0.82 | [+0.61, +0.92] |
| Fn1 | -0.17 | [-0.50, +0.18] | -0.40 | [-0.65, -0.02] |
| Hspg2 | +0.52 | [+0.24, +0.73] | +0.51 | [+0.21, +0.73] |
| Lama1 | +0.57 | [+0.28, +0.77] | +0.60 | [+0.31, +0.79] |
| Lama2 | +0.58 | [+0.28, +0.77] | +0.65 | [+0.39, +0.81] |
| Lama4 | +0.23 | [-0.15, +0.57] | +0.25 | [-0.14, +0.59] |
| Lama5 | +0.49 | [+0.13, +0.77] | +0.48 | [+0.14, +0.77] |
| Ptx3 | -0.20 | [-0.51, +0.13] | -0.35 | [-0.64, +0.01] |
| Serpine2 | +0.37 | [+0.04, +0.63] | +0.45 | [+0.13, +0.71] |
| Thbs2 | +0.48 | [+0.16, +0.69] | +0.50 | [+0.17, +0.72] |

## immune fraction vs day

- all immune: median 21.7% (range 3.5%-51.3%); rho with day -0.10 [-0.44, +0.23]; partial on score -0.35 [-0.66, +0.04]
- infiltrating (no Myeloid): median 8.4% (range 0.3%-25.4%); rho with day -0.16 [-0.48, +0.17]; partial on score -0.35 [-0.64, +0.02]
- Myeloid (microglia + macrophages): median 12.9% (range 3.2%-37.5%); rho with day +0.03 [-0.32, +0.37]; partial on score -0.18 [-0.51, +0.17]

## resolution per focus gene

- **Col4a1: genuine within-cell-type decline** — whole -0.32 [-0.56, -0.01]; top expressing: Endothelial 59.0% pos, 1.767/cell; Fibroblast 47.4% pos, 1.625/cell; within those: Endothelial -0.48 [-0.71, -0.17]; Fibroblast -0.27 [-0.55, +0.07]
- **Col4a2: genuine within-cell-type decline** — whole -0.30 [-0.54, -0.00]; top expressing: Endothelial 48.2% pos, 1.044/cell; VSMC 50.1% pos, 1.013/cell; within those: Endothelial -0.52 [-0.74, -0.22]; VSMC -0.35 [-0.63, -0.02]
- **Lama1: no clear trend** — whole +0.57 [+0.28, +0.77]; top expressing: Fibroblast 20.0% pos, 0.373/cell; VSMC 13.4% pos, 0.329/cell; within those: Fibroblast +0.42 [+0.09, +0.67]; VSMC +0.26 [-0.10, +0.57]
- **Lama2: no clear trend** — whole +0.58 [+0.28, +0.77]; top expressing: VSMC 39.5% pos, 0.648/cell; Schwann cell 30.8% pos, 0.555/cell; within those: Schwann cell +0.42 [+0.09, +0.68]; VSMC +0.15 [-0.23, +0.48]
- **Lama4: no clear trend** — whole +0.23 [-0.15, +0.57]; top expressing: Fibroblast 28.6% pos, 0.428/cell; VSMC 25.2% pos, 0.336/cell; within those: Fibroblast -0.22 [-0.55, +0.15]; VSMC -0.25 [-0.55, +0.12]
- **Lama5: no clear trend** — whole +0.49 [+0.13, +0.77]; top expressing: Ependymal cell 52.7% pos, 1.007/cell; VSMC 31.0% pos, 0.579/cell; within those: Ependymal cell +0.52 [+0.20, +0.76]; VSMC +0.32 [-0.05, +0.62]
- **Fn1: no clear trend** — whole -0.17 [-0.50, +0.18]; top expressing: Endothelial 79.1% pos, 3.640/cell; Myeloid 50.1% pos, 3.316/cell; within those: Endothelial -0.30 [-0.61, +0.07]; Myeloid -0.26 [-0.55, +0.09]
- **Hspg2: no clear trend** — whole +0.52 [+0.24, +0.73]; top expressing: Schwann cell 61.8% pos, 1.966/cell; Fibroblast 58.7% pos, 1.522/cell; within those: Fibroblast +0.41 [+0.03, +0.69]; Schwann cell +0.57 [+0.24, +0.79]
- **Thbs2: no clear trend** — whole +0.48 [+0.16, +0.69]; top expressing: Astrocyte 52.5% pos, 1.399/cell; Fibroblast 41.3% pos, 0.837/cell; within those: Astrocyte -0.06 [-0.39, +0.28]; Fibroblast +0.49 [+0.16, +0.75]
- **Serpine2: no clear trend** — whole +0.37 [+0.04, +0.63]; top expressing: Astrocyte 83.4% pos, 3.161/cell; OPC 76.5% pos, 2.789/cell; within those: Astrocyte -0.05 [-0.37, +0.29]; OPC +0.18 [-0.14, +0.46]
- **Eln: genuine within-cell-type decline** — whole -0.43 [-0.68, -0.11]; top expressing: Fibroblast 28.0% pos, 0.691/cell; VSMC 23.0% pos, 0.690/cell; within those: Fibroblast -0.65 [-0.82, -0.38]; VSMC +0.08 [-0.23, +0.38]
- **Fbln2: no clear trend** — whole +0.22 [-0.10, +0.49]; top expressing: Fibroblast 20.8% pos, 0.371/cell; Schwann cell 17.7% pos, 0.343/cell; within those: Fibroblast +0.01 [-0.33, +0.33]; Schwann cell -0.32 [-0.60, +0.02]
- **Ptx3: no clear trend** — whole -0.20 [-0.51, +0.13]; top expressing: Astrocyte 8.4% pos, 0.227/cell; Ependymal cell 2.9% pos, 0.056/cell; within those: Astrocyte -0.23 [-0.52, +0.10]; Ependymal cell -0.14 [-0.48, +0.26]
- **Fmod: no clear trend** — whole +0.72 [+0.44, +0.87]; top expressing: Fibroblast 37.7% pos, 1.307/cell; Astrocyte 23.9% pos, 0.597/cell; within those: Astrocyte +0.69 [+0.43, +0.86]; Fibroblast +0.48 [+0.14, +0.73]

Full within-type table: results.json -> within; detection: detection_by_celltype.csv
