# Datasets at a glance

Three mouse CNS spatial transcriptomics datasets, all 10x Xenium with the same 5,101-gene
panel, single-cell resolution, µm coordinates. Numbers verified from the files on 2026-09-30.

| | RRMAP2 spinal cord | Optic nerve | mtDNA-DSB brain |
|---|---|---|---|
| Disease | EAE (autoimmune), two models | EAE (autoimmune) | oligodendrocyte mitochondrial DNA damage, no immune trigger |
| Cells | 1,384,881 | 288,023 | 980,474 |
| Genes | 5,101 | 5,101 | 5,101 |
| Animals | 67 | not recorded (30 nerve pieces) | 12 |
| Sections | 158 | 30 polygons | 12 (one per mouse) |
| Slides | 9 (54 capture regions) | 6 | 2 |
| CNS region | spinal cord: lumbar, thoracic, cervical | optic nerve | whole brain, 12 annotated regions |
| Timepoints | 19 stages, sacrifice day 8 to 50 | Onset, Peak | age 21 and 60 (units to confirm) |
| Severity | clinical score 0 to 3.25 | Control, Sham, Mild, Severe | control vs mtDSB |
| Cell types | 19 lineages, 42 subtypes, 137 states | none yet | 13 classes, 44 types |
| Niches | CellCharter 5 to 100 clusters, 27 named niches, 7 disease states | CellCharter 6 to 30 clusters | Novae domains 15 to 50, brain anatomy |
| Role | main dataset | cross-tissue test | cross-etiology test |

## How the data is formatted

Each dataset is one **AnnData** object saved as `.h5ad` (HDF5). AnnData is the standard
Python container for single-cell data: a cells × genes matrix with aligned per-cell and
per-gene tables, plus named slots for embeddings, graphs and metadata. Read with
`anndata.read_h5ad(path)` or lazily with `backed="r"`; `scanpy` and `squidpy` operate on
it directly. Metadata can be read without touching the matrix via `h5py`.

```
adata                       AnnData, n_obs cells × n_vars genes
├── X                       sparse CSR, log1p(CP10k)-normalised expression
├── layers['counts']        sparse CSR, raw transcript counts (same shape as X)
├── obs                     DataFrame, one row per cell  → all cell-level and sample-level labels
├── var                     DataFrame, one row per gene  → gene symbol, Ensembl id, HVG flag
├── obsm                    per-cell arrays
│   ├── spatial             (n_cells, 2)   x, y centroid in µm
│   ├── X_pca               (n_cells, 50)
│   ├── X_umap              (n_cells, 2)
│   ├── X_scVI              (n_cells, 10)  batch-corrected expression embedding   [RRMAP2, optic nerve]
│   ├── X_cellcharter       (n_cells, 40)  neighbourhood-aggregated scVI         [RRMAP2, optic nerve]
│   └── novae_latent        (n_cells, 64)  Novae spatial embedding               [mtDNA-DSB]
├── obsp                    cell × cell sparse graphs
│   ├── spatial_connectivities / spatial_distances    spatial neighbours (radius 50–80 µm, ~6 per cell)
│   └── connectivities / distances                    expression kNN (k = 15)
├── varm['PCs']             (n_genes, 50) PCA loadings
└── uns                     unstructured: clustering params, colours, DE tables, polygon annotations
```

**Where things live in `obs`.** Every column is stored per cell, even if it is really a
sample-level property. Sample-level columns (stage, score, day, condition, model, sex,
run) are simply repeated for every cell of that sample; group by the sample column and
take the first value to get an animal-level table. Categorical columns are stored as
pandas categoricals.

| kind of information | RRMAP2 | Optic nerve | mtDNA-DSB |
|---|---|---|---|
| animal | `sample_name` | not recorded | `sample_id` |
| section / bag | `meta_sample_id` | `karospace_polygon_labels` | `sample_id` |
| slide / batch | `sample_id`, `cassette_or_slide_id`, `run_date` | `sample_id` | `run` |
| disease labels | `stage`, `score_sacrifice`, `day_of_sacrifice`, `condition`, `model` | `condition`, `timepoint` | `condition`, `genotype`, `age` |
| anatomy | `region` (L/T/C), `Global_anatomical_region` | — | `RBD_compartment_simplified` |
| cell type | `Anno_L1_curated` … `Anno_L4` | — (clusters only) | `cell_class_updated`, `cell_type` |
| niche | `Curated_niche_state`, `Global_niche`, `CellCharter_*` | `CellCharter_*` | `novae_domains_*` |
| QC | `transcript_counts`, `n_genes`, `cell_area`, control-probe counts, `segmentation_method` | same | same |

**Matrix conventions.** `X` is already normalised (counts per 10,000 then log1p), so do
not normalise it again; models that want raw counts should read `layers['counts']`. Both
are scipy CSR sparse. The RRMAP2 matrix has 821 M non-zeros; loading `X` and `counts`
fully into memory needs roughly 15 to 20 GB.

**Coordinates.** `obsm['spatial']` is in µm but the origin is per slide or per capture
region, not global, so coordinates from different sections overlap. Always subset to one
section before computing distances or neighbourhoods. The optic nerve and mtDNA-DSB
spatial graphs already respect section boundaries; the RRMAP2 one does not and must be
rebuilt per section.

**Gene space.** All three files have the same 5,101 genes in the same order (`var` index
is the gene symbol; RRMAP2 also carries Ensembl ids in `var['gene_ids']`). Gene-space
alignment across datasets is therefore trivial.

## RRMAP2 spinal cord

**Two EAE models, 67 animals, one timepoint per animal.**

| | Chronic | Relapse-remitting |
|---|---|---|
| Strain, antigen | C57BL/6, MOG35-55 | SJL/J, PLP139-151 |
| Animals | 34 (29 EAE, 5 control) | 33 (30 EAE, 3 control) |
| Course | onset, peak, chronic plateau | onset, peak 1, remission 1, peak 2, remission 2, peak 3 |
| Stages | 9 | 11 |
| Sacrifice day | 8 to 50 | 11 to 49 |
| Cord levels sampled | 34 L, 22 T, 3 C | 33 L, 33 T, 32 C |

**Chronic stages (animals, score, day):** MOG CFA (3, 0, d8-9), NONSYMPTOM (5, 0, d8-9), OS1 (5, 0.5, d10-14), PEAK1 (6, 3.0, d13-18), CFA (2, 0, d16-17), MILD16 (3, 1.25, d27-28), SEVERE16 (3, 2.5, d28-29), SEVERE30 (2, 2.4, d41), MILD30 (5, 1.0, d43-50).

**RR stages (animals, score, day):** ONSET1 (2, 0.25, d11-15), ONSET2 (2, 1.0, d13), PEAK1 (4, 2.75, d14-18), REMISSION1 (5, 0.75, d20-25), PEAK2_MILD (2, 2.0, d31), PEAK2 (2, 2.9, d32-33), MONOPHASIC (4, 0.6, d32-33), PLP CFA (3, 0, d33), PEAK3 (5, 2.5, d38-49), REMISSION2 (3, 1.25, d42-47), REMISSION2_LONG (1, 0.75, d48).

**Structure**
- One animal gives 1 to 3 tissue sections, one per cord level. 158 sections of 2,500 to 20,000 cells (median 8,000).
- Section = `meta_sample_id`, animal = `sample_name`. The `sample_id` column is a Xenium capture region that holds 2 to 3 animals, so it must not be used as a label unit.
- Per animal: stage, clinical score, sacrifice day, sex (48 M, 19 F), model, condition, run date.
- Three runs: Jan 2025 (chronic only), Jun 2025 (RR only), May 2026 (both).

**Cell annotation:** `Anno_L1_curated` with 19 lineages (oligodendrocyte 281k, myeloid 217k, neuron 217k, fibroblast 143k, astrocyte 142k, endothelial 112k, DC 61k, Schwann 59k, OPC 46k, T cell 45k, VSMC 29k, B cell 13k, rest small). Finer levels `Anno_L2` (42), `Anno_L3` (137), `Anno_L4` (174).

**Spatial annotation:** anatomy (`Global_anatomical_region`, 10: white matter, grey matter, dorsal horn, ventral horn, meninges, DRG, vasculature, central canal), named niches (`Global_niche`, 27, e.g. Lesion_myeloid, Lesion_Foamy, Lesion_Tcell, Lesion_Bcell), and disease state (`Curated_niche_state`, 7: Physiological, Lesion_active, Lesion_progressive, Lesion_mix, Lesion_reactive, Meninges, Vasculature). Lesion_active goes from 0 % of cells in controls to 70 to 76 % at RR peaks and back to 2 to 11 % in remission.

**Embeddings:** scVI (10), CellCharter (40), PCA (50), UMAP, spatial kNN graph (needs rebuilding per section, 14.6 % of edges cross sections).

## Optic nerve

- 288k cells, 30 nerve pieces (polygons) on 6 slides, longitudinal sections of about 4 by 1.5 mm.
- Each polygon has one condition and one timepoint: Control 8, Sham 8, Mild 9, Severe 5; Onset 16, Peak 14.
- Mouse identity is not stored. The `animal` column (CM1 to CM9) is a slide group, not a mouse.
- Timepoint is nearly aliased with slide (3 Onset-only slides, 2 Peak-only, 1 mixed). Condition is spread across slides.
- No cell-type labels, only leiden and CellCharter clusters. Spatial graph is clean.

## mtDNA-DSB brain

- 980k cells, 12 mice, one coronal brain section each (70k to 135k cells).
- Design: 3 mice per cell of condition (control vs mtDSB) by age (21 vs 60). 6 female, 6 male.
- Genotypes: mtDSB = PlptTA:mtPst1; controls = PlptTA (5) or mtPst1 (1).
- All age-60 mice on one slide, all age-21 on the other, so age equals slide.
- Cell types: `cell_class_updated` (13: excitatory neurons, neurons, astrocytes, oligodendrocytes, endothelial, GABAergic, medium spiny neurons, microglia, immature oligodendrocytes, disease-associated oligodendrocytes, neural stem cells, endocrine, myocytes) and `cell_type` (44). Almost no T, B or dendritic cells.
- Brain regions: `RBD_compartment_simplified` (12: cortex, thalamus, striatum, hypothalamus, fiber tracts, olfactory areas, meningeal border, ventricular system, hippocampus, pallidum, vasculature, unknown).
- Novae spatial domains (15 to 50) and a 64-d Novae embedding. Spatial graph is clean.

## Files

| dataset | file | size | where |
|---|---|---|---|
| RRMAP2 | `RRMAP2_xenium_all_samples.cellcharter.companion.ready.with_metadata.rerun.with_AnnoL1Curated_with_Region_Anno2to4Updated.h5ad` | 4.3 GB | laptop `~/Downloads` |
| Optic nerve | `optic_nerve_merged.scanpy.companion.ready_with_polygons.h5ad` | 2.6 GB | laptop `~/Downloads`, remote `/Volumes/moldiassd` |
| mtDNA-DSB | `mtDNA_DSB_5k_clustered_annotation_with_rbd_2_cytetype_brain_novae4.companion.ready.h5ad` | 10 GB | remote `/Volumes/moldiassd/oligo-mtDSB/data` |

Detailed write-ups: `rrmap2-data-structure.md`, `optic-nerve-data-structure.md`, `mtdna-dsb-data-structure.md`.
