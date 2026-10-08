# Optic nerve EAE — data structure

*Written 2026-09-30 by direct inspection. Companion to `rrmap2-data-structure.md`.
Corrects the `animal` interpretation in `dataset-hierarchy.md` and the transfer configs.*

**Canonical file (2.6 GB):**
`~/Downloads/optic_nerve_merged.scanpy.companion.ready_with_polygons.h5ad`
(`optic_nerve_merged.scanpy.companion.ready.h5ad` is the same object before polygon
annotation was added; do not use it.)

## 1. What the experiment is

Mouse **optic nerve**, EAE, sampled at two clinical timepoints (Onset, Peak) across four
condition groups (Control, Sham, Mild, Severe). Same 10x Xenium 5,101-gene mouse panel as
RRMAP2, so gene space is identical and models transfer without panel alignment.

Cross-sectional, one timepoint per nerve. No cell-type annotation has been done on this
file yet (only unsupervised clusters).

## 2. Size

| unit | n | notes |
|---|---|---|
| cells | 288,023 | |
| genes | 5,101 | identical panel to RRMAP2; `var` index is `gene` |
| nerve pieces (`karospace_polygon_labels`) | 30 | manually drawn polygons; **the experimental unit**. 2.2k–28k cells each |
| slides (`sample_id` ≡ `sample_batch`) | 6 | ON_TL_1, ON_TR_2, ON_ML_3, ON_MR_4, ON_BL_5, ON_BR_6 (top/middle/bottom × left/right) |
| `animal` | 6 | CM1, CM2, CM3, CM4, CM7, CM9 — see §3, this is **not** a mouse |

Median 897 transcripts and 499 genes per cell; median cell area 64 µm².

## 3. Sample hierarchy and the `animal` trap

```
sample_id / sample_batch (6 slides)
   └── karospace_polygon_labels (30 nerve pieces)     ← LABEL UNIT, likely one mouse each
          label format:  <polygon_index>_<CMx>_<condition>_<timepoint>
          └── cell
animal (6, CM*)  ≈ slide-level group id; 1:1 with slide except CM1 (spans ON_TL_1 + ON_ML_3)
```

Verified facts:
- Every polygon has exactly one `animal`, `condition`, `timepoint`, and (with one
  exception) one slide. Labels are clean at polygon level.
- **`animal` is not a biological replicate.** CM1 contains Control ×4, Mild ×1, Severe ×1
  polygons; CM4 contains Control, Mild, Severe, Sham and both Onset and Peak. One mouse
  cannot be all of those. `animal` maps 1:1 onto slide (CM1 → ON_TL_1, CM2 → ON_ML_3,
  CM3 → ON_BL_5, CM4 → ON_TR_2, CM7 → ON_BR_6, CM9 → ON_MR_4) and is best read as a
  **cassette / slide-group identifier**.
- `polygon_index` (1–6, the numeric prefix) encodes **layout position on the slide** and
  correlates with condition: positions 1–3 are mostly Mild/Severe/Control EAE nerves,
  positions 4–6 are Sham/Control. This is a plating design, not a biological variable.
- The true mouse identity is not recorded. Most likely each polygon is one nerve from one
  mouse (30 mice), but a mouse has two optic nerves, so two polygons could share a mouse.
  **This needs confirming with the wet-lab metadata before choosing a split key.**

### Design table (polygons)

| condition | Onset | Peak | total |
|---|---|---|---|
| Control | 4 | 4 | 8 |
| Sham | 4 | 4 | 8 |
| Mild | 5 | 4 | 9 |
| Severe | 3 | 2 | 5 |
| **total** | 16 | 14 | 30 |

Slides ON_TL_1, ON_ML_3, ON_BL_5 are Onset-only; ON_BR_6, ON_MR_4 are Peak-only;
ON_TR_2 mixes both. **Timepoint is therefore almost perfectly confounded with slide.**
Any Onset-vs-Peak model must be validated across slides, and the honest reading is that
timepoint and slide batch cannot be separated in this design. Condition, in contrast, is
spread across all slides and is the cleaner target.

## 4. Labels

| column | levels | notes |
|---|---|---|
| `condition` | Control / Sham / Mild / Severe | severity axis; Control and Sham are both non-EAE (Sham = adjuvant-only) |
| `timepoint` | Onset / Peak | slide-confounded (see above) |
| `karospace_polygon_labels` | 30 | encodes both, plus layout |
| `tissue_type` | optic_nerve | constant |

No clinical score, day of sacrifice, sex, or run date is stored. If these exist in the lab
records they should be joined on the polygon label.

## 5. Cell-level annotations

- **No cell-type labels.** Only unsupervised clusters: `leiden` (17, ≡ `leiden_0_8`),
  `leiden_0_2` … `leiden_4_0` (9–67 clusters), and CellCharter neighbourhood clusters
  `CellCharter_6` … `CellCharter_30` (8 resolutions).
- Cell typing could be transferred from RRMAP2 `Anno_L1_curated` / `Anno_L2` via the
  shared panel, but optic nerve lacks neurons/grey matter and has its own
  astrocyte/oligodendrocyte/meningeal composition, so expect some label mismatch.
- QC columns as in RRMAP2 (`transcript_counts`, `n_genes`, `cell_area`, `nucleus_area`,
  negative-control counts, `segmentation_method`). 97 % of cells are interior-stain
  segmented (vs a three-way mix in RRMAP2).
- `karospace_polygon_count` (1 or 2): a few cells fall in two overlapping polygons.

## 6. Matrices and embeddings

| slot | content |
|---|---|
| `X` | log1p(CP10k), CSR |
| `layers['counts']` | raw counts, CSR |
| `var` | 5,101 genes; `highly_variable` = 4,000 (seurat_v3, batch-aware) |
| `obsm['spatial']` | µm, **per-slide frames** (~0–5,400 × 0–9,000 µm, all 6 slides overlap) |
| `obsm['X_scVI']` (10), `X_cellcharter` (40), `X_pca` (50), `X_umap` (2), `X_karo_comp` (10) | as RRMAP2 |
| `obsp['spatial_connectivities' / 'spatial_distances']` | 6-NN within 50 µm radius; **0 cross-slide edges, 0.14 % cross-polygon edges** — usable as is |
| `obsp['connectivities' / 'distances']` | expression kNN |
| `uns['karospace_polygon_annotations']` | the polygon vertex definitions |

Nerve pieces are elongated: typical polygon ~3.5–4.5 mm long × 1–2 mm wide, i.e.
longitudinal nerve sections.

## 7. Role in the project

- Same disease, same panel, different CNS tissue → the **cross-tissue generalisation**
  test for anything trained on RRMAP2 spinal cord.
- Bag unit = polygon (30 bags). Too few for training a spatial model from scratch; use as
  held-out evaluation.
- Recommended target = `condition` ordered Control ≈ Sham < Mild < Severe, evaluated
  with leave-one-slide-out. Treat `timepoint` results as exploratory because of the
  slide confound.
- **Split key is unresolved** until mouse identity is confirmed. Do not use `animal`.

## 8. Not on this machine

The third documented dataset, **mtDNA-DSB brain** (980k cells, oligodendrocyte-intrinsic
demyelination, 12 mice, control vs mtDSB), lives at
`/Volumes/processing2/oligo-mtDSB/data/mtDNA_DSB_5k_clustered_annotation_with_rbd_2_cytetype_brain.h5ad`
and was not reachable on 2026-09-30. Its documented layout is in `dataset-hierarchy.md`.
