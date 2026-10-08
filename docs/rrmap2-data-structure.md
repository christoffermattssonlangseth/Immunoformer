# RRMAP2 — data structure for the ImmunoTransformer project

*Written 2026-09-30 by direct inspection of the current canonical file. Intended as the
onboarding description for the AI/ML consultancy. Supersedes the RRMAP2 section of
`dataset-hierarchy.md` where they differ (this file has 19 stages and a curated
cell-type / niche hierarchy that the older doc predates).*

**Canonical file (4.3 GB, AnnData `.h5ad`):**
`~/Downloads/RRMAP2_xenium_all_samples.cellcharter.companion.ready.with_metadata.rerun.with_AnnoL1Curated_with_Region_Anno2to4Updated.h5ad`

## 1. What the experiment is

Mouse spinal cord, **experimental autoimmune encephalomyelitis (EAE)**, the standard animal
model of multiple sclerosis. Two disease models were run in parallel:

| model | strain / antigen | course | animals |
|---|---|---|---|
| `CHRONIC` | C57BL/6, MOG35-55 | monotonic: onset → peak → chronic plateau | 34 (29 EAE + 5 control) |
| `RELAPSE REMITTING` | SJL/J, PLP139-151 | cyclic: onset → peak → remission → relapse … | 33 (30 EAE + 3 control) |

Every animal is a **terminal, cross-sectional** sample: one animal = one timepoint. There
are no within-animal time series. Disease dynamics are therefore *reconstructed* across
animals, not observed.

Measurement: **10x Xenium** in situ spatial transcriptomics, **5,101-gene mouse panel**,
single-cell resolution with segmented cell polygons and µm coordinates.

## 2. Size

| unit | n | notes |
|---|---|---|
| cells | 1,384,881 | rows of the matrix |
| genes | 5,101 | columns; `var` index is gene symbol, Ensembl IDs in `var.gene_ids` |
| animals (`sample_name`) | 67 | biological replicate, the unit of independence |
| tissue sections (`meta_sample_id`) | 158 | one contiguous cord cross-section; 2.5k–20k cells each (median 8k) |
| Xenium capture regions (`sample_id`) | 54 | one imaged region on a slide; carries 1–3 animals |
| physical slides (`cassette_or_slide_id`) | 9 | |
| runs (`run_id` / `run_date`) | 5 / 3 | batch |

Cells per animal: 4.6k–47k (median 20k). Nonzero entries in the matrix: 821M. Median
993 transcripts and 567 detected genes per cell.

## 3. Sample hierarchy (the part that is easy to get wrong)

```
run_date (3)  ─┐
run_id (5)     ├─ batch
slide (9)     ─┘
   └── sample_id / xenium_output (54)      Xenium capture region; MULTIPLEXES 1–3 animals
          └── meta_sample_id (158)         one tissue section  ← BAG / SECTION UNIT
                 └── cell (1.38M)
sample_name (67)                           one animal          ← SPLIT / INDEPENDENCE UNIT
   ├── 1–3 meta_sample_id (one per spinal region: L, T, C)
   └── constant: stage, score_sacrifice, day_of_sacrifice, condition, model, sex, run_date
```

Verified invariants:
- Within a `meta_sample_id`: `stage`, `region`, `score_sacrifice`, `sample_name`,
  `sample_id` are all constant.
- Within a `sample_name`: all disease labels constant; spans up to 3 regions and 3
  sections; never spans runs.
- `sample_id` is **not** a valid label unit: 53 of 54 capture regions contain 2–3 animals
  with different stages. Any grouping by `sample_id` mixes labels.
- `kmeans_split_id` ≡ `meta_sample_id`; `sample_label` ≡ `sample_id` ≡ `xenium_output`.

Regions per animal: 32 animals have all three (L/T/C), 26 have two, 9 have one. The RR
arm is nearly complete (33/33/32 animals with L/T/C); the chronic arm is lumbar-heavy
(34 L, 22 T, only 3 C).

## 4. Disease labels (animal level)

All of these live in `obs` but are constant per animal. Use animal-level tables.

### 4.1 `stage` — 19 levels, the experimental design cell

**Chronic arm (B6/MOG), ordered by day post-induction**

| stage | n animals | clinical score (median, range) | day of sacrifice | meaning |
|---|---|---|---|---|
| MOG CFA | 3 | 0 | 8–9 | adjuvant-only control, early |
| NONSYMPTOM | 5 | 0 | 8–9 | immunised, pre-symptomatic |
| OS1 | 5 | 0.5 (0.25–0.5) | 10–14 | onset |
| PEAK1 | 6 | 3.0 (2.5–3.25) | 13–18 | peak |
| CFA | 2 | 0 | 16–17 | adjuvant-only control, late |
| MILD16 | 3 | 1.25 (1.25–1.5) | 27–28 | chronic, mild, "16-day" cohort |
| SEVERE16 | 3 | 2.5 | 28–29 | chronic, severe, "16-day" cohort |
| SEVERE30 | 2 | 2.4 (2.25–2.5) | 41 | chronic, severe, "30-day" cohort |
| MILD30 | 5 | 1.0 | 43–50 | chronic, mild, "30-day" cohort |

The 16/30 suffix indexes the chronic cohort, not the sacrifice day. The two chronic
cohorts were run in different batches (MILD30 entirely in run 2025-01-10; MILD16 /
SEVERE16 / SEVERE30 entirely in run 2026-05-06), so "duration at matched severity" in the
chronic arm is confounded with batch.

**Relapse-remitting arm (SJL/PLP), ordered by day post-induction**

| stage | n animals | clinical score (median, range) | day of sacrifice | meaning |
|---|---|---|---|---|
| ONSET1 | 2 | 0.25 | 11–15 | first attack, ascending |
| ONSET2 | 2 | 1.0 | 13 | first attack, ascending (later) |
| PEAK1 | 4 | 2.75 (2.25–3.25) | 14–18 | first peak |
| REMISSION1 | 5 | 0.75 (0.5–1.0) | 20–25 | first remission |
| PEAK2_MILD | 2 | 2.0 (1.75–2.25) | 31 | second attack, mild |
| PEAK2 | 2 | 2.9 (2.75–3.0) | 32–33 | second peak |
| MONOPHASIC | 4 | 0.6 (0.5–0.75) | 32–33 | never relapsed after first attack |
| PLP CFA | 3 | 0 | 33 | adjuvant-only control |
| PEAK3 | 5 | 2.5 (2.25–3.0) | 38–49 | third peak |
| REMISSION2 | 3 | 1.25 (1.0–1.25) | 42–47 | second remission |
| REMISSION2_LONG | 1 | 0.75 | 48 | extended second remission |

Note `PEAK1` is the one stage name shared by both arms (6 chronic + 4 RR animals). Always
stratify by `model`.

### 4.2 Continuous / secondary labels

| column | type | values | use |
|---|---|---|---|
| `score_sacrifice` | float | 0–3.25 in 0.25 steps (13 values) | clinical EAE score at sacrifice; the **severity** axis. Clean numeric ordinal. |
| `day_of_sacrifice` | float | 8–50 days post-induction | **duration** axis. Within RR it is decoupled from severity (relapses recur at similar score, later days). |
| `condition` | cat | EAE / CONTROL | 59 / 8 animals |
| `model` | cat | CHRONIC / RELAPSE REMITTING | strain + antigen + course |
| `sex` | cat | M / F | 48 / 19 animals; the panel has `Xist` but no Y genes |
| `region` | cat | L / T / C | lumbar / thoracic / cervical, constant per section, within-animal axis |

### 4.3 Batch structure

| `run_date` | chronic animals | RR animals |
|---|---|---|
| 2025-01-10 | 18 | 0 |
| 2025-06-05 | 0 | 24 |
| 2026-05-06 | 16 | 9 |

The 2026 run is the first to mix both models on the same slides, which partially breaks
the model×batch confound the earlier analyses had to work around. Within the RR arm each
stage is spread across multiple capture regions, so stage is not batch-aliased. Two
instruments (`instrument_or_flowcell`, 49 vs 18 animals). Three segmentation methods
appear in every animal (Xenium multimodal segmentation: boundary stain, interior stain,
nucleus expansion), recorded per cell in `segmentation_method`.

## 5. Cell-level annotations

### 5.1 Cell type — a four-level hierarchy

| column | levels | content |
|---|---|---|
| `Anno_L1` | 13 | coarse lineage (original) |
| **`Anno_L1_curated`** | 19 | coarse lineage, curated. Splits DC and NK/DC out of Myeloid, adds Neutrophil, Muscle, Epithelial, and flags Doublet / T_B_doublet. **Use this one.** |
| `Anno_L2` | 42 | subtype (e.g. MOL / DAO / NFOL; Microglia / MDM / CAM; Homo_AST / Reactive_AST; CD4+ / CD8+ T; Exc / Inh neurons; EAE-associated / meningeal fibroblasts) |
| `Anno_L3` | 137 | fine state (e.g. DAA_Immuno_MHC, CAM_Remodelling, AST_Protoplasmic) |
| `Anno_L4` | 174 | finest state, some slide-specific artefact labels |
| `Anno_L2/3/4_clusters` | 9 / 65 / 403 | the numeric clusters the labels were derived from; `Anno_L2_clusters` is mostly empty string |

The hierarchy is *mostly* nested: 34 of 42 L2 labels map to a single L1, 124 of 137 L3
labels map to a single L2. Treat it as a soft tree, not a strict one.

`Anno_L1_curated` composition (cells): Oligodendrocyte 281k, Myeloid 217k, Neuron 217k,
Fibroblast 143k, Astrocyte 142k, Endothelial 112k, DC 61k, Schwann 59k, OPC 46k,
T cell 45k, VSMC 29k, B cell 13k, Ependymal 9k, NK/DC 6k, Doublet 3k, T_B_doublet 1k,
Neutrophil 359, Muscle 180, Epithelial 124.

Legacy unsupervised clusterings also present: `leiden_1.5` … `leiden_4.0` (36–80
clusters). `leiden_1` (25 clusters, used throughout the existing atlas scripts) is
**no longer in `obs`** in this version, only its parameters in `uns`.

### 5.2 Spatial niches — three families

**(a) CellCharter neighbourhood clusters (unsupervised).**
`CellCharter_5` … `CellCharter_100` (15 resolutions: 5, 10, 15, …, 50, 60, 70, 80, 90,
100 clusters). Computed on `obsm['X_cellcharter']` (40-d neighbourhood-aggregated scVI
embedding). Integer labels, no names.

**(b) Curated disease-niche state (the most useful for disease modelling).**

| column | levels | content |
|---|---|---|
| **`Curated_niche_state`** | 7 | Physiological 467k · Lesion_active 362k · Lesion_progressive 305k · Lesion_mix 210k · Lesion_reactive 19k · Meninges 13k · Vasculature 9k |
| `Niche_state_unbiased` ≡ `Auto_niche_state` | 5 | same idea before curation; has an `Ambiguous` class (246k) |
| **`Global_niche`** | 27 | named niches: Lesion_myeloid I/II/III, Lesion_Foamy, Lesion_Lipid, Lesion_DAA, Lesion_DAO I/II, Lesion_Tcell, Lesion_Bcell, Disease_GM I/II/III, Dorsal/Ventral_Rim_OL, Dorsal_Rim_DAA, DRG_reactive, plus anatomical GM/WM/Meninges/DRG/Vasculature/Central_canal |
| `Global_niche_group` | 22 | `Global_niche` with roman-numeral variants merged |
| `Combined_niche_group` | 25 | `Global_niche_group` with WM split into dorsal/lateral/ventral funiculus |
| `Auto_niche_id` / `unbiased_group` / `Auto_region` | 80 / 20 / 22 | intermediate unsupervised objects behind the curated columns |

`Curated_niche_state` tracks disease stage strongly: Lesion_active is ~0 % of cells in
controls, 30–45 % at onset, 70–76 % at RR PEAK1/PEAK2, and falls back to 2–11 % in
remissions; Lesion_progressive rises through remissions and chronic stages (36–45 %).
Lesion_active is myeloid/DC-rich; Lesion_progressive is fibroblast/myeloid/oligo;
Lesion_reactive is 73 % neurons (reactive grey matter).

**(c) Anatomy.**

| column | levels | content |
|---|---|---|
| **`Global_anatomical_region`** | 10 | WM 423k · GM 290k · DorsalHorn 198k · WM_Meningeal 142k · Meninges 142k · DRG 69k · VentralHorn 47k · Unassigned 34k · Vasculature 25k · Central_canal 14k |
| `Physiological_niche` | 12 | finer anatomy: Intermediate grey, DorsalHorn_sensory, VentralHorn_motor, Dorsal/Lateral/Ventral funiculus, Meninges, DRG, Vasculature, Central canal, Mixed, WM (unclassified) |

So each cell carries three orthogonal spatial descriptors: **where anatomically**
(`Global_anatomical_region`), **what neighbourhood** (`Global_niche`), and **what disease
state that neighbourhood is in** (`Curated_niche_state`), plus the cord level (`region`).

### 5.3 Per-cell QC

`transcript_counts`, `total_counts`, `n_counts`, `n_genes`, `cell_area` (µm², median 71),
`nucleus_area` (all NaN), `nucleus_count`, negative-control counts
(`control_probe_counts`, `control_codeword_counts`, `unassigned_codeword_counts`,
`deprecated_codeword_counts`, `genomic_control_counts`), `polygon_remove*` (manual
exclusion flags from Karospace polygons, 19 polygons drawn), `cell_id` (Xenium id,
unique within a capture region, not globally: 1,384,652 unique for 1,384,881 cells; use
`obs_names`/`obs_id`).

## 6. Matrices and embeddings

| slot | shape / type | content |
|---|---|---|
| `X` | 1.38M × 5101, CSR float32 | **log1p(CP10k)**: normalised to 10,000 counts per cell, then log1p |
| `layers['counts']` | same, CSR int | **raw counts**. Use this for any model that does its own normalisation |
| `var` | 5101 rows | `gene_symbol` (index), `gene_ids` (Ensembl), `highly_variable` (500 HVGs, seurat_v3), means/variances |
| `obsm['spatial']` | 1.38M × 2 float | x/y in **µm**, identical to `obs.x_centroid/y_centroid` |
| `obsm['X_scVI']` | × 10 | scVI latent (batch-corrected expression embedding) |
| `obsm['X_cellcharter']` | × 40 | CellCharter neighbourhood aggregate of the scVI embedding (input to the CellCharter_* clusterings) |
| `obsm['X_pca']` | × 50 | PCA on `X` (67 % variance in 50 PCs); loadings in `varm['PCs']` |
| `obsm['X_umap']` | × 2 | UMAP on the 30-PC kNN graph |
| `obsm['X_karo_comp']` | × 10 | Karospace neighbourhood composition over `CellCharter_10` |
| `obsp['connectivities']`, `['distances']` | 1.38M² sparse | expression kNN graph (k=15, 30 PCs) |
| `obsp['spatial_connectivities']`, `['spatial_distances']` | 1.38M² sparse | spatial graph, squidpy generic, 6 neighbours, radius 77.5 µm. Median neighbour distance 14 µm |
| `uns` | dict | clustering params, colours, `rank_genes_groups` (Wilcoxon on `CellCharter_80`), Karospace companion analytics (per-niche DE, neighbour stats, spatially variable genes), Karospace polygon annotations |

### Two traps in the spatial slots

1. **Coordinates are per capture region, not global.** Every `sample_id` has its own
   frame (roughly 0–10,000 × 0–4,500 µm) and all 54 bounding boxes overlap. Any spatial
   model must condition on `sample_id` (or `meta_sample_id`) before using coordinates;
   never compute distances across sections. A typical section is ~2.2 × 2.3 mm.
2. **The stored `spatial_connectivities` graph has 14.6 % cross-section edges**, because
   it was built on the overlapping coordinate frames. Rebuild it block-diagonally per
   `meta_sample_id` (the `lesion_signaling.py` script already does this) before using it.

## 7. How the existing analyses map onto this structure

- **Bag / multiple-instance unit** = `meta_sample_id` (158 sections). One label per bag.
- **Train/validation split key** = `sample_name` (67 animals). An animal is entirely in
  train or in validation. Splitting on sections or on `sample_id` leaks.
- **Severity target** = `score_sacrifice` (continuous ordinal) or `stage` (ordinal
  within one model arm; do not order across arms).
- **Duration target** = `day_of_sacrifice`, valid within RR only (batch-confounded in
  chronic).
- **Spatial axis** = `region` (L→T→C), a within-animal contrast.
- **Interpretability anchors** = `Anno_L1_curated` / `Anno_L2` for cell identity,
  `Curated_niche_state` / `Global_niche` for lesion architecture.
- Realistic per-stage n is 2–6 animals. Everything downstream is small-n at the
  animal level, however large the cell count.

## 8. Companion datasets (same 5,101-gene panel; not in this file)

Documented in `dataset-hierarchy.md`; files currently on the external volumes only.

| dataset | cells | tissue | design | grouping keys |
|---|---|---|---|---|
| Optic nerve EAE | 288k | optic nerve | Onset / Peak; Control / Sham / Mild / Severe; 6 animals multiplexed as polygons | bag = `karospace_polygon_labels`, animal = `animal` |
| mtDNA-DSB | 980k | brain | oligodendrocyte-intrinsic (non-immune) demyelination; control vs mtDSB; 12 mice, one section each | `sample_id` is both animal and section |

## 9. Reading it

```python
import anndata as ad, h5py
# metadata only (fast, ~seconds):
with h5py.File(path) as f: obs_keys = list(f["obs"].keys())
# full object needs ~15–20 GB RAM for X + counts; use backed mode or ad.read_h5ad(path, backed="r")
a = ad.read_h5ad(path, backed="r")
```

For anything spatial, subset by `meta_sample_id` first.
