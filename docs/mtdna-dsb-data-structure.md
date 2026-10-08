# mtDNA-DSB brain — data structure

*Written 2026-09-30 by direct inspection on the remote Mac
(`ssh christoffer@ki-cljjykhfx7.taileaaab1.ts.net`). Companion to
`rrmap2-data-structure.md` and `optic-nerve-data-structure.md`.*

**Canonical file (10 GB), remote only:**
`/Volumes/moldiassd/oligo-mtDSB/data/mtDNA_DSB_5k_clustered_annotation_with_rbd_2_cytetype_brain_novae4.companion.ready.h5ad`
(newest, 2026-06-23). Earlier generations live in `/Volumes/processing2/oligo-mtDSB/data/`
(`..._cytetype_brain.h5ad` Dec 2025 is the one the transfer configs reference; `novae`,
`novae2`, `novae4` Jan–Feb 2026 add Novae spatial domains). Same cells and annotations
throughout; the June file additionally carries the Karospace companion analytics.

## 1. What the experiment is

Mouse **whole brain**, coronal sections, a **non-immune, oligodendrocyte-intrinsic**
demyelination model: `PlptTA:mtPst1` double-transgenic mice express a mitochondrially
targeted restriction enzyme (PstI) in oligodendrocytes under the Plp promoter, inducing
mitochondrial DNA double-strand breaks (mtDSB). Controls are single-transgenic
(`PlptTA` or `mtPst1`). Two time/age groups, `21` and `60` (units not stored in the file,
presumably days after induction; **confirm with the lab**).

This is the **cross-etiology contrast** to EAE: primary oligodendrocyte damage without an
autoimmune trigger. Same 10x Xenium 5,101-gene mouse panel as RRMAP2 and optic nerve.

## 2. Size

| unit | n | notes |
|---|---|---|
| cells | 980,474 | |
| genes | 5,101 | identical panel; `var` index is gene symbol |
| mice = sections (`sample_id`, RB####) | 12 | one coronal section per mouse; 70k–135k cells each |
| slides | 2 | `0060539` (all six age-60 mice), `0060541` (all six age-21 mice) |
| runs (`run`) | 12 | one Xenium output folder per section, all run 2025-09-17 |

Median 815 transcripts, 514 genes per cell; median cell area 69 µm².

## 3. Design (one row per mouse)

| sample_id | age | condition | genotype | sex | slide |
|---|---|---|---|---|---|
| RB4282 | 60 | control | PlptTA | M | 0060539 |
| RB4350 | 60 | control | PlptTA | M | 0060539 |
| RB4498 | 60 | control | PlptTA | F | 0060539 |
| RB4401 | 60 | mtDSB | PlptTA:mtPst1 | F | 0060539 |
| RB4403 | 60 | mtDSB | PlptTA:mtPst1 | M | 0060539 |
| RB4405 | 60 | mtDSB | PlptTA:mtPst1 | M | 0060539 |
| RB4620 | 21 | control | mtPst1 | M | 0060541 |
| RB4658 | 21 | control | PlptTA | F | 0060541 |
| RB4676 | 21 | control | PlptTA | F | 0060541 |
| RB4627 | 21 | mtDSB | PlptTA:mtPst1 | M | 0060541 |
| RB4630 | 21 | mtDSB | PlptTA:mtPst1 | M | 0060541 |
| RB4653 | 21 | mtDSB | PlptTA:mtPst1 | F | 0060541 |

A balanced 2 × 2 (condition × age), 3 mice per cell, 6 F / 6 M overall. `age`, `condition`,
`genotype`, `sex`, `run` are all constant per `sample_id`; `sample_id` is simultaneously
the **mouse, the section, the bag and the split unit**. `novae_sid` ≡ `sample_id`.

Traps:
- **Age is perfectly confounded with slide** (all age-60 on one slide, all age-21 on the
  other). Any age effect is also a slide effect. Condition is balanced within each slide,
  so the condition contrast is clean.
- **n = 12 bags.** Condition is binary; there is no ordinal or continuous disease axis
  (no clinical score, no per-mouse severity). With 6 vs 6 the achievable statistics are
  rank tests with wide confidence intervals. To get more units, subdivide each section by
  brain region (`RBD_compartment_simplified`, 12 regions → up to 144 region-bags) while
  keeping condition as the label and mouse as the split key.
- One control (RB4620) is `mtPst1`, the other five `PlptTA`; the genotype column
  therefore has 3 levels but `condition` is the intended contrast.

## 4. Cell-level annotations

### Cell type

| column | levels | content |
|---|---|---|
| **`cell_class_updated`** | 13 | coarse: Excitatory Neurons 210k · Neurons 160k · Astrocytes 155k · Oligodendrocytes 146k · Endothelial 128k · GABAergic Neurons 50k · Medium Spiny Neurons 47k · Microglia 29k · Immature Oligodendrocytes 27k · DA-Oligodendrocytes 15k (disease-associated) · Neural Stem Cells 5k · Endocrine 5k · Myocytes 2.5k |
| `cell_class` | 15 | same with Immature Oligodendrocytes split I/II/III |
| **`cell_type`** | 44 | fine: Mature oligodendrocytes I–VIII, Telencephalon/Olfactory astrocytes, Excitatory neurons (cortex/thalamus) I–VII, Inhibitory I–III, Striatal, D1 MSN, Cholinergic, Pericytes, VLMC, Vascular endothelial, Choroid plexus, Ependymal, OPC, Neural progenitors, Microglia, Unknown (8.7k) |
| `cytetype_annotation_leiden_3` / `cytetype_cellOntologyTerm_leiden_3` | 61 / 34 | LLM-assisted (CyteType) labels + Cell Ontology IDs on the `leiden_3` clusters |
| `leiden_0.5` … `leiden_3` | 17–61 | unsupervised expression clusters |

Compared with RRMAP2 this brain object has no separate T/B/DC/fibroblast classes; immune
cells beyond microglia are essentially absent (as expected for a non-immune model), so a
model trained on EAE immune infiltrates will find few matching cells here.

### Anatomy / spatial domains

| column | levels | content |
|---|---|---|
| **`RBD_compartment_simplified`** | 12 | Cortex 259k · Thalamus 131k · Striatum 125k · Hypothalamus 110k · Fiber tracts 78k · Olfactory areas 72k · Meningeal border/glia limitans 62k · Ventricular system 45k · Hippocampus 35k · Pallidum 34k · Unknown 22k · Vasculature 8k |
| `RBD_compartment` | 34 | finer (Cortex I–VI, Thalamus I–IV, Caudoputamen, Dentate gyrus, Corticospinal tract, …) |
| `rbd_domain_0.1` … `rbd_domain_1.0` | 12–52 | the unsupervised region clusters behind RBD |
| `novae_domains_15/20/25/30/50` | 15–50 | **Novae** foundation-model spatial domains (D### ids, 102 cells NaN); `novae_leaves` (512) is the raw leaf code; `neighborhood_valid` flags cells with a complete 2-hop neighbourhood |

The RBD compartments are the anatomical frame for the "which brain region demyelinates
first" question; fiber tracts (white matter) are the expected primary lesion site.

### QC
Same Xenium QC set as RRMAP2 (`transcript_counts`, `n_genes`, `cell_area`,
`nucleus_area`, negative-control counts, three-way `segmentation_method`).

## 5. Matrices and embeddings

| slot | content |
|---|---|
| `X` | log1p(CP10k), CSR |
| `layers['counts']` | raw counts, CSR |
| `var` | 5,101 genes; `highly_variable` 728 (seurat flavour), `novae_use_gene` 692, `in_vocabulary` 4,847 (genes in the Novae vocabulary) |
| `obsm['spatial']` | µm, **per-section frames** (0–5,500 × 0–7,500 µm, all 12 overlap) |
| `obsm['novae_latent']` (64) | Novae cell embedding (spatially aware) |
| `obsm['X_pca']` (50), `X_umap` (2), `X_karo_comp` (44) | PCA / UMAP / Karospace neighbourhood composition over `cell_type` |
| `obsp['spatial_connectivities' / 'spatial_distances']` | radius 80 µm graph, ~6 neighbours; **0 cross-section edges**, usable as is. Median NN distance 16.7 µm |
| `obsp['spatial_distances_local' / '_view']` | Novae 2-hop neighbourhood graphs |
| `obsp['connectivities' / 'distances']` | expression kNN |
| `uns` | `rank_genes_groups` (on `leiden_3`), CyteType job details, Novae attrs, Karospace companion analytics (groupby `sample_id`) |

No scVI or CellCharter embedding in this object, unlike RRMAP2 and optic nerve.

## 6. Role in the project

- The **etiology control**: does a representation learned on immune-driven EAE damage
  transfer to oligodendrocyte-intrinsic damage? A *low* transfer score is the interesting
  result here (distinct trajectory), a *high* one says the readout is generic
  demyelination/glial response.
- Bag = `sample_id` (12) or `sample_id × RBD_compartment_simplified` (~144) with
  leave-one-mouse-out evaluation.
- Age effect cannot be separated from slide.

## 7. Where the three datasets live (2026-09-30)

| dataset | local laptop | remote Mac (Tailscale) |
|---|---|---|
| RRMAP2 (current, 19-stage, curated annos) | `~/Downloads/RRMAP2_...Anno2to4Updated.h5ad` | not yet copied; older `rerun.h5ad` split per section under `/Volumes/moldiassd/RRMAP2_xenium_adata/kmeans_separated/` |
| Optic nerve | `~/Downloads/optic_nerve_merged.scanpy.companion.ready_with_polygons.h5ad` | `/Volumes/moldiassd/` same file |
| mtDNA-DSB | — | `/Volumes/moldiassd/oligo-mtDSB/data/..._novae4.companion.ready.h5ad` and `/Volumes/processing2/oligo-mtDSB/data/` |

Also on the remote under `/Volumes/jamboree/RRMap/data/`: an earlier RR-EAE integration
(`RREAE_5k_raw_only_integration_processed*.h5ad`, 892k cells) and a nuclei-level RR-EAE
object. These predate RRMAP2 and are not part of the current design.
