# Slide text: three Xenium EAE datasets, data structure

Copy-paste source for Google Slides. One heading = one slide. Verified by inspection 2026-09-30.

---

## 1. Title

**Three Xenium EAE datasets: what the data looks like**
Mouse CNS spatial transcriptomics, single-cell resolution, one shared 5,101-gene panel
Karolinska Institutet, 30 Sep 2026

- 2.65 M cells across three datasets
- 5,101 genes, identical panel in all three
- 67 + 12 + (unknown) animals
- ~200 tissue sections

---

## 2. One panel, three tissues, three etiologies

**A. RRMAP2, spinal cord EAE.** Two models: chronic (B6/MOG) and relapse-remitting (SJL/PLP). 1.38 M cells, 67 animals, 158 sections. 19 stages, clinical score, sacrifice day 8 to 50, three cord levels.

**B. Optic nerve EAE.** Same disease, different CNS tissue. 288 k cells, 30 nerve pieces, 6 slides. Onset vs Peak; Control, Sham, Mild, Severe.

**C. mtDNA-DSB brain.** Oligodendrocyte-intrinsic mitochondrial damage, no immune trigger. 980 k cells, 12 mice (one section each), 2 x 2 condition x age.

Common to all: 10x Xenium, mouse, segmented cells with µm coordinates, raw counts in `layers['counts']`, log1p(CP10k) in `X`, AnnData `.h5ad`.

---

## 3. RRMAP2: the experiment

- Terminal, cross-sectional design: one animal = one timepoint. Dynamics are reconstructed across animals, never observed within one.
- CHRONIC arm: C57BL/6, MOG35-55. Monotonic course. 34 animals (29 EAE + 5 control), 9 stages. Lumbar-heavy sampling (34 L, 22 T, 3 C sections).
- RELAPSE-REMITTING arm: SJL/J, PLP139-151. Cyclic course. 33 animals (30 EAE + 3 control), 11 stages. Nearly complete regions (33 L, 33 T, 32 C).
- Different strain, antigen and course: any RR-vs-chronic contrast is also a strain contrast. Compare within-model slopes, not raw levels.
- 2 to 6 animals per stage. Cell counts are huge, but the statistical unit is the animal.

---

## 4. RRMAP2: sample hierarchy

```
run_date (3) / run_id (5) / slide (9)          batch layers
  sample_id = xenium_output (54)               one Xenium capture region, multiplexes 2-3 animals
    meta_sample_id (158)                        one tissue cross-section, 2.5k-20k cells  -> THE BAG
      cell (1,384,881)
sample_name (67)                               one animal, owns 1-3 sections (one per cord level) -> THE SPLIT KEY
```

Rules
- Bag = `meta_sample_id`. Stage, region, score, animal are all constant within it.
- Train/validation split = `sample_name`. An animal is entirely in one fold.
- Never group by `sample_id`: 53 of 54 capture regions contain animals with different stages.
- `kmeans_split_id` = `meta_sample_id`; `sample_label` = `sample_id`.

---

## 5. RRMAP2 timepoints: chronic arm (ordered by day)

| stage | animals | clinical score | day post-induction | meaning |
|---|---|---|---|---|
| MOG CFA | 3 | 0 | 8-9 | adjuvant control, early |
| NONSYMPTOM | 5 | 0 | 8-9 | immunised, pre-symptomatic |
| OS1 | 5 | 0.5 (0.25-0.5) | 10-14 | onset |
| PEAK1 | 6 | 3.0 (2.5-3.25) | 13-18 | peak |
| CFA | 2 | 0 | 16-17 | adjuvant control, late |
| MILD16 | 3 | 1.25 (1.25-1.5) | 27-28 | chronic mild, cohort "16" |
| SEVERE16 | 3 | 2.5 | 28-29 | chronic severe, cohort "16" |
| SEVERE30 | 2 | 2.4 (2.25-2.5) | 41 | chronic severe, cohort "30" |
| MILD30 | 5 | 1.0 | 43-50 | chronic mild, cohort "30" |

Note: the 16/30 suffix is a cohort label, not the sacrifice day. MILD30 was run in a different batch from the other late-chronic stages, so duration at matched severity is batch-confounded in this arm.

---

## 6. RRMAP2 timepoints: relapse-remitting arm (ordered by day)

| stage | animals | clinical score | day | limb |
|---|---|---|---|---|
| ONSET1 | 2 | 0.25 | 11-15 | ascending |
| ONSET2 | 2 | 1.0 | 13 | ascending |
| PEAK1 | 4 | 2.75 (2.25-3.25) | 14-18 | apex |
| REMISSION1 | 5 | 0.75 (0.5-1.0) | 20-25 | descending |
| PEAK2_MILD | 2 | 2.0 (1.75-2.25) | 31 | apex, mild |
| PEAK2 | 2 | 2.9 (2.75-3.0) | 32-33 | apex |
| MONOPHASIC | 4 | 0.6 (0.5-0.75) | 32-33 | never relapsed |
| PLP CFA | 3 | 0 | 33 | adjuvant control |
| PEAK3 | 5 | 2.5 (2.25-3.0) | 38-49 | apex |
| REMISSION2 | 3 | 1.25 (1.0-1.25) | 42-47 | descending |
| REMISSION2_LONG | 1 | 0.75 | 48 | descending, extended |

Why this arm matters
- Severity and time decouple: peaks recur at similar scores on later days. Clinical score cannot tell day 15 from day 45.
- Strain-clean and batch-clean: each stage spans several capture regions.
- The ascending limb is thin: 4 onset animals in total.
- PEAK1 exists in both arms. Always stratify by `model`.

---

## 7. RRMAP2: animal-level labels and batch

| column | values | role |
|---|---|---|
| `score_sacrifice` | 0 to 3.25 in 0.25 steps | severity, clean numeric ordinal |
| `day_of_sacrifice` | 8 to 50 days | duration, identifiable within RR only |
| `condition` | EAE 59, CONTROL 8 | binary |
| `model` | CHRONIC 34, RR 33 | strain + antigen + course |
| `sex` | M 48, F 19 | covariate; panel has Xist, no Y genes |
| `region` | L / T / C | within-animal spatial axis, constant per section |

Batch (animals per run): 2025-01-10 = 18 chronic, 0 RR. 2025-06-05 = 0 chronic, 24 RR. 2026-05-06 = 16 chronic, 9 RR.
The 2026 run is the first to put both models on the same slides and partly breaks the model x batch confound. Two instruments, 9 slides, 3 segmentation modes recorded per cell.

---

## 8. RRMAP2: cell types, a four-level curated hierarchy

| column | levels | granularity |
|---|---|---|
| `Anno_L1_curated` | 19 | lineage (use this) |
| `Anno_L2` | 42 | subtype: MOL / DAO / NFOL, Microglia / MDM / CAM, homeostatic / reactive astrocytes, CD4 / CD8 T, Exc / Inh neurons, EAE-associated fibroblasts |
| `Anno_L3` | 137 | state, e.g. DAA_Immuno_MHC, CAM_Remodelling |
| `Anno_L4` | 174 | finest, some slide-specific artefact labels |

Composition (thousand cells): Oligodendrocyte 281, Myeloid 217, Neuron 217, Fibroblast 143, Astrocyte 142, Endothelial 112, DC 61, Schwann 59, OPC 46, T cell 45, VSMC 29, B cell 13, other 20.
Soft tree: 34/42 L2 labels nest in one L1, 124/137 L3 labels nest in one L2. Legacy `leiden_*` clusters remain; `leiden_1` was dropped in this version.

---

## 9. RRMAP2: spatial niches, three orthogonal descriptors per cell

**Where (anatomy):** `Global_anatomical_region` (10): WM 423k, GM 290k, DorsalHorn 198k, WM_Meningeal 142k, Meninges 142k, DRG 69k, VentralHorn 47k, Vasculature 25k, Central canal 14k. `Physiological_niche` (12) is finer. `region` = cord level.

**What neighbourhood (niche):** `Global_niche` (27) named niches: Lesion_myeloid I-III, Lesion_Foamy, Lesion_Lipid, Lesion_DAA, Lesion_DAO I-II, Lesion_Tcell, Lesion_Bcell, Disease_GM I-III, Dorsal/Ventral_Rim_OL, DRG_reactive, plus anatomical niches. `CellCharter_5` to `CellCharter_100`: 15 unsupervised resolutions.

**What state (disease):** `Curated_niche_state` (7): Physiological 467k, Lesion_active 362k, Lesion_progressive 305k, Lesion_mix 210k, Lesion_reactive 19k, Meninges 13k, Vasculature 9k.
Lesion_active is ~0 % of cells in controls, 30-45 % at onset, 70-76 % at RR peaks, back to 2-11 % in remission. Lesion_progressive climbs through remissions and chronic stages.

---

## 10. RRMAP2: what is in the file, and two spatial traps

| slot | content |
|---|---|
| `X` | log1p(CP10k), CSR float32 |
| `layers['counts']` | raw counts, CSR. Use for any model that normalises itself |
| `obsm['spatial']` | x, y in µm |
| `obsm['X_scVI']` | 10-d batch-corrected expression embedding |
| `obsm['X_cellcharter']` | 40-d neighbourhood-aggregated scVI |
| `obsm['X_pca']`, `X_umap` | 50 PCs (67 % variance), 2-d UMAP |
| `obsp['spatial_connectivities']` | 6-NN within 77.5 µm, median NN distance 14 µm |
| `obsp['connectivities']` | expression kNN, k = 15 |

Trap 1: coordinates are per capture region. Every `sample_id` has its own ~10 x 4.5 mm frame and all 54 bounding boxes overlap. Condition on section before computing any distance.
Trap 2: 14.6 % of edges in the stored spatial graph connect cells from different sections. Rebuild it block-diagonally per `meta_sample_id`.

---

## 11. Optic nerve EAE: 30 nerve pieces on 6 slides

| condition | Onset | Peak | total |
|---|---|---|---|
| Control | 4 | 4 | 8 |
| Sham | 4 | 4 | 8 |
| Mild | 5 | 4 | 9 |
| Severe | 3 | 2 | 5 |
| total | 16 | 14 | 30 |

- Unit = `karospace_polygon_labels`, format `<position>_<CMx>_<condition>_<timepoint>`. 2.2k-28k cells each, longitudinal sections ~4 x 1.5 mm. Labels are clean at polygon level.
- Same 5,101-gene panel. No cell-type annotation yet, only leiden and CellCharter clusters. Spatial graph is clean.
- Trap: `animal` (CM1-CM9) is not an animal. It maps 1:1 onto slide and CM1 holds Control, Mild and Severe polygons at the same timepoint. Mouse identity is not recorded, so the split key is undefined.
- Confound: three slides are Onset-only, two Peak-only, one mixed. Condition is spread across all slides and is the cleaner target (Control ≈ Sham < Mild < Severe), evaluated leave-one-slide-out.

---

## 12. mtDNA-DSB brain: a balanced 2 x 2, twelve mice

| mice | age | condition | genotype | sex | slide |
|---|---|---|---|---|---|
| RB4282, RB4350, RB4498 | 60 | control | PlptTA | M M F | 0060539 |
| RB4401, RB4403, RB4405 | 60 | mtDSB | PlptTA:mtPst1 | F M M | 0060539 |
| RB4620, RB4658, RB4676 | 21 | control | mtPst1 / PlptTA | M F F | 0060541 |
| RB4627, RB4630, RB4653 | 21 | mtDSB | PlptTA:mtPst1 | M M F | 0060541 |

- `sample_id` is mouse, section, bag and split key at once. One coronal section per mouse, 70k-135k cells.
- Oligodendrocyte-restricted mitochondrial DNA double-strand breaks (Plp-tTA driving mito-PstI). No clinical score, no immune trigger.
- Annotations: `cell_class_updated` (13), `cell_type` (44), CyteType ontology terms; includes DA-oligodendrocytes and immature OL states, almost no T/B/DC. `RBD_compartment_simplified` (12 brain regions). Novae spatial domains (15-50) and a 64-d Novae embedding.
- Traps: age = slide (all age-60 on one slide, all age-21 on the other). n = 12 bags with a binary label. For more units, bag = mouse x brain region (~144), split still by mouse.

---

## 13. Side by side

| | A. RRMAP2 spinal cord | B. optic nerve | C. mtDNA-DSB brain |
|---|---|---|---|
| cells | 1.38 M | 288 k | 980 k |
| etiology | autoimmune EAE, 2 models | autoimmune EAE | oligodendrocyte-intrinsic, non-immune |
| independence unit | animal, `sample_name` (67) | unknown, mouse not recorded | mouse, `sample_id` (12) |
| bag / section | `meta_sample_id` (158) | polygon (30) | `sample_id` (12) or x region |
| time axis | 19 stages, score 0-3.25, day 8-50 | Onset / Peak (≈ slide) | age 21 / 60 (= slide) |
| severity axis | clinical score, continuous | Control, Sham, Mild, Severe | none, binary condition |
| spatial axis | L / T / C within animal | none | 12 brain regions |
| cell types | 19 / 42 / 137 / 174 curated | none | 13 / 44 + ontology |
| niches | CellCharter 5-100, 27 named, 7 disease states | CellCharter 6-30 | Novae 15-50, RBD anatomy |
| spatial graph | 14.6 % cross-section edges, rebuild | clean | clean |
| role | train + primary science | cross-tissue test | cross-etiology test |

---

## 14. Five rules every model must respect

1. Split by animal, never by section or slide. Sections of one animal share every label; slides multiplex animals. Section-level splits silently inflate every metric.
2. The label lives at the animal level. 1.38 M cells but 2-6 animals per stage. Use bag- or animal-level statistics, bootstrap CIs, permutation nulls.
3. Normalise from `layers['counts']`, not from `X`, unless the model expects log1p(CP10k) as given.
4. Condition every spatial operation on the section. Coordinate frames overlap across sections in all three datasets. Rebuild the RRMAP2 spatial graph.
5. Stratify by model and by batch. Chronic vs RR is also B6 vs SJL. Chronic duration and mtDNA-DSB age alias slide. Optic-nerve timepoint aliases slide.

Cross-sectional design throughout: one animal, one timepoint. Anything called a trajectory is a reconstruction across animals. The honest claim is "decodable from a snapshot on held-out animals".

---

## 15. Open questions and file locations

To confirm with the wet lab
- Optic nerve: which mouse is each polygon, and can two polygons share a mouse?
- mtDNA-DSB: units and meaning of `age` 21 / 60.
- RRMAP2 chronic: exact meaning of the 16 / 30 cohort suffix.
- Are the human MS Xenium objects (Baló, lesional vs non-lesional) in scope?

Files, 30 Sep 2026
- A: `RRMAP2_xenium_all_samples.cellcharter.companion.ready.with_metadata.rerun.with_AnnoL1Curated_with_Region_Anno2to4Updated.h5ad`, 4.3 GB, laptop
- B: `optic_nerve_merged.scanpy.companion.ready_with_polygons.h5ad`, 2.6 GB, laptop and remote
- C: `mtDNA_DSB_5k_clustered_annotation_with_rbd_2_cytetype_brain_novae4.companion.ready.h5ad`, 10 GB, remote `/Volumes/moldiassd/oligo-mtDSB/data/`

Full write-ups: `docs/rrmap2-data-structure.md`, `docs/optic-nerve-data-structure.md`, `docs/mtdna-dsb-data-structure.md`.
