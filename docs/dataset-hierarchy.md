# Dataset hierarchy — real Xenium `.h5ad` files

> **2026-09-30:** superseded by the per-dataset inspection docs [`rrmap2-data-structure.md`](rrmap2-data-structure.md), [`optic-nerve-data-structure.md`](optic-nerve-data-structure.md), [`mtdna-dsb-data-structure.md`](mtdna-dsb-data-structure.md). Known corrections: RRMAP2 now has 19 stages and `sample_id` is a Xenium capture region (54) rather than a physical slide (9); the optic-nerve `animal` column is a slide-group id, not a mouse.

> The `obs` grouping keys that map each real dataset onto the Stage-1 pipeline
> (bag = section, split = animal). These were reverse-engineered by inspection on
> 2026-06-11 — several are **non-obvious and easy to get wrong**, so they live here.

## Common to all three files

- Full **5101-gene** Xenium panel (mouse).
- `adata.X` is **already log-normalized**; raw counts live in **`layers['counts']`**
  (stored sparse CSR). → configs must set `data.layer: counts`, because the pipeline
  re-normalizes from raw counts itself.
- Per-cell spatial coords are in `obsm['spatial']` (and `obs.x_centroid/y_centroid`).

The pipeline contract that makes the keys below matter:
- **bag / section unit** (`obs.section_id`) must have **one constant label** — every
  cell in a bag shares the section/animal-level label.
- **animal key** (`obs.animal_id`) is the leakage-free split unit — an animal is
  entirely in train OR val. Getting this wrong (e.g. splitting by section when
  sections share an animal) silently inflates every metric.

---

## 1. RRMAP2 — spinal cord EAE (Chronic + Relapse-Remitting)

`/Volumes/moldiassd/RRMAP2_xenium_adata/kmeans_separated/RRMAP2_xenium_all_samples.cellcharter.companion.ready.with_metadata.rerun.h5ad`

1.38M cells. **The hierarchy here is the main trap.** Three sample-ish columns exist
and only the right two are usable:

| Column | n unique | What it actually is | Use as |
|---|---|---|---|
| `meta_sample_id` (≡ `kmeans_split_id`) | 158 | One contiguous tissue piece. `stage`, `score_sacrifice`, `condition`, `region` are all **constant** within it. | **`section_id`** (the bag) |
| `sample_name` | 67 | One animal / biological replicate. Constant disease label; spans multiple regions and 1–3 `meta_sample_id` pieces. | **`animal_id`** (the split key) |
| `sample_id` (≡ `sample_label`) | 54 | A **physical slide that multiplexes several animals**. **52 of 54 contain multiple `stage` values.** | ❌ do **not** use as section or animal |

Verification that caught the trap: grouping by `sample_id` and taking the first cell's
label mislabels ~52/54 bags, because each slide carries multiple animals. Grouping by
`meta_sample_id` gives 0 mixed labels for stage/score/condition/region.

- Bags (`meta_sample_id`) are **2.5k–20k cells** each → none dropped by
  `min_cells_per_bag`. 158 bags / 67 animals.
- With `val_fraction: 0.25` the split is ~119 train / ~39 val bags.

### Label columns
- `stage` — 17 levels, mixes model + severity. Current config orders them by **median
  clinical `score_sacrifice`** (the agreed unbinned "raw score" axis). The order ties
  the relapse peaks (PEAK1/2/3 ≈ score 2.5–3) and interleaves Chronic + RR; swap for a
  biologically-curated order if preferred. Order used:
  `CFA, PLP CFA, MOG CFA, NONSYMPTOM, ONSET1, OS1, MONOPHASIC, REMISSION1, MILD30,
  ONSET2, REMISSION2, MILD16, SEVERE30, SEVERE16, PEAK2, PEAK3, PEAK1`.
- `score_sacrifice` — clinical EAE score 0–3.25 (13 distinct values). Clean numeric
  ordinal; a strong alternative target.
- `condition` — EAE vs CONTROL (binary).
- Covariates: `region` (L/T/C, constant within a bag), `model` (CHRONIC /
  RELAPSE REMITTING), `day_of_sacrifice`, `sex`.

Config: `configs/rrmap2_stage.yaml`.

---

## 2. mtDNA-DSB — brain (cross-etiology model)

`/Volumes/processing2/oligo-mtDSB/data/mtDNA_DSB_5k_clustered_annotation_with_rbd_2_cytetype_brain.h5ad`

980k cells. Simpler layout.

| Column | n unique | Use as |
|---|---|---|
| `sample_id` (RB####) | 12 | both `animal_id` and `section_id` (one section per animal) |

- Target = `condition` (**control vs mtDSB**) — binary, **not** an ordinal timecourse,
  so CORAL collapses to a single threshold. Alternatives: `genotype` (PlptTA,
  PlptTA:mtPst1, mtPst1), `age` (21 vs 60).
- Rich annotations available: `cell_class` (12), `cell_type` (44),
  `RBD_compartment_simplified` (brain regions, 12).
- This is the cross-etiology transfer target (primary oligo damage vs autoimmune EAE).

---

## 3. Optic nerve (cross-tissue model)

`/Volumes/moldiassd/optic_nerve_merged.scanpy.companion.ready_with_polygons.h5ad`

288k cells. Small but has its own trap: **animals are multiplexed on shared slides as
polygons.**

| Column | n unique | What it is | Use as |
|---|---|---|---|
| `animal` | 6 (CM1–CM9) | the biological replicate | **`animal_id`** AND the bag key |
| `sample_id` (≡ `sample_batch`) | 6 | a physical slide carrying several animals' polygons | ❌ not a single-label section |

- Because each `sample_id` holds multiple animals (see `karospace_polygon_labels`),
  the bag must key on **`animal`**, not `sample_id`, or one bag mixes labels.
- Target = `condition` (Mild/Severe/Sham/Control) or `timepoint` (Onset/Peak).
- Only 6 animals → tight for an animal-level split; best used as a **generalization
  test** (spinal cord → optic nerve, same panel/disease), not a primary training run.
