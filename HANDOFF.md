# Handoff — RRMAP2 Stage-1 baseline (2026-06-11)

Picking this up this evening. Goal: get the Stage-1 attention-MIL + CORAL baseline
running on the **RRMAP2 spinal-cord EAE** dataset, target = `stage`.

## Where we are

- Inspected all three real `.h5ad` files and figured out their `obs` layout.
- Wrote a config for RRMAP2: **`configs/rrmap2_stage.yaml`**.
- Applied one memory fix to `train.py`.
- **Blocked:** training still segfaults (exit 139) early in the run. Root cause
  not yet confirmed — leading hypothesis is memory pressure (see below).

## The three datasets (verified mappings)

> Full reference with grouping-key tables and the traps: **`docs/dataset-hierarchy.md`**.

All are the full 5101-gene Xenium panel. `X` is already log-normalized; **raw counts
are in `layers['counts']`** in every file (both stored sparse CSR). So configs must
set `data.layer: counts`.

### 1. RRMAP2 — spinal cord EAE (the one we're working on)
`/Volumes/moldiassd/RRMAP2_xenium_adata/kmeans_separated/RRMAP2_xenium_all_samples.cellcharter.companion.ready.with_metadata.rerun.h5ad`
- 1.38M cells. **Critical, non-obvious hierarchy** (verified):
  - `meta_sample_id` (158) = one tissue piece → **the bag/section unit**. stage,
    score, condition, region are all constant within it.
  - `sample_name` (67) = one animal / biological replicate → **the animal split key**
    (constant disease label; spans multiple regions + 1–3 meta pieces).
  - `sample_id` (54) = physical slide; **MULTIPLEXES several animals** → do NOT use
    it as the section or animal key (52/54 contain multiple stages).
- Bags are 2.5k–20k cells each (none dropped by `min_cells_per_bag`).
- Target `stage` = 17 levels. `label_order` in the config is sorted by **median
  clinical `score_sacrifice`** (this was the agreed "raw score" ordering — unbinned).
  Order mixes Chronic + RR models and ties the relapse peaks; edit if a
  biologically-curated order is preferred.
- Other useful columns: `score_sacrifice` (clinical EAE score 0–3.25, clean ordinal
  alternative target), `condition` (EAE/CONTROL), `region` (L/T/C), `model`
  (CHRONIC / RELAPSE REMITTING), `day_of_sacrifice`, `sex`.

### 2. mtDNA-DSB — brain (cross-etiology, for later)
`/Volumes/processing2/oligo-mtDSB/data/mtDNA_DSB_5k_clustered_annotation_with_rbd_2_cytetype_brain.h5ad`
- 980k cells. `sample_id` (RB####, 12) = animal AND section (one section/animal).
- Target = `condition` (control vs mtDSB) — **binary**, not an ordinal timecourse.
  Also `genotype` (3), `age` (21/60). Rich annots: `cell_class`,
  `RBD_compartment_simplified` (brain regions).

### 3. Optic nerve (cross-tissue, for later)
`/Volumes/moldiassd/optic_nerve_merged.scanpy.companion.ready_with_polygons.h5ad`
- 288k cells. **animal key = `animal`** (CM1–CM9, only 6 animals). Animals are
  **multiplexed on shared slides** (`sample_id` has multiple animals as polygons), so
  the bag must be keyed on `animal`, not `sample_id`. Target = `condition`
  (Mild/Severe/Sham/Control) or `timepoint` (Onset/Peak). Small → use as a
  generalization test, not the first run.

## What's been changed in the repo

- **NEW** `configs/rrmap2_stage.yaml` — the RRMAP2 config (animal=`sample_name`,
  section=`meta_sample_id`, label=`stage`, layer=`counts`, `n_hvg: 2000`).
- **EDITED** `immunotransformer/train.py` — HVG branch now extracts the HVG mask and
  `del tmp; gc.collect()` *before* the dense densify, to avoid holding the full-panel
  copy in memory. (This was necessary but did NOT fully fix the segfault.)

## The blocker: segfault (exit code 139)

Machine has **48 GB RAM**. Full panel densifies to 28 GB; `n_hvg: 2000` → 11 GB
dense (that's why the config uses 2000 HVGs).

What we ruled out by running each step standalone (`/tmp/diag.py`, `/tmp/diag2.py` —
both ran with `PYTHONPATH=.`):
- read_h5ad, HVG select, build_bags (→ **119 train / 39 val bags**), the 11 GB
  densify, and PCA fit/transform **all succeed in isolation**.
- The model forward/backward runs fine on **both CPU and MPS** (5 steps each).

So individually everything works, but the end-to-end `train.py` run segfaults right
after `[load]` / during the HVG step. The diagnostic survived because it does
`del tmp; gc.collect()` — we added the same to train.py but it STILL crashes,
which suggests the peak is still too high (read 13 GB + `tmp` copy 13 GB ≈ 26 GB
during HVG, before any densify) and is tipping over under whatever else is resident.

### Next things to try (in order)
1. **Confirm cause with a traceback.** Relaunch with the fault handler on:
   `PYTHONFAULTHANDLER=1 python -u -m immunotransformer.train --config configs/rrmap2_stage.yaml`
   and check whether the C-level traceback points at memory vs a native lib (HDF5/
   sklearn/OMP). Was about to run `memory_pressure` / `vm_stat` and check for leftover
   python processes when we stopped — do that first (a prior segfaulted run may have
   left memory resident).
2. **Cut peak memory in the HVG step.** Don't `adata.copy()` the whole object (it
   copies the `counts` layer too). Instead compute HVG on a counts-only view, or read
   the file with `backed='r'` and only pull the genes we keep. Target: never hold two
   full-panel copies at once.
3. **Try `encoder: identity` + smaller `n_hvg` (e.g. 1000)** as a cheaper smoke test
   to confirm the training loop end-to-end, then scale back up.
4. Consider forcing `train.device: cpu` to rule out MPS (model test passed on MPS,
   but worth isolating).
5. If memory stays tight, subsample cells up front or process the densify/PCA in
   chunks (IncrementalPCA), or skip the full-matrix densify in
   `data.normalize_expression` (keep it sparse and let PCA consume sparse input).

## Useful scratch files (in /tmp, may be cleared on reboot)
- `/tmp/inspect_h5ad.py` — generic obs/var/X inspector (`python /tmp/inspect_h5ad.py <path>`).
- `/tmp/diag.py` — step-by-step pipeline reproduction with faulthandler.
- `/tmp/diag2.py` — model train-step test on CPU + MPS.
- Run log: `runs_rrmap2_stage.log` in the repo root.

## Quick resume command
```bash
cd /Users/christoffer/work/karolinska/development/Immunoformer
PYTHONFAULTHANDLER=1 python -u -m immunotransformer.train --config configs/rrmap2_stage.yaml 2>&1 | tee runs_rrmap2_stage.log
```
