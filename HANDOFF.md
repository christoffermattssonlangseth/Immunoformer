# Handoff — RRMAP2 Stage-1 baseline (2026-06-11)

Picking this up this evening. Goal: get the Stage-1 attention-MIL + CORAL baseline
running on the **RRMAP2 spinal-cord EAE** dataset, target = `stage`.

## Update (2026-06-12) — integrated the Codon transfer work

Merged the `immunoformer-work` improvements on branch `integrate-codon-transfer`:

- **HVG memory fix applied** — this is next-step #2 below. `train.py` now computes
  HVGs on a 50k-cell counts-only subsample (`_select_hvg`) instead of
  `adata.copy()`, and HVG-subsets via a view (no second full-panel copy). New
  `data.hvg_subsample` config field (default 50_000). **Not yet re-run on the full
  1.38M-cell RRMAP2 object** — still needs the `PYTHONFAULTHANDLER` confirmation run
  in "Quick resume command" to verify the segfault is actually gone.
- **Transfer pipeline added** — `train.py` now serialises `encoder.pkl`,
  `hvg_genes.npy`, `run_manifest.json` (incl. model arch). New modules
  `evaluate.py` (apply a frozen model to a new dataset, gene-panel aligned,
  label-rank remapped) and `stats.py` (bag-level bootstrap CI / permutation /
  Fisher z). New scripts `inspect_labels.py`, `transfer_experiment.py`.
- **Configs** — kept the verified `rrmap2_stage.yaml` (only added `hvg_subsample`).
  Added `transfer_experiment.yaml` + per-dataset transfer templates, populated
  with the verified obs keys below (optic nerve animal=`animal`; mtDNA-DSB
  key=`sample_id`, binary `condition`). Their `label_order` strings still need
  confirming via `inspect_labels.py`.
- Fixed a bug in the imported `evaluate.py` (it indexed the section id as a
  batched list; `collate_single` returns a bare string). Smoke-tested the full
  train→transfer→stats loop on `data/synthetic.h5ad`.

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

## The blocker: segfault (exit code 139) — RESOLVED 2026-06-12

**Root cause was NOT memory — it was an OpenMP runtime conflict.** With
`PYTHONFAULTHANDLER=1` the crash prints right after `[load]`, *before* the HVG
step runs, with `OMP: Info #276` on the preceding line. The base conda env ships
four OpenMP runtimes in `miniconda3/lib` — `libgomp`, Intel `libiomp5`, LLVM
`libomp`, plus bundled copies inside `torch` and `sklearn`. When numba (scanpy),
sklearn (PCA), and torch initialise their thread pools in one process the
duplicate runtimes collide and hard-segfault. The earlier memory hypothesis was
a red herring; on a machine with ~19 GB free it crashed at the 13 GB load.

**Fix:** tolerate the duplicate runtime *and* pin threads, set before numpy/torch
load. Baked into `immunotransformer/__init__.py` (runs on package import, so it
works no matter how you launch — no env vars to remember):
`KMP_DUPLICATE_LIB_OK=TRUE`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, `NUMBA_NUM_THREADS=1` (all `setdefault`, so an explicit
env var still wins). Note: `KMP_DUPLICATE_LIB_OK` alone is NOT enough — verified
it still crashes without the thread pins.

**Verified:** full RRMAP2 run completes end-to-end (40 epochs, ~1 min wall —
training is fast because the model is a light MIL head on PCA features; the ~40 s
is the one-time load + HVG + PCA). Best @epoch 33: **MAE 3.41, Spearman ρ 0.753**
on 17 held-out val bags. Artifacts in `runs/rrmap2_stage/`.

Cleaner permanent option (not required): rebuild the env with a single OpenMP
runtime (`conda install nomkl` / drop `intel-openmp`) so the duplicate never
loads — then the thread pins could be relaxed for speed.

The HVG memory fix from the Codon integration is still a good idea on its own
(lower peak RSS on the full panel) but was never what caused the segfault.

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
