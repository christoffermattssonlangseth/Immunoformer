# ImmunoTransformer

Cell-token transformer and duration-clock analyses for EAE / RR-EAE Xenium spatial
transcriptomics (RRMAP2: 1.38M cells, 67 animals; 33 relapsing-remitting, 34 chronic).
Design rationale: [`docs/le-quesne-framework-applications.md`](docs/le-quesne-framework-applications.md).

## Where the project stands (2026-10-09)

The main result is a **duration clock**. A linear model on animal-level pseudobulk predicts
how long an animal has had EAE (`day_of_sacrifice`, adjusted for severity). It reaches
ρ ≈ 0.83 in RR and 0.85 in chronic, scored leave-one-animal-out (one animal held out per fold).

What we have learned about it:

| question | answer | details |
|---|---|---|
| Does a cell-level attention model (Immunoformer) beat the pseudobulk clock? | **Not shown.** At ~30 animals the two can't be told apart, and switching attention off doesn't hurt. Stage 2 was not built. | [`docs/negative-result-scaling.md`](docs/negative-result-scaling.md), [`runs/baseline_ladder/WRITEUP.md`](runs/baseline_ladder/WRITEUP.md) |
| Is the clock a sequencing-run artefact? | No. It survives run as a covariate and leave-one-run-out. | `runs/clock_run_robustness/` |
| Is it carried by one gene block (circadian, matrix/scar, sex) or one cell type? | No. Once genes picked from the clock itself are removed, each block predicts no better than random genes. No cell type, spatial feature or DA-glia state beats the all-cells clock. The signal is broad and tissue-wide. | `runs/clock_composition/` |
| Is it lesions or the whole tissue? | Non-lesion tissue dates the animal as well as lesions do (RR). | [`runs/closing_block/WRITEUP.md`](runs/closing_block/WRITEUP.md) |
| Cell proportions or cell state? | Mainly cell state. Coarse composition alone is weaker, and expression keeps signal after composition is removed. | same |
| Collagen IV? | Endothelial Col4a1/Col4a2 fall with duration in RR, even without severity adjustment. Not robust in chronic. An IHC test is pre-specified. | same; [`runs/closing_block/IHC_prespecification.md`](runs/closing_block/IHC_prespecification.md) |
| Open | Five circadian genes (Nr1d1, Dbp, Bhlhe40, Per1, Hlf) are strong clock genes. Whether that reflects time of day at sacrifice can't be settled: sacrifice times were not recorded. | `runs/clock_composition/` |

Still running: the tuned attention model (`runs/arm6_tuned/`) and a 100-permutation
confirmation null (`runs/clock_composition/perm/`).

Task-by-task plan: [`WORKORDER.md`](WORKORDER.md). Earlier overnight summary:
[`runs/OVERNIGHT_SUMMARY.md`](runs/OVERNIGHT_SUMMARY.md).

## Running the analyses

- **Data:** the canonical atlas is
  `RRMAP2_xenium_all_samples.cellcharter.companion.ready.with_metadata.rerun.with_AnnoL1Curated_with_Region_Anno2to4Updated.h5ad`.
  Scripts read it from `~/Downloads/` by default; set `RRMAP2_H5AD` to point elsewhere. Daily
  clinical scores and weights are in `spreadsheets/`.
- **Environment:** a conda env with torch, scanpy and scikit-learn (locally
  `~/miniforge3/envs/cellcharter`). **Always import `immunotransformer` before numpy or torch**;
  its `__init__` sets the OpenMP guards that prevent a known segfault (see `HANDOFF.md`).
- **Run a script from the repo root:**
  ```bash
  PYTHONPATH="$PWD:$PWD/scripts" python analysis/<script>.py
  ```
  Each writes `runs/<task>/` (`report.txt` or `WRITEUP.md`, `results.json`, `figures/`).
  Large caches (`*.npz`) are gitignored and rebuilt on first run.
- **Shared rules for every analysis:**
  - the animal (`sample_name`) is the unit;
  - cross-validation is leave-one-animal-out;
  - every fitted transform is fit inside the training fold;
  - CIs are bootstraps over animals;
  - comparisons on the same animals use a paired bootstrap of the difference;
  - the predefined `Global_niche` labels are not used.

## Stage 1 model code

Frozen/pluggable per-cell encoder → **gated attention-MIL** → **CORAL ordinal head** or a
**regression head**, with **animal-level splitting** to avoid leakage. `pooling="mean"`
switches attention off (the control used in the ladder). The attention weights are the
interpretable readout: which cells drive a prediction.

### Install

```bash
pip install -e .
```

### Smoke test on synthetic data (~1–2 min on CPU)

```bash
python scripts/make_synthetic.py --out data/synthetic.h5ad
python -m immunotransformer.train --config configs/baseline.yaml
```

A working run reaches Spearman ρ > 0 on held-out animals; outputs land in `runs/baseline/`.

### Pointing it at a real Xenium `.h5ad`

Edit `configs/baseline.yaml`:
- `data.h5ad_path`
- `obs.animal_id`, `obs.section_id`, `obs.label`, `obs.label_order`
- `data.layer` (the raw-counts layer)
- `model.head` (`coral` or `regression`)

For the full 5101-gene panel use `data.n_hvg: 2000` and `encoder: pca`.

### Not built

- **Stage 2 transformer:** not built, by the pre-registered rule in Task 5; see the negative-result doc.
- **Foundation-model encoder:** `encoders.FrozenFMEncoder` is still a stub.

## Layout

```
immunotransformer/   model code (config, data, encoders, model, losses, train, evaluate, stats)
analysis/            current analyses: trajectory features, baseline ladder, attention-MIL
                     (LOAO, tuned), clock composition and controls, lesion clock, DA-glia,
                     closing block
scripts/             earlier analyses (duration clock, batch identifiability, run robustness,
                     mtDNA-DSB) and model utilities
runs/<task>/         outputs per analysis (WRITEUP.md / report.txt, results.json, figures/)
docs/                write-ups and data documentation
configs/             training configs
spreadsheets/        daily clinical score and weight sheets
```
