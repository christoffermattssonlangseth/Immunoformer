# ImmunoTransformer

Cell-token transformer for EAE / RR-EAE Xenium spatial transcriptomics.
Design rationale: [`docs/le-quesne-framework-applications.md`](docs/le-quesne-framework-applications.md).

## Stage 1 — cheap baseline (this code)

Frozen/pluggable per-cell encoder → **gated attention-MIL** → **ordinal (CORAL)
head** predicting a section-level label (EAE timepoint), with **animal-level
train/val splitting** to avoid leakage. Establishes a real number on the timepoint
task at near-zero pretraining cost, before committing to the full transformer
(Stage 2 — see docs).

The attention weights are the interpretable readout (which cells drive the
prediction → cross-check vs. perivascular T cells / DAM microglia / reactive
astrocytes).

### Install

```bash
pip install -e .
```

### Smoke-test on synthetic data (runs in ~1–2 min on CPU)

```bash
python scripts/make_synthetic.py --out data/synthetic.h5ad
python -m immunotransformer.train --config configs/baseline.yaml
```

A working run reaches Spearman ρ > 0 on held-out animals (the synthetic data has
a real injected timepoint signal). Outputs land in `runs/baseline/`:
`best.pt`, `best_metrics.json`, `val_attention.npz`.

### Point it at your real Xenium `.h5ad`

Edit `configs/baseline.yaml`:
- `data.h5ad_path` → your file.
- `obs.animal_id`, `obs.section_id`, `obs.label` → your `adata.obs` column names.
- `obs.label_order` → the ordered stage values exactly as they appear in `obs`.
- `data.layer` → your raw-counts layer (leave `null` if `adata.X` is counts).
- For the full 5101-gene panel, consider `data.n_hvg: 2000` and `encoder: pca`.

Same command runs it. Start with `timepoint` as the target (cleanest ordinal
axis); swap `obs.label` + `label_order` to model relapse/remission phase later.

### What's deliberately NOT here yet

- The Stage 2 transformer (cell-as-token, spatial PE, dual masking) — see docs.
- A real foundation-model encoder: `encoders.FrozenFMEncoder` is a stub to fill
  in Nicheformer/scGPT (the "fm" encoder option).

## Layout

```
immunotransformer/
  config.py     # all data-layout assumptions live here
  data.py       # AnnData -> animal-split section bags
  encoders.py   # identity | pca | fm(stub) frozen cell encoders
  model.py      # gated attention-MIL
  losses.py     # CORAL ordinal regression
  train.py      # training loop + held-out-animal metrics
scripts/make_synthetic.py
configs/baseline.yaml
```
