"""Stage 1 training loop.

    python -m immunotransformer.train --config configs/baseline.yaml

Reports ordinal metrics on held-out ANIMALS: MAE (in stage units), exact
accuracy, and Spearman ρ between predicted and true stage.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import anndata as ad
import numpy as np
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from .config import Config
from .data import (
    Bag, BagDataset, build_bags, collate_single, fit_encoder_on_train,
    normalize_expression, split_animals,
)
from .losses import coral_loss, coral_predict
from .model import GatedAttentionMIL


def resolve_device(choice: str) -> torch.device:
    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    preds, trues, attn_dump = [], [], []
    for cells, label, sec in loader:
        cells = cells.to(device)
        logits, attn = model(cells)
        p = coral_predict(logits).item()
        preds.append(p)
        trues.append(label.item())
        if len(attn_dump) < 8:  # save a few attention maps for inspection
            attn_dump.append({"section": sec, "label": label.item(),
                              "pred": p, "attn": attn.detach().cpu().numpy()})
    preds, trues = np.array(preds), np.array(trues)
    mae = float(np.abs(preds - trues).mean())
    acc = float((preds == trues).mean())
    rho = float(spearmanr(preds, trues).correlation) if len(set(trues)) > 1 else float("nan")
    return {"mae": mae, "acc": acc, "spearman": rho, "n": len(preds)}, attn_dump


def _select_hvg(adata: ad.AnnData, cfg: Config) -> np.ndarray:
    """Return a boolean HVG mask computed on a random subsample of cells.

    The old code did `adata.copy()` (duplicating the full panel *and* the counts
    layer) before HVG selection — the ~2x RAM spike that tipped the 1.38M-cell
    RRMAP2 run over 48 GB and segfaulted. Here we slice a counts-only subsample
    into a tiny scratch AnnData instead, so we never hold two full-panel copies.
    For dispersion-based HVG selection on a dataset this size, 50k cells is
    statistically equivalent to using all of them.
    """
    import gc
    import scanpy as sc

    n_sample = min(cfg.data.hvg_subsample, adata.n_obs)
    rng = np.random.default_rng(cfg.data.seed + 99)  # distinct from the split RNG
    sample_idx = rng.choice(adata.n_obs, n_sample, replace=False)

    counts = adata.layers[cfg.data.layer] if cfg.data.layer else adata.X
    scratch = ad.AnnData(X=counts[sample_idx].copy())  # sparse slice, cheap
    sc.pp.normalize_total(scratch, target_sum=1e4)
    sc.pp.log1p(scratch)
    sc.pp.highly_variable_genes(scratch, n_top_genes=cfg.data.n_hvg)
    hvg_mask = scratch.var.highly_variable.values.copy()

    del scratch, counts
    gc.collect()
    print(f"[hvg] {hvg_mask.sum()} genes selected "
          f"(computed on {n_sample:,} / {adata.n_obs:,} cells)")
    return hvg_mask


def _save_run_artifacts(
    out_dir: str, enc, gene_names: np.ndarray, cfg: Config, val_animals
) -> None:
    """Persist the encoder, HVG gene panel, and a manifest so evaluate.py can
    apply identical preprocessing to a transfer dataset without re-fitting."""
    with open(os.path.join(out_dir, "encoder.pkl"), "wb") as fh:
        pickle.dump(enc, fh, protocol=pickle.HIGHEST_PROTOCOL)
    np.save(os.path.join(out_dir, "hvg_genes.npy"), gene_names)

    manifest = {
        "label_order": cfg.obs.label_order,
        "encoder": cfg.data.encoder,
        "n_hvg": cfg.data.n_hvg,
        "pca_dim": cfg.data.pca_dim,
        "obs_animal_id": cfg.obs.animal_id,
        "obs_section_id": cfg.obs.section_id,
        "obs_label": cfg.obs.label,
        "data_layer": cfg.data.layer,
        # Held-out val animals, so a within-domain re-eval scores only these
        # bags instead of the whole source set (which includes training bags).
        "val_animals": sorted(str(a) for a in val_animals),
        # Model architecture, so the evaluator reconstructs the exact net rather
        # than assuming ModelConfig defaults.
        "model": {
            "proj_dim": cfg.model.proj_dim,
            "attn_dim": cfg.model.attn_dim,
            "dropout": cfg.model.dropout,
        },
    }
    with open(os.path.join(out_dir, "run_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[artifacts] encoder, HVG genes, manifest -> {out_dir}/")


def train(cfg: Config):
    device = resolve_device(cfg.train.device)
    os.makedirs(cfg.train.out_dir, exist_ok=True)
    torch.manual_seed(cfg.data.seed)

    print(f"[load] {cfg.data.h5ad_path}")
    adata = ad.read_h5ad(cfg.data.h5ad_path)
    print(f"       {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    if cfg.data.n_hvg and cfg.data.n_hvg < adata.n_vars:
        hvg_mask = _select_hvg(adata, cfg)
        adata = adata[:, hvg_mask]  # view — no second full-panel copy

    X = normalize_expression(adata, cfg.data.layer)
    bags = build_bags(adata, cfg)
    train_bags, val_bags, val_animals = split_animals(
        bags, cfg.data.val_fraction, cfg.data.seed)
    print(f"[split] {len(train_bags)} train bags / {len(val_bags)} val bags "
          f"| val animals: {sorted(val_animals)}")

    enc, X_enc = fit_encoder_on_train(X, train_bags, cfg)
    print(f"[encoder] {cfg.data.encoder} -> dim {enc.out_dim}")

    gene_names = np.array(adata.var_names, dtype=str)
    _save_run_artifacts(cfg.train.out_dir, enc, gene_names, cfg, val_animals)

    train_ds = BagDataset(train_bags, X_enc, cfg, train=True)
    val_ds = BagDataset(val_bags, X_enc, cfg, train=False)
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_single)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_single)

    model = GatedAttentionMIL(
        in_dim=enc.out_dim, num_classes=cfg.num_classes,
        proj_dim=cfg.model.proj_dim, attn_dim=cfg.model.attn_dim,
        dropout=cfg.model.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)

    def selection_score(m: dict) -> tuple:
        # Rank by Spearman, break ties by lower MAE (NaN rho sorts last).
        rho = m["spearman"]
        rho = -2.0 if rho != rho else rho  # NaN guard
        return (round(rho, 4), -m["mae"])

    best = {"spearman": -2.0, "mae": float("inf")}
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        opt.zero_grad()
        running = 0.0
        for step, (cells, label, _sec) in enumerate(train_dl, 1):
            cells, label = cells.to(device), label.to(device)
            logits, _ = model(cells)
            loss = coral_loss(logits, label.unsqueeze(0)) / cfg.train.grad_accum
            loss.backward()
            running += loss.item() * cfg.train.grad_accum
            if step % cfg.train.grad_accum == 0:
                opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()

        metrics, attn_dump = evaluate(model, val_dl, device, cfg.num_classes)
        print(f"[epoch {epoch:3d}] loss {running/len(train_dl):.4f} | "
              f"val MAE {metrics['mae']:.3f} acc {metrics['acc']:.3f} "
              f"rho {metrics['spearman']:.3f}")
        if selection_score(metrics) > selection_score(best):
            best = {**metrics, "epoch": epoch}
            torch.save(model.state_dict(), os.path.join(cfg.train.out_dir, "best.pt"))
            np.savez(os.path.join(cfg.train.out_dir, "val_attention.npz"),
                     **{f"bag{i}_{d['section']}": d["attn"]
                        for i, d in enumerate(attn_dump)})

    with open(os.path.join(cfg.train.out_dir, "best_metrics.json"), "w") as fh:
        json.dump(best, fh, indent=2)
    print(f"[done] best @epoch {best.get('epoch')}: "
          f"MAE {best['mae']:.3f} acc {best['acc']:.3f} rho {best['spearman']:.3f}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="YAML config (defaults if omitted)")
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config) if args.config else Config()
    train(cfg)


if __name__ == "__main__":
    main()
