"""Stage 1 training loop.

    python -m immunotransformer.train --config configs/baseline.yaml

Reports ordinal metrics on held-out ANIMALS: MAE (in stage units), exact
accuracy, and Spearman ρ between predicted and true stage.
"""
from __future__ import annotations

import argparse
import json
import os

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


def train(cfg: Config):
    import anndata as ad

    device = resolve_device(cfg.train.device)
    os.makedirs(cfg.train.out_dir, exist_ok=True)
    torch.manual_seed(cfg.data.seed)

    print(f"[load] {cfg.data.h5ad_path}")
    adata = ad.read_h5ad(cfg.data.h5ad_path)

    if cfg.data.n_hvg and cfg.data.n_hvg < adata.n_vars:
        import scanpy as sc
        tmp = adata.copy()
        sc.pp.normalize_total(tmp, target_sum=1e4); sc.pp.log1p(tmp)
        sc.pp.highly_variable_genes(tmp, n_top_genes=cfg.data.n_hvg)
        adata = adata[:, tmp.var.highly_variable.values].copy()
        print(f"[hvg] kept {adata.n_vars} genes")

    X = normalize_expression(adata, cfg.data.layer)
    bags = build_bags(adata, cfg)
    train_bags, val_bags, val_animals = split_animals(
        bags, cfg.data.val_fraction, cfg.data.seed)
    print(f"[split] {len(train_bags)} train bags / {len(val_bags)} val bags "
          f"| val animals: {sorted(val_animals)}")

    enc, X_enc = fit_encoder_on_train(X, train_bags, cfg)
    print(f"[encoder] {cfg.data.encoder} -> dim {enc.out_dim}")

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
