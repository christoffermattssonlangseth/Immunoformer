"""RR vs chronic: bag-level attention-MIL classifier with region-aware CV.

Question: is there a transcriptomic/spatial signature that distinguishes the
relapse-remitting from the chronic EAE course, and which cell populations drive
it? RRMAP2 is terminal (one stage per animal), so this is NOT longitudinal
relapse prediction — it's "does the tissue carry a model-of-disease signature".

Design (all reuse the existing immunotransformer pipeline):
  * Bags = meta_sample_id (158); label = `model` (CHRONIC=0, RELAPSE REMITTING=1);
    binary via the existing CoralHead with num_classes=2 (single logit + BCE).
  * StratifiedGroupKFold over ANIMALS (sample_name) — no animal leakage, label
    balanced across folds. The PCA encoder is refit on each fold's train cells.
  * Out-of-fold AUC, reported overall AND per spinal region. Lumbar (L) is the
    DE-CONFOUNDED test: it is the one region where the two models are balanced
    (cervical is ~32 RR vs 3 chronic, so whole-cord AUC partly reads "is this
    cervical").
  * Region-only baseline: a logistic regression on the region one-hot, same CV.
    The tissue model's lift over this baseline is the part NOT explained by the
    L->T->C inflammation gradient.
  * Attention -> cluster enrichment: for each bag, the softmax attention is the
    interpretable readout; we pool the attention mass per cluster separately for
    RR and chronic val bags. RRMAP2 has only numbered clusters (leiden/CellCharter)
    — annotate the top ones to cell types afterwards.

    python scripts/rr_vs_chronic.py
    python scripts/rr_vs_chronic.py --cluster-col CellCharter_10 --folds 5 --epochs 30
"""

from __future__ import annotations

import argparse
import json
import os

import anndata as ad
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from immunotransformer.config import Config, DataConfig, ModelConfig, ObsSchema, TrainConfig
from immunotransformer.data import BagDataset, build_bags, collate_single, normalize_expression
from immunotransformer.encoders import build_encoder
from immunotransformer.losses import coral_loss
from immunotransformer.model import GatedAttentionMIL
from immunotransformer.train import _select_hvg, resolve_device

RRMAP2 = (
    "/Volumes/moldiassd/RRMAP2_xenium_adata/kmeans_separated/"
    "RRMAP2_xenium_all_samples.cellcharter.companion.ready.with_metadata.rerun.h5ad"
)
LABEL_ORDER = ["CHRONIC", "RELAPSE REMITTING"]  # -> 0, 1


def make_cfg(args) -> Config:
    return Config(
        obs=ObsSchema(
            animal_id="sample_name",
            section_id="meta_sample_id",
            label="model",
            label_order=LABEL_ORDER,
            region="region",
        ),
        data=DataConfig(
            h5ad_path=args.h5ad, layer="counts", encoder="pca", pca_dim=args.pca_dim,
            n_hvg=args.n_hvg, hvg_subsample=50_000, max_cells_per_bag=args.max_cells,
            min_cells_per_bag=64, seed=args.seed,
        ),
        model=ModelConfig(proj_dim=128, attn_dim=64, dropout=0.1),
        train=TrainConfig(epochs=args.epochs, lr=1e-3, weight_decay=1e-4,
                          grad_accum=4, device=args.device, out_dir=args.out_dir),
    )


def train_fold(X_enc, train_bags, cfg, device) -> GatedAttentionMIL:
    """Fixed-epoch training (no val-based selection, so the OOF score is clean)."""
    model = GatedAttentionMIL(
        in_dim=X_enc.shape[1], num_classes=2,
        proj_dim=cfg.model.proj_dim, attn_dim=cfg.model.attn_dim, dropout=cfg.model.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    dl = DataLoader(BagDataset(train_bags, X_enc, cfg, train=True),
                    batch_size=1, shuffle=True, collate_fn=collate_single)
    for _ in range(cfg.train.epochs):
        model.train()
        opt.zero_grad()
        for step, (cells, label, _sec) in enumerate(dl, 1):
            cells, label = cells.to(device), label.to(device)
            logits, _ = model(cells)
            loss = coral_loss(logits, label.unsqueeze(0)) / cfg.train.grad_accum
            loss.backward()
            if step % cfg.train.grad_accum == 0:
                opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
    return model


@torch.no_grad()
def predict_fold(model, val_bags, X_enc, cfg, device):
    """Return per-bag dicts with prob, attention, and the cell indices it covers."""
    model.eval()
    dl = DataLoader(BagDataset(val_bags, X_enc, cfg, train=False),
                    batch_size=1, shuffle=False, collate_fn=collate_single)
    bag_by_sec = {b.section_id: b for b in val_bags}
    out = []
    for cells, label, sec in dl:
        logits, attn = model(cells.to(device))
        prob = torch.sigmoid(logits[0, 0]).item()  # P(RELAPSE REMITTING)
        b = bag_by_sec[sec]
        idx_used = b.cell_idx[: cfg.data.max_cells_per_bag]  # val = deterministic head slice
        out.append({
            "section": sec, "animal": b.animal_id, "true": int(label.item()),
            "prob": prob, "attn": attn.cpu().numpy(), "cell_idx": idx_used,
        })
    return out


def main():
    ap = argparse.ArgumentParser(description="RR vs chronic MIL classifier")
    ap.add_argument("--h5ad", default=RRMAP2)
    ap.add_argument("--cluster-col", default="leiden_1",
                    help="obs column used for attention enrichment (unnamed clusters; annotate later)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n-hvg", type=int, default=2000)
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--max-cells", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-dir", default="runs/rr_vs_chronic")
    args = ap.parse_args()

    cfg = make_cfg(args)
    device = resolve_device(args.device)
    os.makedirs(os.path.join(args.out_dir, "figures"), exist_ok=True)
    torch.manual_seed(args.seed)

    # ---- load + preprocess once -------------------------------------------
    print(f"[load] {args.h5ad}")
    adata = ad.read_h5ad(args.h5ad)
    print(f"       {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    if args.n_hvg and args.n_hvg < adata.n_vars:
        adata = adata[:, _select_hvg(adata, cfg)]

    X = normalize_expression(adata, cfg.data.layer)            # dense [cells, hvg]
    bags = build_bags(adata, cfg)
    obs = adata.obs
    region_by_sec = {str(k): str(v) for k, v in
                     obs.groupby(cfg.obs.section_id, observed=True)[cfg.obs.region].first().items()}
    clusters = obs[args.cluster_col].astype(str).to_numpy() if args.cluster_col in obs.columns else None
    if clusters is None:
        print(f"[warn] cluster column '{args.cluster_col}' not found — skipping enrichment")
    del adata
    print(f"[bags] {len(bags)} bags | label balance: "
          f"{sum(b.label==0 for b in bags)} chronic / {sum(b.label==1 for b in bags)} RR")

    # ---- CV setup: stratified by label, grouped by animal -----------------
    y = np.array([b.label for b in bags])
    groups = np.array([b.animal_id for b in bags])
    regions = np.array([region_by_sec.get(b.section_id, "?") for b in bags])
    skf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    oof = {}            # section -> dict(true, prob, region)
    region_oof = {}     # section -> region-only baseline prob
    attn_records = []   # per-bag attention detail for enrichment

    for fold, (tr, va) in enumerate(skf.split(np.zeros(len(bags)), y, groups), 1):
        train_bags = [bags[i] for i in tr]
        val_bags = [bags[i] for i in va]
        # encoder refit on this fold's train cells only (no leakage)
        enc = build_encoder(cfg.data.encoder, pca_dim=cfg.data.pca_dim, seed=cfg.data.seed)
        enc.fit(X[np.concatenate([b.cell_idx for b in train_bags])])
        X_enc = enc.transform(X)

        model = train_fold(X_enc, train_bags, cfg, device)
        preds = predict_fold(model, val_bags, X_enc, cfg, device)
        for p in preds:
            oof[p["section"]] = {"true": p["true"], "prob": p["prob"],
                                 "region": region_by_sec.get(p["section"], "?")}
            if clusters is not None:
                attn_records.append(p)

        # region-only baseline (logistic on region one-hot), same fold split
        reg_levels = sorted(set(regions))
        onehot = lambda rs: np.array([[r == lv for lv in reg_levels] for r in rs], dtype=float)
        lr = LogisticRegression(max_iter=200)
        lr.fit(onehot(regions[tr]), y[tr])
        for i, prob in zip(va, lr.predict_proba(onehot(regions[va]))[:, 1]):
            region_oof[bags[i].section_id] = float(prob)

        # fold-local AUC for a progress line
        vt = np.array([p["true"] for p in preds]); vp = np.array([p["prob"] for p in preds])
        fa = roc_auc_score(vt, vp) if len(set(vt)) > 1 else float("nan")
        print(f"[fold {fold}/{args.folds}] {len(val_bags)} val bags  AUC={fa:.3f}")

    # ---- pooled OOF metrics ----------------------------------------------
    secs = list(oof)
    true = np.array([oof[s]["true"] for s in secs])
    prob = np.array([oof[s]["prob"] for s in secs])
    reg = np.array([oof[s]["region"] for s in secs])
    rprob = np.array([region_oof[s] for s in secs])

    def auc(mask):
        t, p = true[mask], prob[mask]
        return roc_auc_score(t, p) if len(set(t)) > 1 and mask.sum() > 1 else float("nan")

    results = {
        "n_bags": len(secs),
        "auc_overall": float(roc_auc_score(true, prob)),
        "acc_overall": float(accuracy_score(true, (prob > 0.5).astype(int))),
        "auc_region_only_baseline": float(roc_auc_score(true, rprob)),
        "auc_by_region": {r: float(auc(reg == r)) for r in sorted(set(reg))},
        "region_counts": {r: {"chronic": int(((reg == r) & (true == 0)).sum()),
                              "RR": int(((reg == r) & (true == 1)).sum())} for r in sorted(set(reg))},
    }

    # ---- attention -> cluster enrichment ---------------------------------
    enrichment = None
    if clusters is not None and attn_records:
        # mean attention fraction per cluster, by class (attn sums to 1 within a bag)
        from collections import defaultdict
        mass = {0: defaultdict(float), 1: defaultdict(float)}
        comp = {0: defaultdict(float), 1: defaultdict(float)}  # unweighted cell composition
        nbag = {0: 0, 1: 0}
        for p in attn_records:
            cl = clusters[p["cell_idx"]]
            a = p["attn"]
            lab = p["true"]; nbag[lab] += 1
            n = len(cl)
            for c in set(cl):
                m = (cl == c)
                mass[lab][c] += a[m].sum()
                comp[lab][c] += m.sum() / n
        allc = sorted(set(list(mass[0]) + list(mass[1])))
        rows = []
        for c in allc:
            rr_attn = mass[1][c] / max(nbag[1], 1)
            ch_attn = mass[0][c] / max(nbag[0], 1)
            rr_comp = comp[1][c] / max(nbag[1], 1)
            ch_comp = comp[0][c] / max(nbag[0], 1)
            rows.append({"cluster": str(c), "rr_attn": float(rr_attn), "chronic_attn": float(ch_attn),
                         "attn_diff": float(rr_attn - ch_attn),
                         "rr_comp": float(rr_comp), "chronic_comp": float(ch_comp)})
        rows.sort(key=lambda r: r["attn_diff"])
        enrichment = {"cluster_col": args.cluster_col, "rows": rows}

    # ---- report ----------------------------------------------------------
    lines = []
    lines.append("\n=== RR vs CHRONIC — bag-level MIL, animal-grouped %d-fold CV ===\n" % args.folds)
    lines.append(f"bags={results['n_bags']}  overall AUC={results['auc_overall']:.3f}  "
                 f"acc={results['acc_overall']:.3f}")
    lines.append(f"region-only baseline AUC={results['auc_region_only_baseline']:.3f}  "
                 f"(lift = {results['auc_overall']-results['auc_region_only_baseline']:+.3f})")
    lines.append("\nAUC by spinal region (lumbar L = de-confounded; balanced models):")
    lines.append(f"  {'region':<8}{'AUC':>7}{'chronic':>9}{'RR':>5}")
    for r in sorted(results["auc_by_region"]):
        c = results["region_counts"][r]
        lines.append(f"  {r:<8}{results['auc_by_region'][r]:>7.3f}{c['chronic']:>9}{c['RR']:>5}")
    if enrichment:
        lines.append(f"\nAttention enrichment by '{enrichment['cluster_col']}' "
                     f"(mean attention fraction per bag; + = more attended in RR):")
        lines.append(f"  {'cluster':<10}{'RR_attn':>9}{'chr_attn':>9}{'diff':>8}{'RR_comp':>9}{'chr_comp':>9}")
        top = enrichment["rows"][:5] + enrichment["rows"][-5:]
        for r in top:
            lines.append(f"  {r['cluster']:<10}{r['rr_attn']:>9.3f}{r['chronic_attn']:>9.3f}"
                         f"{r['attn_diff']:>+8.3f}{r['rr_comp']:>9.3f}{r['chronic_comp']:>9.3f}")
    report = "\n".join(lines)
    print(report)

    with open(os.path.join(args.out_dir, "report.txt"), "w") as fh:
        fh.write(report + "\n")
    with open(os.path.join(args.out_dir, "results.json"), "w") as fh:
        json.dump({"results": results, "enrichment": enrichment}, fh, indent=2)
    np.savez(os.path.join(args.out_dir, "oof_predictions.npz"),
             section=np.array(secs), true=true, prob=prob, region=reg, region_baseline_prob=rprob)
    _plot(results, enrichment, args.out_dir)
    print(f"\n[done] -> {args.out_dir}/  (report.txt, results.json, oof_predictions.npz, figures/)")


def _plot(results, enrichment, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[skip] matplotlib unavailable")
        return
    # AUC by region + baseline
    fig, ax = plt.subplots(figsize=(5, 4))
    regs = sorted(results["auc_by_region"])
    vals = [results["auc_by_region"][r] for r in regs]
    ax.bar(range(len(regs)), vals, color=["#4C72B0" if r == "L" else "#B0B0B0" for r in regs])
    ax.axhline(results["auc_overall"], color="black", ls="--", lw=1, label=f"overall {results['auc_overall']:.2f}")
    ax.axhline(results["auc_region_only_baseline"], color="#C44E52", ls=":", lw=1.5,
               label=f"region-only {results['auc_region_only_baseline']:.2f}")
    ax.axhline(0.5, color="gray", lw=0.8)
    ax.set_xticks(range(len(regs))); ax.set_xticklabels([f"{r}\n(C{results['region_counts'][r]['chronic']}/R{results['region_counts'][r]['RR']})" for r in regs])
    ax.set_ylim(0, 1.05); ax.set_ylabel("OOF AUC (RR vs chronic)")
    ax.set_title("RR vs chronic — AUC by region (L = de-confounded)")
    ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "figures", "auc_by_region.pdf"), bbox_inches="tight")
    plt.close(fig)
    # attention enrichment
    if enrichment:
        rows = enrichment["rows"]
        top = rows[:6] + rows[-6:]
        fig, ax = plt.subplots(figsize=(5, max(3, len(top) * 0.4)))
        diffs = [r["attn_diff"] for r in top]
        ax.barh(range(len(top)), diffs, color=["#C44E52" if d > 0 else "#4C72B0" for d in diffs])
        ax.set_yticks(range(len(top))); ax.set_yticklabels([str(r["cluster"]) for r in top])
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("attention(RR) - attention(chronic)")
        ax.set_title(f"Attention enrichment by {enrichment['cluster_col']}")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "figures", "attention_by_cluster.pdf"), bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
