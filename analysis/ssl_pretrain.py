"""SELF-SUPERVISED CELL + NICHE PRETRAINING (exploratory; ladder arm 7)

Learns a per-cell embedding (z_cell) and a spatial-niche embedding (z_niche) from every
cell in the atlas, with no disease label (immunotransformer/ssl.py has the model and the
two objectives). Writes per-section mean embeddings that analysis/baseline_ladder.py picks
up as arm 7 (z_cell), 7n (z_niche) and 7p (pseudobulk + z_niche), on the same LOAO folds
and targets as every other arm.

STATUS. This re-opens the learned-spatial-context direction that the pre-registered B3 test
shelved (runs/clock_composition/paired_difference.md: nested spatial vs pseudobulk, diff
+0.029 [-0.073, +0.162]). Arm 7 is exploratory and is read only with the paired animal
bootstrap of rho(arm) - rho(arm 2) that the ladder reports. Decision rule, fixed before
the first real run: arm 7n counts as a gain only if that CI excludes zero on
day_of_sacrifice (RR); anything else is "not shown", as for arm 6.

LEAKAGE. Pretraining is label-free but transductive: by default it sees the cells of every
animal, including the one each LOAO fold later holds out. To check that this does not
matter, rerun with --exclude-animals (those animals' cells are never trained on, only
embedded) and compare how well held-out vs seen animals are predicted.

Needs ~20 GB RAM at peak for the full atlas (sparse counts, briefly two copies) and a GPU or Apple MPS for the
default 20k steps; --steps 2000 is a quick CPU check.

    PYTHONPATH="$PWD:$PWD/scripts" python analysis/ssl_pretrain.py
    PYTHONPATH="$PWD:$PWD/scripts" python analysis/ssl_pretrain.py --h5ad data/synthetic_spatial.h5ad \
        --layer none --animal-key animal_id --section-key section_id --steps 300 --out runs/ssl_smoke
"""

from __future__ import annotations

import immunotransformer  # noqa: F401  (OpenMP guard — must precede numpy; see HANDOFF.md)

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from immunotransformer import ssl
from immunotransformer.h5io import read_cells
from immunotransformer.train import resolve_device

ATLAS = os.environ.get(
    "RRMAP2_H5AD",
    os.path.expanduser(
        "~/Downloads/RRMAP2_xenium_all_samples.cellcharter.companion.ready.with_metadata."
        "rerun.with_AnnoL1Curated_with_Region_Anno2to4Updated.h5ad"),
)


def parse_args():
    d = ssl.SSLConfig()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--h5ad", default=ATLAS)
    ap.add_argument("--layer", default="counts", help="raw-count layer; 'none' reads X")
    ap.add_argument("--animal-key", default="sample_name")
    ap.add_argument("--section-key", default="meta_sample_id")
    ap.add_argument("--spatial-key", default="spatial")
    ap.add_argument("--n-genes", type=int, default=0,
                    help="top-variance genes to model (0 = whole panel)")
    ap.add_argument("--exclude-animals", default="",
                    help="comma-separated animals, or a file with one per line: never trained on")
    ap.add_argument("--val-frac", type=float, default=0.02,
                    help="random cells held out for the validation loss")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save-cells", action="store_true",
                    help="also write per-cell embeddings (float16; ~350 MB for the atlas)")
    ap.add_argument("--out", default="runs/ssl_pretrain")
    for f, v in vars(d).items():
        ap.add_argument("--" + f.replace("_", "-"), type=type(v), default=v)
    return ap.parse_args()


def excluded(arg: str) -> set[str]:
    if not arg:
        return set()
    if os.path.exists(arg):
        with open(arg) as fh:
            return {ln.strip() for ln in fh if ln.strip()}
    return {a.strip() for a in arg.split(",") if a.strip()}


def main():
    args = parse_args()
    cfg = ssl.SSLConfig(**{f: getattr(args, f) for f in vars(ssl.SSLConfig())})
    os.makedirs(args.out, exist_ok=True)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    t0 = time.time()

    layer = None if args.layer.lower() == "none" else args.layer
    X, genes, obs, xy = read_cells(args.h5ad, layer=layer, animal_key=args.animal_key,
                                   section_key=args.section_key, spatial_key=args.spatial_key)
    print(f"[load] {X.shape[0]:,} cells x {X.shape[1]:,} genes, {obs.section.nunique()} "
          f"sections, {obs.animal.nunique()} animals ({time.time() - t0:.0f}s)", flush=True)

    held = excluded(args.exclude_animals)
    unknown = held - set(obs.animal)
    if unknown:
        raise SystemExit(f"--exclude-animals not in the data: {sorted(unknown)[:5]}")
    rng = np.random.default_rng(cfg.seed)
    is_held = obs.animal.isin(held).to_numpy()
    val_cell = rng.random(len(obs)) < args.val_frac
    train_rows = np.flatnonzero(~is_held & ~val_cell)
    val_rows = np.flatnonzero(is_held | val_cell)

    X = ssl.normalize_log(X)                       # library size on the FULL panel
    keep = ssl.top_variance_genes(X, train_rows, args.n_genes, seed=cfg.seed)
    if len(keep) < X.shape[1]:
        X = X[:, keep].tocsr()
    mean, std = ssl.gene_stats(X, train_rows, seed=cfg.seed)
    nbr = ssl.knn_by_section(xy, obs.section.to_numpy(), cfg.k)
    pos_scale = ssl.pos_scale_from(xy, nbr, seed=cfg.seed)
    device = resolve_device(args.device)
    print(f"[prep] {len(keep):,} genes, k={cfg.k}, pos_scale {pos_scale:.2f} (coord units), "
          f"train {len(train_rows):,} / val {len(val_rows):,} cells, "
          f"{len(held)} excluded animals, device {device} ({time.time() - t0:.0f}s)", flush=True)

    batcher = ssl.CellBatcher(X, nbr, xy, mean, std, pos_scale, device)
    model = ssl.NicheSSL(len(keep), cfg).to(device)
    history = ssl.pretrain(model, batcher, train_rows, val_rows, cfg,
                           log_every=max(1, min(200, cfg.steps // 10)))

    all_rows = np.arange(len(obs))
    z_cell, z_niche = ssl.embed(model, batcher, all_rows)
    print(f"[embed] {len(all_rows):,} cells ({time.time() - t0:.0f}s)", flush=True)

    secs, zc_mean, n_cells = ssl.group_means(z_cell, obs.section.to_numpy())
    _, zn_mean, _ = ssl.group_means(z_niche, obs.section.to_numpy())
    sec_animal = obs.groupby("section").animal.agg(lambda s: s.unique().tolist())
    multi = sec_animal[sec_animal.map(len) > 1]
    if len(multi):
        raise SystemExit(f"sections spanning several animals: {list(multi.index[:5])}")
    d = cfg.dim
    table = pd.DataFrame(
        np.hstack([zc_mean, zn_mean]),
        columns=[f"zc{i}" for i in range(d)] + [f"zn{i}" for i in range(d)])
    table.insert(0, "n_cells", n_cells)
    table.insert(0, "excluded_from_training", [sec_animal[s][0] in held for s in secs])
    table.insert(0, "animal", [sec_animal[s][0] for s in secs])
    table.insert(0, "section", secs)
    table.to_csv(os.path.join(args.out, "section_embeddings.csv"), index=False,
                 float_format="%.6g")

    torch.save({"state_dict": model.state_dict(), "config": ssl.config_dict(cfg),
                "genes": genes[keep].tolist(), "gene_mean": mean, "gene_std": std,
                "pos_scale": pos_scale}, os.path.join(args.out, "model.pt"))
    if args.save_cells:
        np.savez_compressed(os.path.join(args.out, "cell_embeddings.npz"),
                            obs_names=obs.index.to_numpy().astype(str),
                            z_cell=z_cell.astype(np.float16), z_niche=z_niche.astype(np.float16))
    first, last = history[0], history[-1]
    summary = {
        "h5ad": os.path.basename(args.h5ad), "layer": layer, "n_cells": int(len(obs)),
        "n_genes": int(len(keep)), "n_sections": int(len(secs)),
        "n_animals": int(obs.animal.nunique()), "excluded_animals": sorted(held),
        "pos_scale": pos_scale, "device": str(device), "config": ssl.config_dict(cfg),
        "loss_first": first, "loss_last": last, "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    with open(os.path.join(args.out, "history.json"), "w") as fh:
        json.dump(history, fh)
    print(f"[done] val masked-recon {first.get('val_masked_recon', float('nan')):.4f} -> "
          f"{last.get('val_masked_recon', float('nan')):.4f}, val niche-pred "
          f"{first.get('val_niche_pred', float('nan')):.4f} -> "
          f"{last.get('val_niche_pred', float('nan')):.4f}; outputs in {args.out}/")


if __name__ == "__main__":
    main()
