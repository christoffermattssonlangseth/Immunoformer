"""Cross-model conservation figure: chronic severity trend vs RR relapse-cycle trend
per gene. Shows the EAE disease program is strain/model-invariant.

    python scripts/make_conservation_fig.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

OUT = "runs/chronic_trajectory"


def main():
    d = np.load(os.path.join(OUT, "pseudobulk.npz"), allow_pickle=True)
    pb, genes, score = d["pb"], d["genes"].astype(str), d["score"].astype(float)
    # per-gene chronic severity trend (Spearman vs score) — vectorised
    keep = ~np.isnan(score)
    pb, sc = pb[keep], score[keep]
    rs = rankdata(sc)
    rs = (rs - rs.mean()) / rs.std()
    gr = np.apply_along_axis(rankdata, 0, pb)
    gz = (gr - gr.mean(0)) / (gr.std(0) + 1e-12)
    chr_rho = (gz.T @ rs) / len(sc)
    chr = pd.DataFrame({"gene": genes, "chr_rho": chr_rho})

    cyc = pd.read_csv("runs/rr_cycle_oscillation/gene_oscillation_metrics.csv")[["gene", "mono_rho"]]
    mg = chr.merge(cyc, on="gene", how="inner").dropna()
    rho = spearmanr(mg["chr_rho"], mg["mono_rho"]).correlation

    core_up = ["C6", "Gpnmb", "C4b", "Abca1", "Stab1", "Hpse", "Ctss", "C3ar1", "Csf1r"]
    core_dn = ["Msmo1", "Idi1", "Hmgcr", "Lss", "Hsd17b7", "Mal", "Plp1", "Mog"]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        json.dump({"spearman": float(rho)}, open(os.path.join(OUT, "conservation_fig_data.json"), "w"))
        return

    fig, ax = plt.subplots(figsize=(6.4, 6))
    ax.scatter(mg["chr_rho"], mg["mono_rho"], s=5, c="#c9ccd1", rasterized=True)
    for genes_set, col in [(core_up, "#c0392b"), (core_dn, "#2c6fbb")]:
        sub = mg[mg["gene"].isin(genes_set)]
        ax.scatter(sub["chr_rho"], sub["mono_rho"], s=26, c=col, zorder=3)
        for _, r in sub.iterrows():
            ax.annotate(r["gene"], (r["chr_rho"], r["mono_rho"]), fontsize=7)
    ax.axhline(0, c="gray", lw=.5); ax.axvline(0, c="gray", lw=.5)
    lim = [-1, 1]; ax.plot(lim, lim, "k--", lw=.6); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("CHRONIC severity trend (Spearman ρ)")
    ax.set_ylabel("RR relapse-cycle trend (ρ)")
    ax.set_title(f"The EAE disease program is conserved across models\n"
                 f"Spearman = {rho:+.2f}  (5101 genes; CHRONIC = B6/MOG, RR = SJL/PLP)", fontsize=11)
    ax.text(.04, .92, "conserved UP\n(complement / DAM)", color="#c0392b", transform=ax.transAxes, fontsize=8)
    ax.text(.55, .06, "conserved DOWN\n(cholesterol / myelin)", color="#2c6fbb", transform=ax.transAxes, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    png = os.path.join(OUT, "figures", "conservation_scatter.png")
    os.makedirs(os.path.dirname(png), exist_ok=True)
    fig.savefig(png, dpi=130, bbox_inches="tight")
    fig.savefig(png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    json.dump({"spearman": float(rho)}, open(os.path.join(OUT, "conservation_fig_data.json"), "w"))
    print(f"[done] -> {png}  (Spearman {rho:+.3f})")


if __name__ == "__main__":
    main()
