"""Hal cell-type localization figure + data, from the cached per-cluster pseudobulk.

Panel A: mean Hal per leiden_1 cluster (Hal-high cluster highlighted) -> shows the
~10x concentration. Panel B: the Hal-high cluster vs all-other-clusters mean for a
curated marker panel, coloured by lineage -> shows it is an inflammatory myeloid /
macrophage state (Arg1/Chil3/Cd14/Cd74 high) with homeostatic-microglia markers low.

    python scripts/make_hal_celltype_fig.py
"""

from __future__ import annotations

import json
import os

import numpy as np

CACHE = "runs/rr_phase_niche/agg_leiden_1.npz"
OUT_DIR = "runs/rr_phase_niche"
MARKERS = [
    ("Arg1", "myeloid/mac"), ("Chil3", "myeloid/mac"), ("Cd14", "myeloid/mac"),
    ("Fn1", "myeloid/mac"), ("Ccr2", "myeloid/mac"), ("Plac8", "myeloid/mac"),
    ("Cd74", "antigen-pres"), ("H2-Aa", "antigen-pres"), ("Gpnmb", "DAM/foamy"),
    ("Cx3cr1", "microglia"), ("P2ry12", "microglia"), ("Tmem119", "microglia"),
    ("Fcrls", "microglia"),
]
LINE_COLOR = {"myeloid/mac": "#c0392b", "antigen-pres": "#e8590c",
              "DAM/foamy": "#7048e8", "microglia": "#2c6fbb"}


def main():
    os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)
    d = np.load(CACHE, allow_pickle=True)
    pb, genes, clusters = d["pb"], d["genes"].astype(str), d["clusters"].astype(str)
    gi = {g: i for i, g in enumerate(genes)}
    cl_mean = np.nanmean(pb, axis=0)                  # [cluster, gene]
    hal = cl_mean[:, gi["Hal"]]
    top = int(np.argmax(hal))
    other = np.arange(len(clusters)) != top

    data = {
        "hal_high_cluster": clusters[top],
        "hal_top": round(float(hal[top]), 3),
        "hal_median": round(float(np.median(hal)), 3),
        "fold_vs_median": round(float(hal[top] / max(np.median(hal), 1e-6)), 1),
        "per_cluster_hal": {clusters[i]: round(float(hal[i]), 3)
                            for i in np.argsort(-hal)},
        "markers_top_vs_other": [
            {"gene": g, "lineage": ln,
             "top": round(float(cl_mean[top, gi[g]]), 2),
             "other_mean": round(float(cl_mean[other][:, gi[g]].mean()), 2)}
            for g, ln in MARKERS if g in gi],
    }
    with open(os.path.join(OUT_DIR, "hal_celltype.json"), "w") as fh:
        json.dump(data, fh, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib missing; wrote json only")
        return

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 5),
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    # Panel A: Hal per cluster
    order = np.argsort(hal)
    colors = ["#c0392b" if i == top else "#c9ccd1" for i in order]
    axA.barh(range(len(clusters)), hal[order], color=colors)
    axA.set_yticks(range(len(clusters)))
    axA.set_yticklabels([f"c{clusters[i]}" for i in order], fontsize=7)
    axA.set_xlabel("mean Hal (log CP10k)")
    axA.set_title(f"Hal is concentrated in cluster {clusters[top]}\n"
                  f"({data['fold_vs_median']}x the median cluster)", fontsize=11)
    axA.spines[["top", "right"]].set_visible(False)

    # Panel B: marker panel, top cluster vs others
    ms = data["markers_top_vs_other"]
    yp = np.arange(len(ms))
    axB.barh(yp + 0.2, [m["top"] for m in ms], height=0.4,
             color=[LINE_COLOR[m["lineage"]] for m in ms], label=f"cluster {clusters[top]}")
    axB.barh(yp - 0.2, [m["other_mean"] for m in ms], height=0.4,
             color="#d9dce0", label="other clusters (mean)")
    axB.set_yticks(yp)
    axB.set_yticklabels([m["gene"] for m in ms], fontsize=9)
    axB.invert_yaxis()
    axB.set_xlabel("mean expression (log CP10k)")
    axB.set_title("Identity: inflammatory myeloid / antigen-presenting\n"
                  "(homeostatic-microglia markers low)", fontsize=11)
    axB.legend(fontsize=8, frameon=False, loc="lower right")
    axB.spines[["top", "right"]].set_visible(False)
    # lineage colour key
    for i, (ln, c) in enumerate(LINE_COLOR.items()):
        axB.text(1.02, 0.95 - i * 0.07, ln, color=c, transform=axB.transAxes,
                 fontsize=8, fontweight="bold")

    fig.tight_layout()
    png = os.path.join(OUT_DIR, "figures", "hal_celltype.png")
    fig.savefig(png, dpi=130, bbox_inches="tight")
    fig.savefig(png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[done] -> {png} + hal_celltype.json")
    print(f"        Hal-high cluster {clusters[top]}: {data['hal_top']} vs median {data['hal_median']}")


if __name__ == "__main__":
    main()
