"""Region-axis figures for the atlas: gene-program means and cell-type composition
along L -> T -> C (within-animal), plus a small data json.

    python scripts/make_region_fig.py
"""
from __future__ import annotations

import json
import os

import numpy as np

CACHE = "runs/rr_region_gradient/agg_region.npz"
LABELS = "runs/rr_region_gradient/cluster_labels.json"
OUT = "runs/rr_region_gradient"
MIN_CELLS = 50

PROGRAMS = {
    "Inflammation/complement": ["C6", "C3", "C4b", "Cd74", "B2m", "Ctss"],
    "Inflammatory myeloid": ["Arg1", "Chil3", "Cd14", "Gpnmb", "Fn1"],
    "Myelin/oligo": ["Plp1", "Mbp", "Mog", "Mobp", "Cldn11"],
    "Acute oscillators": ["Hal", "Acod1", "Cxcl10", "Gbp2", "Timp1"],
    "Hox posterior (lumbar ID)": ["Hoxa9", "Hoxb9", "Hoxa10", "Hoxc8", "Hoxc10"],
}


def main():
    d = np.load(CACHE, allow_pickle=True)
    pb, ncell, clcount = d["pb"], d["ncell"], d["clcount"]
    genes, regions, clusters = d["genes"].astype(str), d["regions"].astype(str), d["clusters"].astype(str)
    gi = {g: i for i, g in enumerate(genes)}
    labels = json.load(open(LABELS)) if os.path.exists(LABELS) else {}
    present = ncell >= MIN_CELLS

    # program score per (animal,region): mean of genes, z within-animal across regions
    def prog_region(gs):
        cols = [gi[g] for g in gs if g in gi]
        sub = pb[:, :, cols].mean(2)                     # [animal, region]
        out = np.full(len(regions), np.nan)
        for j in range(len(regions)):
            vals = []
            for ai in range(pb.shape[0]):
                pres = present[ai]
                if pres.sum() < 2 or not pres[j]:
                    continue
                row = sub[ai, pres]
                zr = (sub[ai, j] - row.mean()) / (row.std() + 1e-9)
                vals.append(zr)
            out[j] = np.mean(vals) if vals else np.nan
        return out

    prog = {name: prog_region(gs).tolist() for name, gs in PROGRAMS.items()}

    # composition by lineage along region (mean fraction per region)
    frac = clcount / np.clip(clcount.sum(2, keepdims=True), 1, None)
    lineages = {}
    for k, cl in enumerate(clusters):
        ln = labels.get(cl, "other")
        lineages.setdefault(ln, []).append(k)
    comp = {}
    for ln, ks in lineages.items():
        per_region = []
        for j in range(len(regions)):
            vals = [frac[ai, j, ks].sum() for ai in range(frac.shape[0]) if present[ai, j]]
            per_region.append(float(np.mean(vals)) if vals else np.nan)
        comp[ln] = per_region

    json.dump({"regions": regions.tolist(), "programs": prog, "composition": comp},
              open(os.path.join(OUT, "region_fig_data.json"), "w"), indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] no matplotlib; json only"); return

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(regions))
    colors = plt.cm.tab10(np.linspace(0, 1, len(PROGRAMS)))
    for (name, vals), c in zip(prog.items(), colors):
        axA.plot(x, vals, "-o", lw=2, color=c, label=name)
    axA.set_xticks(x); axA.set_xticklabels(["Lumbar", "Thoracic", "Cervical"])
    axA.axhline(0, c="gray", lw=.5)
    axA.set_ylabel("program score (within-animal z)")
    axA.set_title("Disease programs peak in LUMBAR, decline rostrally\n"
                  "(myelin/positional ID run the other way)", fontsize=11)
    axA.legend(fontsize=7.5, frameon=False)
    axA.spines[["top", "right"]].set_visible(False)

    # composition: ordered lineages, lines by region
    order = sorted(comp, key=lambda l: -np.nanmean(comp[l]))
    order = [l for l in order if np.nanmean(comp[l]) > 0.01][:8]
    width = 0.25
    for j, reg in enumerate(["Lumbar", "Thoracic", "Cervical"]):
        axB.bar(np.arange(len(order)) + (j - 1) * width,
                [comp[l][j] for l in order], width=width, label=reg)
    axB.set_xticks(np.arange(len(order)))
    axB.set_xticklabels(order, rotation=35, ha="right", fontsize=8)
    axB.set_ylabel("mean cell fraction")
    axB.set_title("Cell-type composition by region\n(neurons/oligo gain toward cervical)", fontsize=11)
    axB.legend(fontsize=8, frameon=False)
    axB.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    png = os.path.join(OUT, "figures", "region_programs.png")
    os.makedirs(os.path.dirname(png), exist_ok=True)
    fig.savefig(png, dpi=130, bbox_inches="tight")
    fig.savefig(png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[done] -> {png} + region_fig_data.json")


if __name__ == "__main__":
    main()
