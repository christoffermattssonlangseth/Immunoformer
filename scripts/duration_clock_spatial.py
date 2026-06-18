"""TIER 2 — localize the duration clock in space: which cell types / niches drive it?

The pseudobulk duration clock (scripts/duration_clock.py) shows THAT tissue tracks disease
duration beyond severity and WHICH genes carry it, but not WHERE. This localizes it.

With only n=33 animals (bags), a black-box attention-MIL over ~894k cells would overfit.
The honest decomposition is to split the bag by KNOWN instance type and ask which
compartment carries the signal: build the SAME severity-orthogonalized LOAO clock
independently within each cell type and each spatial niche, and rank by held-out Spearman.

Inputs (cached, no volume needed):
  runs/rr_phase_niche/agg_leiden_1.npz       per (animal x leiden-cluster x gene) pseudobulk + counts
  runs/rr_phase_niche/agg_CellCharter_10.npz per (animal x spatial-niche x gene) pseudobulk + counts
  runs/rr_region_gradient/cluster_labels.json leiden-cluster -> cell-type label
  runs/duration_clock/rr_meta.csv            per-animal day_of_sacrifice / score

Analyses:
  1. CELL-TYPE clocks   : clock B within each cell type (count-weighted over its clusters).
  2. SPATIAL-NICHE clocks: clock B within each populated CellCharter niche (labeled by its
     dominant cell-type composition via profile correlation).
  3. COMPOSITION clock  : do cell-type FRACTIONS alone predict duration? (who-is-there vs
     what-they-express).
  4. PERM NULL on the winners (best cell type, best niche, composition).
  5. MODULE -> COMPARTMENT attribution: where each headline gene is expressed (z by gene).

Outputs: runs/duration_clock_spatial/{results.json, report.txt, figures/}.

    PYTHONPATH="$PWD" python scripts/duration_clock_spatial.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.preprocessing import StandardScaler

from immunotransformer.train import resolve_device  # noqa: F401  (OpenMP guard)

LEIDEN = "runs/rr_phase_niche/agg_leiden_1.npz"
CC = "runs/rr_phase_niche/agg_CellCharter_10.npz"
LABELS = "runs/rr_region_gradient/cluster_labels.json"
META = "runs/duration_clock/rr_meta.csv"
OUT = "runs/duration_clock_spatial"

N_PERM = 50
TOP_VAR = 800
# Leaner than the headline clock (cv=3, single l1_ratio) — we run ~17 compartment clocks,
# and this is a localization scan; the winners get a full permutation null.
ENET_KW = dict(l1_ratio=0.5, cv=3, n_alphas=30, max_iter=20000, n_jobs=None)
MIN_CELLS = 20          # an animal needs >= this many cells in a compartment to be usable
MIN_ANIMALS = 30        # a compartment needs >= this many usable animals to be modeled
# headline clock modules (subset) for the compartment attribution heatmap
MODULE_GENES = {
    "neuro↓": ["Mog", "Mal", "Uchl1", "Gad1", "Slc17a6", "Nptx2"],
    "innate/IFN↓": ["Ccr2", "Tmem173", "Ifit3", "Tmem119", "Siglech", "Irf5"],
    "ECM/scar": ["Col4a2", "Col4a1", "Thbs2", "Serpine2", "Ptx3", "Fbln2", "Eln"],
    "lymphoid": ["Cxcl13", "Cxcl12", "Tcf7", "Mzb1", "Jchain"],
    "lipid": ["Lpl", "Pltp", "Abca8a", "Hmgcr", "Idi1", "Msmo1"],
    "circ/repair": ["Nr1d1", "Dbp", "Bhlhe40", "Igf1", "Il33", "S1pr3"],
}


# ----------------------------------------------------------------- LOAO clock
def _topvar(Xtr, k=TOP_VAR):
    if Xtr.shape[1] <= k:
        return np.arange(Xtr.shape[1])
    return np.sort(np.argsort(-Xtr.var(axis=0))[:k])


def loao_clock(X, y, covar, seed=0):
    """Leave-one-animal-out severity-orthogonalized clock; returns held-out predictions."""
    n = len(y)
    pred = np.full(n, np.nan)
    for i in range(n):
        tr = np.ones(n, bool); tr[i] = False
        Xtr, Xte, ytr = X[tr], X[i:i + 1], y[tr]
        ctr, cte = covar[tr], covar[i:i + 1]
        fx = LinearRegression().fit(ctr, Xtr)
        fy = LinearRegression().fit(ctr, ytr)
        Xtr = Xtr - fx.predict(ctr); Xte = Xte - fx.predict(cte)
        ytr = ytr - fy.predict(ctr)
        sel = _topvar(Xtr)
        Xtr, Xte = Xtr[:, sel], Xte[:, sel]
        sc = StandardScaler().fit(Xtr)
        m = ElasticNetCV(random_state=seed, **ENET_KW).fit(sc.transform(Xtr), ytr)
        pred[i] = m.predict(sc.transform(Xte))[0]
    return pred


def resid(y, covar):
    f = LinearRegression().fit(covar, y)
    return y - f.predict(covar).ravel()


def clock_score(X, day, score):
    """LOAO Spearman + R2 of clock B on compartment matrix X (rows aligned to day/score)."""
    cov = score.reshape(-1, 1)
    pred = loao_clock(X, day, cov)
    dres = resid(day, cov)
    rho = spearmanr(pred, dres).statistic
    r2 = 1 - np.sum((pred - dres) ** 2) / np.sum((dres - dres.mean()) ** 2)
    return float(rho), float(r2), pred, dres


def perm_null(X, day, score, obs_rho):
    cov = score.reshape(-1, 1)
    rng = np.random.default_rng(0)
    n = len(day)
    pr = np.empty(N_PERM)
    for k in range(N_PERM):
        yp = day[rng.permutation(n)]
        pk = loao_clock(X, yp, cov)
        pr[k] = spearmanr(pk, resid(yp, cov)).statistic
    return float((1 + np.sum(pr >= obs_rho)) / (1 + N_PERM)), pr


# ----------------------------------------------------------------- IO
def load():
    labels = json.load(open(LABELS))
    d = np.load(LEIDEN, allow_pickle=True)
    pb, counts, genes = d["pb"].astype(float), d["counts"].astype(float), d["genes"].astype(str)
    clusters, animals = d["clusters"].astype(str), d["animals"].astype(str)
    meta = pd.read_csv(META).set_index("sample_name").reindex(pd.Index(animals))
    day = meta["day_of_sacrifice"].to_numpy(float)
    score = meta["score_sacrifice"].to_numpy(float)
    keep = ~np.isnan(day) & ~np.isnan(score)
    return (pb[keep], counts[keep], genes, clusters, animals[keep], labels,
            day[keep], score[keep])


def celltype_compartments(pb, counts, clusters, labels):
    """Count-weighted aggregate of leiden clusters into cell-type pseudobulk.
    Returns dict name -> (X (n_animals x genes), cells_per_animal)."""
    ct = {}
    names = sorted(set(labels.get(c, "?") for c in clusters))
    for name in names:
        idx = [j for j, c in enumerate(clusters) if labels.get(c, "?") == name]
        w = counts[:, idx]                              # (n, k)
        tot = w.sum(1)                                  # cells per animal
        wn = np.where(tot[:, None] > 0, w / np.where(tot[:, None] == 0, 1, tot[:, None]), 0)
        X = np.einsum("nk,nkg->ng", wn, pb[:, idx, :])  # weighted mean over clusters
        ct[name] = (X, tot)
    return ct, names


def main():
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    pb, counts, genes, clusters, animals, labels, day, score = load()
    n = len(day)
    print(f"[load] {n} animals, {pb.shape[1]} clusters, {pb.shape[2]} genes", flush=True)
    ct, ct_names = celltype_compartments(pb, counts, clusters, labels)

    # --- 1. cell-type clocks ---
    cell_rows = []
    for name in ct_names:
        X, cells = ct[name]
        rho, r2, _, _ = clock_score(X, day, score)
        cell_rows.append({"compartment": name, "kind": "cell_type",
                          "median_cells": float(np.median(cells)),
                          "min_cells": float(cells.min()),
                          "spearman": round(rho, 3), "r2_loao": round(r2, 3)})
        print(f"[celltype] {name:16s} Spearman={rho:+.3f} R2={r2:+.3f} "
              f"(median {np.median(cells):.0f} cells)", flush=True)
    cell_rows.sort(key=lambda r: -r["spearman"])

    # --- 2. composition clock: cell-type fractions -> day ---
    frac = np.column_stack([ct[nm][1] for nm in ct_names])
    frac = frac / frac.sum(1, keepdims=True)
    rhoC, r2C, _, _ = clock_score(frac, day, score)
    print(f"[composition] fractions->day Spearman={rhoC:+.3f} R2={r2C:+.3f}", flush=True)

    # --- 3. spatial-niche clocks (label niches by nearest cell-type profile) ---
    dn = np.load(CC, allow_pickle=True)
    npb, nco = dn["pb"].astype(float), dn["counts"].astype(float)
    ncls = dn["clusters"].astype(str)
    # align niche animals to leiden animals (same source/order assumed; guard by length)
    nan = dn["animals"].astype(str)
    meta = pd.read_csv(META).set_index("sample_name").reindex(pd.Index(nan))
    nday = meta["day_of_sacrifice"].to_numpy(float)
    nscore = meta["score_sacrifice"].to_numpy(float)
    nkeep = ~np.isnan(nday) & ~np.isnan(nscore)
    npb, nco, nday, nscore = npb[nkeep], nco[nkeep], nday[nkeep], nscore[nkeep]
    # mean cell-type profiles for niche labeling
    ct_prof = {nm: ct[nm][0].mean(0) for nm in ct_names}
    niche_rows = []
    for j, nm in enumerate(ncls):
        cells = nco[:, j]
        usable = int((cells >= MIN_CELLS).sum())
        if usable < MIN_ANIMALS:
            print(f"[niche {nm}] SKIP (only {usable} animals >= {MIN_CELLS} cells)", flush=True)
            continue
        Xn = npb[:, j, :]
        prof = Xn.mean(0)
        nearest = max(ct_names, key=lambda c: spearmanr(prof, ct_prof[c]).statistic)
        rho, r2, _, _ = clock_score(Xn, nday, nscore)
        niche_rows.append({"compartment": f"niche{nm} ({nearest})", "kind": "spatial_niche",
                           "median_cells": float(np.median(cells)),
                           "min_cells": float(cells.min()),
                           "spearman": round(rho, 3), "r2_loao": round(r2, 3)})
        print(f"[niche {nm}] ~{nearest:14s} Spearman={rho:+.3f} R2={r2:+.3f}", flush=True)
    niche_rows.sort(key=lambda r: -r["spearman"])

    # --- 4. perm nulls on winners ---
    perms = {}
    best_ct = cell_rows[0]
    p_ct, _ = perm_null(ct[best_ct["compartment"]][0], day, score, best_ct["spearman"])
    perms["best_cell_type"] = {"compartment": best_ct["compartment"],
                               "spearman": best_ct["spearman"], "perm_p": round(p_ct, 4)}
    print(f"[perm] best cell type {best_ct['compartment']}: p={p_ct:.4f}", flush=True)
    p_comp, _ = perm_null(frac, day, score, rhoC)
    perms["composition"] = {"spearman": round(rhoC, 3), "perm_p": round(p_comp, 4)}
    print(f"[perm] composition: p={p_comp:.4f}", flush=True)
    if niche_rows:
        bn = niche_rows[0]
        # recover niche index from label
        nm = bn["compartment"].split()[0].replace("niche", "")
        jj = int(np.where(ncls == nm)[0][0])
        p_n, _ = perm_null(npb[:, jj, :], nday, nscore, bn["spearman"])
        perms["best_niche"] = {"compartment": bn["compartment"],
                               "spearman": bn["spearman"], "perm_p": round(p_n, 4)}
        print(f"[perm] best niche {bn['compartment']}: p={p_n:.4f}", flush=True)

    # --- 5. module -> compartment attribution (z by gene across cell types) ---
    attribution = {}
    ctmat = np.vstack([ct[nm][0].mean(0) for nm in ct_names])     # (n_celltypes, genes)
    gidx = {g: i for i, g in enumerate(genes)}
    for mod, glist in MODULE_GENES.items():
        rows = {}
        for g in glist:
            if g not in gidx:
                continue
            v = ctmat[:, gidx[g]]
            z = (v - v.mean()) / (v.std() + 1e-9)
            rows[g] = {ct_names[k]: round(float(z[k]), 2) for k in range(len(ct_names))}
        attribution[mod] = rows

    results = {
        "n_animals": n, "n_cell_types": len(ct_names), "cell_type_clocks": cell_rows,
        "composition_clock": {"spearman": round(rhoC, 3), "r2_loao": round(r2C, 3)},
        "spatial_niche_clocks": niche_rows, "perm_nulls": perms,
        "whole_tissue_reference": {"clock_B_spearman": 0.799, "fully_controlled": 0.864},
        "module_compartment_attribution": attribution,
        "interpretation": (
            "Per-compartment LOAO clock B Spearman localizes the duration signal. Compartments "
            "with high Spearman carry duration-beyond-severity in their transcriptome; the "
            "composition clock tests whether shifting cell-type proportions alone suffices."),
        "caveats": ("Animal-level pseudobulk per compartment; n=33. Sparse compartments noisy "
                    "(B/plasma, T/NK low in some animals). Niches labeled by nearest cell-type "
                    "profile, not annotated. Per-compartment clocks share the leaned settings; "
                    "only winners have a permutation null."),
    }
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    _report(results)
    _plot(results, ct_names)
    print(f"\n[done] -> {OUT}/", flush=True)


def _report(r):
    L = ["\n=== DURATION CLOCK — Tier 2: spatial localization (RR) ===",
         f"n={r['n_animals']} animals. Whole-tissue clock B Spearman="
         f"{r['whole_tissue_reference']['clock_B_spearman']:+.2f}.\n",
         "Cell-type clocks (duration beyond severity, LOAO):"]
    for row in r["cell_type_clocks"]:
        L.append(f"  {row['compartment']:16s} Spearman={row['spearman']:+.3f}  "
                 f"R2={row['r2_loao']:+.3f}  (median {row['median_cells']:.0f} cells)")
    L.append(f"\nComposition clock (cell-type fractions -> day): "
             f"Spearman={r['composition_clock']['spearman']:+.3f}")
    if r["spatial_niche_clocks"]:
        L.append("\nSpatial-niche clocks (labeled by nearest cell type):")
        for row in r["spatial_niche_clocks"]:
            L.append(f"  {row['compartment']:24s} Spearman={row['spearman']:+.3f}  R2={row['r2_loao']:+.3f}")
    L.append("\nPermutation nulls (winners):")
    for k, v in r["perm_nulls"].items():
        L.append(f"  {k}: {v.get('compartment','')} Spearman={v['spearman']:+.3f} perm_p={v['perm_p']}")
    report = "\n".join(L)
    print(report)
    with open(os.path.join(OUT, "report.txt"), "w") as fh:
        fh.write(report + "\n")


def _plot(r, ct_names):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rows = r["cell_type_clocks"] + r["spatial_niche_clocks"] + [
        {"compartment": "composition (fractions)", "spearman": r["composition_clock"]["spearman"],
         "kind": "composition"}]
    rows = sorted(rows, key=lambda x: x["spearman"])
    labels = [x["compartment"] for x in rows]
    vals = [x["spearman"] for x in rows]
    cmap = {"cell_type": "#3b5bdb", "spatial_niche": "#2f9e44", "composition": "#e8590c"}
    colors = [cmap.get(x.get("kind", "cell_type"), "#888") for x in rows]
    fig, ax = plt.subplots(figsize=(8.2, max(4, 0.42 * len(rows))))
    ax.barh(labels, vals, color=colors, edgecolor="k", linewidth=.4)
    ax.axvline(r["whole_tissue_reference"]["clock_B_spearman"], color="#c0392b", ls="--", lw=1.4,
               label=f"whole-tissue clock B ({r['whole_tissue_reference']['clock_B_spearman']:+.2f})")
    ax.axvline(0, color="k", lw=.6)
    for y, v in enumerate(vals):
        ax.text(v + (0.015 if v >= 0 else -0.015), y, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8.5)
    ax.set_xlabel("LOAO Spearman ρ — duration beyond severity, within compartment")
    ax.set_title("Where the duration clock lives — per-compartment localization", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures", "compartment_localization.png"), dpi=135, bbox_inches="tight")
    plt.close(fig)

    # module -> compartment heatmap
    attr = r["module_compartment_attribution"]
    gene_rows, mod_of = [], []
    for mod, genes in attr.items():
        for g in genes:
            gene_rows.append(g); mod_of.append(mod)
    if gene_rows:
        M = np.array([[attr[m][g][c] for c in ct_names] for m, g in zip(mod_of, gene_rows)])
        fig, ax = plt.subplots(figsize=(1.1 * len(ct_names) + 2, 0.32 * len(gene_rows) + 1.5))
        im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
        ax.set_xticks(range(len(ct_names))); ax.set_xticklabels(ct_names, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(gene_rows)))
        ax.set_yticklabels([f"{g}  ·{m}" for g, m in zip(gene_rows, mod_of)], fontsize=8)
        ax.set_title("Clock genes — expression by cell type (z per gene)", fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.025, label="z")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "figures", "module_compartment_heatmap.png"), dpi=135, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
