"""RRMAP2 atlas — the spatial L->T->C region gradient, WITHIN animals, and whether
it recapitulates the temporal relapse cycle.

Region is a WITHIN-animal axis: 32/33 RR animals span all of lumbar/thoracic/cervical
(~9k cells each). Centering each gene by its animal mean removes every animal-level
confound (severity, batch, strain) — so the region gradient is far cleaner and
better powered than the cross-sectional stage trajectory.

Outputs:
  1. Within-animal region slope per gene (rank L=0,T=1,C=2; per-animal slope ->
     one-sample t across animals; BH-FDR). Top genes UP and DOWN along L->T->C.
  2. CROSS-AXIS concordance: correlate the region slope with the temporal cycle
     trend (mono_rho from rr_cycle_oscillation). Do space and time resolve the SAME
     programs in the SAME direction?
  3. Cell-type composition along region: which clusters expand L->T->C.
  4. Hal and the two named programs (acute oscillators, cumulative ratchet) along
     the region axis.

    python scripts/rr_region_gradient.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import ttest_1samp, spearmanr

from immunotransformer.train import resolve_device  # noqa: F401  (OpenMP guard)

RRMAP2 = ("/Volumes/moldiassd/RRMAP2_xenium_adata/kmeans_separated/"
          "RRMAP2_xenium_all_samples.cellcharter.companion.ready.with_metadata.rerun.h5ad")
OUT_DIR = "runs/rr_region_gradient"
CACHE = os.path.join(OUT_DIR, "agg_region.npz")
REGION_RANK = {"L": 0, "T": 1, "C": 2}
CLUSTER_COL = "leiden_1"
MIN_CELLS = 50


def bh_fdr(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = p[o] * n / (np.arange(n) + 1)
    q[o] = np.minimum.accumulate(q[o][::-1])[::-1]
    return np.clip(q, 0, 1)


def build_or_load():
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        print(f"[cache] {CACHE}")
        return (d["pb"], d["ncell"], d["clcount"], d["genes"].astype(str),
                d["animals"].astype(str), d["regions"].astype(str), d["clusters"].astype(str))
    import anndata as ad
    print(f"[load] {RRMAP2}")
    a = ad.read_h5ad(RRMAP2)
    a = a[a.obs["model"] == "RELAPSE REMITTING"].copy()
    o = a.obs
    print(f"       RR: {a.n_obs:,} cells")
    animals = list(pd.unique(o["sample_name"]))
    regions = ["L", "T", "C"]
    clusters = [str(c) for c in pd.unique(o[CLUSTER_COL])]
    a_code = {x: i for i, x in enumerate(animals)}
    r_code = {x: i for i, x in enumerate(regions)}
    c_code = {x: i for i, x in enumerate(clusters)}
    ac = o["sample_name"].map(a_code).to_numpy()
    rc = o["region"].map(r_code).to_numpy()
    cc = o[CLUSTER_COL].astype(str).map(c_code).to_numpy()
    nA, nR, nC = len(animals), len(regions), len(clusters)
    valid = ~np.isnan(rc)
    counts = a.layers["counts"]
    if not sp.issparse(counts):
        counts = sp.csr_matrix(counts)

    # per (animal,region) pseudobulk
    grp = (ac * nR + rc).astype(int)
    G = sp.csr_matrix((valid.astype(float), (np.where(valid, grp, 0), np.arange(a.n_obs))),
                      shape=(nA * nR, a.n_obs))
    S = np.asarray((G @ counts).todense())
    ncell = np.asarray(G.sum(1)).ravel()
    lib = S.sum(1, keepdims=True); lib[lib == 0] = 1.0
    pb = np.log1p(S / lib * 1e4).reshape(nA, nR, -1)
    ncell = ncell.reshape(nA, nR)

    # per (animal,region,cluster) cell counts for composition
    grp2 = ((ac * nR + rc) * nC + cc).astype(int)
    G2 = sp.csr_matrix((valid.astype(float), (np.where(valid, grp2, 0), np.arange(a.n_obs))),
                       shape=(nA * nR * nC, a.n_obs))
    clcount = np.asarray(G2.sum(1)).ravel().reshape(nA, nR, nC)

    genes = np.array(a.var_names, dtype=str)
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(CACHE, pb=pb, ncell=ncell, clcount=clcount, genes=genes,
             animals=np.array(animals), regions=np.array(regions), clusters=np.array(clusters))
    print(f"[cache] wrote {CACHE}")
    return pb, ncell, clcount, genes, np.array(animals), np.array(regions), np.array(clusters)


def within_animal_slope(pb, ncell, regions):
    """Per-gene within-animal region slope (rank L,T,C) -> one-sample t across animals."""
    ranks = np.array([REGION_RANK[r] for r in regions], float)
    nA, nR, nG = pb.shape
    slopes = np.full((nA, nG), np.nan)
    for ai in range(nA):
        present = ncell[ai] >= MIN_CELLS
        if present.sum() < 2:
            continue
        x = ranks[present]
        xc = x - x.mean()
        denom = (xc ** 2).sum()
        Y = pb[ai, present, :]                     # [r_present, genes]
        Yc = Y - Y.mean(0, keepdims=True)
        slopes[ai] = (xc[:, None] * Yc).sum(0) / denom
    valid = ~np.isnan(slopes).all(0)
    mean_slope = np.nanmean(slopes, axis=0)
    t, p = ttest_1samp(slopes, 0.0, axis=0, nan_policy="omit")
    p = np.where(np.isfinite(p), p, 1.0)
    n_used = (~np.isnan(slopes)).sum(0)
    return mean_slope, np.asarray(p), n_used


def main():
    os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)
    pb, ncell, clcount, genes, animals, regions, clusters = build_or_load()
    gi = {g: i for i, g in enumerate(genes)}
    n_span = int((( ncell >= MIN_CELLS).sum(1) >= 2).sum())
    print(f"[setup] {pb.shape[0]} RR animals; {n_span} span >=2 regions; "
          f"{int(((ncell>=MIN_CELLS).sum(1)==3).sum())} span all 3")

    mean_slope, p, n_used = within_animal_slope(pb, ncell, regions)
    q = bh_fdr(p)
    df = pd.DataFrame({"gene": genes, "region_slope": mean_slope, "p": p, "q": q})
    df = df[df["region_slope"].notna()].copy()
    n_sig = int((df["q"] < 0.05).sum())

    up = df.sort_values("region_slope", ascending=False).head(20)
    down = df.sort_values("region_slope").head(20)

    # ---- cross-axis concordance with the temporal cycle ----
    cross = None
    osc_csv = "runs/rr_cycle_oscillation/gene_oscillation_metrics.csv"
    if os.path.exists(osc_csv):
        cyc = pd.read_csv(osc_csv)[["gene", "mono_rho", "amplitude", "floor_drift"]]
        mg = df.merge(cyc, on="gene", how="inner")
        rho_all = spearmanr(mg["region_slope"], mg["mono_rho"]).correlation
        # restrict to genes that actually move on either axis (signal, not noise floor)
        movers = mg[(mg["q"] < 0.05) | (mg["mono_rho"].abs() > 0.5)]
        rho_movers = spearmanr(movers["region_slope"], movers["mono_rho"]).correlation
        cross = {"spearman_all_genes": round(float(rho_all), 3),
                 "spearman_movers": round(float(rho_movers), 3),
                 "n_movers": int(len(movers))}
        mg.to_csv(os.path.join(OUT_DIR, "region_vs_cycle.csv"), index=False)

    # ---- composition along region ----
    frac = clcount / np.clip(clcount.sum(2, keepdims=True), 1, None)   # [animal,region,cluster]
    comp_slope = np.full(len(clusters), np.nan)
    comp_p = np.full(len(clusters), np.nan)
    ranks = np.array([REGION_RANK[r] for r in regions], float)
    for k in range(len(clusters)):
        sl = []
        for ai in range(frac.shape[0]):
            present = ncell[ai] >= MIN_CELLS
            if present.sum() < 2:
                continue
            x = ranks[present]; xc = x - x.mean()
            y = frac[ai, present, k]; yc = y - y.mean()
            sl.append((xc * yc).sum() / (xc ** 2).sum())
        if len(sl) >= 3:
            comp_slope[k] = np.mean(sl)
            comp_p[k] = ttest_1samp(sl, 0.0).pvalue
    comp = pd.DataFrame({"cluster": clusters, "frac_slope_LtoC": comp_slope, "p": comp_p}).dropna()
    comp["q"] = bh_fdr(comp["p"].to_numpy())
    comp = comp.sort_values("frac_slope_LtoC", ascending=False)

    # ---- Hal + named programs along region ----
    def region_profile(gene):
        m = ncell >= MIN_CELLS
        return {r: round(float(np.nanmean(np.where(m[:, j], pb[:, j, gi[gene]], np.nan))), 3)
                for j, r in enumerate(regions)}
    hal_prof = region_profile("Hal")

    results = {
        "n_animals": int(pb.shape[0]), "n_span_ge2": n_span,
        "n_genes_q05_region": n_sig,
        "region_up_LtoC": up[["gene", "region_slope", "q"]].round(4).to_dict("records"),
        "region_down_LtoC": down[["gene", "region_slope", "q"]].round(4).to_dict("records"),
        "cross_axis": cross,
        "composition_expand_LtoC": comp.head(8).round(4).to_dict("records"),
        "hal_region_profile": hal_prof,
    }
    with open(os.path.join(OUT_DIR, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    df.sort_values("region_slope", ascending=False).to_csv(
        os.path.join(OUT_DIR, "region_slopes.csv"), index=False)

    # ---- report ----
    L = [f"\n=== RRMAP2 region gradient L->T->C (within-animal, n={n_span} animals span >=2 regions) ===",
         f"Within-animal centering removes severity/batch/strain. {n_sig} genes at BH-q<0.05.",
         "\n-- UP along L->T->C (higher cervical) --"]
    for r in results["region_up_LtoC"][:12]:
        L.append(f"   {r['gene']:<14} slope={r['region_slope']:+.3f}  q={r['q']:.2g}")
    L.append("-- DOWN along L->T->C (higher lumbar) --")
    for r in results["region_down_LtoC"][:12]:
        L.append(f"   {r['gene']:<14} slope={r['region_slope']:+.3f}  q={r['q']:.2g}")
    if cross:
        L.append(f"\n-- CROSS-AXIS: does space recapitulate time? --")
        L.append(f"   Spearman(region_slope, cycle_rho) all genes = {cross['spearman_all_genes']:+.3f}")
        L.append(f"   among {cross['n_movers']} movers = {cross['spearman_movers']:+.3f}")
        L.append("   (positive => same programs rise on BOTH the spatial and temporal axes)")
    L.append(f"\n-- Cell-type fractions expanding L->T->C (top) --")
    for r in results["composition_expand_LtoC"][:6]:
        L.append(f"   cluster {r['cluster']:<4} frac_slope={r['frac_slope_LtoC']:+.4f}  q={r['q']:.2g}")
    L.append(f"\n-- Hal along region --  " + "  ".join(f"{r}={hal_prof[r]:+.2f}" for r in regions))
    report = "\n".join(L)
    print(report)
    with open(os.path.join(OUT_DIR, "report.txt"), "w") as fh:
        fh.write(report + "\n")

    _plot(df, cross, OUT_DIR)
    print(f"\n[done] -> {OUT_DIR}/ (results.json, region_slopes.csv, region_vs_cycle.csv, figures/)")


def _plot(df, cross, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not os.path.exists("runs/rr_cycle_oscillation/gene_oscillation_metrics.csv"):
        return
    cyc = pd.read_csv("runs/rr_cycle_oscillation/gene_oscillation_metrics.csv")[["gene", "mono_rho"]]
    mg = df.merge(cyc, on="gene", how="inner")
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(mg["region_slope"], mg["mono_rho"], s=6, c="#c9ccd1")
    sig = mg[mg["q"] < 0.05]
    ax.scatter(sig["region_slope"], sig["mono_rho"], s=10, c="#3b5bdb", label="region q<0.05")
    label = ["C6", "Fcrls", "Igf1", "Gpnmb", "Cd74", "Hal", "Arg1", "Chil3",
             "Hmgcr", "Msmo1", "Mog", "Plp1", "Mbp", "Cx3cr1"]
    for _, r in mg[mg["gene"].isin(label)].iterrows():
        ax.scatter(r["region_slope"], r["mono_rho"], s=24, c="#c0392b", zorder=3)
        ax.annotate(r["gene"], (r["region_slope"], r["mono_rho"]), fontsize=7)
    ax.axhline(0, c="gray", lw=.5); ax.axvline(0, c="gray", lw=.5)
    ttl = "Space vs time: region gradient vs relapse-cycle trend"
    if cross:
        ttl += f"\nSpearman(movers) = {cross['spearman_movers']:+.2f}"
    ax.set_xlabel("region slope  (L → T → C)")
    ax.set_ylabel("relapse-cycle trend (rho)")
    ax.set_title(ttl, fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figures", "region_vs_cycle.png"), dpi=130, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "figures", "region_vs_cycle.pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
