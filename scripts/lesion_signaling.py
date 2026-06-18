"""Spatially-informed ligand-receptor signaling wiring of the EAE lesion (RRMAP2).

The lesion atlas (docs/rrmap2-relapse-atlas.md) shows a concentric architecture:
inflammatory-myeloid + demyelinated CORE -> lymphocyte + complement MARGIN -> reactive
astrocyte RIM -> spared parenchyma. This script asks HOW that structure is WIRED:
which cell types sit next to each other in space, and which ligand->receptor axes
connect them. This is the lesion's signaling STRUCTURE; it is NOT a claim beyond
severity (everything co-varies with disease activity -- that question is closed in the
atlas). We are mapping the sender->receiver graph of an established severity-driven state.

Two squidpy analyses on leiden_1 (25 niches), RR cohort (model=="RELAPSE REMITTING"):

  1. nhood_enrichment -- cluster x cluster spatial-adjacency z-scores on a 6-NN
     spatial graph built PER SECTION (meta_sample_id) and stitched block-diagonal,
     so there are genuinely 0 cross-section edges.
     NOTE: the file's PRECOMPUTED obsp['spatial_connectivities'] is NOT intra-section
     here -- 14.6% of its edges cross meta_sample_id boundaries (sections share the
     same xy coordinate frame, so a global k-NN linked cells across sections; cross
     edges have median length 7.9um, i.e. genuinely-close-in-frame but biologically
     different sections). Using it would fabricate adjacency between cell types that
     are never physically near each other. We therefore rebuild per section.

  2. ligrec -- permutation LR test (CellPhoneDB-style) on a curated panel covering the
     atlas axes (recruitment Ccl2-Ccr2 / Cxcl10/9-Cxcr3; lymphoid Cxcl13-Cxcr5,
     Cxcl12-Cxcr4; complement C3-C3ar1/C5ar1; costim Cd80/86-Ctla4/Cd28; Tnf, Trem2).
     The full panel ligrec on 894k cells is heavy; we run on a stratified subsample
     (see SUBSAMPLE) for speed and report exactly what was sampled.

Then INTEGRATE: for the strongest spatially-adjacent cluster pairs, name the LR axes
whose ligand is high in the (sender) cluster, receptor high in the (receiver) cluster,
AND the two clusters are spatially adjacent. That sender->receiver wiring is the deliverable.

    PYTHONPATH="$PWD" python scripts/lesion_signaling.py

Outputs -> runs/lesion_signaling/ (figures/, *.csv, results.json; large cache gitignored).
"""
from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd

from immunotransformer.train import resolve_device  # noqa: F401  (OpenMP guard)

warnings.filterwarnings("ignore")

RRMAP2 = ("/Volumes/moldiassd/RRMAP2_xenium_adata/kmeans_separated/"
          "RRMAP2_xenium_all_samples.cellcharter.companion.ready.with_metadata.rerun.h5ad")
OUT = "runs/lesion_signaling"
CACHE = os.path.join(OUT, "rr_subset.h5ad")

# leiden_1 lineage labels (runs/rr_region_gradient/cluster_labels.json)
LABELS = {"4": "neuron", "14": "neuron", "6": "neuron", "23": "neuron",
          "13": "T/NK", "3": "infl-myeloid", "5": "infl-myeloid", "18": "infl-myeloid",
          "7": "vascular", "10": "vascular", "19": "vascular", "15": "vascular",
          "1": "oligo", "12": "oligo", "2": "oligo", "22": "oligo", "17": "oligo",
          "8": "astrocyte", "9": "astrocyte", "24": "ependymal",
          "0": "microglia", "20": "microglia", "11": "OPC", "16": "OPC",
          "21": "Bcell/plasma"}

# lesion cell types of interest (core myeloid -> T -> B -> astro rim)
LESION = {"5": "myeloid-acute(5)", "3": "myeloid-repair(3)", "18": "myeloid-APC/IFN(18)",
          "13": "T/NK(13)", "21": "B/plasma(21)", "9": "astro-reactive(9)", "8": "astro(8)",
          "0": "microglia(0)", "20": "microglia(20)"}

# Curated LR panel covering the atlas-established present axes. Both partners on panel.
LR_AXES = [
    ("Ccl2", "Ccr2"),        # monocyte recruitment
    ("Ccl5", "Ccr5"),        # T/myeloid recruitment
    ("Cxcl10", "Cxcr3"),     # IFN-driven T recruitment
    ("Cxcl9", "Cxcr3"),
    ("Cxcl13", "Cxcr5"),     # B/lymphoid follicle organization
    ("Cxcl12", "Cxcr4"),     # stromal homing
    ("C3", "C3ar1"),         # complement
    ("C3", "C5ar1"),
    ("Cd80", "Ctla4"),       # costimulation
    ("Cd86", "Ctla4"),
    ("Cd80", "Cd28"),
    ("Cd86", "Cd28"),
    ("Tnf", "Tnfrsf1a"),     # TNF
    ("Tnf", "Tnfrsf1b"),
    ("Csf1", "Csf1r"),       # myeloid maintenance
    ("Il34", "Csf1r"),
    ("Trem2", "Tyrobp"),     # DAM (Tyrobp may be absent -> dropped if so)
    ("Cxcl16", "Cxcr6"),     # tissue-resident T
    ("Ccl19", "Ccr7"),       # lymphoid homing
    ("Ccl21a", "Ccr7"),
    ("Igf1", "Igf1r"),       # repair-myeloid trophic
]

SUBSAMPLE = 150_000   # stratified by cluster for ligrec
N_PERMS = 1000
SEED = 0


def build_section_graph(a):
    """6-NN spatial graph built PER section, stitched block-diagonal (0 cross edges).

    Overwrites obsp['spatial_connectivities']/['spatial_distances'] so squidpy uses it.
    """
    import squidpy as sq
    from scipy.sparse import coo_matrix
    sec = a.obs["meta_sample_id"].astype(str).to_numpy()
    n = a.n_obs
    rows, cols, cvals, dvals = [], [], [], []
    for s in pd.unique(sec):
        idx = np.where(sec == s)[0]
        if len(idx) < 7:
            continue
        sub = a[idx].copy()
        sq.gr.spatial_neighbors(sub, coord_type="generic", n_neighs=6)
        gc = sub.obsp["spatial_connectivities"].tocsr()
        gd = sub.obsp["spatial_distances"].tocsr()
        gc_coo = gc.tocoo()
        # read distances at the exact connectivity positions (aligned)
        d_at = np.asarray(gd[gc_coo.row, gc_coo.col]).ravel()
        rows.append(idx[gc_coo.row]); cols.append(idx[gc_coo.col])
        cvals.append(gc_coo.data); dvals.append(d_at)
    rows = np.concatenate(rows); cols = np.concatenate(cols)
    cvals = np.concatenate(cvals); dvals = np.concatenate(dvals)
    a.obsp["spatial_connectivities"] = coo_matrix((cvals, (rows, cols)), shape=(n, n)).tocsr()
    a.obsp["spatial_distances"] = coo_matrix((dvals, (rows, cols)), shape=(n, n)).tocsr()
    a.uns["spatial_neighbors"] = {
        "connectivities_key": "spatial_connectivities",
        "distances_key": "spatial_distances",
        "params": {"coord_type": "generic", "n_neighbors": 6, "radius": None},
    }
    G = a.obsp["spatial_connectivities"].tocoo()
    cross = int((sec[G.row] != sec[G.col]).sum())
    print(f"       per-section graph: {G.nnz:,} edges, {cross} cross-section "
          f"(must be 0), avg deg {G.nnz / a.n_obs:.2f}")
    assert cross == 0, "cross-section edges leaked into per-section graph"
    return a


def load_rr():
    """RR subset with a freshly built PER-SECTION spatial graph (0 cross edges)."""
    import anndata as ad
    if os.path.exists(CACHE):
        print(f"[cache] {CACHE}")
        return ad.read_h5ad(CACHE)
    print(f"[load] {RRMAP2}")
    a = ad.read_h5ad(RRMAP2)
    a = a[a.obs["model"] == "RELAPSE REMITTING"].copy()
    print(f"       RR: {a.n_obs:,} cells x {a.n_vars} genes")
    a.obs["leiden_1"] = a.obs["leiden_1"].astype("category")
    a = build_section_graph(a)
    os.makedirs(OUT, exist_ok=True)
    a.write_h5ad(CACHE)
    return a


def nhood(a):
    import squidpy as sq
    print("[nhood] nhood_enrichment on leiden_1 (per-section graph)")
    sq.gr.nhood_enrichment(a, cluster_key="leiden_1", seed=SEED, n_perms=N_PERMS,
                           show_progress_bar=False)
    z = a.uns["leiden_1_nhood_enrichment"]["zscore"]
    cats = list(a.obs["leiden_1"].cat.categories)
    zdf = pd.DataFrame(z, index=cats, columns=cats)
    zdf.to_csv(os.path.join(OUT, "nhood_zscore.csv"))
    return zdf, cats


def label_axis(cats):
    return [f"{c}:{LABELS.get(c, '?')}" for c in cats]


def nhood_fig(zdf, cats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # order clusters by lineage for readability
    order = sorted(cats, key=lambda c: (LABELS.get(c, "z"), int(c)))
    Z = zdf.loc[order, order]
    vmax = np.nanpercentile(np.abs(Z.values), 98)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(Z.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    lab = [f"{c}:{LABELS.get(c, '?')}" for c in order]
    ax.set_xticks(range(len(order))); ax.set_xticklabels(lab, rotation=90, fontsize=7)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(lab, fontsize=7)
    ax.set_title("Spatial neighborhood enrichment (z-score), RR EAE — leiden_1\n"
                 "per-section 6-NN graph (0 cross-section edges), 1000 perms", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.6, label="z-score")
    fig.tight_layout()
    p = os.path.join(OUT, "figures", "nhood_enrichment.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"[fig] {p}")


def test_radial_model(zdf):
    """Test the concrete radial-architecture predictions with adjacency z-scores."""
    def z(a, b):
        return float(zdf.loc[a, b])
    tests = {
        "astro-rim(9) ~ myeloid-core(5)": z("9", "5"),
        "astro-rim(9) ~ myeloid-repair(3)": z("9", "3"),
        "astro(8) ~ myeloid-core(5)": z("8", "5"),
        "T/NK(13) ~ myeloid-APC/IFN(18)": z("13", "18"),
        "T/NK(13) ~ myeloid-acute(5)": z("13", "5"),
        "T/NK(13) ~ myeloid-repair(3)": z("13", "3"),
        "B/plasma(21) self-clustering": z("21", "21"),
        "B/plasma(21) ~ T/NK(13)": z("21", "13"),
        "B/plasma(21) ~ myeloid-APC/IFN(18)": z("21", "18"),
        "myeloid-acute(5) ~ myeloid-repair(3)": z("5", "3"),
        "myeloid-acute(5) ~ myeloid-APC(18)": z("5", "18"),
        "microglia(0) ~ myeloid-acute(5)": z("0", "5"),
        "astro-rim(9) ~ T/NK(13)": z("9", "13"),
    }
    return tests


def mean_expr_per_cluster(a, genes):
    """log-norm mean expression of genes per leiden_1 cluster (for sender/receiver)."""
    from scipy.sparse import issparse
    present = [g for g in genes if g in a.var_names]
    X = a[:, present].X
    X = np.asarray(X.todense()) if issparse(X) else np.asarray(X)
    df = pd.DataFrame(X, columns=present)
    df["cl"] = a.obs["leiden_1"].astype(str).to_numpy()
    m = df.groupby("cl").mean()
    return m  # rows=cluster, cols=gene


def run_ligrec(a):
    import squidpy as sq
    rng = np.random.default_rng(SEED)
    # stratified subsample by cluster
    idx = []
    cl = a.obs["leiden_1"].astype(str).to_numpy()
    frac = min(1.0, SUBSAMPLE / a.n_obs)
    for c in np.unique(cl):
        ci = np.where(cl == c)[0]
        take = max(1, int(round(len(ci) * frac)))
        idx.append(rng.choice(ci, size=min(take, len(ci)), replace=False))
    idx = np.sort(np.concatenate(idx))
    sub = a[idx].copy()
    print(f"[ligrec] subsample {sub.n_obs:,} cells "
          f"(stratified ~{frac:.1%} per cluster)")

    # restrict to curated interactions whose BOTH partners are on the panel
    inter = [(l, r) for (l, r) in LR_AXES
             if l in a.var_names and r in a.var_names]
    dropped = [(l, r) for (l, r) in LR_AXES
               if l not in a.var_names or r not in a.var_names]
    print(f"[ligrec] {len(inter)} curated LR pairs present; dropped {dropped}")
    interactions = pd.DataFrame(inter, columns=["source", "target"])

    res = sq.gr.ligrec(
        sub, cluster_key="leiden_1", interactions=interactions,
        n_perms=N_PERMS, seed=SEED, threshold=0.01, use_raw=False,
        corr_method="fdr_bh", show_progress_bar=False, copy=True,
    )
    return res, inter, sub.n_obs


def summarize_ligrec(res, inter, expr_mean):
    """Flatten ligrec into significant lesion-relevant sender->receiver rows."""
    pvals = res["pvalues"]            # MultiIndex rows (source,target gene), cols (cl_s, cl_r)
    means = res["means"]
    lesion_cls = set(LESION)
    # ligrec uppercases gene symbols; map back to expr_mean's original casing
    gmap = {g.upper(): g for g in expr_mean.columns}
    rows = []
    for (lg, rg) in pvals.index:
        for (cs, cr) in pvals.columns:
            if cs not in lesion_cls or cr not in lesion_cls:
                continue
            p = pvals.loc[(lg, rg), (cs, cr)]
            if pd.isna(p):
                continue
            m = means.loc[(lg, rg), (cs, cr)]
            lg0, rg0 = gmap.get(lg.upper()), gmap.get(rg.upper())
            rows.append({
                "ligand": lg, "receptor": rg,
                "sender": cs, "sender_label": LESION.get(cs, cs),
                "receiver": cr, "receiver_label": LESION.get(cr, cr),
                "pval": float(p), "mean_expr": float(m) if pd.notna(m) else np.nan,
                "lig_in_sender": float(expr_mean.loc[cs, lg0]) if lg0 else np.nan,
                "rec_in_receiver": float(expr_mean.loc[cr, rg0]) if rg0 else np.nan,
            })
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["pval", "mean_expr"], ascending=[True, False])
    return df


def integrate(zdf, lr_df, adj_thresh=2.0, cross_only=True):
    """Wiring: significant LR axes between spatially-adjacent lesion cluster pairs.

    cross_only drops sender==receiver self-edges (trivially most-adjacent and
    uninformative about INTER-cell-type wiring); the deliverable is the cross-type graph.
    """
    if not len(lr_df):
        return pd.DataFrame()
    sig = lr_df[lr_df["pval"] <= 0.05].copy()
    if cross_only:
        sig = sig[sig["sender"] != sig["receiver"]].copy()
    adj = [float(zdf.loc[r["sender"], r["receiver"]]) for _, r in sig.iterrows()]
    sig["adjacency_z"] = adj
    wired = sig[sig["adjacency_z"] >= adj_thresh].copy()
    # rank by interaction strength (mean of ligand+receptor expr) among adjacent pairs
    wired = wired.sort_values(["mean_expr", "adjacency_z"], ascending=[False, False])
    return wired


def main():
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    a = load_rr()

    # 1. spatial adjacency
    zdf, cats = nhood(a)
    nhood_fig(zdf, cats)
    radial = test_radial_model(zdf)
    print("\n[radial-model tests] adjacency z-scores:")
    for k, v in radial.items():
        print(f"   {v:+7.2f}  {k}")

    # expression per cluster for sender/receiver direction
    lr_genes = sorted({g for pair in LR_AXES for g in pair})
    expr_mean = mean_expr_per_cluster(a, lr_genes)

    # 2. ligand-receptor
    res, inter, n_sub = run_ligrec(a)
    lr_df = summarize_ligrec(res, inter, expr_mean)
    lr_df.to_csv(os.path.join(OUT, "ligrec_lesion_pairs.csv"), index=False)
    n_sig = int((lr_df["pval"] <= 0.05).sum()) if len(lr_df) else 0
    print(f"\n[ligrec] {n_sig} significant (p<=0.05) lesion-pair LR interactions")

    # 3. integrate -> wiring
    wired = integrate(zdf, lr_df)
    wired.to_csv(os.path.join(OUT, "wiring_axes.csv"), index=False)
    print(f"\n[wiring] {len(wired)} LR axes between spatially-adjacent (z>=2) lesion clusters:")
    for _, r in wired.head(30).iterrows():
        print(f"   {r['ligand']:>7}->{r['receptor']:<9} "
              f"{r['sender_label']:>18} -> {r['receiver_label']:<18} "
              f"p={r['pval']:.3g} adj_z={r['adjacency_z']:+.1f}")

    # 4. wiring dotplot
    wiring_fig(wired)

    # results.json summary
    summary = {
        "n_cells_rr": int(a.n_obs),
        "n_cells_ligrec_subsample": int(n_sub),
        "n_perms": N_PERMS,
        "lr_pairs_tested": [f"{l}-{r}" for l, r in inter],
        "lr_pairs_dropped_absent": [f"{l}-{r}" for l, r in LR_AXES
                                    if l not in a.var_names or r not in a.var_names],
        "radial_model_adjacency_z": radial,
        "n_significant_lesion_lr": n_sig,
        "n_wired_axes": int(len(wired)),
        "top_wired_axes": (wired.head(20)[["ligand", "receptor", "sender_label",
                           "receiver_label", "pval", "adjacency_z"]].to_dict("records")
                           if len(wired) else []),
    }
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] -> {OUT}/results.json")


def wiring_fig(wired):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not len(wired):
        return
    d = wired.head(25).iloc[::-1]
    labels = [f"{r.ligand}->{r.receptor}: {r.sender_label}->{r.receiver_label}"
              for r in d.itertuples()]
    y = np.arange(len(d))
    neglogp = -np.log10(np.clip(d["pval"].values, 1e-4, None))
    sizes = 40 + 200 * (d["adjacency_z"].clip(0) / max(1, d["adjacency_z"].max()))
    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(d))))
    sc = ax.scatter(neglogp, y, s=sizes, c=d["adjacency_z"].values,
                    cmap="viridis", edgecolor="k", linewidth=0.4)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("-log10(ligrec permutation p)")
    ax.set_title("Lesion signaling wiring: LR axes between spatially-adjacent cell types\n"
                 "(dot size & color = spatial adjacency z-score)", fontsize=10)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="adjacency z")
    fig.tight_layout()
    p = os.path.join(OUT, "figures", "wiring_axes.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"[fig] {p}")


if __name__ == "__main__":
    main()
