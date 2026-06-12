"""Within-RR relapse biology — descriptive, animal-level (NOT a trained model).

RR-vs-chronic is uninterpretable (model is 100% confounded with slide+strain).
Within the RR cohort, by contrast, stage is spread across slides (each stage on
9-12 slides; slides mix stages), so these contrasts are not batch artifacts.

Two analyses, both at the ANIMAL level (the unit of independence), n is small so
treat as exploratory:

  1. Relapse-cycle TRAJECTORY (primary). Order the RR disease stages along the
     relapse cycle and find cell populations (leiden clusters) and genes whose
     animal-level pseudobulk trends monotonically along it (Spearman vs cycle
     rank, BH-FDR). Cycle order assumed:
        PLP CFA < ONSET1 < PEAK1 < REMISSION1 < ONSET2 < PEAK2 < REMISSION2 < PEAK3
     (edit CYCLE_ORDER if your stage naming differs). MONOPHASIC is a separate
     course, excluded from the trajectory.

  2. MONOPHASIC vs RELAPSED (secondary, underpowered: 4 vs 15 animals).
     Mann-Whitney per cluster / per gene. Caveat: monophasic animals are all on
     the earlier run date, so residual batch can't be fully excluded.

    python scripts/rr_within_relapse.py
    python scripts/rr_within_relapse.py --cluster-col CellCharter_10 --top 25
"""

from __future__ import annotations

import argparse
import json
import os

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.stats import mannwhitneyu, rankdata, spearmanr
from scipy.stats import t as tdist

from immunotransformer.train import resolve_device  # noqa: F401  (imports package OpenMP guard)

RRMAP2 = (
    "/Volumes/moldiassd/RRMAP2_xenium_adata/kmeans_separated/"
    "RRMAP2_xenium_all_samples.cellcharter.companion.ready.with_metadata.rerun.h5ad"
)
CYCLE_ORDER = ["PLP CFA", "ONSET1", "PEAK1", "REMISSION1",
               "ONSET2", "PEAK2", "REMISSION2", "PEAK3"]
RELAPSED = {"ONSET2", "REMISSION2", "PEAK2", "PEAK3"}  # evidence of a 2nd attack


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    q[order] = (p[order] * n) / (np.arange(n) + 1)
    # enforce monotonicity
    q[order] = np.minimum.accumulate(q[order][::-1])[::-1]
    return np.clip(q, 0, 1)


def partial_trend(pb, names, mask, cyc, score):
    """Partial Spearman of each gene vs cycle position, controlling for severity.

    Removes the linear (rank) effect of score_sacrifice from both gene and cycle
    rank, so a surviving partial_rho means a cycle-specific signal beyond acute
    severity. Returns (df sorted by partial_rho, r_cycle_vs_severity, n_used).
    """
    keep = mask & ~np.isnan(score)
    sub = pb[keep]                                   # [m, g]
    m = sub.shape[0]
    rr, rs = rankdata(cyc[keep]), rankdata(score[keep])

    def zc(v):
        v = v - v.mean(); s = v.std()
        return v / s if s > 0 else np.zeros_like(v)

    zr, zs = zc(rr), zc(rs)
    r_rs = float(zr @ zs / m)                         # cycle vs severity (rank)
    gr = np.apply_along_axis(rankdata, 0, sub)        # rank genes across animals
    zx = (gr - gr.mean(0)) / (gr.std(0) + 1e-12)
    r_xr = (zx.T @ zr) / m
    r_xs = (zx.T @ zs) / m
    denom = np.sqrt(np.clip((1 - r_xs ** 2) * (1 - r_rs ** 2), 1e-12, None))
    pr = (r_xr - r_xs * r_rs) / denom
    df = m - 3
    tstat = pr * np.sqrt(df / np.clip(1 - pr ** 2, 1e-12, None))
    pval = 2 * tdist.sf(np.abs(tstat), df)
    out = pd.DataFrame({"name": names, "partial_rho": pr, "raw_p": pval})
    out = out[sub.std(0) > 0].copy()
    out["q"] = bh_fdr(out["raw_p"].to_numpy())
    return out.sort_values("partial_rho"), r_rs, m


def pseudobulk(counts, codes, n_groups):
    """Sum counts rows by integer group code -> [n_groups, n_genes], then CPM+log1p."""
    g = counts.shape[1]
    pb = np.zeros((n_groups, g), dtype=np.float64)
    for k in range(n_groups):
        rows = counts[codes == k]
        s = np.asarray(rows.sum(axis=0)).ravel() if issparse(rows) else rows.sum(axis=0)
        pb[k] = s
    libsize = pb.sum(axis=1, keepdims=True)
    libsize[libsize == 0] = 1.0
    return np.log1p(pb / libsize * 1e4)  # log CP10k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", default=RRMAP2)
    ap.add_argument("--cluster-col", default="leiden_1")
    ap.add_argument("--layer", default="counts")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out-dir", default="runs/rr_within_relapse")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out_dir, "figures"), exist_ok=True)

    print(f"[load] {args.h5ad}")
    adata = ad.read_h5ad(args.h5ad)
    adata = adata[adata.obs["model"] == "RELAPSE REMITTING"].copy()
    print(f"       RR subset: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    o = adata.obs
    counts = adata.layers[args.layer] if args.layer else adata.X
    genes = np.array(adata.var_names, dtype=str)

    # animal-level frame
    animals = list(pd.unique(o["sample_name"]))
    a_code = {a: i for i, a in enumerate(animals)}
    codes = o["sample_name"].map(a_code).to_numpy()
    a_meta = o.drop_duplicates("sample_name").set_index("sample_name")
    a_stage = a_meta["stage"].astype(str).reindex(animals)
    a_score = pd.to_numeric(a_meta["score_sacrifice"], errors="coerce").reindex(animals)

    print("[pseudobulk] summing counts per animal ...")
    pb = pseudobulk(counts, codes, len(animals))     # [n_animals, n_genes], log CP10k

    # cluster composition per animal
    have_clusters = args.cluster_col in o.columns
    if have_clusters:
        comp = pd.crosstab(o["sample_name"], o[args.cluster_col].astype(str))
        comp = comp.reindex(animals).fillna(0)
        comp = comp.div(comp.sum(axis=1), axis=0)     # fractions
        clusters = list(comp.columns)
    else:
        print(f"[warn] cluster col '{args.cluster_col}' missing — skipping composition")
        clusters = []

    results = {"n_rr_animals": len(animals), "cluster_col": args.cluster_col}

    # ---------------------------------------------------------------- trajectory
    rank_map = {s: i for i, s in enumerate(CYCLE_ORDER)}
    traj_mask = np.array([s in rank_map for s in a_stage.values])
    ranks = np.array([rank_map.get(s, -1) for s in a_stage.values])
    n_traj = int(traj_mask.sum())
    print(f"[trajectory] {n_traj} animals on the relapse cycle "
          f"(excludes {int((a_stage.values=='MONOPHASIC').sum())} monophasic)")

    def spearman_trend(M, names):
        r = ranks[traj_mask]
        rows = []
        for j in range(M.shape[1]):
            x = M[traj_mask, j]
            if np.ptp(x) == 0:
                continue
            rho, p = spearmanr(x, r)
            if not np.isnan(rho):
                rows.append((names[j], float(rho), float(p)))
        df = pd.DataFrame(rows, columns=["name", "rho", "p"])
        df["q"] = bh_fdr(df["p"].to_numpy()) if len(df) else []
        return df.sort_values("rho")

    gene_traj = spearman_trend(pb, genes)
    results["trajectory_genes"] = {
        "increasing": gene_traj.tail(args.top)[::-1].to_dict("records"),
        "decreasing": gene_traj.head(args.top).to_dict("records"),
        "n_sig_q05": int((gene_traj["q"] < 0.05).sum()),
    }
    if have_clusters:
        clu_traj = spearman_trend(comp.to_numpy(), clusters)
        results["trajectory_clusters"] = clu_traj.sort_values("rho").to_dict("records")

    # ---- severity-adjusted: partial Spearman controlling for score_sacrifice
    gene_adj, r_cyc_sev, n_adj = partial_trend(pb, genes, traj_mask,
                                               ranks.astype(float), a_score.to_numpy(float))
    raw_sig = set(gene_traj.loc[gene_traj["q"] < 0.05, "name"])
    adj_sig = set(gene_adj.loc[gene_adj["q"] < 0.05, "name"])
    results["trajectory_severity_adjusted"] = {
        "cycle_vs_severity_rho": r_cyc_sev,
        "n_animals_used": n_adj,
        "n_sig_raw_q05": len(raw_sig),
        "n_sig_partial_q05": len(adj_sig),
        "n_survive_adjustment": len(raw_sig & adj_sig),
        "increasing_cycle_specific": gene_adj.tail(args.top)[::-1].to_dict("records"),
        "decreasing_cycle_specific": gene_adj.head(args.top).to_dict("records"),
        # raw-significant genes that DROP OUT after severity adjustment (severity-driven)
        "lost_after_adjustment": sorted(raw_sig - adj_sig)[:40],
    }
    # targeted readout: the cholesterol-biosynthesis genes, raw vs partial
    chol = ["Hmgcr", "Msmo1", "Idi1", "Ldlr", "Lss", "Hsd17b7"]
    gt = gene_traj.set_index("name"); ga = gene_adj.set_index("name")
    results["trajectory_severity_adjusted"]["cholesterol_genes"] = [
        {"gene": g,
         "raw_rho": float(gt.loc[g, "rho"]) if g in gt.index else None,
         "raw_q": float(gt.loc[g, "q"]) if g in gt.index else None,
         "partial_rho": float(ga.loc[g, "partial_rho"]) if g in ga.index else None,
         "partial_q": float(ga.loc[g, "q"]) if g in ga.index else None}
        for g in chol]

    # cache pseudobulk so future iterations skip the 14GB reload
    np.savez(os.path.join(args.out_dir, "pseudobulk.npz"),
             pb=pb, genes=genes, animals=np.array(animals),
             stage=a_stage.to_numpy().astype(str), score=a_score.to_numpy(float),
             cycle_rank=ranks)

    # ---------------------------------------------------------------- mono vs relapsed
    grp = np.where(a_stage.values == "MONOPHASIC", "mono",
                   np.where(np.isin(a_stage.values, list(RELAPSED)), "relapsed", "other"))
    mono = grp == "mono"
    rel = grp == "relapsed"
    print(f"[mono-vs-relapsed] {mono.sum()} monophasic vs {rel.sum()} relapsed")

    def mwu(M, names):
        rows = []
        for j in range(M.shape[1]):
            xm, xr = M[mono, j], M[rel, j]
            if np.ptp(np.concatenate([xm, xr])) == 0:
                continue
            try:
                U, p = mannwhitneyu(xr, xm, alternative="two-sided")
            except ValueError:
                continue
            rows.append((names[j], float(np.median(xr) - np.median(xm)), float(p)))
        df = pd.DataFrame(rows, columns=["name", "relapsed_minus_mono", "p"])
        df["q"] = bh_fdr(df["p"].to_numpy()) if len(df) else []
        return df.sort_values("relapsed_minus_mono")

    if mono.sum() >= 2 and rel.sum() >= 2:
        gene_mr = mwu(pb, genes)
        results["mono_vs_relapsed_genes"] = {
            "up_in_relapsed": gene_mr.tail(args.top)[::-1].to_dict("records"),
            "up_in_monophasic": gene_mr.head(args.top).to_dict("records"),
            "n_sig_q05": int((gene_mr["q"] < 0.05).sum()),
            "note": "n=4 monophasic; monophasic confined to run_date 20250605 — exploratory",
        }
        if have_clusters:
            clu_mr = mwu(comp.to_numpy(), clusters)
            results["mono_vs_relapsed_clusters"] = clu_mr.sort_values("relapsed_minus_mono").to_dict("records")

    # ---------------------------------------------------------------- report
    L = []
    L.append(f"\n=== WITHIN-RR relapse biology (n={len(animals)} RR animals, animal-level) ===")
    L.append("Descriptive only; stage is de-confounded from slide within RR (each stage on 9-12 slides).")
    L.append(f"\n-- Relapse-cycle trajectory ({n_traj} animals) — top gene trends (Spearman vs cycle rank) --")
    L.append(f"   genes at BH-q<0.05: {results['trajectory_genes']['n_sig_q05']}")
    L.append("   INCREASING along cycle:")
    for r in results["trajectory_genes"]["increasing"][:10]:
        L.append(f"     {r['name']:<16} rho={r['rho']:+.3f}  q={r['q']:.3g}")
    L.append("   DECREASING along cycle:")
    for r in results["trajectory_genes"]["decreasing"][:10]:
        L.append(f"     {r['name']:<16} rho={r['rho']:+.3f}  q={r['q']:.3g}")
    if have_clusters and "trajectory_clusters" in results:
        L.append(f"\n   cluster ({args.cluster_col}) composition trends along cycle (top |rho|):")
        ct = sorted(results["trajectory_clusters"], key=lambda r: -abs(r["rho"]))[:6]
        for r in ct:
            L.append(f"     cluster {r['name']:<6} rho={r['rho']:+.3f}  q={r['q']:.3g}")

    sa = results["trajectory_severity_adjusted"]
    L.append(f"\n-- Severity-adjusted trajectory (partial Spearman | score_sacrifice, n={sa['n_animals_used']}) --")
    L.append(f"   cycle vs severity rho = {sa['cycle_vs_severity_rho']:+.3f} "
             f"(collinearity: how much the cycle axis IS severity)")
    L.append(f"   genes q<0.05: raw {sa['n_sig_raw_q05']} -> partial {sa['n_sig_partial_q05']} "
             f"({sa['n_survive_adjustment']} survive adjustment)")
    L.append("   cycle-SPECIFIC increasing (beyond severity):")
    for r in sa["increasing_cycle_specific"][:8]:
        L.append(f"     {r['name']:<16} partial_rho={r['partial_rho']:+.3f}  q={r['q']:.3g}")
    L.append("   cycle-SPECIFIC decreasing (beyond severity):")
    for r in sa["decreasing_cycle_specific"][:8]:
        L.append(f"     {r['name']:<16} partial_rho={r['partial_rho']:+.3f}  q={r['q']:.3g}")
    L.append("   cholesterol-biosynthesis genes (raw -> severity-adjusted):")
    for c in sa["cholesterol_genes"]:
        if c["raw_rho"] is None:
            continue
        L.append(f"     {c['gene']:<10} raw rho={c['raw_rho']:+.3f} (q={c['raw_q']:.2g})  ->  "
                 f"partial rho={c['partial_rho']:+.3f} (q={c['partial_q']:.2g})")
    if "mono_vs_relapsed_genes" in results:
        L.append(f"\n-- Monophasic (4) vs relapsed (15) — EXPLORATORY (run-date imbalance) --")
        L.append(f"   genes at BH-q<0.05: {results['mono_vs_relapsed_genes']['n_sig_q05']}")
        L.append("   up in RELAPSED:")
        for r in results["mono_vs_relapsed_genes"]["up_in_relapsed"][:8]:
            L.append(f"     {r['name']:<16} dmedian={r['relapsed_minus_mono']:+.3f}  q={r['q']:.3g}")
        L.append("   up in MONOPHASIC:")
        for r in results["mono_vs_relapsed_genes"]["up_in_monophasic"][:8]:
            L.append(f"     {r['name']:<16} dmedian={r['relapsed_minus_mono']:+.3f}  q={r['q']:.3g}")
    report = "\n".join(L)
    print(report)

    with open(os.path.join(args.out_dir, "report.txt"), "w") as fh:
        fh.write(report + "\n")
    with open(os.path.join(args.out_dir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    _plot(pb, genes, a_stage.values, ranks, traj_mask, gene_traj, gene_adj, args.out_dir)
    print(f"\n[done] -> {args.out_dir}/  (report.txt, results.json, pseudobulk.npz, figures/)")


def _plot(pb, genes, stages, ranks, traj_mask, gene_traj, gene_adj, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # raw cycle-rho vs severity-adjusted partial-rho: points below the diagonal
    # lost their trend to severity; points on it are cycle-specific
    mg = gene_traj.set_index("name").join(gene_adj.set_index("name")[["partial_rho", "q"]],
                                          rsuffix="_adj")
    mg = mg.dropna(subset=["partial_rho"])
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    surv = mg["q_adj"] < 0.05
    ax.scatter(mg.loc[~surv, "rho"], mg.loc[~surv, "partial_rho"], s=6, c="#C0C0C0", label="ns after adj")
    ax.scatter(mg.loc[surv, "rho"], mg.loc[surv, "partial_rho"], s=10, c="#4C72B0", label="cycle-specific (q<.05)")
    for g in ["Hmgcr", "Msmo1", "Idi1", "Ldlr", "C6", "Igf1", "Fcrls", "Igkc"]:
        if g in mg.index:
            ax.annotate(g, (mg.loc[g, "rho"], mg.loc[g, "partial_rho"]), fontsize=7)
    lim = [-1, 1]
    ax.plot(lim, lim, "k--", lw=0.8); ax.axhline(0, c="gray", lw=0.5); ax.axvline(0, c="gray", lw=0.5)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("raw cycle rho"); ax.set_ylabel("severity-adjusted partial rho")
    ax.set_title("Relapse-cycle trends: raw vs severity-adjusted")
    ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "figures", "raw_vs_severity_adjusted.pdf"), bbox_inches="tight")
    plt.close(fig)
    # heatmap: top trending genes (rows) x ordered cycle stages (cols), stage-mean z-scored
    top = pd.concat([gene_traj.head(15), gene_traj.tail(15)])
    gidx = {g: i for i, g in enumerate(genes)}
    rows = [gidx[n] for n in top["name"]]
    present = [s for s in CYCLE_ORDER if s in set(stages[traj_mask])]
    M = np.zeros((len(rows), len(present)))
    for cj, st in enumerate(present):
        m = stages == st
        M[:, cj] = pb[np.ix_(m, rows)].mean(axis=0) if m.sum() else np.nan
    # z-score per gene across stages
    M = (M - M.mean(axis=1, keepdims=True)) / (M.std(axis=1, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(figsize=(max(5, len(present) * 0.9), 8))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(range(len(present))); ax.set_xticklabels(present, rotation=40, ha="right")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(list(top["name"]), fontsize=7)
    ax.set_title("Top relapse-cycle trending genes (z-scored stage means)")
    fig.colorbar(im, ax=ax, fraction=0.025, label="z")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "figures", "trajectory_heatmap.pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
