"""RR vs chronic — DIFFERENTIAL DISEASE DYNAMICS (not naive DE).

A direct RR-vs-chronic comparison is uninterpretable: model is 100% confounded
with slide/run/strain (all 54 slides are model-pure; SJL/PLP vs B6/MOG). A naive
DE just reads strain+batch (that is what the AUC-0.998 classifier showed).

Instead we ask how the two courses DIFFER IN DYNAMICS along the one axis both
share — clinical severity `score_sacrifice`. Per gene, animal-level OLS:

    expr ~ score + model + score:model      (score centered)

The score:model INTERACTION = "does this gene change with severity differently
in RR vs chronic". A constant strain/batch offset cancels in the slope, so the
interaction is the de-confounded readout; the model MAIN effect (baseline level)
is dominated by strain/batch and is reported only for context.

CAVEAT this does NOT fully de-confound: batch is still 100% aligned with model,
so a batch effect that itself correlates with severity within a cohort would
leak into the interaction. This is the best the design allows; treat as
exploratory / hypothesis-generating.

    python scripts/model_difference.py
"""

from __future__ import annotations

import argparse
import json
import os

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.stats import t as tdist

from immunotransformer.train import resolve_device  # noqa: F401  (package OpenMP guard)

RRMAP2 = (
    "/Volumes/moldiassd/RRMAP2_xenium_adata/kmeans_separated/"
    "RRMAP2_xenium_all_samples.cellcharter.companion.ready.with_metadata.rerun.h5ad"
)


def bh_fdr(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = (p[o] * n) / (np.arange(n) + 1)
    q[o] = np.minimum.accumulate(q[o][::-1])[::-1]
    return np.clip(q, 0, 1)


def pseudobulk(counts, codes, n_groups):
    g = counts.shape[1]
    pb = np.zeros((n_groups, g))
    for k in range(n_groups):
        rows = counts[codes == k]
        s = np.asarray(rows.sum(0)).ravel() if issparse(rows) else rows.sum(0)
        pb[k] = s
    lib = pb.sum(1, keepdims=True); lib[lib == 0] = 1
    return np.log1p(pb / lib * 1e4)


def ols_interaction(Y, score, model):
    """Vectorised OLS  y ~ 1 + score_c + model + score_c:model  for all columns of Y.

    Returns dict of arrays (one per gene): interaction effect (RR_slope-chronic_slope),
    interaction p, chronic_slope, rr_slope, model_main effect, model_main p.
    """
    sc = score - score.mean()
    X = np.column_stack([np.ones_like(sc), sc, model, sc * model])  # [n,4]
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ Y                                         # [4, G]
    resid = Y - X @ beta
    sigma2 = (resid ** 2).sum(0) / (n - k)                           # [G]
    se = np.sqrt(np.outer(np.diag(XtX_inv), sigma2))                # [4, G]
    tvals = beta / se
    pvals = 2 * tdist.sf(np.abs(tvals), n - k)
    return {
        "inter_effect": beta[3], "inter_p": pvals[3],
        "model_effect": beta[2], "model_p": pvals[2],
        "chronic_slope": beta[1], "rr_slope": beta[1] + beta[3],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", default=RRMAP2)
    ap.add_argument("--cluster-col", default="leiden_1")
    ap.add_argument("--layer", default="counts")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out-dir", default="runs/model_difference")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out_dir, "figures"), exist_ok=True)

    print(f"[load] {args.h5ad}")
    adata = ad.read_h5ad(args.h5ad)
    print(f"       {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    o = adata.obs
    counts = adata.layers[args.layer] if args.layer else adata.X
    genes = np.array(adata.var_names, dtype=str)

    animals = list(pd.unique(o["sample_name"]))
    a_code = {a: i for i, a in enumerate(animals)}
    codes = o["sample_name"].map(a_code).to_numpy()
    meta = o.drop_duplicates("sample_name").set_index("sample_name").reindex(animals)
    a_model = (meta["model"].astype(str) == "RELAPSE REMITTING").to_numpy(float)  # 1=RR,0=chronic
    a_score = pd.to_numeric(meta["score_sacrifice"], errors="coerce").to_numpy(float)

    print("[pseudobulk] per animal ...")
    pb = pseudobulk(counts, codes, len(animals))

    # composition per animal
    have_clu = args.cluster_col in o.columns
    if have_clu:
        comp = pd.crosstab(o["sample_name"], o[args.cluster_col].astype(str)).reindex(animals).fillna(0)
        comp = comp.div(comp.sum(1), axis=0)

    keep = ~np.isnan(a_score)
    print(f"[design] {int(keep.sum())} animals "
          f"({int(a_model[keep].sum())} RR / {int((1-a_model[keep]).sum())} chronic) with score")

    fit = ols_interaction(pb[keep], a_score[keep], a_model[keep])
    nonconst = pb[keep].std(0) > 0
    q = np.full(len(genes), np.nan); q[nonconst] = bh_fdr(fit["inter_p"][nonconst])
    df = pd.DataFrame({
        "gene": genes, "inter_effect": fit["inter_effect"], "inter_q": q,
        "chronic_slope": fit["chronic_slope"], "rr_slope": fit["rr_slope"],
        "model_effect": fit["model_effect"], "model_q": bh_fdr(np.nan_to_num(fit["model_p"], nan=1.0)),
    })[nonconst].sort_values("inter_effect")

    results = {
        "n_animals": int(keep.sum()),
        "n_rr": int(a_model[keep].sum()), "n_chronic": int((1 - a_model[keep]).sum()),
        "interaction_sig_q05": int((df["inter_q"] < 0.05).sum()),
        "model_main_sig_q05": int((df["model_q"] < 0.05).sum()),
        "steeper_in_RR": df.tail(args.top)[::-1].to_dict("records"),       # rises more with severity in RR
        "steeper_in_chronic": df.head(args.top).to_dict("records"),       # rises more in chronic (or falls in RR)
    }

    if have_clu:
        cfit = ols_interaction(comp.to_numpy()[keep], a_score[keep], a_model[keep])
        cq = bh_fdr(np.nan_to_num(cfit["inter_p"], nan=1.0))
        cdf = pd.DataFrame({"cluster": list(comp.columns), "inter_effect": cfit["inter_effect"],
                            "inter_q": cq, "chronic_slope": cfit["chronic_slope"],
                            "rr_slope": cfit["rr_slope"]}).sort_values("inter_effect")
        results["cluster_interaction"] = cdf.to_dict("records")

    # ---- report
    L = [f"\n=== RR vs CHRONIC differential dynamics (interaction score:model, n={results['n_animals']} "
         f"= {results['n_rr']} RR / {results['n_chronic']} chronic) ===",
         "Interaction = gene changes with severity DIFFERENTLY between courses (de-confounds constant strain/batch offset).",
         f"\ngenes with significant score:model interaction (BH-q<0.05): {results['interaction_sig_q05']}",
         f"(model MAIN effect q<0.05: {results['model_main_sig_q05']} — strain/batch baseline, NOT interpreted)",
         "\n-- Rises with severity MORE STEEPLY IN RR (RR_slope > chronic_slope) --",
         f"   {'gene':<14}{'effect':>8}{'q':>9}{'chr_slope':>11}{'rr_slope':>10}"]
    for r in results["steeper_in_RR"][:12]:
        L.append(f"   {r['gene']:<14}{r['inter_effect']:>+8.3f}{r['inter_q']:>9.2g}"
                 f"{r['chronic_slope']:>+11.3f}{r['rr_slope']:>+10.3f}")
    L.append("\n-- Rises with severity MORE STEEPLY IN CHRONIC --")
    L.append(f"   {'gene':<14}{'effect':>8}{'q':>9}{'chr_slope':>11}{'rr_slope':>10}")
    for r in results["steeper_in_chronic"][:12]:
        L.append(f"   {r['gene']:<14}{r['inter_effect']:>+8.3f}{r['inter_q']:>9.2g}"
                 f"{r['chronic_slope']:>+11.3f}{r['rr_slope']:>+10.3f}")
    if have_clu:
        L.append(f"\n-- cluster ({args.cluster_col}) composition: differential severity dynamics --")
        sig = [r for r in results["cluster_interaction"] if r["inter_q"] < 0.1]
        for r in sorted(sig, key=lambda r: r["inter_effect"]):
            L.append(f"   cluster {r['cluster']:<5} effect={r['inter_effect']:+.4f} q={r['inter_q']:.2g} "
                     f"(chr slope {r['chronic_slope']:+.4f} / RR slope {r['rr_slope']:+.4f})")
    report = "\n".join(L)
    print(report)

    with open(os.path.join(args.out_dir, "report.txt"), "w") as fh:
        fh.write(report + "\n")
    with open(os.path.join(args.out_dir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    np.savez(os.path.join(args.out_dir, "pseudobulk_both.npz"),
             pb=pb, genes=genes, animals=np.array(animals), model=a_model, score=a_score)
    _plot(df, pb, genes, a_score, a_model, keep, args.out_dir)
    print(f"\n[done] -> {args.out_dir}/  (report.txt, results.json, pseudobulk_both.npz, figures/)")


def _plot(df, pb, genes, score, model, keep, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    # volcano of interaction
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    eff = df["inter_effect"].to_numpy(); ql = -np.log10(np.clip(df["inter_q"].to_numpy(), 1e-12, 1))
    sig = df["inter_q"].to_numpy() < 0.05
    ax.scatter(eff[~sig], ql[~sig], s=6, c="#c8ccd4")
    ax.scatter(eff[sig], ql[sig], s=12, c="#3b5bdb")
    ax.axhline(-np.log10(0.05), color="#c0392b", ls=":", lw=1)
    ax.set_xlabel("interaction effect  (RR slope − chronic slope)")
    ax.set_ylabel("-log10 BH q")
    ax.set_title("Differential severity dynamics: RR vs chronic")
    top = pd.concat([df.head(6), df.tail(6)])
    for _, r in top.iterrows():
        ax.annotate(r["gene"], (r["inter_effect"], -np.log10(max(r["inter_q"], 1e-12))), fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "figures", "interaction_volcano.pdf"), bbox_inches="tight")
    plt.close(fig)
    # example genes: expr vs severity by model
    gi = {g: i for i, g in enumerate(genes)}
    examples = [df.tail(2)["gene"].tolist() + df.head(2)["gene"].tolist()][0]
    s, m = score[keep], model[keep]
    fig, axes = plt.subplots(1, len(examples), figsize=(3.4 * len(examples), 3.2), sharex=True)
    for ax, g in zip(np.atleast_1d(axes), examples):
        y = pb[keep][:, gi[g]]
        for mv, col, lab in [(0, "#e8590c", "chronic"), (1, "#3b5bdb", "RR")]:
            mask = m == mv
            ax.scatter(s[mask], y[mask], s=14, c=col, label=lab, alpha=.8)
            if mask.sum() > 2:
                b = np.polyfit(s[mask], y[mask], 1)
                xs = np.linspace(s.min(), s.max(), 20); ax.plot(xs, np.polyval(b, xs), c=col, lw=1.3)
        ax.set_title(g, fontsize=10); ax.set_xlabel("score_sacrifice")
        ax.spines[["top", "right"]].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("log CP10k"); np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "figures", "example_genes.pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
