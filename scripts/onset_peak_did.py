"""Onset -> peak transition: RR vs chronic DIFFERENCE-IN-DIFFERENCES.

Motivation: the continuous severity-slope interaction blends disease phases, and
a single matched-stage contrast (RR-PEAK1 vs chronic-PEAK1) is still fully
strain/batch confounded. The clean, phase-anchored alternative is the DiD:

    (RR_peak - RR_onset) - (chronic_peak - chronic_onset)

Each within-model (peak - onset) change is batch-clean (within a model, stage is
spread across slides); subtracting the two within-model changes cancels the
constant strain/batch baseline. So this isolates how the onset->peak transition
DIFFERS between courses.

Matched phases (the only ones available):
    onset:  chronic OS1 (n=5)     vs  RR ONSET1 (n=2)   <- RR onset n=2 is limiting
    peak:   chronic PEAK1 (n=6)   vs  RR PEAK1 (n=4)
So this is EXPLORATORY / directional, not FDR-grade.

Reuses the cached pseudobulk from model_difference.py (no 14GB reload); only the
per-animal stage labels are pulled via a light backed read.

    python scripts/onset_peak_did.py
"""

from __future__ import annotations

import argparse
import json
import os

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import t as tdist

from immunotransformer.train import resolve_device  # noqa: F401  (package OpenMP guard)

RRMAP2 = (
    "/Volumes/moldiassd/RRMAP2_xenium_adata/kmeans_separated/"
    "RRMAP2_xenium_all_samples.cellcharter.companion.ready.with_metadata.rerun.h5ad"
)
CHR_ONSET, RR_ONSET, PEAK = "OS1", "ONSET1", "PEAK1"


def bh_fdr(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = (p[o] * n) / (np.arange(n) + 1)
    q[o] = np.minimum.accumulate(q[o][::-1])[::-1]
    return np.clip(q, 0, 1)


def ols_coef_p(Y, X, row):
    """Vectorised OLS; return (coef, p) for design-matrix column `row`, all genes."""
    n, k = X.shape
    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ X.T @ Y
    resid = Y - X @ beta
    sigma2 = (resid ** 2).sum(0) / (n - k)
    se = np.sqrt(np.outer(np.diag(XtXi), sigma2))
    p = 2 * tdist.sf(np.abs(beta / se), n - k)
    return beta[row], p[row]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", default=RRMAP2)
    ap.add_argument("--pb", default="runs/model_difference/pseudobulk_both.npz")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out-dir", default="runs/onset_peak_did")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out_dir, "figures"), exist_ok=True)

    npz = np.load(args.pb, allow_pickle=True)
    pb, genes, animals = npz["pb"], npz["genes"].astype(str), npz["animals"].astype(str)
    model = npz["model"].astype(int)  # 1=RR, 0=chronic
    print(f"[pb] cached pseudobulk: {pb.shape[0]} animals x {pb.shape[1]} genes")

    print(f"[stage] backed read of obs from {args.h5ad}")
    adata = ad.read_h5ad(args.h5ad, backed="r")
    st = adata.obs.drop_duplicates("sample_name").set_index("sample_name")["stage"].astype(str)
    stage = st.reindex(animals).to_numpy()

    rr, chrom = model == 1, model == 0
    g = {
        "chr_onset": (stage == CHR_ONSET) & chrom,
        "chr_peak": (stage == PEAK) & chrom,
        "rr_onset": (stage == RR_ONSET) & rr,
        "rr_peak": (stage == PEAK) & rr,
    }
    counts = {k: int(v.sum()) for k, v in g.items()}
    print(f"[groups] {counts}")
    for k, n in counts.items():
        if n < 2:
            raise SystemExit(f"group {k} has n={n} (<2) — cannot run")

    mean = lambda mask: pb[mask].mean(0)
    chr_delta = mean(g["chr_peak"]) - mean(g["chr_onset"])
    rr_delta = mean(g["rr_peak"]) - mean(g["rr_onset"])
    did = rr_delta - chr_delta

    # nominal interaction test: 2x2 (peak vs onset) x (RR vs chronic) OLS
    sel = g["chr_onset"] | g["chr_peak"] | g["rr_onset"] | g["rr_peak"]
    Y = pb[sel]
    is_peak = (stage[sel] == PEAK).astype(float)
    is_rr = (model[sel] == 1).astype(float)
    X = np.column_stack([np.ones(sel.sum()), is_peak, is_rr, is_peak * is_rr])
    inter_coef, inter_p = ols_coef_p(Y, X, 3)   # interaction == DiD
    nonconst = Y.std(0) > 0
    q = np.full(len(genes), np.nan)
    q[nonconst] = bh_fdr(inter_p[nonconst])

    df = pd.DataFrame({
        "gene": genes, "did": did, "rr_delta": rr_delta, "chr_delta": chr_delta,
        "inter_p": inter_p, "q": q,
    })[nonconst].sort_values("did")

    # the "trap": single-stage PEAK1 RR-vs-chronic contrast (strain/batch-confounded)
    pk = g["chr_peak"] | g["rr_peak"]
    Xpk = np.column_stack([np.ones(pk.sum()), (model[pk] == 1).astype(float)])
    _, p_pk = ols_coef_p(pb[pk], Xpk, 1)
    q_pk = bh_fdr(p_pk[pb[pk].std(0) > 0])

    results = {
        "groups": counts,
        "did_sig_q05": int((df["q"] < 0.05).sum()),
        "peak1_single_contrast_sig_q05": int((q_pk < 0.05).sum()),
        "note": f"RR onset n={counts['rr_onset']} is the limiting factor — exploratory/directional.",
        "stronger_onset_to_peak_in_RR": df.tail(args.top)[::-1].to_dict("records"),
        "stronger_onset_to_peak_in_chronic": df.head(args.top).to_dict("records"),
    }

    L = [f"\n=== Onset->Peak transition: RR vs CHRONIC difference-in-differences ===",
         f"groups: {counts}",
         f"\nDiD = (RR_peak - RR_onset) - (chr_peak - chr_onset).  Interaction cancels the constant strain/batch offset.",
         f"\nTHE TRAP (for contrast): a single matched PEAK1 RR-vs-chronic contrast calls "
         f"{results['peak1_single_contrast_sig_q05']} genes different (q<0.05) — that is strain/batch.",
         f"The phase-anchored DiD calls {results['did_sig_q05']} genes different (q<0.05) "
         f"[exploratory: RR onset n={counts['rr_onset']}].",
         f"\n-- onset->peak rises MORE IN RR than chronic (top |DiD|) --",
         f"   {'gene':<14}{'DiD':>8}{'rr_d':>8}{'chr_d':>8}{'q':>9}"]
    for r in results["stronger_onset_to_peak_in_RR"][:12]:
        L.append(f"   {r['gene']:<14}{r['did']:>+8.3f}{r['rr_delta']:>+8.3f}{r['chr_delta']:>+8.3f}{r['q']:>9.2g}")
    L.append("\n-- onset->peak rises MORE IN CHRONIC than RR --")
    L.append(f"   {'gene':<14}{'DiD':>8}{'rr_d':>8}{'chr_d':>8}{'q':>9}")
    for r in results["stronger_onset_to_peak_in_chronic"][:12]:
        L.append(f"   {r['gene']:<14}{r['did']:>+8.3f}{r['rr_delta']:>+8.3f}{r['chr_delta']:>+8.3f}{r['q']:>9.2g}")
    report = "\n".join(L)
    print(report)

    with open(os.path.join(args.out_dir, "report.txt"), "w") as fh:
        fh.write(report + "\n")
    with open(os.path.join(args.out_dir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    np.savez(os.path.join(args.out_dir, "did_full.npz"),
             genes=genes, did=did, q=q, rr_delta=rr_delta, chr_delta=chr_delta,
             chr_onset=mean(g["chr_onset"]), chr_peak=mean(g["chr_peak"]),
             rr_onset=mean(g["rr_onset"]), rr_peak=mean(g["rr_peak"]),
             groups=json.dumps(counts))
    _plot(df, pb, genes, g, args.out_dir)
    print(f"\n[done] -> {args.out_dir}/  (report.txt, results.json, figures/)")


def _plot(df, pb, genes, g, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    # DiD volcano
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    eff = df["did"].to_numpy(); ql = -np.log10(np.clip(df["q"].to_numpy(), 1e-12, 1))
    sig = df["q"].to_numpy() < 0.05
    ax.scatter(eff[~sig], ql[~sig], s=6, c="#c8ccd4")
    ax.scatter(eff[sig], ql[sig], s=13, c="#3b5bdb")
    ax.axhline(-np.log10(0.05), color="#c0392b", ls=":", lw=1)
    for _, r in pd.concat([df.head(6), df.tail(6)]).iterrows():
        ax.annotate(r["gene"], (r["did"], -np.log10(max(r["q"], 1e-12))), fontsize=7)
    ax.set_xlabel("DiD  (← stronger onset→peak in chronic | stronger in RR →)")
    ax.set_ylabel("-log10 BH q"); ax.set_title("Onset→peak transition: RR vs chronic (DiD)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "figures", "did_volcano.pdf"), bbox_inches="tight")
    plt.close(fig)
    # interaction plots for top DiD genes (4 group means)
    gi = {g_: i for i, g_ in enumerate(genes)}
    examples = df.tail(2)["gene"].tolist() + df.head(2)["gene"].tolist()
    fig, axes = plt.subplots(1, len(examples), figsize=(3.3 * len(examples), 3.2))
    for ax, gene in zip(np.atleast_1d(axes), examples):
        j = gi[gene]
        for mod, col, on, pk in [("chronic", "#e8590c", "chr_onset", "chr_peak"),
                                 ("RR", "#3b5bdb", "rr_onset", "rr_peak")]:
            ax.plot([0, 1], [pb[g[on]][:, j].mean(), pb[g[pk]][:, j].mean()],
                    "-o", c=col, label=mod)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["onset", "peak"])
        ax.set_title(gene, fontsize=10); ax.spines[["top", "right"]].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("log CP10k"); np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "figures", "did_examples.pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
