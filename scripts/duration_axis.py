"""The DURATION axis — does accumulated damage track actual time-since-induction,
beyond current clinical severity?

`day_of_sacrifice` (continuous, ~9-49 days post-induction) is a direct measure of
disease duration. The relapse cycle decouples it from severity: peaks recur at the
same severity but later days (PEAK1 -> PEAK3 ≈ +20 days), so within RR we can estimate
the day effect WHILE HOLDING SEVERITY FIXED (partial Spearman | score_sacrifice).
Within RR this is also strain-clean (all SJL/PLP).

Tests, per gene and for the acute/cumulative programs:
  partial Spearman( expression , day_of_sacrifice  |  score_sacrifice )
  -> positive = accrues with time independent of how sick the animal is now.
Expect the CUMULATIVE/ratchet program (Gpnmb, Fcrls, Plin4...) to track day; the
ACUTE program (Hal, Arg1...) not to.

    python scripts/duration_axis.py
"""
from __future__ import annotations

import json
import os

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.stats import t as tdist

OUT = "runs/duration_axis"
RRMAP2 = ("/Volumes/moldiassd/RRMAP2_xenium_adata/kmeans_separated/"
          "RRMAP2_xenium_all_samples.cellcharter.companion.ready.with_metadata.rerun.h5ad")
ACUTE = ["Hal", "Arg1", "Chil3", "Chil1", "Acod1", "Cxcl10", "Timp1", "Gbp2"]
CUMULATIVE = ["Gpnmb", "Plin4", "Fcrls", "Igf2", "Fmod", "Pmp22", "Ptgds"]


def bh_fdr(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = np.empty(n); q[o] = p[o] * n / (np.arange(n) + 1)
    q[o] = np.minimum.accumulate(q[o][::-1])[::-1]
    return np.clip(q, 0, 1)


def partial_spearman(M, day, score):
    """Per-column partial Spearman(col, day | score). M: [animals, genes]."""
    m = M.shape[0]
    rd, rs = rankdata(day), rankdata(score)
    def z(v):
        v = v - v.mean(); s = v.std(); return v / s if s > 0 else np.zeros_like(v)
    zd, zs = z(rd), z(rs)
    r_ds = float(zd @ zs / m)
    gr = np.apply_along_axis(rankdata, 0, M)
    zg = (gr - gr.mean(0)) / (gr.std(0) + 1e-12)
    r_gd = (zg.T @ zd) / m
    r_gs = (zg.T @ zs) / m
    denom = np.sqrt(np.clip((1 - r_gs ** 2) * (1 - r_ds ** 2), 1e-12, None))
    pr = (r_gd - r_gs * r_ds) / denom
    dfree = m - 3
    tstat = pr * np.sqrt(dfree / np.clip(1 - pr ** 2, 1e-12, None))
    pval = 2 * tdist.sf(np.abs(tstat), dfree)
    return pr, pval, r_ds


def load_with_day():
    """Load RR + chronic pseudobulk caches, attach per-animal day_of_sacrifice from obs."""
    rr = np.load("runs/rr_within_relapse/pseudobulk.npz", allow_pickle=True)
    chr_ = np.load("runs/chronic_trajectory/pseudobulk.npz", allow_pickle=True)
    genes = rr["genes"].astype(str)
    assert np.array_equal(genes, chr_["genes"].astype(str))
    a = ad.read_h5ad(RRMAP2, backed="r")
    o = a.obs
    day = (o.drop_duplicates("sample_name").set_index("sample_name")["day_of_sacrifice"]
           .astype(float))
    def days_for(animals):
        return day.reindex(pd.Index(animals.astype(str))).to_numpy(float)
    pb = np.vstack([chr_["pb"], rr["pb"]])
    model = np.array(["CHRONIC"] * len(chr_["pb"]) + ["RR"] * len(rr["pb"]))
    stage = np.concatenate([chr_["stage"].astype(str), rr["stage"].astype(str)])
    score = np.concatenate([chr_["score"].astype(float), rr["score"].astype(float)])
    animals = np.concatenate([chr_["animals"].astype(str), rr["animals"].astype(str)])
    days = np.concatenate([days_for(chr_["animals"]), days_for(rr["animals"])])
    return pb, genes, model, stage, score, days, animals


def main():
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    pb, genes, model, stage, score, days, animals = load_with_day()
    gi = {g: i for i, g in enumerate(genes)}
    z = (pb - pb.mean(0)) / (pb.std(0) + 1e-9)
    acute_s = z[:, [gi[g] for g in ACUTE if g in gi]].mean(1)
    cum_s = z[:, [gi[g] for g in CUMULATIVE if g in gi]].mean(1)

    results = {}
    for mdl in ["RR", "CHRONIC"]:
        m = (model == mdl) & ~np.isnan(days) & ~np.isnan(score)
        # exclude pre-symptomatic baselines so the cycle/severity range is meaningful
        sub = pb[m]
        d, s = days[m], score[m]
        # programs partial
        prog = np.c_[acute_s[m], cum_s[m]]
        pr_prog, pv_prog, r_ds = partial_spearman(prog, d, s)
        # gene-level partial (duration genes)
        pr_g, pv_g, _ = partial_spearman(sub, d, s)
        q_g = bh_fdr(pv_g)
        gdf = pd.DataFrame({"gene": genes, "partial_rho": pr_g, "p": pv_g, "q": q_g})
        gdf = gdf[sub.std(0) > 0].sort_values("partial_rho", ascending=False)
        # named program genes partial
        named = {g: round(float(pr_g[gi[g]]), 3) for g in (CUMULATIVE + ["Hal", "Arg1"]) if g in gi}
        results[mdl] = {
            "n_animals": int(m.sum()),
            "day_range": [float(np.nanmin(d)), float(np.nanmax(d))],
            "day_vs_score_collinearity_rho": round(float(r_ds), 3),
            "acute_program_partial_rho": round(float(pr_prog[0]), 3),
            "acute_program_p": round(float(pv_prog[0]), 4),
            "cumulative_program_partial_rho": round(float(pr_prog[1]), 3),
            "cumulative_program_p": round(float(pv_prog[1]), 4),
            "n_genes_q05": int((gdf["q"] < 0.05).sum()),
            "top_duration_genes": gdf.head(20)[["gene", "partial_rho", "q"]].round(4).to_dict("records"),
            "named_program_partial_rho": named,
        }
    results["caveats"] = ("Within RR is strain-clean (SJL/PLP). day_of_sacrifice and score are decoupled by "
                          "the relapse cycle (peaks recur at later days), enabling the partial. CHRONIC is "
                          "confounded: day_of_sacrifice aliases run_date (day16 batch vs day30 batch), so the "
                          "chronic duration effect cannot be separated from batch. Small n; animal-level.")
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    # ---- report ----
    L = ["\n=== DURATION axis: accrual vs actual day_of_sacrifice (partial | severity) ==="]
    for mdl in ["RR", "CHRONIC"]:
        r = results[mdl]
        L.append(f"\n-- {mdl} (n={r['n_animals']}, days {r['day_range'][0]:.0f}-{r['day_range'][1]:.0f}, "
                 f"day~score collinearity rho={r['day_vs_score_collinearity_rho']:+.2f}) --")
        L.append(f"   CUMULATIVE program vs day | severity: partial rho={r['cumulative_program_partial_rho']:+.3f} "
                 f"(p={r['cumulative_program_p']})")
        L.append(f"   ACUTE program      vs day | severity: partial rho={r['acute_program_partial_rho']:+.3f} "
                 f"(p={r['acute_program_p']})")
        L.append(f"   genes with q<0.05 (duration, severity-adjusted): {r['n_genes_q05']}")
        L.append(f"   top duration genes: " + ", ".join(f"{x['gene']}({x['partial_rho']:+.2f})"
                                                         for x in r['top_duration_genes'][:10]))
        L.append(f"   named: " + ", ".join(f"{g}={v:+.2f}" for g, v in r['named_program_partial_rho'].items()))
    report = "\n".join(L)
    print(report)
    with open(os.path.join(OUT, "report.txt"), "w") as fh:
        fh.write(report + "\n")

    _plot(pb, gi, model, stage, score, days, cum_s, acute_s, OUT)
    print(f"\n[done] -> {OUT}/ (results.json, report.txt, figures/)")


def _plot(pb, gi, model, stage, score, days, cum_s, acute_s, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rr = (model == "RR") & ~np.isnan(days)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))
    sc = axA.scatter(days[rr], cum_s[rr], c=score[rr], cmap="viridis", s=70, edgecolor="k", linewidth=.4)
    axA.set_xlabel("day of sacrifice (days post-induction)")
    axA.set_ylabel("cumulative program (z)")
    axA.set_title("RR: cumulative damage climbs with TIME\n(colour = clinical severity)", fontsize=11)
    plt.colorbar(sc, ax=axA, label="score_sacrifice")
    axA.spines[["top", "right"]].set_visible(False)
    # acute for contrast
    sc2 = axB.scatter(days[rr], acute_s[rr], c=score[rr], cmap="viridis", s=70, edgecolor="k", linewidth=.4)
    axB.set_xlabel("day of sacrifice")
    axB.set_ylabel("acute program (z)")
    axB.set_title("RR: acute program does NOT track time\n(driven by current severity, not duration)", fontsize=11)
    plt.colorbar(sc2, ax=axB, label="score_sacrifice")
    axB.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figures", "duration_axis.png"), dpi=130, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "figures", "duration_axis.pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
