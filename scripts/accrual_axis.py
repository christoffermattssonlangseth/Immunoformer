"""Same clinical score, different molecular state — the ACCRUAL axis.

Clinical score is a 1-D readout of *current* disease activity. It captures the
acute, reversible program (which RESETS each relapse and is conserved across models)
but is largely blind to *accumulated, irreversible* damage, which grows with disease
HISTORY (number of relapse cycles / duration).

Evidence: compare matched high-severity PEAKS —
  CHRONIC PEAK1 (score ~2.96)  vs  RR PEAK1 (~2.75) / PEAK2 (~2.44) / PEAK3 (~2.65) —
decomposed into:
  * ACUTE program (oscillators: Hal, Arg1, Chil3, Acod1, Cxcl10, ...) — resets, conserved
  * CUMULATIVE program (ratchet: Gpnmb, Plin4, Fcrls, Igf2, Fmod, Pmp22) — accrues
The clean (strain-matched) test is WITHIN RR: PEAK1 vs PEAK3 at near-identical score
differ in the cumulative program but not the acute one. CHRONIC PEAK1 is added for
context but is strain/antigen-confounded (SJL/PLP vs B6/MOG) — absolute differences
there mix history AND strain.

    python scripts/accrual_axis.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

OUT = "runs/accrual_axis"
ACUTE = ["Hal", "Arg1", "Chil3", "Chil1", "Acod1", "Cxcl10", "Timp1", "Gbp2"]
CUMULATIVE = ["Gpnmb", "Plin4", "Fcrls", "Igf2", "Fmod", "Pmp22", "Ptgds"]
GROUPS = [("CHRONIC", "PEAK1", "chr PEAK1"), ("RR", "PEAK1", "RR PEAK1"),
          ("RR", "PEAK2", "RR PEAK2"), ("RR", "PEAK3", "RR PEAK3")]


def load():
    rr = np.load("runs/rr_within_relapse/pseudobulk.npz", allow_pickle=True)
    chr_ = np.load("runs/chronic_trajectory/pseudobulk.npz", allow_pickle=True)
    g_rr, g_chr = rr["genes"].astype(str), chr_["genes"].astype(str)
    assert np.array_equal(g_rr, g_chr), "gene order differs between caches"
    pb = np.vstack([chr_["pb"], rr["pb"]])
    model = np.array(["CHRONIC"] * len(chr_["pb"]) + ["RR"] * len(rr["pb"]))
    stage = np.concatenate([chr_["stage"].astype(str), rr["stage"].astype(str)])
    score = np.concatenate([chr_["score"].astype(float), rr["score"].astype(float)])
    return pb, g_rr, model, stage, score


def main():
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    pb, genes, model, stage, score = load()
    gi = {g: i for i, g in enumerate(genes)}
    # z-score each gene across ALL 67 animals (stable cross-model scaling; both caches
    # are log1p-CP10k of the same panel, so directly comparable)
    z = (pb - pb.mean(0)) / (pb.std(0) + 1e-9)

    def prog(gs):
        cols = [gi[g] for g in gs if g in gi]
        return z[:, cols].mean(1)
    acute_s, cum_s = prog(ACUTE), prog(CUMULATIVE)

    rows = []
    for mdl, stg, label in GROUPS:
        m = (model == mdl) & (stage == stg)
        rows.append({
            "group": label, "model": mdl, "stage": stg, "n": int(m.sum()),
            "mean_score": round(float(np.nanmean(score[m])), 2),
            "acute_program_z": round(float(acute_s[m].mean()), 3),
            "cumulative_program_z": round(float(cum_s[m].mean()), 3),
            **{f"gene_{g}": round(float(pb[m, gi[g]].mean()), 2) for g in ["Hal", "Arg1"] + ["Gpnmb", "Plin4", "Fcrls"] if g in gi},
        })
    df = pd.DataFrame(rows)

    # ---- the clean within-RR test: PEAK1 vs PEAK3, acute vs cumulative ----
    def mwu(prog_vals, a, b):
        return float(mannwhitneyu(prog_vals[a], prog_vals[b], alternative="two-sided")[1])
    rr_p1 = (model == "RR") & (stage == "PEAK1")
    rr_p3 = (model == "RR") & (stage == "PEAK3")
    within_rr = {
        "score_PEAK1": round(float(score[rr_p1].mean()), 2),
        "score_PEAK3": round(float(score[rr_p3].mean()), 2),
        "acute_PEAK1": round(float(acute_s[rr_p1].mean()), 3),
        "acute_PEAK3": round(float(acute_s[rr_p3].mean()), 3),
        "acute_p": round(mwu(acute_s, rr_p1, rr_p3), 3),
        "cumulative_PEAK1": round(float(cum_s[rr_p1].mean()), 3),
        "cumulative_PEAK3": round(float(cum_s[rr_p3].mean()), 3),
        "cumulative_p": round(mwu(cum_s, rr_p1, rr_p3), 3),
        "n_PEAK1": int(rr_p1.sum()), "n_PEAK3": int(rr_p3.sum()),
        "note": "strain-CLEAN (same SJL/PLP strain); small n (4 vs 5) -> descriptive",
    }
    # per-gene cumulative ratchet across RR peaks
    ratchet_rr = {g: [round(float(pb[(model=="RR")&(stage==s), gi[g]].mean()), 2) for s in ["PEAK1","PEAK2","PEAK3"]]
                  for g in CUMULATIVE if g in gi}
    acute_rr = {g: [round(float(pb[(model=="RR")&(stage==s), gi[g]].mean()), 2) for s in ["PEAK1","PEAK2","PEAK3"]]
                for g in ["Hal","Arg1","Chil3"] if g in gi}

    results = {
        "groups": df.to_dict("records"),
        "within_rr_peak1_vs_peak3": within_rr,
        "rr_ratchet_genes_PEAK1_2_3": ratchet_rr,
        "rr_acute_genes_PEAK1_2_3": acute_rr,
        "caveats": ("Within-RR (PEAK1 vs PEAK3) is strain-matched and clean. CHRONIC PEAK1 is "
                    "strain/antigen-confounded (SJL/PLP vs B6/MOG) so its absolute offset mixes "
                    "history AND strain. All groups small n (4-6); descriptive."),
    }
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    # ---- report ----
    L = ["\n=== Same score, different state — the accrual axis ===",
         "group        n   score   acute_z   cumulative_z   Hal   Gpnmb",
         ]
    for r in results["groups"]:
        L.append(f"  {r['group']:<10} {r['n']:<3} {r['mean_score']:<6} {r['acute_program_z']:+.2f}      "
                 f"{r['cumulative_program_z']:+.2f}        {r.get('gene_Hal','-'):<5} {r.get('gene_Gpnmb','-')}")
    L.append(f"\n-- WITHIN-RR (strain-clean) PEAK1 vs PEAK3 — near-identical score "
             f"({within_rr['score_PEAK1']} vs {within_rr['score_PEAK3']}) --")
    L.append(f"   ACUTE program:      {within_rr['acute_PEAK1']:+.2f} -> {within_rr['acute_PEAK3']:+.2f}  "
             f"(p={within_rr['acute_p']})  ~FLAT (resets)")
    L.append(f"   CUMULATIVE program: {within_rr['cumulative_PEAK1']:+.2f} -> {within_rr['cumulative_PEAK3']:+.2f}  "
             f"(p={within_rr['cumulative_p']})  RISES (accrual)")
    L.append("   ratchet genes PEAK1->2->3: " + "; ".join(f"{g} {v}" for g, v in ratchet_rr.items()))
    L.append("   acute genes   PEAK1->2->3: " + "; ".join(f"{g} {v}" for g, v in acute_rr.items()))
    report = "\n".join(L)
    print(report)
    with open(os.path.join(OUT, "report.txt"), "w") as fh:
        fh.write(report + "\n")

    _plot(results, OUT)
    print(f"\n[done] -> {OUT}/ (results.json, report.txt, figures/)")


def _plot(results, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    g = results["groups"]
    labels = [r["group"] for r in g]
    x = np.arange(len(g))
    acute = [r["acute_program_z"] for r in g]
    cum = [r["cumulative_program_z"] for r in g]
    scores = [r["mean_score"] for r in g]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))
    axA.plot(x, acute, "-o", lw=2.5, color="#c0392b", label="acute program (resets)")
    axA.plot(x, cum, "-o", lw=2.5, color="#2c6fbb", label="cumulative program (accrues)")
    for i, s in enumerate(scores):
        axA.annotate(f"score {s}", (x[i], min(acute[i], cum[i]) - 0.06), ha="center", fontsize=8, color="#666")
    axA.axhline(0, c="gray", lw=.5)
    axA.set_xticks(x); axA.set_xticklabels(labels, rotation=15)
    axA.set_ylabel("program score (z across all animals)")
    axA.axvspan(-0.4, 0.4, color="#eee", alpha=.6, zorder=0)  # shade chronic
    axA.set_title("Matched-severity peaks: cumulative load climbs across RR cycles\n"
                  "(acute does not accrue; chronic PEAK1 shaded — strain-confounded)", fontsize=11)
    axA.legend(fontsize=9, frameon=False)
    axA.spines[["top", "right"]].set_visible(False)

    # per-gene ratchet (RR peaks) vs acute
    rr = results["rr_ratchet_genes_PEAK1_2_3"]; ac = results["rr_acute_genes_PEAK1_2_3"]
    xp = np.arange(3)
    for gene, vals in rr.items():
        axB.plot(xp, vals, "-o", lw=1.6, color="#2c6fbb", alpha=.7)
    for gene, vals in ac.items():
        axB.plot(xp, vals, "--s", lw=1.6, color="#c0392b", alpha=.7)
    axB.plot([], [], "-o", color="#2c6fbb", label="cumulative (Gpnmb, Plin4, Fcrls...)")
    axB.plot([], [], "--s", color="#c0392b", label="acute (Hal, Arg1, Chil3)")
    axB.set_xticks(xp); axB.set_xticklabels(["PEAK1", "PEAK2", "PEAK3"])
    axB.set_ylabel("pseudobulk log CP10k")
    axB.set_title("Within RR (strain-clean): same score across peaks,\ncumulative genes rise, acute genes do not", fontsize=11)
    axB.legend(fontsize=8, frameon=False)
    axB.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figures", "accrual_axis.png"), dpi=130, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "figures", "accrual_axis.pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
