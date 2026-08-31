"""The duration clock + gene decomposition for the CHRONIC arm, and RR↔chronic relating.

The headline duration clock was built on RR only, because the chronic arm's day axis was
believed to alias run_date. With the corrected metadata
(FINAL_..._META_RRMap2.Main.filled.stage_fixed.csv) the chronic cohort has a CONTINUOUS
day_of_sacrifice (8–50 dpi, 18 values) that is clean of batch (Kruskal day~run_date
p≈0.57; day~batch ρ≈0.10) and decoupled from severity (r≈0.31) — the earlier "alias" was
an artifact of the coarse MILD16/MILD30 stage grouping, not the underlying day.

So we can run the SAME analysis on chronic and relate the two models:
  * Clock A / Clock B (severity-orthogonalized), LOAO + permutation null  — reuses
    scripts/duration_clock.py.
  * Gene-level day decomposition: r(gene, day|score), r(gene, score|day)    — reuses
    scripts/duration_gene_decomposition.py.
  * CROSS-MODEL: correlate the chronic duration axis against the RR duration axis
    (does the same program accrue with duration in both models / strains?), and against
    the RR severity-oscillation axis as a contrast.

Chronic is B6/MOG; RR is SJL/PLP. A direct level comparison is confounded by strain+slide,
so we relate per-gene SLOPES (the within-model day trends), which cancels the strain offset.

    PYTHONPATH="$PWD" python scripts/duration_clock_chronic.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kruskal

# reuse the exact RR clock + decomposition machinery (pure functions; main() is guarded)
from duration_clock import loao_clock, loao_target, final_coefs, N_PERM
from duration_gene_decomposition import partial_corr_vec, bh_fdr

CHRONIC_PB = "runs/chronic_trajectory/pseudobulk.npz"
META = "runs/duration_clock_chronic/chronic_meta.csv"
RR_CLOCK = "runs/duration_clock/clock_genes.csv"
RR_DECOMP = "runs/duration_gene_decomposition/gene_day_decomposition.csv"
RR_OSC = "runs/rr_cycle_oscillation/gene_oscillation_metrics.csv"
OUT = "runs/duration_clock_chronic"


def main():
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    d = np.load(CHRONIC_PB, allow_pickle=True)
    pb, genes, animals = d["pb"].astype(float), d["genes"].astype(str), d["animals"].astype(str)
    run_date = d["run_date"].astype(str)
    meta = pd.read_csv(META).set_index("sample_name").reindex(animals)
    day = meta["day_of_sacrifice"].to_numpy(float)
    score = meta["score_sacrifice"].to_numpy(float)
    keep = np.isfinite(day) & np.isfinite(score)
    pb, animals, day, score, run_date = pb[keep], animals[keep], day[keep], score[keep], run_date[keep]
    n = len(day)
    sc_col = score.reshape(-1, 1)
    print(f"[load] CHRONIC n={n}, {pb.shape[1]} genes, day {day.min():.0f}-{day.max():.0f}, "
          f"corr(day,score)={np.corrcoef(day, score)[0, 1]:+.2f}", flush=True)

    # ---- clocks (identical pipeline to RR) ----
    predA = loao_clock(pb, day)
    rhoA = spearmanr(predA, day).statistic
    predB = loao_clock(pb, day, covar=sc_col)
    dayB = loao_target(day, sc_col)
    rhoB = spearmanr(predB, dayB).statistic
    r2B = 1 - np.sum((predB - dayB) ** 2) / np.sum((dayB - dayB.mean()) ** 2)
    predS = loao_clock(sc_col, day)
    rhoS = spearmanr(predS, day).statistic
    predX = loao_clock(pb, score, covar=day.reshape(-1, 1))
    rhoX = spearmanr(predX, loao_target(score, day.reshape(-1, 1))).statistic
    print(f"[clock] A={rhoA:+.3f}  B(sev-orth)={rhoB:+.3f}  baseline(day~score)={rhoS:+.3f}  "
          f"cross={rhoX:+.3f}", flush=True)

    # ---- permutation null on clock B ----
    rng = np.random.default_rng(0)
    perm = np.empty(N_PERM)
    for k in range(N_PERM):
        yp = day[rng.permutation(n)]
        perm[k] = spearmanr(loao_clock(pb, yp, covar=sc_col), loao_target(yp, sc_col)).statistic
        if (k + 1) % 10 == 0:
            print(f"[perm] {k + 1}/{N_PERM}", flush=True)
    p_perm = (1 + np.sum(perm >= rhoB)) / (1 + N_PERM)

    # ---- batch leakage: day ~ run_date ----
    grp = [day[run_date == u] for u in np.unique(run_date) if (run_date == u).sum() >= 2]
    kruskal_p = float(kruskal(*grp).pvalue) if len(grp) >= 2 else None

    # ---- clock-gene signature ----
    coefs, alphaB = final_coefs(pb, day, genes, covar=sc_col)
    coefs.to_csv(os.path.join(OUT, "chronic_clock_genes.csv"), index=False)

    # ---- gene-level day decomposition (severity-orthogonalized) ----
    r_day, _ = partial_corr_vec(pb, day, score)
    r_score, _ = partial_corr_vec(pb, score, day)
    df = pd.DataFrame({"gene": genes, "r_day_given_score": r_day, "r_score_given_day": r_score})
    df.sort_values("r_day_given_score", ascending=False).to_csv(
        os.path.join(OUT, "chronic_gene_day_decomposition.csv"), index=False)

    # ===================== CROSS-MODEL RELATING =====================
    rel = {}
    rr_dec = pd.read_csv(RR_DECOMP)[["gene", "r_day_given_score"]].rename(
        columns={"r_day_given_score": "rr_day"})
    ch_dec = df[["gene", "r_day_given_score"]].rename(columns={"r_day_given_score": "ch_day"})
    mrg = rr_dec.merge(ch_dec, on="gene")
    rho_day = spearmanr(mrg.rr_day, mrg.ch_day)
    rel["rr_duration_vs_chronic_duration"] = {
        "spearman": float(rho_day.statistic), "p": float(rho_day.pvalue), "n_genes": int(len(mrg)),
        "sign_agreement": float((np.sign(mrg.rr_day) == np.sign(mrg.ch_day)).mean())}

    # contrast: RR severity-oscillation axis vs chronic duration (should be weaker / opposite)
    if os.path.exists(RR_OSC):
        osc = pd.read_csv(RR_OSC)[["gene", "amplitude"]].rename(columns={"amplitude": "rr_amp"})
        m2 = osc.merge(ch_dec, on="gene")
        rho_amp = spearmanr(m2.rr_amp, m2.ch_day)
        rel["rr_oscillation_amp_vs_chronic_duration"] = {
            "spearman": float(rho_amp.statistic), "p": float(rho_amp.pvalue), "n_genes": int(len(m2))}

    # clock-coefficient agreement over the RR clock genes
    rr_clock = pd.read_csv(RR_CLOCK).rename(columns={"coef": "rr_coef"})
    cc = rr_clock.merge(df, on="gene")
    if len(cc):
        rho_cc = spearmanr(cc.rr_coef, cc.r_day_given_score)
        rel["rr_clock_coef_vs_chronic_day_partial"] = {
            "spearman": float(rho_cc.statistic), "p": float(rho_cc.pvalue),
            "n_clock_genes": int(len(cc)),
            "sign_agreement": float((np.sign(cc.rr_coef) == np.sign(cc.r_day_given_score)).mean())}

    # conserved accrual / decline (strong day trend, same direction in BOTH models)
    TH = 0.30
    mrg["conserved"] = np.where((mrg.rr_day >= TH) & (mrg.ch_day >= TH), "accrue_both",
                        np.where((mrg.rr_day <= -TH) & (mrg.ch_day <= -TH), "decline_both", "other"))
    acc = mrg[mrg.conserved == "accrue_both"].assign(m=lambda x: x[["rr_day", "ch_day"]].min(1)) \
        .sort_values("m", ascending=False)
    dec = mrg[mrg.conserved == "decline_both"].assign(m=lambda x: x[["rr_day", "ch_day"]].max(1)) \
        .sort_values("m")
    rel["n_conserved_accrue"] = int((mrg.conserved == "accrue_both").sum())
    rel["n_conserved_decline"] = int((mrg.conserved == "decline_both").sum())
    rel["top_conserved_accrue"] = acc.head(25)[["gene", "rr_day", "ch_day"]].to_dict("records")
    rel["top_conserved_decline"] = dec.head(25)[["gene", "rr_day", "ch_day"]].to_dict("records")
    # model-specific: accrues with duration in chronic but not RR, and vice versa
    rel["chronic_specific_accrue"] = mrg[(mrg.ch_day >= 0.45) & (mrg.rr_day.abs() < 0.15)] \
        .sort_values("ch_day", ascending=False).head(15)[["gene", "ch_day", "rr_day"]].to_dict("records")
    rel["rr_specific_accrue"] = mrg[(mrg.rr_day >= 0.45) & (mrg.ch_day.abs() < 0.15)] \
        .sort_values("rr_day", ascending=False).head(15)[["gene", "rr_day", "ch_day"]].to_dict("records")

    results = {
        "n_animals": n, "day_range": [float(day.min()), float(day.max())],
        "corr_day_score": float(np.corrcoef(day, score)[0, 1]),
        "model": "CHRONIC (B6/MOG)",
        "clock_A_raw": {"spearman": round(float(rhoA), 3)},
        "clock_B_severity_orthogonalized": {
            "spearman": round(float(rhoB), 3), "r2_loao": round(float(r2B), 3),
            "perm_p": round(float(p_perm), 4), "perm_rho_mean": round(float(perm.mean()), 3),
            "n_clock_genes": int(len(coefs)), "alpha": round(alphaB, 5)},
        "severity_only_baseline_spearman": round(float(rhoS), 3),
        "cross_target_predict_score_spearman": round(float(rhoX), 3),
        "batch_leakage_day_run_date_kruskal_p": kruskal_p,
        "top_clock_genes": coefs.head(25).round(4).to_dict("records"),
        "cross_model": rel,
    }
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    _plot(mrg, OUT)
    _report(results)
    print(f"[done] -> {OUT}/")


def _plot(mrg, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    ax.scatter(mrg.rr_day, mrg.ch_day, s=7, c="#cfcfcf", zorder=1)
    cols = {"accrue_both": "#c0392b", "decline_both": "#2c6fbb"}
    for cls, c in cols.items():
        s = mrg[mrg.conserved == cls]
        ax.scatter(s.rr_day, s.ch_day, s=16, c=c, alpha=.7, zorder=2,
                   label=f"{cls.replace('_', ' ')} (n={len(s)})")
    lab = pd.concat([
        mrg[mrg.conserved == "accrue_both"].assign(m=lambda x: x[["rr_day", "ch_day"]].min(1))
            .sort_values("m", ascending=False).head(10),
        mrg[mrg.conserved == "decline_both"].assign(m=lambda x: x[["rr_day", "ch_day"]].max(1))
            .sort_values("m").head(10)])
    for r in lab.itertuples():
        ax.annotate(r.gene, (r.rr_day, r.ch_day), fontsize=7, xytext=(3, 2),
                    textcoords="offset points")
    rho = spearmanr(mrg.rr_day, mrg.ch_day).statistic
    ax.axhline(0, color="#bbb", lw=.6); ax.axvline(0, color="#bbb", lw=.6)
    ax.plot([-.8, .9], [-.8, .9], "--", color="grey", lw=.8)
    ax.set_xlabel("RR duration axis   r(gene, day | score)")
    ax.set_ylabel("CHRONIC duration axis   r(gene, day | score)")
    ax.set_title(f"The duration program is conserved across models\n"
                 f"RR (SJL/PLP) vs chronic (B6/MOG)  ·  Spearman = {rho:+.2f}", fontsize=12)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figures", "rr_vs_chronic_duration.png"),
                dpi=135, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "figures", "rr_vs_chronic_duration.pdf"), bbox_inches="tight")
    plt.close(fig)


def _report(r):
    cb = r["clock_B_severity_orthogonalized"]
    cm = r["cross_model"]
    L = [f"\n=== CHRONIC duration clock + RR↔chronic relating (n={r['n_animals']}) ==="]
    L.append(f"day {r['day_range'][0]:.0f}-{r['day_range'][1]:.0f} dpi  corr(day,score)={r['corr_day_score']:+.2f}  "
             f"batch day~run_date Kruskal p={r['batch_leakage_day_run_date_kruskal_p']}")
    L.append(f"Clock A raw            : Spearman={r['clock_A_raw']['spearman']:+.3f}")
    L.append(f"Clock B (sev-orth)     : Spearman={cb['spearman']:+.3f}  R2={cb['r2_loao']:+.3f}  "
             f"perm p={cb['perm_p']:.4f}  ({cb['n_clock_genes']} genes)")
    L.append(f"severity-only baseline : Spearman={r['severity_only_baseline_spearman']:+.3f}")
    L.append(f"cross-target           : Spearman={r['cross_target_predict_score_spearman']:+.3f}")
    d = cm["rr_duration_vs_chronic_duration"]
    L.append(f"\n-- CROSS-MODEL --")
    L.append(f"RR duration axis  vs  chronic duration axis : Spearman={d['spearman']:+.3f} "
             f"(p={d['p']:.1e}, {d['n_genes']} genes, sign-agree {d['sign_agreement']:.0%})")
    if "rr_oscillation_amp_vs_chronic_duration" in cm:
        o = cm["rr_oscillation_amp_vs_chronic_duration"]
        L.append(f"RR severity-oscillation vs chronic duration : Spearman={o['spearman']:+.3f} (contrast)")
    if "rr_clock_coef_vs_chronic_day_partial" in cm:
        c = cm["rr_clock_coef_vs_chronic_day_partial"]
        L.append(f"RR clock coef vs chronic day-partial        : Spearman={c['spearman']:+.3f} "
                 f"({c['n_clock_genes']} clock genes, sign-agree {c['sign_agreement']:.0%})")
    L.append(f"\nconserved ACCRUE in both (n={cm['n_conserved_accrue']}): "
             + ", ".join(x["gene"] for x in cm["top_conserved_accrue"][:15]))
    L.append(f"conserved DECLINE in both (n={cm['n_conserved_decline']}): "
             + ", ".join(x["gene"] for x in cm["top_conserved_decline"][:15]))
    L.append("chronic-specific accrue: " + ", ".join(x["gene"] for x in cm["chronic_specific_accrue"][:10]))
    L.append("RR-specific accrue: " + ", ".join(x["gene"] for x in cm["rr_specific_accrue"][:10]))
    report = "\n".join(L)
    print(report)
    with open(os.path.join(OUT, "report.txt"), "w") as fh:
        fh.write(report + "\n")


if __name__ == "__main__":
    main()
