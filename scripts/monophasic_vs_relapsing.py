"""Why does an SJL/PLP animal relapse or stay monophasic? — REM1-anchored trajectories.

The only identifiable "why relapse vs not" contrast is WITHIN the relapsing-remitting cohort
(RR vs chronic is strain-confounded). Experimental design (confirmed in the data):

    REMISSION1 (~day 22)  ──>  PEAK2      (~day 32, score 2.44)   = the animal RELAPSED
                          └──>  MONOPHASIC (~day 33, score 0.62)   = the animal did NOT relapse

Monophasic animals are collected after remission I and at ~the same timepoint as PEAK2
(day MW p≈0.16), confirming the non-relapsing phenotype — at the time a relapser is in its
second attack, the monophasic animal is still low-severity. So:

  * the TIME-MATCHED endpoint contrast is MONOPHASIC vs PEAK2 (same day, opposite course);
  * the biology of the relapse decision lives in the two trajectories FROM the common REM1
    origin: REM1->MONOPHASIC (no relapse) vs REM1->PEAK2 (relapse). Genes/modules that rise
    into PEAK2 but not into MONOPHASIC are relapse-associated; the reverse are
    resolution/non-relapse-associated.

IMPORTANT framing: this is terminal cross-sectional data — REM1, MONOPHASIC and PEAK2 are
DIFFERENT animals, so these are population-level group trajectories, not within-animal paths,
and we cannot label a REM1 animal as future-relapsing. The MONO-vs-PEAK2 endpoint also differs
in severity (PEAK2 is a peak), so that axis is partly the acute relapse/severity program by
construction. n is small (4 / 4 / 5) -> EXPLORATORY, effect sizes not significance.

    PYTHONPATH="$PWD" python scripts/monophasic_vs_relapsing.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

RRPB = "runs/rr_within_relapse/pseudobulk.npz"
META = "runs/duration_clock/rr_meta.csv"
DECOMP = "runs/duration_gene_decomposition/gene_day_decomposition.csv"
OSC = "runs/rr_cycle_oscillation/gene_oscillation_metrics.csv"
OUT = "runs/monophasic_vs_relapsing"

# biological modules (duration-clock modules + acute-oscillator + glucocorticoid/stress)
MODULES = {
    "neuro/myelin": ["Mog", "Mal", "Uchl1", "Syn1", "Nptx2", "Gad1", "Gad2", "Slc17a6",
                     "Kif5a", "Tubb2b", "Opalin"],
    "innate/IFN": ["Ccr2", "Tmem173", "Ifit1", "Ifit3", "Irf5", "Tlr2", "Tmem119", "Siglech",
                   "Gbp2", "Cxcl10", "Irgm1", "Isg15", "Usp18", "Stat2", "Oasl2"],
    "acute-oscillator": ["Arg1", "Chil1", "Chil3", "Timp1", "Acod1", "Ccl2", "Serpina3n", "Hal"],
    "ECM/scar": ["Thbs2", "Serpine2", "Ptx3", "Fbln2", "Hpse", "Klf5", "Col4a1", "Col4a2", "Eln"],
    "lymphoid": ["Cxcl13", "Cxcl12", "Tcf7", "Il2ra", "Mzb1", "Jchain"],
    "lipid-scavenge": ["Lpl", "Pltp", "Abca8a", "Srebf1", "Hmgcr", "Hsd17b7", "Idi1", "Msmo1",
                       "Ldlr", "Acss2", "Plin4", "Abca1", "Mertk"],
    "circ/repair": ["Nr1d1", "Dbp", "Bhlhe40", "Per1", "Hlf", "Txnip", "Igf1", "Igfbp2",
                    "Il33", "S1pr3", "Mlc1"],
    "GC/stress": ["Sgk1", "Fkbp5", "Tsc22d3", "Zbtb16", "Ddit4", "Hspa1a", "Hspa1b", "Mt1"],
}


def mw_effect(A, B):
    nA, nB = A.shape[0], B.shape[0]
    p = np.ones(A.shape[1]); auc = np.full(A.shape[1], 0.5)
    for j in range(A.shape[1]):
        a, b = A[:, j], B[:, j]
        if np.ptp(np.concatenate([a, b])) == 0:
            continue
        try:
            u, pj = mannwhitneyu(a, b, alternative="two-sided")
            p[j] = pj; auc[j] = u / (nA * nB)
        except ValueError:
            pass
    return p, auc, A.mean(0) - B.mean(0)


def main():
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    d = np.load(RRPB, allow_pickle=True)
    pb, genes, animals = d["pb"].astype(float), d["genes"].astype(str), d["animals"].astype(str)
    stage = d["stage"].astype(str)
    meta = pd.read_csv(META).set_index("sample_name").reindex(animals)
    day = meta["day_of_sacrifice"].to_numpy(float)
    score = meta["score_sacrifice"].to_numpy(float)
    gi = {g: i for i, g in enumerate(genes)}

    m_rem1 = stage == "REMISSION1"
    m_mono = stage == "MONOPHASIC"
    m_pk2 = stage == "PEAK2"
    grp = {"REM1": m_rem1, "MONO": m_mono, "PEAK2": m_pk2}
    print("[groups] " + "  ".join(f"{k} n={v.sum()}" for k, v in grp.items()))

    timing = {k: {"n": int(v.sum()), "day_mean": float(day[v].mean()),
                  "score_mean": float(score[v].mean())} for k, v in grp.items()}
    timing["MONO_vs_PEAK2_day_MW_p"] = float(mannwhitneyu(day[m_mono], day[m_pk2]).pvalue)
    timing["MONO_vs_PEAK2_score_MW_p"] = float(mannwhitneyu(score[m_mono], score[m_pk2]).pvalue)

    # ---- per-gene group means + the two trajectories from REM1 ----
    mean_rem1 = pb[m_rem1].mean(0)
    mean_mono = pb[m_mono].mean(0)
    mean_pk2 = pb[m_pk2].mean(0)
    d_mono = mean_mono - mean_rem1          # REM1 -> monophasic (no relapse)
    d_pk2 = mean_pk2 - mean_rem1            # REM1 -> peak2 (relapse)
    divergence = d_pk2 - d_mono             # relapse-specific direction (= PEAK2 - MONO)

    # time-matched endpoint contrast (effect size + MW)
    p_endpoint, auc_endpoint, diff_endpoint = mw_effect(pb[m_pk2], pb[m_mono])

    df = pd.DataFrame({
        "gene": genes, "mean_REM1": mean_rem1, "mean_MONO": mean_mono, "mean_PEAK2": mean_pk2,
        "d_REM1_to_MONO": d_mono, "d_REM1_to_PEAK2": d_pk2, "divergence_PEAK2_minus_MONO": divergence,
        "endpoint_PEAK2_minus_MONO": diff_endpoint, "endpoint_auc_PEAK2_gt_MONO": auc_endpoint,
        "endpoint_mw_p": p_endpoint})

    # classify by trajectory (threshold on the REM1-anchored deltas)
    TH = 0.25
    relapse_up = (d_pk2 >= TH) & (d_mono < TH / 2)        # rises into relapse only
    mono_up = (d_mono >= TH) & (d_pk2 < TH / 2)           # rises into non-relapse only
    shared_up = (d_pk2 >= TH) & (d_mono >= TH)            # rises into both (time/duration)
    df["trajectory_class"] = np.select(
        [relapse_up, mono_up, shared_up], ["relapse_specific", "monophasic_specific", "shared_up"],
        default="other")

    # ---- module trajectories (z-scored within RR, averaged per module per group) ----
    z = (pb - pb.mean(0)) / (pb.std(0) + 1e-9)
    mod_traj = {}
    for name, gs in MODULES.items():
        idx = [gi[g] for g in gs if g in gi]
        if not idx:
            continue
        zs = z[:, idx].mean(1)
        mod_traj[name] = {"REM1": float(zs[m_rem1].mean()), "MONO": float(zs[m_mono].mean()),
                          "PEAK2": float(zs[m_pk2].mean()), "n_genes": len(idx)}

    # ---- relate the relapse-specific direction to the disease axes ----
    axis = {}
    if os.path.exists(DECOMP):
        dec = pd.read_csv(DECOMP)[["gene", "r_day_given_score", "r_score_given_day"]]
        df = df.merge(dec, on="gene", how="left")
    if os.path.exists(OSC):
        osc = pd.read_csv(OSC)[["gene", "amplitude"]].rename(columns={"amplitude": "rr_osc_amp"})
        df = df.merge(osc, on="gene", how="left")
    sub = df.dropna(subset=["rr_osc_amp", "r_score_given_day"])
    if len(sub):
        axis["divergence_vs_oscillation_amp"] = float(spearmanr(sub.divergence_PEAK2_minus_MONO, sub.rr_osc_amp).statistic)
        axis["divergence_vs_severity_axis"] = float(spearmanr(sub.divergence_PEAK2_minus_MONO, sub.r_score_given_day).statistic)
        axis["divergence_vs_duration_axis"] = float(spearmanr(sub.divergence_PEAK2_minus_MONO, sub.r_day_given_score).statistic)

    df.sort_values("divergence_PEAK2_minus_MONO", ascending=False).to_csv(
        os.path.join(OUT, "rem1_trajectory_genes.csv"), index=False)

    relapse_genes = df[df.trajectory_class == "relapse_specific"].sort_values(
        "d_REM1_to_PEAK2", ascending=False)
    mono_genes = df[df.trajectory_class == "monophasic_specific"].sort_values(
        "d_REM1_to_MONO", ascending=False)

    results = {
        "design": ("REM1 (~day22) splits into MONOPHASIC (~day33, no relapse) and PEAK2 "
                   "(~day32, relapse). MONO and PEAK2 are time-matched (day MW p="
                   f"{timing['MONO_vs_PEAK2_day_MW_p']:.2f}) but differ in severity (p="
                   f"{timing['MONO_vs_PEAK2_score_MW_p']:.3f}). Cross-sectional: different "
                   "animals, population trajectories."),
        "timing": timing,
        "n_relapse_specific": int(relapse_up.sum()),
        "n_monophasic_specific": int(mono_up.sum()),
        "n_shared_up": int(shared_up.sum()),
        "module_trajectories": mod_traj,
        "axis_alignment": axis,
        "top_relapse_specific": relapse_genes.head(25)[
            ["gene", "d_REM1_to_PEAK2", "d_REM1_to_MONO", "rr_osc_amp"]].round(3).to_dict("records"),
        "top_monophasic_specific": mono_genes.head(25)[
            ["gene", "d_REM1_to_MONO", "d_REM1_to_PEAK2"]].round(3).to_dict("records"),
        "note": ("EXPLORATORY n=4/4/5; effect sizes not significance. The relapse endpoint "
                 "(PEAK2) is severe by construction, so relapse_specific genes are expected to "
                 "overlap the acute severity/oscillation program — that is the point, not a bug."),
    }
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    _plot(mod_traj, relapse_genes, mono_genes, df, timing, OUT)
    _report(results)
    print(f"[done] -> {OUT}/")


def _plot(mod_traj, relapse_genes, mono_genes, df, timing, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.2, 6.0),
                                 gridspec_kw={"width_ratios": [1.05, 1]})

    # left: module trajectories REM1 -> {MONO, PEAK2}
    xs = [0, 1]
    import itertools
    palette = ["#c0392b", "#e8590c", "#d6336c", "#1098ad", "#2f9e44", "#e88a00",
               "#2c6fbb", "#7048e8"]
    for (name, t), c in zip(mod_traj.items(), itertools.cycle(palette)):
        a1.plot(xs, [t["REM1"], t["PEAK2"]], "-o", color=c, lw=2, ms=5)
        a1.plot(xs, [t["REM1"], t["MONO"]], "--s", color=c, lw=1.6, ms=4, alpha=.8)
        a1.annotate(f" {name}", (1, t["PEAK2"]), fontsize=7.5, color=c, va="center")
    a1.set_xticks(xs); a1.set_xticklabels(["REM1 (origin)", "endpoint (~day 32)"])
    a1.set_ylabel("module score (z, RR-wide)")
    a1.set_title("Module trajectories from REM1\nsolid = →PEAK2 (relapse) · dashed = →MONO "
                 "(no relapse)", fontsize=10.5)
    a1.axhline(0, color="#bbb", lw=.6)
    a1.spines[["top", "right"]].set_visible(False)

    # right: per-gene trajectory scatter d_mono (x) vs d_pk2 (y)
    cmap = {"relapse_specific": "#c0392b", "monophasic_specific": "#2f9e44",
            "shared_up": "#e8950c", "other": "#cfcfcf"}
    for cls, c in cmap.items():
        s = df[df.trajectory_class == cls]
        a2.scatter(s.d_REM1_to_MONO, s.d_REM1_to_PEAK2, s=12 if cls != "other" else 6,
                   c=c, alpha=.8 if cls != "other" else .4,
                   label=f"{cls.replace('_', ' ')}" + (f" (n={len(s)})" if cls != "other" else ""),
                   zorder=3 if cls != "other" else 1)
    lab = pd.concat([relapse_genes.head(10), mono_genes.head(8)])
    for r in lab.itertuples():
        a2.annotate(r.gene, (r.d_REM1_to_MONO, r.d_REM1_to_PEAK2), fontsize=7,
                    xytext=(2, 2), textcoords="offset points")
    lim = 1.05 * max(df.d_REM1_to_MONO.abs().max(), df.d_REM1_to_PEAK2.abs().max())
    a2.plot([-lim, lim], [-lim, lim], "--", color="grey", lw=.7)
    a2.axhline(0, color="#bbb", lw=.6); a2.axvline(0, color="#bbb", lw=.6)
    a2.set_xlabel("REM1 → MONOPHASIC  (no relapse)")
    a2.set_ylabel("REM1 → PEAK2  (relapse)")
    a2.set_title("Per-gene divergence from the REM1 origin\nabove diagonal = relapse-specific",
                 fontsize=10.5)
    a2.legend(fontsize=8, frameon=False, loc="lower right")
    a2.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Why relapse or not — within SJL/PLP (MONO & PEAK2 time-matched, "
                 f"day MW p={timing['MONO_vs_PEAK2_day_MW_p']:.2f}; exploratory n=4/4/5)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figures", "monophasic_vs_relapsing.png"),
                dpi=135, bbox_inches="tight")
    plt.close(fig)


def _report(r):
    t = r["timing"]; ax = r["axis_alignment"]
    L = ["\n=== Why relapse or not — REM1-anchored, within SJL/PLP (EXPLORATORY) ==="]
    L.append(r["design"])
    L.append("  " + "  ".join(f"{k}: n={t[k]['n']} day~{t[k]['day_mean']:.0f} score~{t[k]['score_mean']:.2f}"
                              for k in ["REM1", "MONO", "PEAK2"]))
    L.append(f"  trajectory genes: relapse-specific={r['n_relapse_specific']}  "
             f"monophasic-specific={r['n_monophasic_specific']}  shared-up={r['n_shared_up']}")
    if ax:
        L.append("\nrelapse-specific direction (divergence = PEAK2-MONO) alignment:")
        L.append(f"   vs oscillation amplitude : {ax['divergence_vs_oscillation_amp']:+.2f}")
        L.append(f"   vs severity axis         : {ax['divergence_vs_severity_axis']:+.2f}")
        L.append(f"   vs duration axis         : {ax['divergence_vs_duration_axis']:+.2f}")
    L.append("\n-- module trajectories (z): REM1 -> MONO / PEAK2 --")
    for name, m in r["module_trajectories"].items():
        L.append(f"   {name:<18} REM1={m['REM1']:+.2f}  MONO={m['MONO']:+.2f}  PEAK2={m['PEAK2']:+.2f}")
    L.append("\n-- RELAPSE-specific (rise REM1->PEAK2, not REM1->MONO) --")
    for g in r["top_relapse_specific"][:12]:
        L.append(f"   {g['gene']:<14} d->PEAK2={g['d_REM1_to_PEAK2']:+.2f}  d->MONO={g['d_REM1_to_MONO']:+.2f}")
    L.append("\n-- MONOPHASIC-specific (rise REM1->MONO, not REM1->PEAK2) --")
    for g in r["top_monophasic_specific"][:12]:
        L.append(f"   {g['gene']:<14} d->MONO={g['d_REM1_to_MONO']:+.2f}  d->PEAK2={g['d_REM1_to_PEAK2']:+.2f}")
    report = "\n".join(L)
    print(report)
    with open(os.path.join(OUT, "report.txt"), "w") as fh:
        fh.write(report + "\n")


if __name__ == "__main__":
    main()
