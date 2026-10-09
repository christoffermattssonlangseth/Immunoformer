"""Figure 4 — what is changing.
4a cell-type proportions vs day; 4b endothelial Col4a1+Col4a2 vs day in RR coloured by severity
(chronic shown and flagged as not interpretable); 4c within severity tertiles; 4d severity-
matched pairs; 4e rank of Col4a1/Col4a2 in the all-gene and severity-matched nulls.

    PYTHONPATH="$PWD" python analysis/figures/fig4.py
"""

from __future__ import annotations

import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from common import COL, INK, MUTED, animals, corner, letter, load_json, plt, rho_label, save  # noqa: E402

PAIR_TOL = 0.25


def main():
    A = animals()
    S = load_json("runs/closing_block/severity_adjustment/results.json")
    fig = plt.figure(figsize=(7.4, 11))
    gs = fig.add_gridspec(4, 3, hspace=0.95, wspace=0.55, height_ratios=[0.85, 1, 0.8, 0.85])
    # 4a
    for k, (c, lab) in enumerate((("prop_Fibroblast", "fibroblasts"), ("prop_Astrocyte", "astrocytes"),
                                  ("prop_Endothelial", "endothelial cells"))):
        ax = fig.add_subplot(gs[0, k])
        txt = []
        for ck in ("RR", "chronic"):
            d = A[A.cohort == ck]
            ax.scatter(d.day, 100 * d[c], s=9, color=COL[ck], edgecolor="white", lw=0.3, label=ck)
            txt.append(f"{ck} {rho_label(d[c], d.day)}")
        ax.set_xlabel("day of sacrifice"); ax.set_ylabel(f"% {lab}")
        ax.set_title(lab + "\n" + "\n".join(txt), loc="left", fontsize=6)
        if k == 0:
            letter(ax, "a")
    # 4b RR + chronic (flagged)
    for k, ck in enumerate(("RR", "chronic")):
        ax = fig.add_subplot(gs[1, k])
        d = A[A.cohort == ck]
        sc = ax.scatter(d.day, d.endo_Col4_mean, c=d.score, cmap="cividis", s=18, edgecolor="white", lw=0.4,
                        vmin=0, vmax=3.25)
        ax.set_xlabel("day of sacrifice"); ax.set_ylabel("endothelial Col4a1+Col4a2\n(log CP10k, animal mean)")
        u = S[ck]["1_unadjusted"]["Col4a1+Col4a2 mean"]
        stat = f"unadjusted ρ {u['rho_day']:+.2f} [{u['ci_day'][0]:+.2f}, {u['ci_day'][1]:+.2f}], n = {S[ck]['n']}"
        if ck == "RR":
            ax.set_title("RR: falls with duration\n" + stat, loc="left", fontsize=6.5)
        else:
            ax.set_title("chronic: NOT INTERPRETABLE\n(severity~day ρ +0.58)\n" + stat, loc="left", fontsize=6.5,
                         color="#7a3b00")
        if k == 0:
            letter(ax, "b")
            rr_ax = ax
    cax = rr_ax.inset_axes([0.78, 0.55, 0.04, 0.4])
    cb_ = fig.colorbar(sc, cax=cax)
    cb_.set_label("score", fontsize=5.5); cb_.ax.tick_params(labelsize=5)
    # 4d matched pairs (RR)
    ax = fig.add_subplot(gs[1, 2])
    d = A[A.cohort == "RR"].reset_index(drop=True)
    for i, j in itertools.combinations(range(len(d)), 2):
        if abs(d.score[i] - d.score[j]) <= PAIR_TOL and d.day[i] != d.day[j]:
            e, l_ = (i, j) if d.day[i] < d.day[j] else (j, i)
            down = d.endo_Col4_mean[l_] < d.endo_Col4_mean[e]
            ax.annotate("", xy=(d.day[l_], d.endo_Col4_mean[l_]), xytext=(d.day[e], d.endo_Col4_mean[e]),
                        arrowprops=dict(arrowstyle="-|>", lw=0.5, color=COL["RR"] if down else "#b0b0b0",
                                        alpha=0.55, mutation_scale=5))
    ax.scatter(d.day, d.endo_Col4_mean, s=6, color=INK, zorder=3)
    mp = S["RR"]["2_matched_pairs"]
    ax.set_xlabel("day of sacrifice"); ax.set_ylabel("endothelial Col4a1+Col4a2")
    ax.set_title(f"RR matched pairs (|Δscore| ≤ 0.25), early→late\n"
                 f"later lower in {mp['frac_longer_duration_lower']:.0%} of {mp['n_pairs']} pairs, "
                 f"CI {mp['animal_bootstrap_ci_frac'][0]:.0%}–{mp['animal_bootstrap_ci_frac'][1]:.0%}\n"
                 f"({mp['n_animals_in_pairs']} animals; blue = lower)", loc="left", fontsize=6.3)
    letter(ax, "d")
    # 4c tertiles (RR)
    rk = rankdata(d.score, method="average")
    q = np.quantile(rk, [1 / 3, 2 / 3])
    bands = [(-np.inf, q[0]), (q[0], q[1]), (q[1], np.inf)]
    tert = S["RR"]["2_tertiles"]
    for k, ((lo, hi), b, rng_) in enumerate(zip(bands, tert["bands"], tert["score_ranges"])):
        ax = fig.add_subplot(gs[2, k])
        m = (rk > lo) & (rk <= hi)
        ax.scatter(d.day[m], d.endo_Col4_mean[m], s=14, color=COL["RR"], edgecolor="white", lw=0.4)
        x, y = d.day[m].to_numpy(), d.endo_Col4_mean[m].to_numpy()
        f = np.polyfit(x, y, 1); tt = np.linspace(x.min(), x.max(), 10)
        ax.plot(tt, np.polyval(f, tt), color=COL["RR"], lw=0.8, alpha=0.7)
        ax.set_title(f"score {rng_[0]:g}–{rng_[1]:g}: ρ {b['rho']:+.2f} [{b['ci'][0]:+.2f}, {b['ci'][1]:+.2f}], n = {b['n']}",
                     loc="left", fontsize=6.3)
        ax.set_xlabel("day of sacrifice")
        if k == 0:
            ax.set_ylabel("endothelial Col4a1+Col4a2")
            letter(ax, "c")
        if k == 0:
            ax.text(0, 1.28, f"RR within severity tertiles, no regression: combined ρ {tert['combined_rho']:+.2f} "
                    f"[{tert['combined_ci'][0]:+.2f}, {tert['combined_ci'][1]:+.2f}]", transform=ax.transAxes,
                    fontsize=6.8, color=INK)
    # 4e nulls
    G = pd.read_csv("runs/closing_block/wrapup/gene_stats_RR_Endothelial.csv", index_col=0)
    W = load_json("runs/closing_block/wrapup/results.json")["1"]["RR/Endothelial"]
    for k, (title, sub) in enumerate((("null 1: all expressed genes", G),
                                      ("null 2: severity-matched genes",
                                       G[(G.rho_sev - G.loc["Col4a2", "rho_sev"]).abs() <= W["Col4a2"]["tolerance"]]))):
        ax = fig.add_subplot(gs[3, k])
        ax.hist(sub.rho_day_given_sev.abs(), bins=40, color="#c7cfdb", edgecolor="white", lw=0.3)
        for g, ls in (("Col4a1", "-"), ("Col4a2", "--")):
            v = abs(G.loc[g, "rho_day_given_sev"])
            pct = (sub.rho_day_given_sev.abs() < v).mean() * 100
            ax.axvline(v, color=COL["RR"], lw=1.2, ls=ls)
            ax.text(v, ax.get_ylim()[1] * (0.92 if g == "Col4a1" else 0.72), f" {g}\n {pct:.1f}th pct", fontsize=6,
                    color=COL["RR"], va="top")
        ax.set_xlabel("|ρ(gene, day | severity)|, RR endothelium")
        ax.set_ylabel("genes")
        ax.set_title(f"{title}\n(n = {len(sub):,} genes)", loc="left", fontsize=6.5)
        if k == 0:
            letter(ax, "e")
    ax = fig.add_subplot(gs[3, 2]); ax.axis("off")
    ax.text(0, 0.95, "Col4a1/Col4a2 were selected from the\nclock's coefficients; these ranks are the\n"
                     "selection-aware null.\n\nRaw (unadjusted) ranks, all genes:\n"
                     f"Col4a1 {W['Col4a1']['whole_set_raw_percentile']:.1f}th, Col4a2 {W['Col4a2']['whole_set_raw_percentile']:.1f}th pct\n"
                     "within the severity-matched set:\n"
                     f"Col4a1 {W['Col4a1']['rank_raw']['percentile']:.1f}th, Col4a2 {W['Col4a2']['rank_raw']['percentile']:.1f}th pct",
            fontsize=6.3, va="top", color=INK)
    fig.text(0.01, 0.06, "Points = animals (pseudobulk of that cell type per animal). CIs = 95% bootstrap over animals "
             "(2000 resamples). Pairs share animals; the pair CI resamples animals.", fontsize=6, color=MUTED)
    save(fig, "fig4_what_changes")


if __name__ == "__main__":
    main()
