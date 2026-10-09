"""Supplementary figures.
S1 composition-only clock, three-type vs seventeen-type (paired); S2 basement-membrane gene
panel (A5) forest, both cohorts; S3 lesion descriptives vs day with the control false-positive
rate; S4 gene-list instability across the three severity adjustments (upset).

    PYTHONPATH="$PWD" python analysis/figures/supp.py
"""

from __future__ import annotations

import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import COL, DATA, INK, LIGHT, MUTED, corner, forest, letter, load_json, plt, rho_label, save  # noqa: E402

COHORT = {"RELAPSE REMITTING": "RR", "CHRONIC": "chronic"}
FLAG = "#7a3b00"


def s1():
    W = load_json("runs/closing_block/wrapup/results.json")["2"]
    fig, axes = plt.subplots(2, 1, figsize=(5.2, 3.6), gridspec_kw={"hspace": 0.9, "height_ratios": [2, 1]})
    ax = axes[0]
    labels, est, lo, hi, cols = [], [], [], [], []
    for ck in ("RR", "chronic"):
        for key, lab in (("three_type_clock", "3 types (Fib/Endo/Astro)"), ("seventeen_type_clock", "all 17 types")):
            r = W[ck][key]
            labels.append(f"{ck}: {lab} (n = {r['n']})")
            est.append(r["rho"]); lo.append(r["ci"][0]); hi.append(r["ci"][1]); cols.append(COL[ck])
    forest(ax, labels, est, lo, hi, cols, ref=0, xlabel="composition-only clock ρ (95% CI)")
    ax.set_xlim(0, 1)
    ax.set_title("composition-only clocks (cell-type proportions)", loc="left")
    letter(ax, "a")
    ax = axes[1]
    rows = [W[ck]["three_vs_seventeen"] for ck in ("RR", "chronic")]
    forest(ax, [f"{ck} (n = {r['n']}, Wilcoxon |err| p = {r['wilcoxon_p']:.3f})" for ck, r in zip(("RR", "chronic"), rows)],
           [r["diff_mean"] for r in rows], [r["diff_ci"][0] for r in rows], [r["diff_ci"][1] for r in rows],
           [COL["RR"], COL["chronic"]], ref=0, xlabel="ρ difference, 3-type − 17-type (paired)")
    ax.set_title("paired difference", loc="left")
    letter(ax, "b")
    fig.text(-0.35, -0.1, "The three types were chosen from the C5 composition-clock coefficients on the same animals: "
             "the 3-type ρ is selection-inflated and is not an independent test.\nCIs = 95% paired bootstrap over "
             "animals (2000 resamples). Target = day of sacrifice | terminal score, inside each fold.",
             fontsize=6, color=MUTED)
    save(fig, "supp_S1_composition_clock")


def s2():
    R = load_json("runs/closing_block/results.json")["A"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True, gridspec_kw={"wspace": 0.9})
    for k, ck in enumerate(("RR", "chronic")):
        A5 = R[f"{ck}/Endothelial"]["A5"]
        genes = list(A5)
        ax = axes[k]
        yy = np.arange(len(genes))[::-1]
        for key, ci, off, mk, lab in (("rho_day_raw", "ci_day_raw", 0.18, "o", "unadjusted"),
                                      ("rho_day_given_severity", "ci_day_given_severity", -0.18, "s",
                                       "given terminal score")):
            e = np.array([A5[g][key] for g in genes])
            lo = np.array([A5[g][ci][0] for g in genes]); hi = np.array([A5[g][ci][1] for g in genes])
            fill = COL[ck] if mk == "s" else "white"
            for y, a, b in zip(yy + off, lo, hi):
                ax.plot([a, b], [y, y], color=COL[ck], lw=1.2)
            ax.plot(e, yy + off, mk, color=COL[ck], mfc=fill, ms=4, mew=0.9, ls="none", label=lab)
        ax.axvline(0, color=LIGHT, lw=0.8, zorder=0)
        ax.set_yticks(yy, [f"{g}{' *' if A5[g]['in_clock_coef_lists'] else ''} ({A5[g]['frac_cells_pos']:.0%} +)"
                           for g in genes])
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(-1, 1)
        ax.set_xlabel("ρ(endothelial expression, day)")
        n = A5[genes[0]]["n"]
        if ck == "RR":
            ax.set_title(f"RR endothelium (n = {n})", loc="left")
            h, l_ = ax.get_legend_handles_labels()
        else:
            ax.set_title(f"chronic endothelium (n = {n}): NOT INTERPRETABLE\nseverity~day coupled; adjusted and "
                         "unadjusted disagree", loc="left", color=FLAG)
        letter(ax, "ab"[k])
    fig.legend(h, l_, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=2, frameon=False, fontsize=6.5)
    fig.text(0.01, -0.1, "Basement-membrane gene panel, animal pseudobulk of endothelial cells. * = gene in the clock "
             "coefficient lists (selected; Col4a1/2). (x% +) = fraction of endothelial cells expressing.\n"
             "Open circles = unadjusted; filled squares = rank-partial on terminal score. CIs = 95% bootstrap over "
             "animals (2000 resamples).", fontsize=6, color=MUTED)
    save(fig, "supp_S2_basement_membrane_panel")


def s3():
    L = pd.read_csv("runs/lesion_clock/lesions_per_animal.csv")
    S = pd.read_csv("runs/lesion_clock/sections.csv")
    res = load_json("runs/lesion_clock/results.json")["1a"]
    L = L.merge(S.groupby("animal").size().rename("n_sections"), left_on="animal", right_index=True)
    L["per_section"] = L.n_lesions / L.n_sections
    L["cohort"] = L.model.map(COHORT)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), gridspec_kw={"hspace": 0.75, "wspace": 0.4})
    for j, ck in enumerate(("RR", "chronic")):
        d = L[L.cohort == ck]
        e, c = d[d.condition == "EAE"], d[d.condition == "CONTROL"]
        for i, (col, ylab, scale) in enumerate((("per_section", "lesions per section", 1),
                                                ("lesion_area_fraction", "lesion area (% of tissue)", 100))):
            ax = axes[i, j]
            ax.scatter(e.day_of_sacrifice, scale * e[col], s=14, color=COL[ck], edgecolor="white", lw=0.4,
                       label=f"EAE (n = {len(e)})")
            ax.scatter(c.day_of_sacrifice, scale * c[col], s=16, facecolor="white", edgecolor=COL[ck], lw=0.9,
                       label=f"CFA control (n = {len(c)})")
            if col == "per_section":
                ax.axhline(res["control_lesions_per_section"], color=MUTED, lw=0.8, ls="--")
                ax.text(ax.get_xlim()[1], res["control_lesions_per_section"],
                        f"control false-positive rate\n{res['control_lesions_per_section']:.2f} per section ",
                        ha="right", va="bottom", fontsize=5.8, color=MUTED)
            ax.set_xlabel("day of sacrifice"); ax.set_ylabel(ylab)
            ax.set_title(f"{ck}: EAE {rho_label(e[col], e.day_of_sacrifice)}", loc="left")
            if i == 0 and j == 0:
                ax.legend(frameon=False, loc="upper left", fontsize=6)
        letter(axes[0, j], "ab"[j]); letter(axes[1, j], "cd"[j])
    fig.text(0.01, -0.02, f"Points = animals (lesions summed over sections, divided by sections). Segmentation: "
             f"lesionSegmenter port, myeloid/DC density proxy ({res['n_lesions']} lesions in {res['n_sections']} "
             f"sections).\nThe control rate ({res['lesions_in_control_animals']} lesions in CFA animals) is the "
             "detection floor. ρ over EAE animals only; CIs = 95% bootstrap over animals.", fontsize=6, color=MUTED)
    save(fig, "supp_S3_lesion_descriptives")


def s4():
    G = json.load(open(os.path.join(DATA, "severity_variant_genesets.json")))
    variants = [("none", "no adjustment"), ("terminal", "terminal score"), ("cumulative", "cumulative score")]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.2), gridspec_kw={"height_ratios": [1.4, 0.7], "hspace": 0.08,
                                                                     "wspace": 0.35})
    for j, ck in enumerate(("RR", "chronic")):
        sets = {v: set(G[f"{ck}/{v}"]) for v, _ in variants}
        combos = []
        for r in (3, 2, 1):
            for c in itertools.combinations([v for v, _ in variants], r):
                inside = set.intersection(*[sets[v] for v in c])
                outside = set().union(*[sets[v] for v, _ in variants if v not in c])
                combos.append((c, len(inside - outside)))
        x = np.arange(len(combos))
        ax = axes[0, j]
        ax.bar(x, [n for _, n in combos], color=COL[ck], width=0.6)
        for xi, (_, n) in zip(x, combos):
            ax.text(xi, n, str(n), ha="center", va="bottom", fontsize=6, color=INK)
        ax.set_xticks([]); ax.spines["bottom"].set_visible(False)
        ax.set_ylabel("genes (exclusive intersection)")
        alln = len(set.union(*sets.values()))
        jac = [len(sets[a] & sets[b]) / len(sets[a] | sets[b]) for (a, _), (b, _) in itertools.combinations(variants, 2)]
        ax.set_title(f"{ck}: {alln} genes in any list\n"
                     f"{', '.join(f'{lab} {len(sets[v])}' for v, lab in variants)}\n"
                     f"pairwise Jaccard {min(jac):.2f}–{max(jac):.2f}", loc="left", fontsize=6.5)
        letter(ax, "ab"[j])
        ax = axes[1, j]
        for yi, (v, lab) in enumerate(variants[::-1]):
            for xi, (c, _) in zip(x, combos):
                ax.plot(xi, yi, "o", ms=4.5, color=INK if v in c else LIGHT)
        for xi, (c, _) in zip(x, combos):
            ys = [len(variants) - 1 - [v for v, _ in variants].index(v) for v in c]
            ax.plot([xi, xi], [min(ys), max(ys)], color=INK, lw=1)
        ax.set_yticks(range(len(variants)), [lab for _, lab in variants[::-1]])
        ax.set_xticks([]); ax.set_xlim(axes[0, j].get_xlim()); ax.set_ylim(-0.5, len(variants) - 0.5)
        for s in ("left", "bottom"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.text(0.01, -0.03, "Genes with non-zero coefficients in the final (all-animal) clock fitted under each severity "
             "adjustment of the target. Bars = genes in exactly that combination of lists.", fontsize=6, color=MUTED)
    save(fig, "supp_S4_gene_list_instability")


if __name__ == "__main__":
    s1(); s2(); s3(); s4()
