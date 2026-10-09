"""Figure 2 — the signal is tissue-wide.
2a lesion vs non-lesion compartment predictions per animal (paired); 2b rho vs exclusion
margin; 2c competing configurations vs the all-cells clock (paired differences); 2d one
representative section coloured by compartment.

    PYTHONPATH="$PWD:$PWD/scripts:$PWD/analysis" python analysis/figures/fig2.py
"""

from __future__ import annotations

import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import COL, INK, LIGHT, MUTED, animals, corner, forest, letter, load_json, plt, save  # noqa: E402

SECTION_2D = "S5_T2_0"      # RR_P3_5 (PEAK3, day 49): a section with several lesions


def paired_vs_arm2():
    """Paired differences on the same 33 RR animals vs the all-cells clock (arm 2)."""
    import clock_composition as CC
    from duration_clock import loao_target
    from paired_difference import paired
    pb, info, meta = CC.load_rr()
    rr = info[info.model == CC.RR]
    ref = loao_target(rr.day.to_numpy(), rr.score.to_numpy().reshape(-1, 1))
    lad = pd.read_csv("runs/baseline_ladder/loao_predictions_arms1-5.csv")
    base = lad[(lad.target == "day_of_sacrifice") & (lad.arm == "arm2")].set_index("sample_name").loc[rr.index, "pred"].to_numpy()
    out = []
    for label, path in (("attention-MIL (arm 6, 3 seeds)", "runs/baseline_ladder/arm6_predictions.csv"),
                        ("mean-pooling MIL (arm 6b, 3 seeds)", "runs/baseline_ladder/arm6b_meanpool_predictions.csv")):
        a = pd.read_csv(path)
        p = a[a.target == "day_of_sacrifice"].set_index("sample_name").loc[rr.index, "pred"].to_numpy()
        out.append((label, paired(label, p, base, ref, label, "arm 2")))
    shards = glob.glob("runs/arm6_tuned/shard_*.csv")
    if shards:
        t = pd.concat([pd.read_csv(s) for s in shards])
        t = t[t.target == "day_of_sacrifice"].drop_duplicates("held_out").set_index("held_out")
        if set(rr.index) <= set(t.index):
            p = t.loc[rr.index, "pred"].to_numpy()
            out.append(("attention-MIL, tuned (arm 6t)", paired("tuned", p, base, ref, "tuned", "arm 2")))
    return out


def section_compartments():
    import lesion_segment as LS
    from closing_block_data import signed_distance
    with h5py.File(LS.CC.ATLAS, "r") as f:
        sc_, scats = LS._obs(f, "meta_sample_id")
        c1_, c1cats = LS._obs(f, "Anno_L1_curated")
        xs, _ = LS._obs(f, "x_centroid")
        ys, _ = LS._obs(f, "y_centroid")
        rows = np.where(sc_ == list(scats).index(SECTION_2D))[0]
        pu1 = np.isin(c1cats[c1_[rows]], LS.PU1_LABELS)
        d = signed_distance(xs[rows], ys[rows], pu1)
        return xs[rows], ys[rows], d


def main():
    A = animals()
    cb = load_json("runs/closing_block/results.json")
    fig = plt.figure(figsize=(7.2, 7.6))
    gs = fig.add_gridspec(3, 4, hspace=0.7, wspace=0.75, height_ratios=[1, 0.95, 1.1])
    # 2a
    for k, ck in enumerate(("RR", "chronic")):
        ax = fig.add_subplot(gs[0, 2 * k:2 * k + 2])
        d = A[(A.cohort == ck)].dropna(subset=["pred_lesion", "pred_nonlesion_m50"])
        for _, r in d.iterrows():
            ax.plot([0, 1], [r.pred_lesion, r.pred_nonlesion_m50], color=COL[ck], alpha=0.45, lw=0.8)
        ax.scatter(np.zeros(len(d)), d.pred_lesion, s=12, color=COL[ck], edgecolor="white", lw=0.4, zorder=3)
        ax.scatter(np.ones(len(d)), d.pred_nonlesion_m50, s=12, color=COL[ck], edgecolor="white", lw=0.4, zorder=3)
        ax.set_xticks([0, 1], ["lesion", "non-lesion\n(> 50 µm)"])
        ax.set_xlim(-0.4, 1.4)
        ax.set_ylabel("predicted day | severity (days)")
        b = cb["B"][ck]["B3_nonlesion50_vs_lesion"]
        ax.set_title(f"{ck}, same {b['n']} animals: ρ lesion {b['rho_B']:+.2f}, non-lesion {b['rho_A']:+.2f}\n"
                     f"paired difference {b['diff_mean']:+.2f} [{b['diff_ci'][0]:+.2f}, {b['diff_ci'][1]:+.2f}]",
                     loc="left")
        if k == 0:
            letter(ax, "a")
    # 2b
    ax = fig.add_subplot(gs[1, 0:2])
    for j, ck in enumerate(("RR", "chronic")):
        B = cb["B"][ck]
        xs_ = np.array([0, 50, 150]) + (j - 0.5) * 6
        e = [B[f"nonlesion_m{m}"]["rho"] for m in (0, 50, 150)]
        lo = [B[f"nonlesion_m{m}"]["ci"][0] for m in (0, 50, 150)]
        hi = [B[f"nonlesion_m{m}"]["ci"][1] for m in (0, 50, 150)]
        ax.errorbar(xs_, e, yerr=[np.subtract(e, lo), np.subtract(hi, e)], fmt="-o", color=COL[ck], ms=4,
                    lw=1.2, capsize=0, label=f"{ck} (n = {B['nonlesion_m50']['n']})")
    ax.set_xticks([0, 50, 150])
    ax.set_xlabel("exclusion margin around lesions (µm)")
    ax.set_ylabel("non-lesion clock ρ (95% CI)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("margin sensitivity — chronic drops: compartments not\ncleanly separable in chronic", loc="left")
    letter(ax, "b")
    # 2d
    ax = fig.add_subplot(gs[1, 2:4])
    x, y, d = section_compartments()
    lab = np.where(d <= 0, "lesion", np.where(d <= 50, "margin (0–50 µm)", "non-lesion"))
    cmap = {"non-lesion": LIGHT, "margin (0–50 µm)": "#f2b880", "lesion": "#b2182b"}
    for k_ in ("non-lesion", "margin (0–50 µm)", "lesion"):
        m = lab == k_
        ax.scatter(x[m], y[m], s=0.3, color=cmap[k_], label=f"{k_} ({m.sum():,} cells)", rasterized=True)
    ax.set_aspect("equal"); ax.axis("off")
    x0, y0 = x.min(), y.min() - 60
    ax.plot([x0, x0 + 500], [y0, y0], color=INK, lw=1.5)
    ax.text(x0 + 250, y0 - 40, "500 µm", ha="center", va="top", fontsize=6)
    ax.legend(frameon=False, loc="upper center", markerscale=12, fontsize=6, bbox_to_anchor=(0.5, -0.02), ncol=1)
    ax.set_title(f"compartments, one section ({SECTION_2D}, RR_P3_5)", loc="left")
    letter(ax, "d")
    # 2c
    ax = fig.add_subplot(gs[2, 0:4])
    pdiff = load_json("runs/clock_composition/paired_difference.json")
    by = {r["comparison"]: r for r in pdiff}
    rows = [("best cell type: astrocytes (selected maximum of 13)", by["1f Astrocyte vs all cells (same 33 animals)"]),
            ("cell type chosen inside each fold (nested)", by["A2 nested cell type vs pseudobulk"]),
            ("spatial configuration chosen inside each fold (nested)", by["B3 nested spatial vs pseudobulk"])]
    rows += paired_vs_arm2()
    forest(ax, [f"{l} (n = {r['n']})" for l, r in rows], [r["diff_mean"] for _, r in rows],
           [r["diff_ci"][0] for _, r in rows], [r["diff_ci"][1] for _, r in rows],
           [COL["RR"]] * len(rows), ref=0, xlabel="ρ difference vs all-cells pseudobulk clock (paired bootstrap, 95% CI)")
    ax.set_title("nothing localises the clock better than all cells (RR, day of sacrifice | severity)", loc="left")
    letter(ax, "c")
    fig.text(0.01, 0.0, "Points = animals; lines join the same animal. Error bars = 95% bootstrap CIs over animals "
             "(2000 resamples; paired resampling in a and c). Frozen leave-one-animal-out clocks in a, b.",
             fontsize=6, color=MUTED)
    save(fig, "fig2_tissue_wide")


if __name__ == "__main__":
    main()
