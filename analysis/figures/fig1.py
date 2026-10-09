"""Figure 1 — the clock.
1a LOAO predicted vs actual severity-adjusted day (RR, chronic); 1b permutation null for the
all-cells clock; 1c clock rho under each severity adjustment; 1d cross-run transfer.

    PYTHONPATH="$PWD" python analysis/figures/fig1.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from common import COL, INK, MUTED, animals, corner, forest, letter, load_json, plt, rho_label, save  # noqa: E402


def main():
    A = animals()
    fig = plt.figure(figsize=(7.2, 8.4))
    gs = fig.add_gridspec(3, 3, hspace=0.75, wspace=0.55, height_ratios=[1, 0.9, 0.9])
    # 1a
    for k, ck in enumerate(("RR", "chronic")):
        ax = fig.add_subplot(gs[0, k])
        d = A[A.cohort == ck]
        ax.scatter(d.clock_target_resid, d.clock_pred_resid, s=16, color=COL[ck], edgecolor="white", lw=0.5)
        lim = [min(d.clock_target_resid.min(), d.clock_pred_resid.min()) - 2,
               max(d.clock_target_resid.max(), d.clock_pred_resid.max()) + 2]
        ax.plot(lim, lim, color=MUTED, lw=0.7, ls="--")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("day of sacrifice | severity (days)")
        ax.set_ylabel("LOAO predicted (days)")
        ax.set_title(f"{ck} clock", loc="left")
        corner(ax, rho_label(d.clock_pred_resid, d.clock_target_resid), loc="lower right")
        if k == 0:
            letter(ax, "a")
    # 1b
    ax = fig.add_subplot(gs[0, 2])
    p = load_json("runs/baseline_ladder/perm_arm2_day_of_sacrifice.json")
    null = np.array(p["null"])
    ax.hist(null, bins=40, color="#b8c4d6", edgecolor="white", lw=0.3)
    ax.axvline(p["observed_rho"], color=COL["RR"], lw=1.5)
    ax.axvline(null.max(), color=MUTED, lw=0.8, ls=":")
    ax.text(p["observed_rho"], ax.get_ylim()[1] * 0.95, f" observed\n {p['observed_rho']:.2f}", fontsize=6,
            color=COL["RR"], va="top")
    ax.text(null.max(), ax.get_ylim()[1] * 0.6, f"null max\n{null.max():.2f} ", fontsize=6, color=MUTED, ha="right")
    ax.set_xlabel("ρ under shuffled day labels")
    ax.set_ylabel("permutations")
    ax.set_title(f"RR all-cells clock, {p['n_perm']} permutations", loc="left")
    corner(ax, f"p = {p['p']:.3f} (floor)", loc="upper left")
    letter(ax, "b")
    # 1c
    ax = fig.add_subplot(gs[1, 0:2])
    S = load_json("runs/closing_block/severity_adjustment/results.json")
    labels, est, lo, hi, cols = [], [], [], [], []
    for ck in ("RR", "chronic"):
        for v, lab in (("none", "no adjustment"), ("terminal", "terminal score (current)"),
                       ("cumulative", "cumulative score")):
            r = S[ck]["4_clock"][v]
            labels.append(f"{ck}: {lab}  (n={r['n']}, zero-feature folds {r['zero_feature_folds']})")
            est.append(r["rho"]); lo.append(r["ci"][0]); hi.append(r["ci"][1]); cols.append(COL[ck])
    forest(ax, labels, est, lo, hi, cols, ref=None, xlabel="clock ρ (95% bootstrap CI over animals)")
    ax.set_xlim(0.2, 1.0)
    ax.set_title("severity adjustment used for the clock", loc="left")
    letter(ax, "c")
    # 1d
    ax = fig.add_subplot(gs[2, 0:2])
    R = load_json("runs/clock_run_robustness/results.json")
    labels, est, lo, hi, cols = [], [], [], [], []
    for key, ck in (("rr", "RR"), ("chronic", "chronic")):
        b = R[key]["baseline_B"]["full_data_ref"]
        labels.append(f"{ck}: all runs, LOAO (n = {b['n']})"); est.append(b["rho"]); lo.append(b["ci"][0]); hi.append(b["ci"][1]); cols.append(COL[ck])
        for k_, v in R[key]["leave_one_run_out"].items():
            s = v["clock_B_sev_orth"]
            tr, te = k_.split("->")
            mon = {"01": "Jan", "05": "May", "06": "Jun"}
            fmt = lambda r: f"{mon[r[4:6]]} {r[:4]}"
            labels.append(f"{ck}: train {fmt(tr)} → test {fmt(te)} (n test = {s['n']})")
            est.append(s["rho"]); lo.append(s["ci"][0]); hi.append(s["ci"][1]); cols.append(COL[ck])
    forest(ax, labels, est, lo, hi, cols, ref=0, xlabel="clock ρ (95% bootstrap CI over animals)")
    ax.set_xlim(-0.3, 1.05)
    ax.set_title("cross-run transfer: train on one Xenium run, predict the other", loc="left")
    letter(ax, "d")
    fig.text(0.01, -0.01, "Points = animals. Error bars = 95% bootstrap CIs over animals (2000 resamples). "
             "Target = day of sacrifice residualised on terminal score inside each fold unless stated.",
             fontsize=6, color=MUTED)
    save(fig, "fig1_clock")


if __name__ == "__main__":
    main()
