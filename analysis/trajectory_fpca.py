"""Functional PCA over the daily clinical-score curves (WORKORDER Task 0, step 6).

Called from analysis/trajectory_features.py. Curves are aligned at onset_day and
right-truncated at sacrifice, on a dense integer grid, so the estimator is the small
correct thing rather than PACE:
  * pairwise-complete mean and covariance surface (each cell uses the animals observed on
    both post-onset days),
  * eigendecomposition; the pairwise-complete matrix is not guaranteed PSD, so negative
    eigenvalues are clipped (nearest PSD in Frobenius norm) and the clipped mass reported,
  * conditional-expectation (BLUP) scores, which use each animal's observed days only.
Fit per cohort (RR, chronic), EAE animals with an onset_day. Capped at MAX_K = 3 components.
Support (animals per post-onset day) is reported; components whose loading mass sits mainly
where support < MIN_SUPPORT_INTERPRET are flagged as confounded with stage/sacrifice design.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MIN_SUPPORT_FIT = 5            # grid days with fewer animals are not estimated at all
MIN_SUPPORT_INTERPRET = 15     # WORKORDER: do not interpret components where support < 15
MAX_K = 3
CLIP_WARN = 0.10               # clipped |negative eigenvalue| mass above this -> fall back


def post_onset_matrix(curves, T):
    """curves: {sample_name: (tt, yy)} on days post-induction. Returns Y[n, G] post-onset."""
    rows, names = [], []
    for _, r in T.iterrows():
        tt, yy = curves[r.sample_name]
        k = tt >= r.onset_day
        rows.append(dict(zip((tt[k] - r.onset_day).astype(int), yy[k])))
        names.append(r.sample_name)
    G = max(max(d) for d in rows if d) + 1
    Y = np.full((len(rows), G), np.nan)
    for i, d in enumerate(rows):
        for t, v in d.items():
            Y[i, t] = v
    return Y, names


def fpca(Y):
    support = (~np.isnan(Y)).sum(0)
    grid = np.arange(np.where(support >= MIN_SUPPORT_FIT)[0].max() + 1)
    Yg = Y[:, grid]
    mu = np.nanmean(Yg, 0)
    R = Yg - mu
    obs = ~np.isnan(R)
    Rz = np.where(obs, R, 0.0)
    pair_n = obs.T.astype(float) @ obs.astype(float)
    C = (Rz.T @ Rz) / np.maximum(pair_n - 1, 1)
    w, V = np.linalg.eigh(C)
    neg = w < 0
    clipped = float(np.abs(w[neg]).sum() / np.abs(w).sum())
    w = np.clip(w, 0, None)
    order = np.argsort(-w)
    w, V = w[order], V[:, order]
    ve = w / w.sum()
    k90 = int(np.searchsorted(np.cumsum(ve), 0.90) + 1)
    K = min(MAX_K, k90)
    # measurement-error variance: raw diagonal not captured by the K leading components
    recon = (V[:, :K] * w[:K]) @ V[:, :K].T
    sigma2 = float(max(np.mean(np.diag(C) - np.diag(recon)), 1e-3))
    scores = np.full((Y.shape[0], K), np.nan)
    for i in range(Y.shape[0]):
        o = obs[i]
        if o.sum() == 0:
            continue
        Phi = V[o][:, :K]
        S = (Phi * w[:K]) @ Phi.T + sigma2 * np.eye(o.sum())
        scores[i] = (w[:K] * (Phi.T @ np.linalg.solve(S, R[i, o])))
    low = support[grid] < MIN_SUPPORT_INTERPRET
    tail_mass = [float((V[low, k] ** 2).sum()) for k in range(K)]  # eigvecs are unit-norm
    return {"grid": grid, "support": support, "mu": mu, "eigvals": w, "eigvecs": V,
            "var_explained": ve, "k90": k90, "K": K, "sigma2": sigma2,
            "clipped_mass": clipped, "scores": scores, "tail_mass": tail_mass,
            "low_support_days": grid[low].tolist()}


def correspondence(scores, T, cols):
    out = {}
    for k in range(scores.shape[1]):
        row = {}
        for c in cols:
            x = T[c].to_numpy(float)
            m = np.isfinite(x) & np.isfinite(scores[:, k])
            if m.sum() >= 5 and np.std(x[m]) > 0:
                row[c] = float(spearmanr(scores[m, k], x[m]).statistic)
        out[f"fpc{k + 1}"] = row
    return out


def run(curves, T, feature_cols, out_dir):
    """Returns (per-animal fpc DataFrame, results dict, report lines). Writes a figure."""
    res, frames, L = {}, [], ["6. FUNCTIONAL PCA (onset-aligned, per cohort)"]
    figs = {}
    for model, g in T[(T.condition == "EAE") & T.onset_day.notna()].groupby("model"):
        g = g.reset_index(drop=True)
        Y, names = post_onset_matrix(curves, g)
        f = fpca(Y)
        dropped = T[(T.model == model) & ((T.condition != "EAE") | T.onset_day.isna())]
        corr = correspondence(f["scores"], g, ["score_sacrifice", "day_of_sacrifice"] +
                              feature_cols)
        fb = f["clipped_mass"] > CLIP_WARN
        flags = {}
        for k in range(f["K"]):
            best = max(corr[f"fpc{k + 1}"].items(), key=lambda kv: abs(kv[1]))
            maps = abs(best[1]) >= 0.7
            partial = 0.5 <= abs(best[1]) < 0.7
            conf = f["tail_mass"][k] > 0.5
            flags[f"fpc{k + 1}"] = {
                "best_match": best[0], "rho": best[1], "maps_onto_existing": bool(maps),
                "partial_match": bool(partial),
                "tail_loading_mass": f["tail_mass"][k], "stage_confounded": bool(conf),
                "candidate_task4_target": bool(not maps and not partial and not conf)}
        key = "rr" if model != "CHRONIC" else "chronic"
        res[key] = {"n": len(names), "dropped": dropped[["sample_name", "stage"]].values.tolist(),
                    "grid_days": [int(f["grid"][0]), int(f["grid"][-1])],
                    "support": f["support"].tolist(), "low_support_days": f["low_support_days"],
                    "var_explained": f["var_explained"][:6].tolist(), "k_for_90pct": f["k90"],
                    "K_used": f["K"], "clipped_mass": f["clipped_mass"],
                    "fallback_needed": bool(fb), "sigma2": f["sigma2"],
                    "correspondence": corr, "flags": flags}
        df = pd.DataFrame(f["scores"], columns=[f"fpc{k + 1}" for k in range(f["K"])])
        df.insert(0, "sample_name", names)
        frames.append(df)
        figs[key] = (f, model)
        sup = f["support"][f["grid"]]
        L += [f"  {model}: n = {len(names)} animals (dropped {len(dropped)}: controls and "
              "never-symptomatic)",
              f"    post-onset grid days {f['grid'][0]}-{f['grid'][-1]} (support >= "
              f"{MIN_SUPPORT_FIT}); support falls below {MIN_SUPPORT_INTERPRET} from day "
              f"{f['grid'][sup < MIN_SUPPORT_INTERPRET][0] if (sup < MIN_SUPPORT_INTERPRET).any() else '-'}",
              f"    variance explained: " + ", ".join(f"{v:.0%}" for v in f["var_explained"][:5]) +
              f"; components to 90%: {f['k90']} -> using {f['K']} (cap {MAX_K})",
              f"    PSD clipping: {f['clipped_mass']:.1%} of eigenvalue mass"
              + ("  -> LARGE: truncation dominates; read as plain PCA on the common window"
                 if fb else " (small)")]
        L.append("    correspondence (top 4 |Spearman rho| per component):")
        for k in range(f["K"]):
            c = corr[f"fpc{k + 1}"]
            fl = flags[f"fpc{k + 1}"]
            top = sorted(c.items(), key=lambda kv: -abs(kv[1]))[:4]
            L.append(f"      fpc{k + 1}: " + ", ".join(f"{a} {v:+.2f}" for a, v in top) +
                     f" | tail loading {fl['tail_loading_mass']:.0%}"
                     + (" | STAGE-CONFOUNDED" if fl["stage_confounded"] else "")
                     + (" | maps onto existing" if fl["maps_onto_existing"] else "")
                     + (f" | partial match ({fl['best_match']})" if fl["partial_match"] else "")
                     + (" | CANDIDATE TASK 4 TARGET" if fl["candidate_task4_target"] else ""))
        L.append("")
    _plot(figs, out_dir)
    L += ["  full correspondence table (all features): results.json -> fpca.<cohort>.correspondence",
          "  Reading: 'maps onto existing' if |rho| >= 0.7 with score_sacrifice, day_of_sacrifice",
          "  or a hand-crafted feature; 'partial match' 0.5-0.7; a candidate Task 4 target needs",
          "  |rho| < 0.5 with all of them AND tail loading <= 50%. FPCA recovering variables in",
          "  hand demonstrates the redundancy Task 0's first pass found one target at a time.", ""]
    return pd.concat(frames, ignore_index=True), res, L


def _plot(figs, out_dir):
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
    COLS = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, axes = plt.subplots(2, len(figs), figsize=(5.5 * len(figs), 6.4), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for j, (key, (f, model)) in enumerate(figs.items()):
        g = f["grid"]
        sup = f["support"][g]
        low = sup < MIN_SUPPORT_INTERPRET
        ax = axes[0, j]
        ax.bar(g, sup, color="#b7b6b0", width=0.8)
        ax.axhline(MIN_SUPPORT_INTERPRET, color=INK, lw=0.8, ls=":")
        ax.set_title(f"{'RR' if key == 'rr' else 'Chronic'} — animals per post-onset day",
                     loc="left", fontsize=9, color=INK)
        ax.set_ylabel("n animals", fontsize=8, color=MUTED)
        ax2 = axes[1, j]
        if low.any():
            ax2.axvspan(g[low][0] - 0.5, g[-1] + 0.5, color="#f0efec", lw=0)
        for k in range(f["K"]):
            ax2.plot(g, f["eigvecs"][:, k], color=COLS[k], lw=2,
                     label=f"φ{k + 1} ({f['var_explained'][k]:.0%})")
        ax2.axhline(0, color=MUTED, lw=0.6)
        ax2.set_title("eigenfunctions (shaded: support < 15, do not interpret)", loc="left",
                      fontsize=9, color=INK)
        ax2.set_xlabel("days after onset", fontsize=8, color=MUTED)
        ax2.legend(fontsize=7, frameon=False)
        for a in (ax, ax2):
            a.set_facecolor(SURFACE)
            for sp_ in ("top", "right"):
                a.spines[sp_].set_visible(False)
            a.tick_params(colors=MUTED, labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figures", "fpca_support_eigenfunctions.png"), dpi=160)
    plt.close(fig)
