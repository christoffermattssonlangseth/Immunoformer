"""Memory timescale and lag profile (WORKORDER Task 0, step 7).

Called from analysis/trajectory_features.py. How far back does the tissue remember the
clinical course?

STATED PREDICTION (written before running; WORKORDER):
  7a  the acute program (Hal, Arg1, Chil3) has a SHORT memory timescale tau; the accrual
      program (Gpnmb, Igf2, Fmod, Fcrls, Plin4, Pmp22, Ptgds) has a LONG tau; the CIs do not
      overlap. If the two taus are indistinguishable, the acute/accrual dissociation of
      runs/accrual_axis is not supported and that is the result.
  7b  the acute lag profile peaks at short lag and decays; the accrual profile stays flat or
      rises going back. The TEST STATISTIC is the difference in SHAPE between the two
      programs (mean rho at lags >= 7 minus mean rho at lags <= 3, accrual minus acute), not
      the level of either: partialling on score_sacrifice suppresses short lags for both
      programs alike (lag 0 is score_sacrifice itself), so the contrast survives it.

7a  For each program and tau in 1..30 days, the exponentially weighted integral of the
    animal's observed past scores, sum_t score(t) * exp(-(T - t) / tau) over observed days
    t <= T (sacrifice). tau* maximises Spearman rho with the animal-level program score;
    95% CI by bootstrap over animals. A tau* at the top of the scan means the program
    tracks unweighted cumulative burden.
7b  For lags 0,1,2,3,5,7,10,14,21,28 days before sacrifice: rho(score at T - lag, program),
    raw and partialled on score_sacrifice (rank-based partial Spearman). An animal enters a
    lag only if lag <= its disease duration (T - onset_day) and the score at T - lag was
    recorded. n and the contributing animals are reported at every lag; the plotted profile
    is solid up to the last lag with n >= 20 and dashed beyond.
7c  Program scores: mean z (within cohort) of animal log-CP10k pseudobulk; section
    C2_G3_Mid_1 excluded. The hand-picked ratchet list is used if >= 5 of its 7 genes are
    on the panel, otherwise the program is rebuilt from runs/duration_clock/clock_genes.csv.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

PB = "runs/rr_within_relapse/pseudobulk_all67.npz"
EXCLUDE_SECTIONS = {"C2_G3_Mid_1"}
RATCHET = ["Gpnmb", "Igf2", "Fmod", "Fcrls", "Plin4", "Pmp22", "Ptgds"]
ACUTE = ["Hal", "Arg1", "Chil3"]
TAUS = np.arange(1, 31)
LAGS = [0, 1, 2, 3, 5, 7, 10, 14, 21, 28]
N_BOOT = 2000
MIN_N_SOLID = 20


def animal_pseudobulk():
    d = np.load(PB, allow_pickle=True)
    meta = pd.DataFrame(d["section_meta"].tolist(), columns=d["section_meta_cols"])
    keep = ~meta.meta_sample_id.isin(EXCLUDE_SECTIONS).to_numpy()
    meta, counts = meta[keep], d["section_counts"][keep]
    animals = meta.sample_name.drop_duplicates().to_numpy()
    pb = np.vstack([counts[(meta.sample_name == a).to_numpy()].sum(0) for a in animals])
    return pd.DataFrame(np.log1p(pb / pb.sum(1, keepdims=True) * 1e4), index=animals,
                        columns=d["genes"].astype(str))


def program_definitions(genes):
    on = [g for g in RATCHET if g in genes]
    if len(on) >= 5:
        return {"accrual": on, "acute": [g for g in ACUTE if g in genes]}, "hand-picked list"
    cg = pd.read_csv("runs/duration_clock/clock_genes.csv")
    acc = cg[cg.coef > 0].gene.head(10).tolist()
    return ({"accrual": acc, "acute": [g for g in ACUTE if g in genes]},
            "rebuilt from top positive clock_genes.csv coefficients")


def program_scores(pb, animals, programs):
    out = {}
    for name, gl in programs.items():
        Z = pb.loc[animals, gl]
        Z = (Z - Z.mean()) / Z.std()
        out[name] = Z.mean(1).to_numpy()
    return out


def ewi(tt, yy, T, tau):
    m = tt <= T
    return float(np.sum(yy[m] * np.exp(-(T - tt[m]) / tau)))


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    A = np.c_[np.ones(len(rz)), rz]
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    if ex.std() == 0 or ey.std() == 0:
        return np.nan
    return float(np.corrcoef(ex, ey)[0, 1])


def memory_timescale(curves, g, prog, rng):
    E = np.array([[ewi(*curves[a], T, tau) for tau in TAUS]
                  for a, T in zip(g.sample_name, g.day_of_sacrifice)])
    rho = np.array([spearmanr(E[:, j], prog).statistic for j in range(len(TAUS))])
    best = int(TAUS[np.nanargmax(rho)])
    eb = E[:, best - 1]
    day = g.day_of_sacrifice.to_numpy(float)
    sev = g.score_sacrifice.to_numpy(float)
    boot = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(prog), len(prog))
        r = [spearmanr(E[i, j], prog[i]).statistic for j in range(len(TAUS))]
        if np.all(np.isnan(r)):
            continue
        boot.append(int(TAUS[np.nanargmax(r)]))
    return {"tau_star": best, "rho_at_tau_star": float(np.nanmax(rho)),
            # is long memory just elapsed time / current severity?
            "rho_partial_on_day": partial_spearman(eb, prog, day),
            "rho_partial_on_score": partial_spearman(eb, prog, sev),
            "rho_ewi_vs_day": float(spearmanr(eb, day).statistic),
            "tau_ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
            "rho_curve": rho.tolist(), "at_scan_limit": best == int(TAUS[-1])}


def lag_profile(curves, g, prog, rng):
    rows = []
    for lag in LAGS:
        x, idx, who = [], [], []
        for i, (a, T, on) in enumerate(zip(g.sample_name, g.day_of_sacrifice, g.onset_day)):
            if not np.isfinite(on) or lag > T - on:
                continue
            tt, yy = curves[a]
            hit = np.where(tt == T - lag)[0]
            if not len(hit):
                continue
            x.append(yy[hit[0]]); idx.append(i); who.append(a)
        x, idx = np.array(x), np.array(idx, int)
        r = {"lag": lag, "n": len(idx), "animals": who}
        for name, p in prog.items():
            if len(idx) >= 5 and np.std(x) > 0:
                raw = float(spearmanr(x, p[idx]).statistic)
                par = partial_spearman(x, p[idx], g.score_sacrifice.to_numpy()[idx])
                br, bp = [], []
                for _ in range(1000):
                    b = rng.integers(0, len(idx), len(idx))
                    if np.std(x[b]) > 0:
                        br.append(spearmanr(x[b], p[idx][b]).statistic)
                        bp.append(partial_spearman(x[b], p[idx][b],
                                                   g.score_sacrifice.to_numpy()[idx][b]))
                r[name] = {"raw": raw, "raw_ci": np.nanquantile(br, [0.025, 0.975]).tolist(),
                           "partial": par,
                           "partial_ci": np.nanquantile(bp, [0.025, 0.975]).tolist()}
            else:
                r[name] = None
        rows.append(r)
    return rows


def shape_contrast(rows, kind):
    def m(name, sel):
        v = [r[name][kind] for r in rows if r[name] and sel(r["lag"]) and r["n"] >= MIN_N_SOLID
             and np.isfinite(r[name][kind])]
        return float(np.mean(v)) if v else np.nan
    acc = m("accrual", lambda L: L >= 7) - m("accrual", lambda L: L <= 3)
    acu = m("acute", lambda L: L >= 7) - m("acute", lambda L: L <= 3)
    return {"accrual_long_minus_short": acc, "acute_long_minus_short": acu,
            "contrast_accrual_minus_acute": acc - acu}


def run(curves, T, out_dir):
    pb = animal_pseudobulk()
    programs, how = program_definitions(set(pb.columns))
    rng = np.random.default_rng(0)
    res = {"program_definition": how, "programs": programs,
           "ratchet_on_panel": f"{sum(g in pb.columns for g in RATCHET)}/{len(RATCHET)}"}
    L = ["7. MEMORY TIMESCALE AND LAG PROFILE",
         "  Prediction (stated before running): acute tau SHORT, accrual tau LONG, CIs not",
         "  overlapping; acute lag profile decays with lag, accrual flat or rising. Test",
         "  statistic for 7b = shape contrast (mean rho lags>=7 minus lags<=3, accrual minus",
         "  acute), not either level.",
         f"  Programs ({how}; ratchet genes on panel {res['ratchet_on_panel']}): "
         f"accrual {programs['accrual']}; acute {programs['acute']}"]
    plots = {}
    for model, g in T[T.condition == "EAE"].groupby("model"):
        g = g[g.sample_name.isin(pb.index)].reset_index(drop=True)
        key = "rr" if model != "CHRONIC" else "chronic"
        prog = program_scores(pb, g.sample_name, programs)
        mt = {k: memory_timescale(curves, g, p, rng) for k, p in prog.items()}
        lp = lag_profile(curves, g, prog, rng)
        sc = {k: shape_contrast(lp, k) for k in ("raw", "partial")}
        res[key] = {"n": len(g), "memory_timescale": mt, "lag_profile": lp, "shape_contrast": sc,
                    "missing_final_days": g.loc[g.sacrifice_point_imputed, "sample_name"].tolist()}
        plots[key] = (lp, model)
        a, c = mt["accrual"], mt["acute"]
        overlap = not (a["tau_ci95"][0] > c["tau_ci95"][1] or c["tau_ci95"][0] > a["tau_ci95"][1])
        L += [f"  {model} (n = {len(g)} EAE animals)",
              f"    7a tau*: accrual {a['tau_star']} d [95% CI {a['tau_ci95'][0]:.0f}-"
              f"{a['tau_ci95'][1]:.0f}], rho {a['rho_at_tau_star']:+.2f}"
              + (" (at scan limit = tracks cumulative burden)" if a["at_scan_limit"] else ""),
              f"           acute   {c['tau_star']} d [95% CI {c['tau_ci95'][0]:.0f}-"
              f"{c['tau_ci95'][1]:.0f}], rho {c['rho_at_tau_star']:+.2f}"
              + (" (at scan limit)" if c["at_scan_limit"] else ""),
              f"       CIs {'OVERLAP -> taus indistinguishable; dissociation NOT supported' if overlap else 'do not overlap'}",
              f"       is it just time / current severity? EWI(tau*) vs day rho: accrual "
              f"{a['rho_ewi_vs_day']:+.2f}, acute {c['rho_ewi_vs_day']:+.2f}; rho with program "
              f"partial on day: accrual {a['rho_partial_on_day']:+.2f}, acute "
              f"{c['rho_partial_on_day']:+.2f}; partial on score_sacrifice: accrual "
              f"{a['rho_partial_on_score']:+.2f}, acute {c['rho_partial_on_score']:+.2f}",
              f"       reading: accrual long-memory {'is LARGELY ELAPSED TIME (collapses on day)' if abs(a['rho_partial_on_day']) < 0.3 else 'survives partialling on day (memory beyond elapsed time)'};"
              f" acute short-memory {'is CURRENT SEVERITY (collapses on score_sacrifice)' if abs(c['rho_partial_on_score']) < 0.3 else 'survives partialling on score_sacrifice'}.",
              "    7b lag profile (rho with program; raw | partial on score_sacrifice):",
              f"      {'lag':>4s} {'n':>3s} {'accrual raw':>12s} {'acute raw':>10s} "
              f"{'accrual part':>13s} {'acute part':>11s}"]
        for r in lp:
            f = lambda k, kind: (f"{r[k][kind]:+.2f}" if r[k] and np.isfinite(r[k][kind])
                                 else "  -")
            L.append(f"      {r['lag']:4d} {r['n']:3d} {f('accrual', 'raw'):>12s} "
                     f"{f('acute', 'raw'):>10s} {f('accrual', 'partial'):>13s} "
                     f"{f('acute', 'partial'):>11s}" + ("   (n < 20: biased long-disease subset)"
                                                        if r["n"] < MIN_N_SOLID else ""))
        fmt = lambda v: (f"{v:+.2f}" if np.isfinite(v) else
                         "not computable (no lag >= 7 with n >= 20)")
        L += [f"      shape contrast (accrual minus acute, long minus short lags; lags with "
              f"n >= 20 only): raw {fmt(sc['raw']['contrast_accrual_minus_acute'])}, partial "
              f"{fmt(sc['partial']['contrast_accrual_minus_acute'])} (positive = predicted)",
              f"      animals scored with the atlas sacrifice score because the sheet lacks the "
              f"final days: {res[key]['missing_final_days'] or 'none'}",
              "      contributing animals per lag: results.json -> memory.<cohort>.lag_profile", ""]
    L += ["  Caveat: n shrinks with lag because short-course animals drop out; the animals left",
          "  at long lags have longer disease and are not a random subset.", ""]
    _plot(plots, out_dir)
    return res, L


def _plot(plots, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
    COL = {"accrual": "#2a78d6", "acute": "#eb6834"}
    fig, axes = plt.subplots(len(plots), 2, figsize=(10, 3.6 * len(plots)), squeeze=False,
                             sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for i, (key, (lp, model)) in enumerate(plots.items()):
        for j, kind in enumerate(("raw", "partial")):
            ax = axes[i, j]
            ax.set_facecolor(SURFACE)
            for name, c in COL.items():
                pts = [(r["lag"], r[name][kind], *r[name][f"{kind}_ci"], r["n"]) for r in lp
                       if r[name] and np.isfinite(r[name][kind])]
                if not pts:
                    continue
                lag, v, lo, hi, n = map(np.array, zip(*pts))
                solid = n >= MIN_N_SOLID
                ax.fill_between(lag[solid], lo[solid], hi[solid], color=c, alpha=0.12, lw=0)
                ax.plot(lag[solid], v[solid], "-o", color=c, lw=2, ms=5, label=name)
                if (~solid).any():
                    k = np.r_[np.where(solid)[0][-1:], np.where(~solid)[0]]
                    ax.plot(lag[k], v[k], "--o", color=c, lw=1.2, ms=4, mfc=SURFACE)
            for q, r in enumerate(lp):
                ax.annotate(f"n={r['n']}", (r["lag"], -0.95 + 0.09 * (q % 2)), fontsize=6,
                            color=MUTED, ha="center")
            ax.axhline(0, color=MUTED, lw=0.6)
            ax.set_ylim(-1, 1)
            ax.set_title(f"{'RR' if key == 'rr' else 'Chronic'} — "
                         f"{'raw' if kind == 'raw' else 'partial on score_sacrifice'}",
                         loc="left", fontsize=9, color=INK)
            ax.set_xlabel("days before sacrifice", fontsize=8, color=MUTED)
            for sp_ in ("top", "right"):
                ax.spines[sp_].set_visible(False)
            ax.tick_params(colors=MUTED, labelsize=7)
        axes[i, 0].set_ylabel("Spearman ρ, score at lag vs program", fontsize=8, color=MUTED)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.text(0.01, 0.005, "solid: n >= 20 · dashed: n < 20 (biased long-disease subset) · band: "
             "95% bootstrap CI", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(out_dir, "figures", "memory_lag_profile.png"), dpi=160)
    plt.close(fig)
