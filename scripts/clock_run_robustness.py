"""CLOCK RUN-ROBUSTNESS — does the duration clock survive the Xenium run? (WORKORDER Task 1b)

Task 1 showed run identity is recoverable from expression even within one model at matched
stages. Both clock cohorts cross a run boundary (RR: Jun 2025 + May 2026; chronic: Jan 2025
+ May 2026), so a within-cohort clock could partly be reading run. Kruskal day~run_date
(p = 0.14 RR) only tests label~run association, not this.

Per cohort (RR, chronic), on animal log-CP10k pseudobulk:
  0. Reference: the incumbent severity-orthogonalised clock B (loao_clock from
     scripts/duration_clock.py, unchanged), refit on the current all-animal pseudobulk.
  1. Aliasing: run_date x day_of_sacrifice and x stage; eta^2 of day on run, Kruskal p.
  2. Run-adjusted clock: clock B with covariates [score_sacrifice, run_date dummy] —
     X and day residualised on both inside each LOAO training fold. Evaluated against
     (a) the full-data residual and (b) a fold-wise residual (held-out animal's day minus
     the training-fold fit), the latter being the leakage-free reference.
     Permutation null: day shuffled WITHIN run (keeps the run structure), 200 shuffles.
  3. Leave-one-run-out: fit on one run, predict the other; raw clock A and severity-
     residualised clock B (residualiser fit on the training run only). Spearman rho in the
     held-out run, bootstrap CI over animals, mean bias.

Interpretation fixed before running (WORKORDER):
  * survives  — run-adjusted rho's 95% CI excludes 0 and its point estimate lies inside the
                baseline's 95% CI ("modest drop");
  * collapses — run-adjusted rho's CI includes 0, or leave-one-run-out rho's CI includes 0
                in both directions;
  * anything else is reported as a partial drop with the numbers.

Section C2_G3_Mid_1 is excluded (WORKORDER Task 0 reassignment rule).

    PYTHONPATH="$PWD:$PWD/scripts" python scripts/clock_run_robustness.py
"""

from __future__ import annotations

import immunotransformer  # noqa: F401  (OpenMP guard — must precede numpy; see HANDOFF.md)

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kruskal, spearmanr
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from duration_clock import loao_clock, loao_target, ENET_KW  # noqa: E402
from immunotransformer.stats import bootstrap_spearman, compare_rhos  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

PB = "runs/rr_within_relapse/pseudobulk_all67.npz"
TRAJ = "runs/trajectory_features/animal_trajectory.csv"
OUT = "runs/clock_run_robustness"
EXCLUDE_SECTIONS = {"C2_G3_Mid_1"}
N_PERM = int(os.environ.get("N_PERM", "200"))
N_JOBS = int(os.environ.get("N_JOBS", "8"))
N_BOOT = 2000
PUBLISHED = {"RELAPSE REMITTING": 0.799, "CHRONIC": 0.856}
COHORTS = [("RELAPSE REMITTING", "rr"), ("CHRONIC", "chronic")]


# ---------------------------------------------------------------- data

def load_cohort(model):
    d = np.load(PB, allow_pickle=True)
    meta = pd.DataFrame(d["section_meta"].tolist(), columns=d["section_meta_cols"])
    counts = d["section_counts"]
    keep = (~meta.meta_sample_id.isin(EXCLUDE_SECTIONS) & (meta.model == model)).to_numpy()
    meta, counts = meta[keep].reset_index(drop=True), counts[keep]
    animals = meta.sample_name.drop_duplicates().to_numpy()
    pb = np.vstack([counts[(meta.sample_name == a).to_numpy()].sum(0) for a in animals])
    pb = np.log1p(pb / pb.sum(1, keepdims=True) * 1e4)
    traj = pd.read_csv(TRAJ).set_index("sample_name").loc[animals]
    am = meta.drop_duplicates("sample_name").set_index("sample_name").loc[animals]
    df = pd.DataFrame({"sample_name": animals, "run_date": am.run_date.to_numpy(),
                       "run_id": am.run_id.to_numpy(), "stage": am.stage.to_numpy(),
                       "day": traj.day_of_sacrifice.to_numpy(float),
                       "score": traj.score_sacrifice.to_numpy(float)})
    ok = np.isfinite(df.day) & np.isfinite(df.score)
    return pb[ok.to_numpy()], df[ok].reset_index(drop=True)


def run_dummies(run):
    levels = sorted(set(run))
    return np.column_stack([(run == lv).astype(float) for lv in levels[1:]])


# ---------------------------------------------------------------- metrics

def foldwise_target(y, covar):
    """Held-out animal's y minus the TRAINING-fold fit of y on covar (no leakage)."""
    out = np.empty(len(y))
    for i in range(len(y)):
        tr = np.ones(len(y), bool); tr[i] = False
        out[i] = y[i] - LinearRegression().fit(covar[tr], y[tr]).predict(covar[i:i + 1])[0]
    return out


def r2(pred, true):
    return float(1 - np.sum((pred - true) ** 2) / np.sum((true - true.mean()) ** 2))


def summarise(pred, true, seed=0):
    b = bootstrap_spearman(pred, true, n_boot=N_BOOT, seed=seed)
    return {"rho": float(b["rho"]), "ci": [float(b["ci_lo"]), float(b["ci_hi"])],
            "r2": r2(pred, true), "n": int(len(true))}


def _perm_within_run(X, day, run, cov, seed):
    rng = np.random.default_rng(seed)
    yp = day.copy()
    for lv in np.unique(run):
        idx = np.where(run == lv)[0]
        yp[idx] = day[rng.permutation(idx)]
    return spearmanr(loao_clock(X, yp, covar=cov), loao_target(yp, cov)).statistic


# ---------------------------------------------------------------- analysis

def aliasing(df):
    groups = [g.day.to_numpy() for _, g in df.groupby("run_date")]
    grand = df.day.mean()
    ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_t = ((df.day - grand) ** 2).sum()
    st = pd.crosstab(df.stage, df.run_date)
    return {
        "n_by_run": df.run_date.value_counts().to_dict(),
        "day_by_run": {r: {"min": float(g.day.min()), "median": float(g.day.median()),
                           "max": float(g.day.max())} for r, g in df.groupby("run_date")},
        "eta2_day_on_run": float(ss_b / ss_t),
        "kruskal_p_day_run": float(kruskal(*groups).pvalue),
        "score_by_run_median": df.groupby("run_date").score.median().to_dict(),
        "stages_in_one_run_only": {r: st.index[(st[r] > 0) & (st.drop(columns=r).sum(1) == 0)]
                                   .tolist() for r in st.columns},
        "stages_shared": st.index[(st > 0).all(1)].tolist(),
        "animals_in_run_only_stages": int(df.stage.isin(
            st.index[(st > 0).sum(1) == 1]).sum()),
    }


def leave_one_run_out(X, df):
    out = {}
    runs = sorted(df.run_date.unique())
    for tr_run in runs:
        te_run = [r for r in runs if r != tr_run][0]
        tr, te = (df.run_date == tr_run).to_numpy(), (df.run_date == te_run).to_numpy()
        day, sc = df.day.to_numpy(), df.score.to_numpy().reshape(-1, 1)
        res = {"train_run": tr_run, "test_run": te_run, "n_train": int(tr.sum()),
               "n_test": int(te.sum())}
        # clock A: raw day ~ genes, fit on the training run only
        pa = _fit_predict(X[tr], day[tr], X[te])
        res["clock_A_raw"] = {**summarise(pa, day[te]),
                              "mean_bias_days": float((pa - day[te]).mean())}
        # clock B: residualise X and day on severity using the training run only
        fx = LinearRegression().fit(sc[tr], X[tr])
        fy = LinearRegression().fit(sc[tr], day[tr])
        Xtr_r, Xte_r = X[tr] - fx.predict(sc[tr]), X[te] - fx.predict(sc[te])
        ytr_r, yte_r = day[tr] - fy.predict(sc[tr]), day[te] - fy.predict(sc[te])
        pb_ = _fit_predict(Xtr_r, ytr_r, Xte_r)
        res["clock_B_sev_orth"] = {**summarise(pb_, yte_r),
                                   "mean_bias_days": float((pb_ - yte_r).mean())}
        res["test_day_range"] = [float(day[te].min()), float(day[te].max())]
        res["train_day_range"] = [float(day[tr].min()), float(day[tr].max())]
        out[f"{tr_run}->{te_run}"] = res
    return out


def _fit_predict(Xtr, ytr, Xte, seed=0):
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler
    from duration_clock import _topvar
    sel = _topvar(Xtr)
    sc = StandardScaler().fit(Xtr[:, sel])
    m = ElasticNetCV(random_state=seed, **{**ENET_KW, "cv": min(5, len(ytr) - 1)})
    m.fit(sc.transform(Xtr[:, sel]), ytr)
    return m.predict(sc.transform(Xte[:, sel]))


def analyse(model, key):
    X, df = load_cohort(model)
    day, run = df.day.to_numpy(), df.run_date.to_numpy()
    sev = df.score.to_numpy().reshape(-1, 1)
    sev_run = np.column_stack([sev, run_dummies(run)])
    print(f"[{key}] n={len(df)} animals, runs {df.run_date.value_counts().to_dict()}", flush=True)

    res = {"n_animals": len(df), "aliasing": aliasing(df),
           "tables": {"run_date x day": pd.crosstab(df.run_date, df.day.astype(int)),
                      "run_date x stage": pd.crosstab(df.stage, df.run_date)}}

    pred_b = loao_clock(X, day, covar=sev)
    res["baseline_B"] = {"full_data_ref": summarise(pred_b, loao_target(day, sev)),
                         "foldwise_ref": summarise(pred_b, foldwise_target(day, sev))}
    print(f"[{key}] baseline clock B rho {res['baseline_B']['full_data_ref']['rho']:+.3f}",
          flush=True)

    pred_r = loao_clock(X, day, covar=sev_run)
    res["run_adjusted_B"] = {"full_data_ref": summarise(pred_r, loao_target(day, sev_run)),
                             "foldwise_ref": summarise(pred_r, foldwise_target(day, sev_run))}
    print(f"[{key}] run-adjusted clock B rho "
          f"{res['run_adjusted_B']['full_data_ref']['rho']:+.3f}", flush=True)
    a, b = res["baseline_B"]["full_data_ref"], res["run_adjusted_B"]["full_data_ref"]
    res["fisher_z_baseline_vs_run_adjusted"] = compare_rhos(a["rho"], a["n"], b["rho"], b["n"])

    print(f"[{key}] {N_PERM} within-run permutations of the run-adjusted clock ...", flush=True)
    null = np.array(Parallel(n_jobs=N_JOBS)(
        delayed(_perm_within_run)(X, day, run, sev_run, 1 + i) for i in range(N_PERM)))
    res["run_adjusted_B"]["perm_within_run"] = {
        "null_mean": float(null.mean()), "null_q95": float(np.quantile(null, 0.95)),
        "p": float((1 + (null >= b["rho"]).sum()) / (1 + N_PERM)), "n_perm": N_PERM}

    res["leave_one_run_out"] = leave_one_run_out(X, df)
    for k, v in res["leave_one_run_out"].items():
        print(f"[{key}] LORO {k}: clock A rho {v['clock_A_raw']['rho']:+.3f}, "
              f"clock B rho {v['clock_B_sev_orth']['rho']:+.3f}", flush=True)
    res["verdict"] = verdict(res)
    return res


def verdict(res):
    base = res["baseline_B"]["full_data_ref"]
    adj = res["run_adjusted_B"]["full_data_ref"]
    loro = [v["clock_B_sev_orth"] for v in res["leave_one_run_out"].values()]
    if adj["ci"][0] <= 0 or all(v["ci"][0] <= 0 for v in loro):
        return "collapses"
    if base["ci"][0] <= adj["rho"] <= base["ci"][1]:
        return "survives"
    return "partial drop"


# ---------------------------------------------------------------- main

def main():
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    results = {"settings": {"n_perm": N_PERM, "n_boot": N_BOOT, "enet": str(ENET_KW),
                            "excluded_sections": sorted(EXCLUDE_SECTIONS),
                            "published_clock_B_rho": PUBLISHED}}
    for model, key in COHORTS:
        results[key] = analyse(model, key)
    tables = {k: results[k].pop("tables") for _, k in COHORTS}
    for k, tb in tables.items():
        for name, t in tb.items():
            t.to_csv(os.path.join(OUT, f"{k}_{name.replace(' ', '_')}.csv"))
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    _plot(results)
    _report(results, tables)


# ---------------------------------------------------------------- figure

INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
ARM_COLORS = {"baseline": "#2a78d6", "run-adjusted": "#eb6834", "LORO": "#1baf7a"}


def _plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (model, key) in zip(axes, COHORTS):
        r = results[key]
        rows = [("baseline clock B\n(LOAO)", "baseline", r["baseline_B"]["full_data_ref"]),
                ("+ run covariate\n(LOAO)", "run-adjusted", r["run_adjusted_B"]["full_data_ref"])]
        for k, v in r["leave_one_run_out"].items():
            tr, te = k.split("->")
            rows.append((f"train {tr}\ntest {te} (n={v['n_test']})", "LORO", v["clock_B_sev_orth"]))
        ax.set_facecolor(SURFACE)
        for i, (lab, kind, s) in enumerate(rows):
            ax.plot([i, i], s["ci"], color=ARM_COLORS[kind], lw=2, solid_capstyle="round")
            ax.plot(i, s["rho"], "o", ms=9, color=ARM_COLORS[kind], mec=SURFACE, mew=2)
            ax.annotate(f"{s['rho']:.2f}", (i, s["rho"]), xytext=(9, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(range(len(rows)), [x[0] for x in rows], fontsize=7, color=INK)
        ax.set_xlim(-0.5, len(rows) - 0.4)
        ax.set_ylim(-1, 1)
        ax.set_title(f"{'RR' if key == 'rr' else 'Chronic'} (n={r['n_animals']}) — "
                     f"{r['verdict']}", loc="left", fontsize=9, color=INK)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(axis="y", color="#e6e5e0", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Spearman ρ, severity-orthogonalised clock\n(95% CI, animal bootstrap)",
                       fontsize=8, color=MUTED)
    fig.legend([Line2D([], [], color=c, lw=2, marker="o") for c in ARM_COLORS.values()],
               list(ARM_COLORS), loc="upper right", ncol=3, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT, "figures", "clock_run_robustness.png"), dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- report

def _s(s):
    return (f"rho {s['rho']:+.3f} [95% CI {s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}]  "
            f"R2 {s['r2']:+.3f}  n={s['n']}")


def _report(results, tables):
    L = ["CLOCK RUN-ROBUSTNESS — does the duration clock survive the Xenium run?", "=" * 72, "",
         "Severity-orthogonalised clock B (scripts/duration_clock.py loao_clock, unchanged),",
         "refit on the current all-animal pseudobulk (section C2_G3_Mid_1 excluded). CIs:",
         f"{N_BOOT} animal bootstrap resamples of the LOAO predictions. Fisher z treats the two",
         "rhos as independent; they share animals, so that p is conservative.",
         "Interpretation fixed in advance: SURVIVES = run-adjusted rho CI excludes 0 and its",
         "point estimate lies inside the baseline CI; COLLAPSES = run-adjusted CI includes 0, or",
         "leave-one-run-out CI includes 0 in both directions; otherwise PARTIAL DROP.", ""]
    for model, key in COHORTS:
        r = results[key]
        al = r["aliasing"]
        L += [f"{model}  (n = {r['n_animals']} animals)", "-" * 60,
              "1. ALIASING",
              f"  animals per run: {al['n_by_run']}",
              "  day by run: " + "; ".join(f"{k}: {v['min']:.0f}-{v['max']:.0f} (median "
                                           f"{v['median']:.0f})" for k, v in al["day_by_run"].items()),
              f"  eta^2 (share of day variance explained by run): {al['eta2_day_on_run']:.3f};"
              f" Kruskal day~run p = {al['kruskal_p_day_run']:.3f}",
              f"  median score by run: {al['score_by_run_median']}",
              f"  stages present in both runs: {al['stages_shared']}",
              f"  stages in one run only: {al['stages_in_one_run_only']} "
              f"({al['animals_in_run_only_stages']} of {r['n_animals']} animals)"]
        for name, t in tables[key].items():
            L += [f"  {name}:", "  " + t.to_string().replace("\n", "\n  ")]
        b, a = r["baseline_B"], r["run_adjusted_B"]
        fz = r["fisher_z_baseline_vs_run_adjusted"]
        pm = a["perm_within_run"]
        L += ["", "2. RUN AS A COVARIATE (residualise X and day on severity + run, per fold)",
              f"  published clock B rho: {PUBLISHED[model]:+.3f} (older atlas pseudobulk)",
              f"  baseline  B, full-data ref : {_s(b['full_data_ref'])}",
              f"  baseline  B, fold-wise ref : {_s(b['foldwise_ref'])}",
              f"  run-adj.  B, full-data ref : {_s(a['full_data_ref'])}",
              f"  run-adj.  B, fold-wise ref : {_s(a['foldwise_ref'])}",
              f"  change in rho: {a['full_data_ref']['rho'] - b['full_data_ref']['rho']:+.3f}; "
              f"change in R2: {a['full_data_ref']['r2'] - b['full_data_ref']['r2']:+.3f}; "
              f"Fisher z p = {fz['p_value']:.3f}",
              f"  within-run permutation null ({pm['n_perm']} shuffles of day inside each run): "
              f"mean {pm['null_mean']:+.3f}, q95 {pm['null_q95']:+.3f}, p = {pm['p']:.3f}",
              "", "3. LEAVE-ONE-RUN-OUT (fit on one run, predict the other)"]
        for k, v in r["leave_one_run_out"].items():
            L += [f"  train {v['train_run']} (n={v['n_train']}, day {v['train_day_range'][0]:.0f}-"
                  f"{v['train_day_range'][1]:.0f})  ->  test {v['test_run']} (n={v['n_test']}, "
                  f"day {v['test_day_range'][0]:.0f}-{v['test_day_range'][1]:.0f})",
                  f"    clock A raw     : {_s(v['clock_A_raw'])}  mean bias "
                  f"{v['clock_A_raw']['mean_bias_days']:+.1f} d",
                  f"    clock B sev-orth: {_s(v['clock_B_sev_orth'])}  mean bias "
                  f"{v['clock_B_sev_orth']['mean_bias_days']:+.1f} d"]
        L += ["", f"VERDICT ({model}): {r['verdict'].upper()}", ""]
    L += ["CONSEQUENCE FOR TASK 4",
          "  If either cohort COLLAPSES, the Task 4 ladder must be reconsidered before running.",
          "  Otherwise Task 4 proceeds; arm 2 should still be reported with and without the run",
          "  covariate so the run share of every target stays visible.",
          "", "figures/: clock_run_robustness.png"]
    report = "\n".join(L)
    print(report)
    with open(os.path.join(OUT, "report.txt"), "w") as fh:
        fh.write(report + "\n")


if __name__ == "__main__":
    main()
