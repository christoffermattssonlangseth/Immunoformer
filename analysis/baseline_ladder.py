"""THE BASELINE LADDER — does cell- and niche-level structure beat animal pseudobulk?
(WORKORDER Task 4, revised targets)

The duration clock is animal pseudobulk: no cell boundaries, no coordinates. Immunoformer's
premise is that cell/niche structure adds information. This puts every arm on the same
leave-one-animal-out (LOAO) folds, per target, and compares them.

Targets (from runs/trajectory_features/animal_trajectory.csv; residualised per fold):
  days_since_last_peak | score + day | RR       primary: time since last insult
  slope_sign           | score       | RR       ascending (1) vs descending (0) at sacrifice
  onset_day            | -           | RR, CHR  latency as a constitutive host property
  cumulative_score     | score + day | CHRONIC  accumulated burden (collinear covariates)
  day_of_sacrifice     | score       | RR       incumbent comparator (published rho 0.799)
Cohorts: EAE animals with a defined target; day_of_sacrifice keeps the incumbent's 33 RR
animals (incl. 3 PLP CFA controls) for continuity. Section C2_G3_Mid_1 is excluded
(Task 0 reassignment rule).

Arms (arms 1-5 here; arm 6 is analysis/mil_loao.py, read back from its predictions file):
  1  score_sacrifice alone — predicts the RAW target (the residualised target is ~0 by
     construction); evaluated against the raw target.
  2  animal pseudobulk, 5101 genes (loao_clock, unchanged)        2r: + run_date covariate
  3  Global_niche composition (27 fractions)
  4  Anno_L1_curated composition (19 fractions)
  5  arm 2 + arm 3 — top-variance prefilter on genes only; niche fractions always kept
     (the variance filter would otherwise drop them: fractions have tiny variance).
Metrics: Spearman rho and LOAO R2 vs the full-data residual target, plus the fold-wise
(leak-free) residual reference; 95% CI from 2000 animal bootstrap resamples; Fisher z vs
arm 2 (treats rhos as independent; they share animals, so p is conservative). slope_sign
also reports AUC. Permutation null (1000 shuffles) for arm 2 on day_of_sacrifice: run with
PERM_ONLY=1 (heavy; meant for the remote machine).

POWER, decided in advance: residualised targets keep 9-60% of their variance on 28-34
animals. "Arm 6 does not beat arm 2" with overlapping CIs is INCONCLUSIVE, not negative.

    PYTHONPATH="$PWD:$PWD/scripts" python analysis/baseline_ladder.py
    PERM_ONLY=1 N_JOBS=14 PYTHONPATH="$PWD:$PWD/scripts" python analysis/baseline_ladder.py
"""

from __future__ import annotations

import immunotransformer  # noqa: F401  (OpenMP guard — must precede numpy; see HANDOFF.md)

import json
import os
import sys
import warnings

import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from duration_clock import ENET_KW, _topvar, loao_clock, loao_target  # noqa: E402
from immunotransformer.stats import bootstrap_spearman, compare_rhos  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

ATLAS = os.environ.get(
    "RRMAP2_H5AD",
    os.path.expanduser(
        "~/Downloads/RRMAP2_xenium_all_samples.cellcharter.companion.ready.with_metadata."
        "rerun.with_AnnoL1Curated_with_Region_Anno2to4Updated.h5ad"),
)
PB = "runs/rr_within_relapse/pseudobulk_all67.npz"
TRAJ = "runs/trajectory_features/animal_trajectory.csv"
OUT = "runs/baseline_ladder"
COMP_CACHE = os.path.join(OUT, "composition_sections.csv")
ARM6 = os.path.join(OUT, "arm6_predictions.csv")
ARM6R = os.path.join(OUT, "arm6_runadj_predictions.csv")
ARM6B = os.path.join(OUT, "arm6b_meanpool_predictions.csv")
EXCLUDE_SECTIONS = {"C2_G3_Mid_1"}
COMP_KEYS = {"arm3": "Global_niche", "arm4": "Anno_L1_curated"}
N_BOOT = 2000
N_PERM = int(os.environ.get("N_PERM", "1000"))
N_JOBS = int(os.environ.get("N_JOBS", "8"))
RR, CHR = "RELAPSE REMITTING", "CHRONIC"

TARGETS = [
    # key,                   column,                 cohort, covariates,          kind
    ("days_since_last_peak", "days_since_last_peak", RR,  ["score", "day"], "continuous"),
    ("slope_sign",           "slope_sign",           RR,  ["score"],        "binary"),
    ("onset_day_rr",         "onset_day",            RR,  [],               "continuous"),
    ("onset_day_chronic",    "onset_day",            CHR, [],               "continuous"),
    ("cumulative_chronic",   "cumulative_score",     CHR, ["score", "day"], "continuous"),
    ("day_of_sacrifice",     "day_of_sacrifice",     RR,  ["score"],        "continuous"),
]
ARMS = ["arm1", "arm2", "arm2r", "arm3", "arm4", "arm5", "arm6", "arm6r", "arm6b"]
ARM_LABEL = {"arm1": "1 score only (raw target)", "arm2": "2 pseudobulk",
             "arm2r": "2r pseudobulk + run", "arm3": "3 niche comp.",
             "arm4": "4 cell-type comp.", "arm5": "5 pseudobulk + niche",
             "arm6": "6 attention-MIL", "arm6r": "6r attention-MIL + run",
             "arm6b": "6b mean-pool MIL (control)"}
PROGRAMS = {"ratchet": ["Gpnmb", "Igf2", "Fmod", "Fcrls", "Plin4"],
            "acute": ["Hal", "Arg1", "Chil3"]}


# ---------------------------------------------------------------- data

def _obs_col(obs, k):
    g = obs[k]
    cats = [c.decode() if isinstance(c, bytes) else c for c in g["categories"][:]]
    return pd.Categorical.from_codes(g["codes"][:], cats)


def section_composition():
    """Cells per (section, category) for both composition keys; cached as a tidy CSV."""
    if os.path.exists(COMP_CACHE):
        return pd.read_csv(COMP_CACHE)
    with h5py.File(ATLAS, "r") as f:
        obs = f["obs"]
        df = pd.DataFrame({k: _obs_col(obs, k) for k in ["meta_sample_id", *COMP_KEYS.values()]})
    rows = []
    for key in COMP_KEYS.values():
        c = df.groupby(["meta_sample_id", key], observed=False).size().rename("n").reset_index()
        rows.append(c.rename(columns={key: "category"}).assign(key=key))
    out = pd.concat(rows, ignore_index=True)
    os.makedirs(OUT, exist_ok=True)
    out.to_csv(COMP_CACHE, index=False)
    return out


def load_all():
    d = np.load(PB, allow_pickle=True)
    meta = pd.DataFrame(d["section_meta"].tolist(), columns=d["section_meta_cols"])
    keep = ~meta.meta_sample_id.isin(EXCLUDE_SECTIONS).to_numpy()
    meta, counts = meta[keep].reset_index(drop=True), d["section_counts"][keep]
    genes = d["genes"].astype(str)
    animals = meta.sample_name.drop_duplicates().to_numpy()
    pb = np.vstack([counts[(meta.sample_name == a).to_numpy()].sum(0) for a in animals])
    pb = np.log1p(pb / pb.sum(1, keepdims=True) * 1e4)

    comp = section_composition()
    comp = comp[comp.meta_sample_id.isin(meta.meta_sample_id)]
    comp = comp.merge(meta[["meta_sample_id", "sample_name"]], on="meta_sample_id")
    fracs = {}
    for arm, key in COMP_KEYS.items():
        t = (comp[comp.key == key].groupby(["sample_name", "category"]).n.sum()
             .unstack(fill_value=0).loc[animals])
        fracs[arm] = (t.div(t.sum(1), axis=0)).to_numpy(float), t.columns.to_numpy()

    traj = pd.read_csv(TRAJ).set_index("sample_name").loc[animals]
    am = meta.drop_duplicates("sample_name").set_index("sample_name").loc[animals]
    info = pd.DataFrame({
        "sample_name": animals, "model": am.model.to_numpy(), "run_date": am.run_date.to_numpy(),
        "stage": am.stage.to_numpy(), "condition": traj.condition.to_numpy(),
        "day": traj.day_of_sacrifice.to_numpy(float),
        "score": traj.score_sacrifice.to_numpy(float),
        "days_since_last_peak": traj.days_since_last_peak.to_numpy(float),
        "onset_day": traj.onset_day.to_numpy(float),
        "cumulative_score": traj.cumulative_score.to_numpy(float),
        "day_of_sacrifice": traj.day_of_sacrifice.to_numpy(float),
        "slope_sign": traj.slope_class.map({"ascending": 1.0, "descending": 0.0}).to_numpy(float),
    })
    return pb, genes, fracs, info


def cohort(info, col, model, key):
    m = (info.model == model) & np.isfinite(info[col]) & np.isfinite(info.score)
    if key != "day_of_sacrifice":
        m &= info.condition == "EAE"
    return np.where(m)[0]


# ---------------------------------------------------------------- arms

def loao_keep(Xg, Xk, y, covar=None, seed=0):
    """loao_clock with the top-variance prefilter applied to Xg only; Xk always kept."""
    n = len(y)
    pred = np.full(n, np.nan)
    X = np.hstack([Xg, Xk])
    ng = Xg.shape[1]
    for i in range(n):
        tr = np.ones(n, bool); tr[i] = False
        Xtr, Xte, ytr = X[tr], X[i:i + 1], y[tr]
        if covar is not None:
            fx = LinearRegression().fit(covar[tr], Xtr)
            fy = LinearRegression().fit(covar[tr], ytr)
            Xtr, Xte = Xtr - fx.predict(covar[tr]), Xte - fx.predict(covar[i:i + 1])
            ytr = ytr - fy.predict(covar[tr])
        sel = np.r_[_topvar(Xtr[:, :ng]), np.arange(ng, X.shape[1])]
        sc = StandardScaler().fit(Xtr[:, sel])
        m = ElasticNetCV(random_state=seed, **ENET_KW).fit(sc.transform(Xtr[:, sel]), ytr)
        pred[i] = m.predict(sc.transform(Xte[:, sel]))[0]
    return pred


def foldwise_target(y, covar):
    out = np.empty(len(y))
    for i in range(len(y)):
        tr = np.ones(len(y), bool); tr[i] = False
        out[i] = y[i] - LinearRegression().fit(covar[tr], y[tr]).predict(covar[i:i + 1])[0]
    return out


def run_dummies(run):
    levels = sorted(set(run))
    return np.column_stack([(run == lv).astype(float) for lv in levels[1:]])


def metrics(pred, ref, ref_fold, binary_y=None, seed=0):
    b = bootstrap_spearman(pred, ref, n_boot=N_BOOT, seed=seed)
    out = {"rho": float(b["rho"]), "ci": [float(b["ci_lo"]), float(b["ci_hi"])],
           "ci_width": float(b["ci_hi"] - b["ci_lo"]),
           "r2": float(1 - np.sum((pred - ref) ** 2) / np.sum((ref - ref.mean()) ** 2)),
           "rho_foldwise_ref": float(spearmanr(pred, ref_fold).statistic), "n": int(len(ref))}
    if binary_y is not None:
        out["auc"] = float(roc_auc_score(binary_y, pred))
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(pred), len(pred))
            if 0 < binary_y[i].sum() < len(i):
                vals.append(roc_auc_score(binary_y[i], pred[i]))
        out["auc_ci"] = [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]
    return out


def run_target(spec, pb, fracs, info, arm6):
    key, col, model, covs, kind = spec
    idx = cohort(info, col, model, key)
    sub = info.iloc[idx].reset_index(drop=True)
    y = sub[col].to_numpy(float)
    covar = sub[covs].to_numpy(float) if covs else None
    ref = loao_target(y, covar) if covs else y
    ref_fold = foldwise_target(y, covar) if covs else y
    biny = y.astype(int) if kind == "binary" else None
    X = pb[idx]
    print(f"[{key}] n={len(y)} ({model}), covariates {covs or 'none'}", flush=True)

    preds = {}
    preds["arm1"] = loao_clock(sub[["score"]].to_numpy(float), y)
    preds["arm2"] = loao_clock(X, y, covar=covar)
    run_cov = run_dummies(sub.run_date.to_numpy())
    covar_r = run_cov if covar is None else np.column_stack([covar, run_cov])
    preds["arm2r"] = loao_clock(X, y, covar=covar_r)
    preds["arm3"] = loao_clock(fracs["arm3"][0][idx], y, covar=covar)
    preds["arm4"] = loao_clock(fracs["arm4"][0][idx], y, covar=covar)
    preds["arm5"] = loao_keep(X, fracs["arm3"][0][idx], y, covar=covar)
    if arm6 is not None:
        a6 = arm6[arm6.target == key].set_index("sample_name")
        if set(sub.sample_name) <= set(a6.index):
            preds["arm6"] = a6.loc[sub.sample_name, "pred"].to_numpy(float)
    if os.path.exists(ARM6B):
        a6b = pd.read_csv(ARM6B)
        a6b = a6b[a6b.target == key].set_index("sample_name")
        if len(a6b) and set(sub.sample_name) <= set(a6b.index):
            preds["arm6b"] = a6b.loc[sub.sample_name, "pred"].to_numpy(float)
    if os.path.exists(ARM6R):
        a6r = pd.read_csv(ARM6R)
        a6r = a6r[a6r.target == key + "_runadj"].set_index("sample_name")
        if len(a6r) and set(sub.sample_name) <= set(a6r.index):
            preds["arm6r"] = a6r.loc[sub.sample_name, "pred"].to_numpy(float)

    res = {"n": int(len(y)), "cohort": model, "covariates": covs, "kind": kind,
           "animals": sub.sample_name.tolist(), "arms": {},
           "by_run": sub.groupby("run_date")[col].agg(["count", "median", "min", "max"])
                        .to_dict(orient="index")}
    loo_mean = (y.sum() - y) / (len(y) - 1)
    ref_r = loao_target(y, covar_r)
    for arm, p in preds.items():
        if arm == "arm1":
            m = metrics(p, y, y, biny)
        elif arm in ("arm2r", "arm6r"):
            m = metrics(p, ref_r, foldwise_target(y, covar_r), biny)
        else:
            m = metrics(p, ref, ref_fold, biny)
        # intercept-only fold: no feature survived, prediction = training mean of the
        # (residualised) target -> LOAO rho is pushed toward -1 by construction
        null = loo_mean if (arm == "arm1" or (arm != "arm2r" and not covs)) else np.zeros(len(y))
        if not arm.startswith("arm6"):
            m["n_null_folds"] = int((np.abs(p - null) < 1e-6 * (np.std(y) + 1)).sum())
        res["arms"][arm] = m
        print(f"  {arm}: rho {m['rho']:+.3f} [{m['ci'][0]:+.2f}, {m['ci'][1]:+.2f}]", flush=True)
    a2 = res["arms"]["arm2"]
    for arm, m in res["arms"].items():
        if arm != "arm2":
            m["fisher_z_vs_arm2_p"] = compare_rhos(m["rho"], m["n"], a2["rho"], a2["n"])["p_value"]
    if covs == ["score", "day"]:
        r = np.corrcoef(sub.score, sub.day)[0, 1]
        res["vif_score_day"] = float(1 / (1 - r ** 2))
        res["corr_score_day"] = float(r)
    res["residual_variance_share"] = float(np.var(ref) / np.var(y)) if covs else 1.0
    res["programs"] = program_matrix(pb, idx, sub, ref)
    return res, {a: p.tolist() for a, p in preds.items()}


def program_matrix(pb, idx, sub, ref, genes_all=None):
    out = {}
    for name, gl in PROGRAMS.items():
        gi = [GENE_INDEX[g] for g in gl if g in GENE_INDEX]
        Z = pb[idx][:, gi]
        Z = (Z - Z.mean(0)) / Z.std(0)
        s = Z.mean(1)
        b = bootstrap_spearman(s, ref, n_boot=N_BOOT, seed=1)
        out[name] = {"rho": float(b["rho"]), "ci": [float(b["ci_lo"]), float(b["ci_hi"])],
                     "genes_used": [g for g in gl if g in GENE_INDEX]}
    return out


GENE_INDEX: dict = {}


# ---------------------------------------------------------------- permutation null

def _perm_one(X, y, covar, seed):
    yp = y[np.random.default_rng(seed).permutation(len(y))]
    return spearmanr(loao_clock(X, yp, covar=covar), loao_target(yp, covar)).statistic


def perm_only(pb, info):
    key, col, model, covs, _ = [t for t in TARGETS if t[0] == "day_of_sacrifice"][0]
    idx = cohort(info, col, model, key)
    sub = info.iloc[idx]
    y, covar = sub[col].to_numpy(float), sub[covs].to_numpy(float)
    obs = spearmanr(loao_clock(pb[idx], y, covar=covar), loao_target(y, covar)).statistic
    print(f"[perm] arm 2 day_of_sacrifice observed rho {obs:+.3f}; {N_PERM} perms on "
          f"{N_JOBS} jobs", flush=True)
    null = np.array(Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(_perm_one)(pb[idx], y, covar, 1 + i) for i in range(N_PERM)))
    out = {"target": key, "arm": "arm2", "observed_rho": float(obs), "n_perm": N_PERM,
           "null_mean": float(null.mean()), "null_q95": float(np.quantile(null, 0.95)),
           "null_max": float(null.max()),
           "p": float((1 + (null >= obs).sum()) / (1 + N_PERM)), "null": null.tolist()}
    with open(os.path.join(OUT, "perm_arm2_day_of_sacrifice.json"), "w") as fh:
        json.dump(out, fh)
    print(f"[perm] p = {out['p']:.4f} (floor {1 / (1 + N_PERM):.4f})")


# ---------------------------------------------------------------- main

def main():
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    pb, genes, fracs, info = load_all()
    GENE_INDEX.update({g: i for i, g in enumerate(genes)})
    if os.environ.get("PERM_ONLY"):
        perm_only(pb, info)
        return
    arm6 = pd.read_csv(ARM6) if os.path.exists(ARM6) else None
    results = {"settings": {"enet": str(ENET_KW), "n_boot": N_BOOT,
                            "excluded_sections": sorted(EXCLUDE_SECTIONS),
                            "programs": PROGRAMS,
                            "panel_check": {k: [g for g in v if g not in GENE_INDEX]
                                            for k, v in PROGRAMS.items()}},
               "targets": {}}
    all_preds = []
    for spec in TARGETS:
        res, preds = run_target(spec, pb, fracs, info, arm6)
        results["targets"][spec[0]] = res
        for arm, p in preds.items():
            all_preds += [{"target": spec[0], "arm": arm, "sample_name": a, "pred": v}
                          for a, v in zip(res["animals"], p)]
    perm_path = os.path.join(OUT, "perm_arm2_day_of_sacrifice.json")
    if os.path.exists(perm_path):
        with open(perm_path) as fh:
            p = json.load(fh)
        results["perm_arm2_day_of_sacrifice"] = {k: v for k, v in p.items() if k != "null"}
    pd.DataFrame(all_preds).to_csv(os.path.join(OUT, "loao_predictions_arms1-5.csv"), index=False)
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    _plot(results)
    _report(results)


# ---------------------------------------------------------------- figure

INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
ARM_COLOR = {"arm1": "#b7b6b0", "arm2": "#2a78d6", "arm2r": "#86b6ef", "arm3": "#1baf7a",
             "arm4": "#eda100", "arm5": "#4a3aa7", "arm6": "#eb6834", "arm6r": "#f4a582",
             "arm6b": "#e87ba4"}


def _plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = results["targets"]
    fig, axes = plt.subplots(1, len(T), figsize=(3.1 * len(T), 4.4), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (key, r) in zip(axes, T.items()):
        ax.set_facecolor(SURFACE)
        arms = [a for a in ARMS if a in r["arms"]]
        for i, a in enumerate(arms):
            m = r["arms"][a]
            ax.plot([i, i], m["ci"], color=ARM_COLOR[a], lw=2, solid_capstyle="round")
            hollow = m.get("n_null_folds", 0) > m["n"] / 2   # mostly intercept-only folds
            ax.plot(i, m["rho"], "o", ms=8, color=SURFACE if hollow else ARM_COLOR[a],
                    mec=ARM_COLOR[a] if hollow else SURFACE, mew=1.5)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(range(len(arms)), [a.replace("arm", "") for a in arms], fontsize=8,
                      color=INK)
        ax.set_ylim(-1, 1)
        cov = "+".join(r["covariates"]) or "none"
        ax.set_title(f"{key}\n{'RR' if r['cohort'] == RR else 'chronic'}, n={r['n']}, "
                     f"resid: {cov}", loc="left", fontsize=8, color=INK)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(axis="y", color="#e6e5e0", lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("arm", fontsize=7, color=MUTED)
    axes[0].set_ylabel("Spearman ρ, LOAO (95% CI, animal bootstrap)", fontsize=8, color=MUTED)
    fig.text(0.01, 0.01, "  ".join(f"{a.replace('arm', '')} = {ARM_LABEL[a][2:].strip()}"
                                   for a in ARMS) + "   ·   hollow = most folds kept no feature "
             "(LOAO mean-fallback; rho near -1 is an artefact, i.e. no signal)",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(OUT, "figures", "baseline_ladder.png"), dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- report

def _report(results):
    L = ["THE BASELINE LADDER — LOAO, same folds for every arm, per target", "=" * 72, "",
         "rho / R2 vs the full-data residual target (arm 1: vs the RAW target); fold-wise =",
         "rho vs the leak-free fold-wise residual. CI = 95% animal bootstrap. Fisher z vs arm 2",
         "treats rhos as independent (they share animals) -> conservative. Arm 2r adds run_date",
         "to the residualisation (Task 1b). Pre-registered: overlapping CIs = INCONCLUSIVE.",
         "null = folds where no feature survived (prediction = leave-one-out mean). Under LOAO",
         "an all-null arm gives rho = -1 BY CONSTRUCTION; strongly negative rho with negative R2",
         "means NO SIGNAL, not anti-signal. Arm 6 has no such fallback.",
         f"Panel check, program genes missing: {results['settings']['panel_check']}", ""]
    for key, r in results["targets"].items():
        cov = " + ".join(r["covariates"]) or "nothing"
        L += [f"TARGET {key}  ({'RR' if r['cohort'] == RR else 'chronic'}, n = {r['n']}, "
              f"residualised on {cov}; residual keeps {r['residual_variance_share']:.0%} of "
              "target variance)"]
        L.append("  target by run: " + "; ".join(
            f"{k}: n {v['count']}, median {v['median']:.1f} ({v['min']:.0f}-{v['max']:.0f})"
            for k, v in r["by_run"].items()))
        if "vif_score_day" in r:
            L.append(f"  corr(score, day) = {r['corr_score_day']:+.2f}, VIF = "
                     f"{r['vif_score_day']:.2f}")
        L.append(f"  {'arm':26s} {'rho':>7s} {'95% CI':>17s} {'width':>6s} {'R2':>7s} "
                 f"{'fold-wise':>9s} {'Fz p vs 2':>9s} {'null':>5s}" + ("  AUC [95% CI]" if r["kind"] ==
                                                         "binary" else ""))
        for a in ARMS:
            if a not in r["arms"]:
                if a == "arm6":
                    L.append(f"  {ARM_LABEL[a]:26s}  pending (analysis/mil_loao.py)")
                elif a == "arm6r" and key in ("days_since_last_peak", "onset_day_rr"):
                    L.append(f"  {ARM_LABEL[a]:26s}  pending (RUN_ADJ_TARGETS)")
                continue
            m = r["arms"][a]
            fz = "-" if a == "arm2" else f"{m['fisher_z_vs_arm2_p']:.3f}"
            line = (f"  {ARM_LABEL[a]:26s} {m['rho']:+7.3f} [{m['ci'][0]:+.2f}, {m['ci'][1]:+.2f}]"
                    f" {m['ci_width']:6.2f} {m['r2']:+7.3f} {m['rho_foldwise_ref']:+9.3f} {fz:>9s}"
                    f" {m.get('n_null_folds', 0):>2d}/{m['n']:<2d}")
            if "auc" in m:
                line += f"  {m['auc']:.2f} [{m['auc_ci'][0]:.2f}, {m['auc_ci'][1]:.2f}]"
            L.append(line)
        pr = r["programs"]
        L.append("  programs vs residual target: " + "; ".join(
            f"{k} rho {v['rho']:+.2f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]" for k, v in pr.items()))
        L.append("")

    L += ["CROSS-TARGET MATRIX — program score (mean z of genes) vs each residualised target",
          f"  {'target':22s} " + " ".join(f"{k:>24s}" for k in PROGRAMS)]
    for key, r in results["targets"].items():
        L.append(f"  {key:22s} " + " ".join(
            f"{r['programs'][k]['rho']:+.2f} [{r['programs'][k]['ci'][0]:+.2f},"
            f"{r['programs'][k]['ci'][1]:+.2f}]".rjust(24) for k in PROGRAMS))
    L += ["  Prediction (WORKORDER): ratchet tracks days_since_last_peak, acute tracks",
          "  slope_sign — not the reverse.", ""]
    if "perm_arm2_day_of_sacrifice" in results:
        p = results["perm_arm2_day_of_sacrifice"]
        L.append(f"PERMUTATION NULL, arm 2 on day_of_sacrifice: {p['n_perm']} shuffles, null mean "
                 f"{p['null_mean']:+.3f}, q95 {p['null_q95']:+.3f}, max {p['null_max']:+.3f}, "
                 f"p = {p['p']:.4f}")
    else:
        L.append("PERMUTATION NULL (1000) for arm 2 on day_of_sacrifice: pending (PERM_ONLY=1).")
    L += ["", "figures/: baseline_ladder.png"]
    report = "\n".join(L)
    print(report)
    with open(os.path.join(OUT, "report.txt"), "w") as fh:
        fh.write(report + "\n")


if __name__ == "__main__":
    main()
