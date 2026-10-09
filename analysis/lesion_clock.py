"""Lesion clock: can individual lesions be dated, and do animals accrue NEW lesions or GROW old ones?

Lesions come from analysis/lesion_segment.py (lesionSegmenter ported to Xenium; lesion
mask + 50 um perilesional margin). Each lesion pseudobulk is scored with the EXISTING
duration-clock pipeline (scripts/duration_clock.py mechanics: residualise X and day on
score_sacrifice, top-1000-variance, scaler, ElasticNetCV; no new architecture). Scoring
uses the LOAO FOLD clock that never saw the lesion's animal (an all-animal clock applied
to its own training animals would make step 2c in-sample); the in-sample version is
reported alongside. RR lesions use the RR clock, chronic lesions the chronic clock.

VALIDITY RULE (fixed before scoring): lesion dating is valid only if the per-animal MEAN
of lesion predictions correlates with the residualised day target at rho >= 0.5 with a
95% CI excluding 0 (RR). If not, stop: the clock does not resolve to lesions.

Steps: 1a lesions per section / size / total (+ control-animal lesions = false-positive
calibration); 1b lesion count per animal vs day and cohort; 2c validation; 2d lesion size
vs predicted age (+ 2c with size residualised); 3a per-animal age distribution; 3b
spread vs day per cohort; 3c RR vs chronic spread at matched duration; 4a random gene-set
null for 3b/3c; 4b age vs immune fraction / cell density; per-lesion measurement-noise
floor from binomial count halving. All tests group by ANIMAL (n animals reported).

    N_JOBS=4 PYTHONPATH="$PWD:$PWD/scripts" python analysis/lesion_clock.py
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
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from duration_clock import ENET_KW, _topvar, loao_target  # noqa: E402
from immunotransformer.stats import bootstrap_spearman  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
OUT = "runs/lesion_clock"
PB = "runs/rr_within_relapse/pseudobulk_all67.npz"
TRAJ = "runs/trajectory_features/animal_trajectory.csv"
EXCLUDE_SECTIONS = {"C2_G3_Mid_1"}
IMMUNE = ["Myeloid", "T cell", "B cell", "DC", "NK/DC", "Neutrophil"]
COHORTS = {"RELAPSE REMITTING": "rr", "CHRONIC": "chronic"}
MIN_LESIONS = 3
N_RANDOM = int(os.environ.get("N_RANDOM", "100"))
N_JOBS = int(os.environ.get("N_JOBS", "4"))
N_BOOT = 2000


def lognorm(c):
    return np.log1p(c / np.maximum(c.sum(1, keepdims=True), 1) * 1e4)


# ---------------------------------------------------------------- clock objects

class Clock:
    """The duration-clock fit (same mechanics as loao_clock's per-fold fit), kept so it can
    score new samples (lesion pseudobulk) at a given severity."""

    def __init__(self, X, y, sev, cols=None):
        self.cols = np.arange(X.shape[1]) if cols is None else np.asarray(cols)
        X = X[:, self.cols]
        self.fx = LinearRegression().fit(sev, X)
        self.fy = LinearRegression().fit(sev, y)
        Xr = X - self.fx.predict(sev)
        self.sel = _topvar(Xr)
        self.sc = StandardScaler().fit(Xr[:, self.sel])
        self.m = ElasticNetCV(random_state=0, **ENET_KW).fit(self.sc.transform(Xr[:, self.sel]),
                                                              y - self.fy.predict(sev))
        self.nonzero = int((self.m.coef_ != 0).sum())

    def predict(self, X, sev):
        """Returns (residual-scale prediction, day-scale prediction)."""
        X = X[:, self.cols]
        sev = np.broadcast_to(np.asarray(sev, float).reshape(-1, 1), (len(X), 1))
        r = self.m.predict(self.sc.transform((X - self.fx.predict(sev))[:, self.sel]))
        return r, r + self.fy.predict(sev)


def load():
    d = np.load(PB, allow_pickle=True)
    meta = pd.DataFrame(d["section_meta"].tolist(), columns=d["section_meta_cols"])
    keep = ~meta.meta_sample_id.isin(EXCLUDE_SECTIONS).to_numpy()
    meta, counts = meta[keep].reset_index(drop=True), d["section_counts"][keep]
    animals = meta.sample_name.drop_duplicates().to_numpy()
    Xa = lognorm(np.vstack([counts[(meta.sample_name == a).to_numpy()].sum(0) for a in animals]))
    traj = pd.read_csv(TRAJ).set_index("sample_name").loc[animals]
    info = traj[["model", "condition", "stage", "day_of_sacrifice", "score_sacrifice",
                 "disease_duration"]].copy()
    info["run_date"] = meta.drop_duplicates("sample_name").set_index("sample_name").loc[animals,
                                                                                      "run_date"].to_numpy()
    les = pd.read_csv(os.path.join(OUT, "lesions.csv"))
    z = np.load(os.path.join(OUT, "lesions.npz"), allow_pickle=True)
    ct = pd.DataFrame(z["celltype_counts"], columns=z["celltypes"].astype(str))
    les["immune_fraction"] = ct[[c for c in IMMUNE if c in ct]].sum(1) / ct.sum(1)
    return Xa, pd.Series(range(len(animals)), index=animals), info, les, z["counts"]


def score_cohort(Xa, aidx, info, les, Lc, model, cols=None, noise=True, seed=0):
    """LOAO fold clocks for one cohort; score each animal's lesions and its whole pseudobulk."""
    coh = info[(info.model == model) & info.day_of_sacrifice.notna() & info.score_sacrifice.notna()]
    animals = coh.index.to_numpy()
    X = Xa[aidx.loc[animals].to_numpy()]
    y, sev = coh.day_of_sacrifice.to_numpy(), coh.score_sacrifice.to_numpy().reshape(-1, 1)
    Ll = lognorm(Lc)
    rng = np.random.default_rng(seed)
    rows, whole = [], []
    for i, a in enumerate(animals):
        tr = np.ones(len(animals), bool); tr[i] = False
        clk = Clock(X[tr], y[tr], sev[tr], cols)
        wr, _ = clk.predict(X[i:i + 1], sev[i])
        whole.append({"animal": a, "pred_whole_resid": float(wr[0])})
        li = np.where(les.animal.to_numpy() == a)[0]
        if not len(li):
            continue
        r, dd = clk.predict(Ll[li], np.full(len(li), sev[i, 0]))
        rec = {"lesion_row": li, "animal": a, "pred_resid": r, "pred_day": dd}
        if noise:
            h1 = rng.binomial(Lc[li].astype(np.int64), 0.5)
            h2 = Lc[li].astype(np.int64) - h1
            p1, _ = clk.predict(lognorm(h1.astype(float)), np.full(len(li), sev[i, 0]))
            p2, _ = clk.predict(lognorm(h2.astype(float)), np.full(len(li), sev[i, 0]))
            rec["noise_sd"] = np.abs(p1 - p2) / 2.0       # half-sample SD/sqrt2 ~ full-lesion SD
        rows.append(rec)
    L = []
    for rec in rows:
        for k, li in enumerate(rec["lesion_row"]):
            L.append({"lesion_row": int(li), "animal": rec["animal"],
                      "pred_resid": float(rec["pred_resid"][k]), "pred_day": float(rec["pred_day"][k]),
                      "noise_sd": float(rec["noise_sd"][k]) if "noise_sd" in rec else np.nan})
    ref = pd.Series(loao_target(y, sev), index=animals)
    return pd.DataFrame(L), pd.DataFrame(whole).set_index("animal"), ref, coh


def rho_ci(x, y, seed=0):
    b = bootstrap_spearman(np.asarray(x, float), np.asarray(y, float), n_boot=N_BOOT, seed=seed)
    return float(b["rho"]), [float(b["ci_lo"]), float(b["ci_hi"])]


def per_animal(lp, coh, key="pred_day"):
    g = lp.groupby("animal")[key]
    s = pd.DataFrame({"n_lesions": g.size(), "mean": g.mean(), "sd": g.std(), "min": g.min(),
                      "max": g.max(), "iqr": g.quantile(0.75) - g.quantile(0.25)})
    s["range"] = s["max"] - s["min"]
    s["noise_sd_median"] = lp.groupby("animal").noise_sd.median()
    return s.join(coh[["day_of_sacrifice", "score_sacrifice", "stage", "condition", "disease_duration"]])


def cohort_spread_test(stats):
    """3c: spread (SD) ~ cohort + day, animals as units; bootstrap animals for the cohort term."""
    d = stats[(stats.condition == "EAE") & (stats.n_lesions >= MIN_LESIONS)].dropna(subset=["sd"])
    Xd = np.column_stack([np.ones(len(d)), (d.cohort == "rr").astype(float), d.day_of_sacrifice])
    beta = np.linalg.lstsq(Xd, d.sd.to_numpy(), rcond=None)[0]
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(d), len(d))
        if d.cohort.iloc[i].nunique() < 2:
            continue
        bs.append(np.linalg.lstsq(Xd[i], d.sd.to_numpy()[i], rcond=None)[0][1])
    lo, hi = d.day_of_sacrifice.groupby(d.cohort).agg(["min", "max"]).pipe(
        lambda t: (t["min"].max(), t["max"].min()))
    o = d[(d.day_of_sacrifice >= lo) & (d.day_of_sacrifice <= hi)]
    mw = mannwhitneyu(o[o.cohort == "rr"].sd, o[o.cohort == "chronic"].sd) if \
        o.cohort.nunique() == 2 and o.groupby("cohort").size().min() >= 3 else None
    return {"n_animals": {k: int(v) for k, v in d.cohort.value_counts().items()},
            "rr_minus_chronic_sd_adj_for_day": float(beta[1]),
            "ci": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "overlap_day_range": [float(lo), float(hi)],
            "overlap_n_animals": {k: int(v) for k, v in o.cohort.value_counts().items()},
            "overlap_mannwhitney_p": float(mw.pvalue) if mw else None}


def spread_vs_day(stats, cohort):
    d = stats[(stats.cohort == cohort) & (stats.condition == "EAE") &
              (stats.n_lesions >= MIN_LESIONS)].dropna(subset=["sd"])
    if len(d) < 5:
        return {"n_animals": int(len(d)), "skipped": True}
    r, ci = rho_ci(d.sd, d.day_of_sacrifice)
    rm, cim = rho_ci(d["mean"], d.day_of_sacrifice)
    rn = float(spearmanr(d.sd, d.n_lesions).statistic)
    return {"n_animals": int(len(d)), "rho_sd_day": r, "ci": ci, "rho_mean_day": rm, "ci_mean": cim,
            "rho_sd_nlesions": rn, "median_sd": float(d.sd.median()),
            "median_noise_sd": float(d.noise_sd_median.median())}


def run_all(Xa, aidx, info, les, Lc, cols=None, noise=True):
    lps, stats, wholes, refs = [], [], {}, {}
    for model, key in COHORTS.items():
        lp, whole, ref, coh = score_cohort(Xa, aidx, info, les, Lc, model, cols, noise)
        lp["cohort"] = key
        st = per_animal(lp, coh)
        st["cohort"] = key
        lps.append(lp); stats.append(st); wholes[key] = whole; refs[key] = ref
    return pd.concat(lps, ignore_index=True), pd.concat(stats), wholes, refs


def _null_one(Xa, aidx, info, les, Lc, size, seed):
    cols = np.random.default_rng(seed).choice(Xa.shape[1], size, replace=False)
    lp, stats, _, _ = run_all(Xa, aidx, info, les, Lc, cols, noise=False)
    return {k: spread_vs_day(stats, k).get("rho_sd_day", np.nan) for k in ("rr", "chronic")} | \
        {"cohort_diff": cohort_spread_test(stats)["rr_minus_chronic_sd_adj_for_day"]}


def main():
    Xa, aidx, info, les, Lc = load()
    res = {}
    # ---- 1a / 1b
    sec = pd.read_csv(os.path.join(OUT, "sections.csv"))
    les = les.join(info[["model", "condition", "stage", "day_of_sacrifice"]], on="animal")
    res["1a"] = {"n_lesions": int(len(les)), "n_sections": int(len(sec)),
                 "lesions_per_section": sec.n_lesions.describe().round(2).to_dict(),
                 "area_um2_quantiles": les.area_um2.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(0).to_dict(),
                 "lesions_in_control_animals": int((les.condition == "CONTROL").sum()),
                 "control_lesions_per_section": float(sec.merge(info, left_on="animal", right_index=True)
                                                      .query("condition == 'CONTROL'").n_lesions.mean())}
    cnt = sec.groupby("animal").agg(n_lesions=("n_lesions", "sum"), lesion_area=("lesion_area_um2", "sum"),
                                     tissue=("tissue_area_um2", "sum")).join(info)
    cnt["lesion_area_fraction"] = cnt.lesion_area / cnt.tissue
    b1 = {}
    for model, key in COHORTS.items():
        d = cnt[(cnt.model == model) & (cnt.condition == "EAE")]
        r, ci = rho_ci(d.n_lesions, d.day_of_sacrifice)
        r2, ci2 = rho_ci(d.lesion_area_fraction, d.day_of_sacrifice)
        b1[key] = {"n_animals": int(len(d)), "median_lesions": float(d.n_lesions.median()),
                   "rho_count_day": r, "ci": ci, "rho_areafrac_day": r2, "ci_areafrac": ci2}
    e = cnt[cnt.condition == "EAE"]
    b1["rr_vs_chronic_count_mannwhitney_p"] = float(mannwhitneyu(
        e[e.model == "RELAPSE REMITTING"].n_lesions, e[e.model == "CHRONIC"].n_lesions).pvalue)
    res["1b"] = b1
    cnt.to_csv(os.path.join(OUT, "lesions_per_animal.csv"))

    # ---- 2: score lesions with fold clocks
    lp, stats, wholes, refs = run_all(Xa, aidx, info, les, Lc)
    lp = lp.join(les[["area_um2", "n_cells", "cells_per_mm2", "immune_fraction", "pu1_fraction"]],
                 on="lesion_row")
    lp.to_csv(os.path.join(OUT, "lesion_predictions.csv"), index=False)
    v = {}
    for key in ("rr", "chronic"):
        m = lp[lp.cohort == key].groupby("animal").pred_resid.mean()
        ref, whole = refs[key].loc[m.index], wholes[key].loc[m.index, "pred_whole_resid"]
        r, ci = rho_ci(m, ref)
        rw, ciw = rho_ci(whole, ref)
        # 2d size-residualised (pooled within cohort)
        sub = lp[lp.cohort == key]
        la = np.log(sub.area_um2.to_numpy())
        res_pred = sub.pred_resid - np.polyval(np.polyfit(la, sub.pred_resid, 1), la)
        ms = res_pred.groupby(sub.animal).mean().loc[m.index]
        rs, cis = rho_ci(ms, ref)
        v[key] = {"n_animals_with_lesions": int(len(m)), "rho_lesion_mean": r, "ci": ci,
                  "rho_section_clock_same_animals": rw, "ci_section": ciw,
                  "rho_lesion_mean_size_residualised": rs, "ci_size_resid": cis,
                  "valid": bool(r >= 0.5 and ci[0] > 0)}
    res["2c"] = v
    # 2d size vs age, grouped by animal
    d2 = {}
    for key in ("rr", "chronic"):
        sub = lp[lp.cohort == key]
        per = sub.groupby("animal").filter(lambda g: len(g) >= 4).groupby("animal").apply(
            lambda g: spearmanr(np.log(g.area_um2), g.pred_day).statistic)
        cen = sub.assign(a=np.log(sub.area_um2) - sub.groupby("animal").area_um2.transform(
            lambda x: np.log(x).mean()), p=sub.pred_day - sub.groupby("animal").pred_day.transform("mean"))
        d2[key] = {"n_animals_ge4": int(per.notna().sum()), "median_within_animal_rho": float(per.median()),
                   "frac_positive": float((per > 0).mean()),
                   "pooled_animal_centred_rho": float(spearmanr(cen.a, cen.p).statistic)}
    res["2d"] = d2
    if not v["rr"]["valid"]:
        res["stopped"] = "lesion dating did not validate in RR (2c); steps 3-4 not interpretable"
        _write(res, stats)
        return

    # ---- 3
    stats.to_csv(os.path.join(OUT, "per_animal_lesion_age_distribution.csv"))
    res["3b"] = {k: spread_vs_day(stats, k) for k in ("rr", "chronic")}
    res["3c"] = cohort_spread_test(stats)
    # ---- 4b immune fraction / density vs predicted age, within animal
    c4 = {}
    for key in ("rr", "chronic"):
        sub = lp[lp.cohort == key].copy()
        for col in ("immune_fraction", "cells_per_mm2"):
            cen_x = sub[col] - sub.groupby("animal")[col].transform("mean")
            cen_p = sub.pred_day - sub.groupby("animal").pred_day.transform("mean")
            c4[f"{key}_{col}_within_animal_rho"] = float(spearmanr(cen_x, cen_p).statistic)
    res["4b"] = c4
    strong = {k: v for k, v in c4.items() if abs(v) > 0.5}
    if strong:
        adj = lp.copy()
        for key in ("rr", "chronic"):
            m = adj.cohort == key
            Z = np.column_stack([np.ones(m.sum()), adj.loc[m, "immune_fraction"], np.log(adj.loc[m, "cells_per_mm2"] + 1)])
            adj.loc[m, "pred_day"] = adj.loc[m, "pred_day"] - Z[:, 1:] @ np.linalg.lstsq(Z, adj.loc[m, "pred_day"], rcond=None)[0][1:]
        st_adj = pd.concat([per_animal(adj[adj.cohort == k], stats[stats.cohort == k][
            ["day_of_sacrifice", "score_sacrifice", "stage", "condition", "disease_duration"]]).assign(cohort=k)
                            for k in ("rr", "chronic")])
        res["4b_adjusted"] = {"3b": {k: spread_vs_day(st_adj, k) for k in ("rr", "chronic")},
                              "3c": cohort_spread_test(st_adj), "triggered_by": strong}
    # ---- 4a random gene-set null (size = non-zero genes of the RR all-animal clock)
    coh = info[(info.model == "RELAPSE REMITTING") & info.day_of_sacrifice.notna()]
    full = Clock(Xa[aidx.loc[coh.index]], coh.day_of_sacrifice.to_numpy(),
                 coh.score_sacrifice.to_numpy().reshape(-1, 1))
    size = full.nonzero
    nulls = Parallel(n_jobs=N_JOBS)(delayed(_null_one)(Xa, aidx, info, les, Lc, size, 100 + i)
                                    for i in range(N_RANDOM))
    nd = pd.DataFrame(nulls)
    obs = {"rr": res["3b"]["rr"].get("rho_sd_day"), "chronic": res["3b"]["chronic"].get("rho_sd_day"),
           "cohort_diff": res["3c"]["rr_minus_chronic_sd_adj_for_day"]}
    res["4a"] = {"gene_set_size": size, "n_sets": N_RANDOM,
                 **{k: {"observed": obs[k], "null_mean": float(nd[k].mean()),
                        "null_q05": float(nd[k].quantile(0.05)), "null_q95": float(nd[k].quantile(0.95)),
                        "percentile": float((nd[k] < obs[k]).mean() * 100) if obs[k] is not None else None}
                    for k in obs}}
    _write(res, stats)
    _plot(stats)


def _write(res, stats):
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    print(json.dumps(res, indent=1, default=str))


def _plot(stats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
    COL = {"rr": "#2a78d6", "chronic": "#eb6834"}
    d = stats[(stats.condition == "EAE") & (stats.n_lesions >= MIN_LESIONS)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor(SURFACE)
    for k, c in COL.items():
        s = d[d.cohort == k]
        axes[0].scatter(s.day_of_sacrifice, s.sd, s=20 + 4 * s.n_lesions, color=c, alpha=0.8,
                        edgecolor=SURFACE, label=f"{'RR' if k == 'rr' else 'chronic'} (n={len(s)} animals)")
        axes[0].scatter(s.day_of_sacrifice, s.noise_sd_median, s=12, marker="x", color=c, alpha=0.6)
        axes[1].errorbar(s.day_of_sacrifice, s["mean"], yerr=[s["mean"] - s["min"], s["max"] - s["mean"]],
                         fmt="o", color=c, alpha=0.7, ms=4, lw=1)
    axes[0].set_xlabel("day of sacrifice", fontsize=8, color=MUTED)
    axes[0].set_ylabel("within-animal SD of lesion age (days)\n× = median measurement-noise SD", fontsize=8, color=MUTED)
    axes[0].legend(fontsize=7, frameon=False)
    axes[0].set_title("3b: lesion-age spread vs disease time", loc="left", fontsize=9, color=INK)
    axes[1].set_xlabel("day of sacrifice", fontsize=8, color=MUTED)
    axes[1].set_ylabel("lesion age (days): mean, min-max", fontsize=8, color=MUTED)
    axes[1].set_title("lesion-age distribution per animal", loc="left", fontsize=9, color=INK)
    for a in axes:
        a.set_facecolor(SURFACE)
        for sp_ in ("top", "right"):
            a.spines[sp_].set_visible(False)
        a.tick_params(colors=MUTED, labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures", "lesion_age_spread.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
