"""TRAJECTORY FEATURES — per-animal disease-course features from the daily score/weight sheets.

Every target used so far is a terminal scalar (`score_sacrifice`, `day_of_sacrifice`). The
daily clinical-score and weight spreadsheets hold the full course for every animal; this
turns them into a tidy per-animal table (WORKORDER Task 0, prerequisite for tasks 4 and 6).

Sheets are SECTION-level (one row per tissue section, curve repeated per section). They are
collapsed to one curve per animal; any disagreement between an animal's sections is reported,
never averaged. The atlas (h5ad obs only, read with h5py — no 14GB load) is the reference for
animal identity, `day_of_sacrifice` and `score_sacrifice`.

Known join facts handled here:
  * alias: atlas `C_L_1..5` == sheet `C_M30_1..5` (verified by section ID below);
  * `C_CFA_5` is in the sheets but not the atlas;
  * `C_M16_2` / section `C2_G3_Mid_1`: its sheet curve and Animal ID are those of `C_M16_1`.
    Resolved with an expression sex check (Xist) — see `section_sex_check()`.

Feature definitions (days are days post-induction, d0..d50 columns):
  onset_day            first day of a run of score >= 0.5 lasting >= 2 consecutive days; a run
                       cut off by sacrifice counts but is flagged `onset_censored`
  disease_duration     day_of_sacrifice - onset_day (NaN if never symptomatic)
  peak_score/peak_day  max score to sacrifice / first day it occurred
  cumulative_score     trapezoidal AUC of the score curve, d0 -> sacrifice
  cumulative_since_onset   same, onset_day -> sacrifice
  n_relapses_r1        drop >= 1 from a peak, then rise >= 1 from the trough (first-pass rule)
  n_relapses_strict    drop >= 0.5 then rise >= 0.5, each sustained >= 2 consecutive days
  days_since_last_peak sacrifice - first day of the last local max (prominence >= 0.5; the
                       sacrifice day itself can be that max)
  slope_at_sacrifice   OLS slope of score over [sacrifice-2, sacrifice] (>= 2 points)
  max_weight_loss      min weight to sacrifice / baseline (mean of d0-d2); 1.0 = no loss
  score_coverage       fraction of days d0..sacrifice with a recorded score in the sheet

If the sacrifice-day score is missing from the sheet, the atlas `score_sacrifice` is added as
that day's point (flagged `sacrifice_point_imputed`).

    PYTHONPATH="$PWD" python analysis/trajectory_features.py
"""

from __future__ import annotations

import immunotransformer  # noqa: F401  (OpenMP guard — must precede numpy; see HANDOFF.md)

import json
import os

import h5py
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import spearmanr

ATLAS = os.environ.get(
    "RRMAP2_H5AD",
    os.path.expanduser(
        "~/Downloads/RRMAP2_xenium_all_samples.cellcharter.companion.ready.with_metadata."
        "rerun.with_AnnoL1Curated_with_Region_Anno2to4Updated.h5ad"),
)
SCORE_XLSX = "spreadsheets/Fixed_RRMap2_FinalSamples_AllScore_curated_20260723_212152.xlsx"
WEIGHT_XLSX = "spreadsheets/Fixed_RRMap2_FinalSamples_AllWeight_curated_20260723_212152.xlsx"
OUT = "runs/trajectory_features"

ALIAS = {f"C_M30_{i}": f"C_L_{i}" for i in range(1, 6)}  # sheet name -> atlas name
ONSET_THR = 0.5
PEAK_PROMINENCE = 0.5
XIST_FEMALE_THR = 0.10  # fraction of Xist+ cells; F sections >= 0.14, M <= 0.019 observed

# expected relapse count implied by the stage label (number of attacks after the first)
STAGE_RELAPSES = {
    "ONSET1": 0, "PEAK1": 0, "REMISSION1": 0, "MONOPHASIC": 0,
    "ONSET2": 1, "PEAK2": 1, "PEAK2_MILD": 1, "REMISSION2": 1, "REMISSION2_LONG": 1,
    "PEAK3": 2,
}

FEATURES = ["onset_day", "disease_duration", "peak_score", "peak_day", "cumulative_score",
            "cumulative_since_onset", "n_relapses_r1", "n_relapses_strict",
            "days_since_last_peak", "slope_at_sacrifice", "max_weight_loss", "score_coverage"]
TERMINAL = ["score_sacrifice", "day_of_sacrifice"]


# ---------------------------------------------------------------- atlas (h5py, obs only)

def _obs_col(obs, k):
    g = obs[k]
    if isinstance(g, h5py.Group):
        cats = [c.decode() if isinstance(c, bytes) else c for c in g["categories"][:]]
        return pd.Categorical.from_codes(g["codes"][:], cats)
    return g[:]


def load_atlas_sections(f):
    obs = f["obs"]
    cols = ["sample_name", "meta_sample_id", "stage", "condition", "model", "region", "sex",
            "day_of_sacrifice", "score_sacrifice"]
    cols = [c for c in cols if c in obs]
    df = pd.DataFrame({k: _obs_col(obs, k) for k in cols})
    return df.drop_duplicates("meta_sample_id").reset_index(drop=True)


def section_sex_check(f):
    """Fraction of Xist+ cells per section (layers/counts, row-sliced CSR — no full load)."""
    obs = f["obs/meta_sample_id"]
    cats = [c.decode() if isinstance(c, bytes) else c for c in obs["categories"][:]]
    codes = obs["codes"][:]
    var = f["var"]
    names = [x.decode() if isinstance(x, bytes) else x for x in var[var.attrs["_index"]][:]]
    xi = names.index("Xist")
    C = f["layers/counts"]
    indptr = C["indptr"][:]
    rows_out = []
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(cats) + 1))
    for ci, sec in enumerate(cats):
        rows = np.sort(order[bounds[ci]:bounds[ci + 1]])
        if not len(rows):
            continue
        pos = 0
        for blk in np.split(rows, np.where(np.diff(rows) != 1)[0] + 1):
            a, e = indptr[blk[0]], indptr[blk[-1] + 1]
            pos += int((C["indices"][a:e] == xi).sum())
        rows_out.append((sec, len(rows), pos / len(rows)))
    d = pd.DataFrame(rows_out, columns=["meta_sample_id", "n_cells", "frac_xist_pos"])
    d["sex_from_xist"] = np.where(d.frac_xist_pos >= XIST_FEMALE_THR, "F", "M")
    return d


# ---------------------------------------------------------------- sheets

def load_sheets():
    s = pd.read_excel(SCORE_XLSX)
    w = pd.read_excel(WEIGHT_XLSX)
    days = [c for c in s.columns if str(c).startswith("d") and str(c)[1:].isdigit()]
    assert (s.sample_id.values == w.sample_id.values).all()
    for d in (s, w):
        d["meta_sample_id"] = d.sample_id.str.replace("-", "_", regex=False)
        d["atlas_name"] = d.sample_name.map(lambda n: ALIAS.get(n, n))
    return s, w, days


def section_disagreements(s, days):
    """Per animal with >1 section: which sections deviate from the modal curve, and by how much."""
    out = []
    for name, g in s.groupby("sample_name"):
        if len(g) < 2:
            continue
        X = g[days].to_numpy(float)
        keys = [tuple(np.nan_to_num(x, nan=-9)) for x in X]
        modal = max(set(keys), key=keys.count)
        ref = np.array(modal); ref[ref == -9] = np.nan
        for (_, r), x, k in zip(g.iterrows(), X, keys):
            if k == modal:
                continue
            both = ~np.isnan(x) & ~np.isnan(ref)
            diff = np.abs(x - ref)
            dd = [days[i] for i in np.where(both & (diff > 0))[0]]
            out.append({
                "sample_name": name, "section": r.sample_id, "region": r.region,
                "animal_id": r["Animal ID"], "modal_animal_id": int(g["Animal ID"].mode()[0]),
                "n_days_differ": len(dd), "days_differ": dd,
                "max_abs_diff": float(np.nanmax(np.where(both, diff, np.nan))),
                "sum_abs_diff": float(np.nansum(np.where(both, diff, 0))),
                "sheet_day_of_sacrifice": str(r.day_of_sacrifice),
                "sheet_score_sacrifice": str(r.score_sacrifice),
            })
    return pd.DataFrame(out)


def identical_curve_owner(s, days, section):
    """Other animals whose curve is identical to `section`'s on all co-recorded days (>= 20)."""
    x = s.loc[s.sample_id == section, days].to_numpy(float)[0]
    hits = []
    for name, g in s[s.sample_id != section].groupby("sample_name"):
        y = g[days].to_numpy(float)[0]
        m = ~np.isnan(x) & ~np.isnan(y)
        if m.sum() >= 20 and np.all(x[m] == y[m]):
            hits.append((name, int(g["Animal ID"].iloc[0]), int(m.sum())))
    return hits


# ---------------------------------------------------------------- features

def build_curve(xrow, dos, score_sac):
    """Observed (t, score) points on d0..dos; append (dos, score_sac) if the sheet lacks it."""
    t = np.arange(len(xrow))
    m = ~np.isnan(xrow) & (t <= dos)
    tt, yy = t[m].astype(float), xrow[m].astype(float)
    imputed = False
    if dos not in tt and np.isfinite(score_sac):
        tt, yy = np.append(tt, dos), np.append(yy, score_sac)
        imputed = True
    return tt, yy, imputed


def onset(tt, yy, dos):
    above = yy >= ONSET_THR
    for i in np.where(above)[0]:
        if i + 1 < len(tt) and above[i + 1] and tt[i + 1] == tt[i] + 1:
            return tt[i], False
        if tt[i] == dos:  # first supra-threshold day is the sacrifice day
            return tt[i], True
        # a single-day blip followed by a gap/drop does not count
    return np.nan, False


def relapses_r1(yy, drop=1.0, rise=1.0):
    peak, trough, falling, n = -np.inf, np.inf, False, 0
    for y in yy:
        if not falling:
            peak = max(peak, y)
            if y <= peak - drop:
                falling, trough = True, y
        else:
            trough = min(trough, y)
            if y >= trough + rise:
                n += 1
                falling, peak = False, y
    return n


def relapses_strict(tt, yy, drop=0.5, rise=0.5, sustain=2):
    """Hysteresis as r1 but each crossing must hold on `sustain` consecutive calendar days."""
    peak, trough, falling, n, run = -np.inf, np.inf, False, 0, 0
    prev_t = None
    for t, y in zip(tt, yy):
        if prev_t is not None and t != prev_t + 1:
            run = 0
        prev_t = t
        if not falling:
            if y > peak:
                peak, run = y, 0
                continue
            run = run + 1 if y <= peak - drop else 0
            if run >= sustain:
                falling, trough, run = True, y, 0
        else:
            if y < trough:
                trough, run = y, 0
                continue
            run = run + 1 if y >= trough + rise else 0
            if run >= sustain:
                n += 1
                falling, peak, run = False, y, 0
    return n


def last_peak_day(tt, yy):
    if yy.max() <= 0:
        return np.nan
    # pad with zeros so a maximum at either end can qualify; plateau -> its first day
    pk, props = find_peaks(np.r_[0, yy, 0], prominence=PEAK_PROMINENCE, plateau_size=1)
    if not len(pk):
        return np.nan
    return tt[props["left_edges"][-1] - 1]


def slope_final(tt, yy, dos):
    m = tt >= dos - 2
    if m.sum() < 2:
        return np.nan
    return float(np.polyfit(tt[m], yy[m], 1)[0])


def animal_features(xs, xw, dos, score_sac):
    tt, yy, imputed = build_curve(xs, dos, score_sac)
    on, censored = onset(tt, yy, dos)
    f = {"sacrifice_point_imputed": imputed, "onset_censored": censored, "onset_day": on,
         "first_symptom_day": float(tt[yy > 0][0]) if (yy > 0).any() else np.nan}
    f["disease_duration"] = dos - on if np.isfinite(on) else np.nan
    f["peak_score"] = float(yy.max())
    f["peak_day"] = float(tt[np.argmax(yy)])
    f["cumulative_score"] = float(np.trapz(yy, tt))
    if np.isfinite(on):
        m = tt >= on
        f["cumulative_since_onset"] = float(np.trapz(yy[m], tt[m]))
    else:
        f["cumulative_since_onset"] = np.nan
    f["n_relapses_r1"] = relapses_r1(yy)
    f["n_relapses_strict"] = relapses_strict(tt, yy)
    lp = last_peak_day(tt, yy)
    f["days_since_last_peak"] = dos - lp if np.isfinite(lp) else np.nan
    sl = slope_final(tt, yy, dos)
    f["slope_at_sacrifice"] = sl
    f["slope_class"] = ("" if not np.isfinite(sl) else
                        "ascending" if sl > 0 else "descending" if sl < 0 else "plateau")
    t = np.arange(len(xw))
    wm = ~np.isnan(xw) & (t <= dos)
    base = np.nanmean(xw[:3])
    f["weight_baseline"] = float(base)
    f["max_weight_loss"] = float(np.nanmin(xw[wm]) / base) if wm.any() else np.nan
    f["score_coverage"] = float((~np.isnan(xs[:dos + 1])).sum() / (dos + 1))
    return f


# ---------------------------------------------------------------- main

def main():
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    s, w, days = load_sheets()

    with h5py.File(ATLAS, "r") as f:
        atlas = load_atlas_sections(f)
        sex_path = os.path.join(OUT, "section_sex_check.csv")
        if os.path.exists(sex_path):
            sex = pd.read_csv(sex_path)
        else:
            sex = section_sex_check(f)
            sex.to_csv(sex_path, index=False)
    atlas_animals = atlas.groupby("sample_name", observed=True).first()

    # ---- 1. join
    sheet_names = set(s.sample_name)
    atlas_names = set(atlas.sample_name.astype(str))
    direct = sorted(sheet_names & atlas_names)
    aliased = sorted(n for n in sheet_names if n in ALIAS and ALIAS[n] in atlas_names)
    sheet_only = sorted(sheet_names - atlas_names - set(aliased))
    atlas_only = sorted(atlas_names - sheet_names - set(ALIAS.values()))
    alias_check = []
    for sn, an in ALIAS.items():
        secs_sheet = set(s.loc[s.sample_name == sn, "meta_sample_id"])
        secs_atlas = set(atlas.loc[atlas.sample_name == an, "meta_sample_id"])
        alias_check.append({"sheet": sn, "atlas": an, "sheet_sections": sorted(secs_sheet),
                            "atlas_sections": sorted(secs_atlas),
                            "verified": secs_sheet == secs_atlas})

    # ---- 4. section disagreement + sex check
    dis = section_disagreements(s, days)
    sex = sex.merge(atlas[["meta_sample_id", "sample_name", "sex"]], on="meta_sample_id",
                    how="left").rename(columns={"sex": "sex_meta"})
    sex_mismatch = sex[sex.sex_from_xist != sex.sex_meta]
    owners = {r.section: identical_curve_owner(s, days, r.section) for r in dis.itertuples()}
    # resolution: an animal's curve = its sections carrying the modal Animal ID
    modal_id = s.groupby("sample_name")["Animal ID"].agg(lambda x: x.mode()[0])
    s_ok = s[s["Animal ID"] == s.sample_name.map(modal_id)]
    w_ok = w.loc[s_ok.index]

    # ---- 2. features
    rows = []
    for an in sorted(set(s_ok.atlas_name) & atlas_names):
        gs = s_ok[s_ok.atlas_name == an]
        gw = w_ok.loc[gs.index]
        a = atlas_animals.loc[an]
        dos, ssac = float(a.day_of_sacrifice), float(a.score_sacrifice)
        sheet_dos = pd.to_numeric(gs.day_of_sacrifice, errors="coerce").iloc[0]
        sheet_ssac = pd.to_numeric(gs.score_sacrifice, errors="coerce").iloc[0]
        r = {"sample_name": an, "sheet_name": gs.sample_name.iloc[0],
             "animal_id": int(gs["Animal ID"].iloc[0]), "model": a.model,
             "condition": a.condition, "stage": a.stage, "sex": a.sex,
             "day_of_sacrifice": dos, "score_sacrifice": ssac,
             "sheet_dos_mismatch": not (sheet_dos == dos),
             "sheet_score_mismatch": not (sheet_ssac == ssac)}
        r.update(animal_features(gs[days].to_numpy(float)[0], gw[days].to_numpy(float)[0],
                                 int(dos), ssac))
        r["expected_relapses_stage"] = STAGE_RELAPSES.get(a.stage, 0 if a.model == "CHRONIC"
                                                          else np.nan)
        rows.append(r)
    T = pd.DataFrame(rows)
    T["section_relabel_flag"] = T.sample_name.isin(dis.sample_name) | T.sheet_name.isin(
        [o[0] for v in owners.values() for o in v])
    T.to_csv(os.path.join(OUT, "animal_trajectory.csv"), index=False)

    # ---- 3. relapse rules vs stage
    rr = T[T.model != "CHRONIC"].dropna(subset=["expected_relapses_stage"])
    relapse = {}
    for col in ("n_relapses_r1", "n_relapses_strict"):
        relapse[col] = {
            "rr_agree_with_stage": int((rr[col] == rr.expected_relapses_stage).sum()),
            "rr_n": int(len(rr)),
            "rr_disagree": rr.loc[rr[col] != rr.expected_relapses_stage,
                                  ["sample_name", "stage", col]].values.tolist(),
            "peak3": T.loc[T.stage == "PEAK3", ["sample_name", col]].values.tolist(),
            "chronic_nonzero": T.loc[(T.model == "CHRONIC") & (T[col] > 0),
                                     ["sample_name", "stage", col]].values.tolist(),
            "crosstab": pd.crosstab(T.stage, T[col]).to_dict(orient="index"),
        }
    rule_agree = int((T.n_relapses_r1 == T.n_relapses_strict).sum())
    rule_disagree = T.loc[T.n_relapses_r1 != T.n_relapses_strict,
                          ["sample_name", "stage", "n_relapses_r1", "n_relapses_strict"]]

    # ---- 5. correlations per cohort (EAE animals only; controls have flat curves)
    corr, unexpl = {}, {}
    feats = [c for c in FEATURES] + TERMINAL
    for model, g in T[T.condition == "EAE"].groupby("model"):
        C = g[feats].astype(float).corr(method="spearman")
        corr[model] = C
        C.to_csv(os.path.join(OUT, f"corr_spearman_{model.replace(' ', '_')}.csv"))
        u = {}
        for ft in FEATURES:
            d = g[[ft, "score_sacrifice", "day_of_sacrifice"]].astype(float).dropna()
            if len(d) < 5 or d[ft].std() == 0:
                continue
            u[ft] = {"n": len(d),
                     "r2_score": _r2(d[ft], d[["score_sacrifice"]]),
                     "r2_score_day": _r2(d[ft], d[["score_sacrifice", "day_of_sacrifice"]]),
                     "rho_score": float(spearmanr(d[ft], d.score_sacrifice)[0])}
        unexpl[model] = u

    _plot_curves(s_ok, days, T)
    _plot_corr(corr)

    results = {
        "join": {"n_atlas": len(atlas_names), "n_sheet": len(sheet_names),
                 "direct": len(direct), "aliased": len(aliased),
                 "joined": len(T), "sheet_only": sheet_only, "atlas_only": atlas_only,
                 "alias_check": alias_check},
        "section_disagreements": dis.to_dict(orient="records"),
        "identical_curve_owner": owners,
        "sex_mismatch_sections": sex_mismatch.to_dict(orient="records"),
        "sex_check_range": {k: [float(v.frac_xist_pos.min()), float(v.frac_xist_pos.max())]
                            for k, v in sex[~sex.meta_sample_id.isin(
                                sex_mismatch.meta_sample_id)].groupby("sex_meta",
                                                                       observed=True)},
        "sheet_vs_atlas_mismatch": T.loc[T.sheet_dos_mismatch | T.sheet_score_mismatch,
                                         ["sample_name", "sheet_dos_mismatch",
                                          "sheet_score_mismatch"]].values.tolist(),
        "n_sacrifice_point_imputed": int(T.sacrifice_point_imputed.sum()),
        "imputed_animals": T.loc[T.sacrifice_point_imputed, "sample_name"].tolist(),
        "onset_censored": T.loc[T.onset_censored, ["sample_name", "stage"]].values.tolist(),
        "never_symptomatic_eae": T.loc[(T.condition == "EAE") & T.onset_day.isna(),
                                       ["sample_name", "stage"]].values.tolist(),
        "below_onset_threshold": T.loc[(T.peak_score > 0) & T.onset_day.isna(),
                                       ["sample_name", "stage", "peak_score"]].values.tolist(),
        "stage_curve_contradiction": T.loc[
            (T.expected_relapses_stage >= 1) & (T.first_symptom_day >= T.day_of_sacrifice - 2),
            ["sample_name", "stage", "first_symptom_day", "day_of_sacrifice"]].values.tolist(),
        "onset_range": T[T.condition == "EAE"].groupby("model").onset_day.agg(
            ["min", "max", "std"]).to_dict(orient="index"),
        "slope_class_counts": T[T.condition == "EAE"].groupby(
            ["model", "slope_class"]).size().unstack(fill_value=0).to_dict(orient="index"),
        "relapse": relapse,
        "relapse_rule_agreement": [rule_agree, len(T)],
        "relapse_rule_disagree": rule_disagree.values.tolist(),
        "unexplained_by_score_sacrifice": unexpl,
    }
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=_json_default)
    _report(results, T, corr)


def _r2(y, X):
    X = np.c_[np.ones(len(X)), X.to_numpy(float)]
    beta, *_ = np.linalg.lstsq(X, y.to_numpy(float), rcond=None)
    res = y.to_numpy(float) - X @ beta
    return float(1 - res.var() / y.to_numpy(float).var())


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


# ---------------------------------------------------------------- figures

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # relapse count 0..3 (fixed order)
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(axis="y", color="#e6e5e0", lw=0.6)
    ax.set_axisbelow(True)


def _plot_curves(s_ok, days, T):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    T = T[T.condition == "EAE"]
    order = (T.groupby(["model", "stage"]).day_of_sacrifice.median()
             .reset_index().sort_values(["model", "day_of_sacrifice"]))
    panels = list(zip(order.model, order.stage))
    nc = 5
    nr = int(np.ceil(len(panels) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(nc * 3.0, nr * 2.3), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (model, stage) in zip(axes.flat, panels):
        _style(ax)
        for _, r in T[(T.model == model) & (T.stage == stage)].iterrows():
            x = s_ok.loc[s_ok.sample_name == r.sheet_name, days].to_numpy(float)[0]
            tt, yy, _ = build_curve(x, int(r.day_of_sacrifice), r.score_sacrifice)
            c = SERIES[min(int(r.n_relapses_r1), 3)]
            ax.plot(tt, yy, color=c, lw=1.5, alpha=0.9)
            if np.isfinite(r.onset_day):
                ax.plot(r.onset_day, ONSET_THR, "o", ms=4, color=c, mec=SURFACE, mew=1)
        ax.set_title(f"{'RR' if model != 'CHRONIC' else 'Chr'} · {stage}", fontsize=8,
                     color=INK, loc="left")
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("day post-induction", fontsize=7, color=MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel("clinical score", fontsize=7, color=MUTED)
    fig.legend([Line2D([], [], color=c, lw=2) for c in SERIES],
               ["0 relapses", "1", "2", "3+"], title="r1 rule (dot = onset)",
               loc="upper right", fontsize=7, title_fontsize=7, frameon=False, ncol=4)
    fig.suptitle("Daily clinical score to sacrifice, EAE animals by stage", x=0.01, ha="left",
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, "figures", "score_curves_by_stage.png"), dpi=160)
    plt.close(fig)


def _plot_corr(corr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    fig, axes = plt.subplots(1, len(corr), figsize=(8.2 * len(corr), 6.8),
                             gridspec_kw={"wspace": 0.55})
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_1d(axes)
    for ax, (model, C) in zip(axes, corr.items()):
        im = ax.imshow(C.values, cmap=cmap, vmin=-1, vmax=1)
        ax.set_xticks(range(len(C)), C.columns, rotation=60, ha="right", fontsize=7, color=INK)
        ax.set_yticks(range(len(C)), C.index, fontsize=7, color=INK)
        for i in range(len(C)):
            for j in range(len(C)):
                v = C.values[i, j]
                if np.isfinite(v) and i != j and abs(v) >= 0.5:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5,
                            color=SURFACE if abs(v) > 0.75 else INK)
        n = len(C)
        ax.axhline(n - 2.5, color=INK, lw=0.8)
        ax.axvline(n - 2.5, color=INK, lw=0.8)
        ax.set_title(f"{model} (EAE animals) — Spearman ρ", fontsize=9, color=INK, loc="left")
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.colorbar(im, ax=axes, shrink=0.6, label="ρ")
    fig.savefig(os.path.join(OUT, "figures", "feature_correlations.png"), dpi=160,
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- report

def _report(r, T, corr):
    j = r["join"]
    L = ["TRAJECTORY FEATURES — per-animal course features from the daily score/weight sheets",
         "=" * 88, "",
         "1. JOIN (sheet -> atlas on sample_name)",
         f"  atlas animals {j['n_atlas']}, sheet animals {j['n_sheet']}",
         f"  joined {j['joined']}/{j['n_atlas']} atlas animals "
         f"({j['direct']} direct + {j['aliased']} aliased C_M30_i -> C_L_i)",
         f"  alias verified by section ID: "
         f"{sum(a['verified'] for a in j['alias_check'])}/{len(j['alias_check'])}",
         f"  sheet-only (not in atlas): {j['sheet_only']}",
         f"  atlas-only (no curve): {j['atlas_only'] or 'none'}",
         f"  sheet vs atlas day/score mismatches: {r['sheet_vs_atlas_mismatch'] or 'none'}",
         f"  sacrifice-day score missing in sheet, atlas score_sacrifice used as last point: "
         f"{r['n_sacrifice_point_imputed']} animals {r['imputed_animals']}", ""]

    L += ["4. SECTION DISAGREEMENT (curves are animal-level; a section that differs is wrong)"]
    for d in r["section_disagreements"]:
        L.append(f"  {d['sample_name']} section {d['section']} (region {d['region']}): differs "
                 f"from the animal's other sections on {d['n_days_differ']} days "
                 f"{d['days_differ']}, max |diff| {d['max_abs_diff']:.2f}, sum |diff| "
                 f"{d['sum_abs_diff']:.2f}")
        L.append(f"    Animal ID on that row {d['animal_id']} vs animal's {d['modal_animal_id']}; "
                 f"sheet sacrifice day/score '{d['sheet_day_of_sacrifice']}' / "
                 f"'{d['sheet_score_sacrifice']}'")
        for o in r["identical_curve_owner"].get(d["section"], []):
            L.append(f"    its curve is IDENTICAL to {o[0]} (Animal ID {o[1]}) on all {o[2]} "
                     "co-recorded days")
    rng = r["sex_check_range"]
    L.append(f"  Xist sex check, all sections: F sections {rng.get('F')} frac Xist+, "
             f"M sections {rng.get('M')}")
    for m in r["sex_mismatch_sections"]:
        L.append(f"  SEX MISMATCH: section {m['meta_sample_id']} labelled {m['sample_name']} "
                 f"({m['sex_meta']}) has {m['frac_xist_pos']:.4f} Xist+ cells -> "
                 f"{m['sex_from_xist']}")
    L += ["  STATUS: RESOLVED for the trajectory table. Section C2_G3_Mid_1 is tissue from a",
          "  MALE animal carrying C_M16_1's Animal ID and C_M16_1's exact score curve; C_M16_2 is",
          "  female. Its sheet curve is correct for the tissue; its sample_name label (C_M16_2) is",
          "  wrong in BOTH the sheet and the atlas. C_M16_2's curve here uses its two",
          "  agreeing sections (Animal ID 1058093); C_M16_1 is unchanged. NOT fixed in the atlas:",
          "  any animal-level pseudobulk built so far pools one male C_M16_1 section into",
          "  C_M16_2 (and C_M16_2 is the only animal with two sections from one region).", ""]

    L += ["2. FEATURES  ->  animal_trajectory.csv",
          f"  {len(T)} animals; onset = first >= {ONSET_THR} run lasting >= 2 days",
          f"  EAE animals never reaching onset: {r['never_symptomatic_eae']}",
          f"    of which symptomatic but below the {ONSET_THR} threshold (sacrificed at first "
          f"sign, score 0.25 -> disease_duration NaN): {r['below_onset_threshold']}",
          f"  onset_day range per cohort: {r['onset_range']}",
          "  STAGE vs CURVE CONTRADICTION (stage implies a prior attack, curve is 0 until the",
          f"  last 3 days): {r['stage_curve_contradiction'] or 'none'}",
          "    -> these 'second onset' animals show only a FIRST attack in the daily sheet;",
          "       either the stage label or the curve is wrong. Unresolved.",
          f"  onset censored at sacrifice (single supra-threshold day = sacrifice day): "
          f"{r['onset_censored'] or 'none'}",
          f"  slope_at_sacrifice classes (EAE): {r['slope_class_counts']}", ""]
    eae = T[T.condition == "EAE"]
    L.append("  per-cohort medians (EAE animals):")
    L.append("  " + eae.groupby("model")[FEATURES + TERMINAL].median().round(2)
             .T.to_string().replace("\n", "\n  "))
    L.append("")

    L += ["3. RELAPSE COUNT — TWO RULES, NO DEFINITION CHOSEN",
          "  The work order says the first-pass rule is 'already noted in docs/'; it is not —",
          "  no doc defines it. Both rules are implemented as specified in WORKORDER.md:",
          "    r1     : drop >= 1 from a peak, then rise >= 1 from the trough",
          "    strict : drop >= 0.5 then rise >= 0.5, each sustained >= 2 consecutive days",
          "  Expected count from stage label (RR): ONSET1/PEAK1/REM1/MONO=0; ONSET2/PEAK2/"
          "PEAK2_MILD/REM2/REM2_LONG=1; PEAK3=2. Chronic expected 0."]
    for col, v in r["relapse"].items():
        L.append(f"  {col}: agrees with stage label in {v['rr_agree_with_stage']}/{v['rr_n']} RR "
                 "animals")
        L.append(f"    PEAK3 animals: {v['peak3']}")
        L.append(f"    RR disagreements: {v['rr_disagree']}")
        L.append(f"    chronic animals with >= 1 relapse: {v['chronic_nonzero']}")
    a, n = r["relapse_rule_agreement"]
    L += ["  r1 misses RR_P3_3's third attack: it rises 1.5 -> 2.75 at d37 straight off a",
          "  plateau with no remission before it, so no drop is seen under either rule.",
          "  MONOPHASIC animals: strict counts a late low-grade rise (trough ~0.25 -> 0.75-1.0)",
          "  as a relapse in all 4; r1 does not. That is the main disagreement between rules."]
    L.append(f"  r1 vs strict agree on {a}/{n} animals; disagree on: {r['relapse_rule_disagree']}")
    L += ["  -> A relapse definition must be agreed with the person who scored the animals",
          "     before n_relapses is used as a target or covariate.", ""]

    L += ["5. HOW MUCH IS NEW INFORMATION (EAE animals, per cohort)",
          "  R2 = variance explained by score_sacrifice alone (and + day_of_sacrifice);",
          "  1-R2 = what the trajectory feature adds beyond terminal state.",
          f"  {'feature':24s} {'cohort':18s} {'n':>3s} {'R2|score':>9s} {'R2|score+day':>13s} "
          f"{'rho(score)':>10s}"]
    for model, u in r["unexplained_by_score_sacrifice"].items():
        for ft, v in u.items():
            L.append(f"  {ft:24s} {model:18s} {v['n']:3d} {v['r2_score']:9.2f} "
                     f"{v['r2_score_day']:13.2f} {v['rho_score']:10.2f}")
    L.append("")
    for model, C in corr.items():
        L.append(f"  Spearman rho, {model} (EAE): features vs terminal scalars")
        L.append("  " + C.loc[FEATURES, TERMINAL].round(2).to_string().replace("\n", "\n  "))
        L.append("")

    L.append("HEADLINE — cumulative_score beyond score_sacrifice:")
    for model, u in r["unexplained_by_score_sacrifice"].items():
        v = u.get("cumulative_score")
        if v:
            L.append(f"  {model}: score_sacrifice explains {v['r2_score']:.0%} of cumulative_score "
                     f"variance, leaving {1 - v['r2_score']:.0%} unexplained "
                     f"({1 - v['r2_score_day']:.0%} after also adding day_of_sacrifice).")
    rr = r["unexplained_by_score_sacrifice"].get("RELAPSE REMITTING", {})
    ch = r["unexplained_by_score_sacrifice"].get("CHRONIC", {})
    L += ["  CAVEAT FOR TASKS 4 AND 6: most of that unexplained variance is day_of_sacrifice.",
          "  cumulative_score is close to a restatement of time-since-induction (score + day",
          f"  explain {rr.get('cumulative_score', {}).get('r2_score_day', np.nan):.0%} RR, "
          f"{ch.get('cumulative_score', {}).get('r2_score_day', np.nan):.0%} chronic). Likewise "
          f"disease_duration: score + day explain "
          f"{rr.get('disease_duration', {}).get('r2_score_day', np.nan):.0%} RR / "
          f"{ch.get('disease_duration', {}).get('r2_score_day', np.nan):.0%} chronic,",
          "  because onset latency varies little relative to the sacrifice-day spread",
          "  (see onset_day range above). The trajectory reframing removes terminal severity,",
          "  but a model that already predicts day_of_sacrifice will largely predict these",
          "  targets too. Genuinely new information beyond (score, day) is mainly in",
          "  onset_day, days_since_last_peak, slope_at_sacrifice, peak_day (RR) and",
          "  max_weight_loss (RR) — the shape-of-course features, not the accumulation ones.",
          ""]
    L.append("figures/: score_curves_by_stage.png, feature_correlations.png")

    report = "\n".join(L)
    print(report)
    with open(os.path.join(OUT, "report.txt"), "w") as fh:
        fh.write(report + "\n")


if __name__ == "__main__":
    main()
