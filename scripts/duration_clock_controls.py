"""Adversarial confound controls for the duration clock: SEX and REGION.

Two covariates could alias day_of_sacrifice within RR:
  - SEX: clock B includes Xist (X-inactivation marker); cohort is mixed-sex. Panel has
    Xist but no Y-genes, so sex is called from Xist bimodality (high=female, low=male).
  - REGION: EAE has a spatial cord gradient (L/T/C). Lesions resolve L->T->C, so rostral
    (C) samples taken later carry more accrued signal -> region partially aligns with both
    severity and duration. We must show the clock predicts day BEYOND region.

Tests:
  1. CONFOUND TESTS: is each covariate associated with day? (sex: Mann-Whitney /
     point-biserial; region: Kruskal + ordinal L<T<C Spearman).
  2. FULLY DE-CONFOUNDED REFIT: clock B with covar = [score, sex, region one-hot],
     Xist dropped from features, LOAO Spearman + R2 + permutation null. If it survives,
     the duration axis is not a sex/region/severity artifact in any combination.

    PYTHONPATH="$PWD:$PWD/scripts" python scripts/duration_clock_controls.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu, pointbiserialr, kruskal

from immunotransformer.train import resolve_device  # noqa: F401  (OpenMP guard)
import duration_clock as dc

OUT = dc.OUT
REGION_ORDER = {"L": 0, "T": 1, "C": 2}  # caudal -> rostral; rostral resolves later


def call_sex(pb, genes):
    xi = pb[:, np.where(genes == "Xist")[0][0]].astype(float)
    s = np.sort(xi)
    thr = (s[np.argmax(np.diff(s))] + s[np.argmax(np.diff(s)) + 1]) / 2
    return xi, (xi >= thr).astype(float), float(thr)


def main():
    pb, genes, animals, stage, score, day, meta = dc.load()
    n = len(day)
    xi, sex, thr = call_sex(pb, genes)
    region = meta["region"].astype(str).to_numpy()

    # --- confound test: sex vs day ---
    nf, nm = int(sex.sum()), int((1 - sex).sum())
    mw = mannwhitneyu(day[sex == 1], day[sex == 0])
    pbi = pointbiserialr(sex, day)
    print(f"[sex] female={nf} male={nm} | MW day~sex p={mw.pvalue:.3f} | "
          f"point-biserial r={pbi.statistic:+.3f} p={pbi.pvalue:.3f}", flush=True)

    # --- confound test: region vs day ---
    groups = [day[region == r] for r in np.unique(region)]
    kw = kruskal(*groups)
    ro = np.array([REGION_ORDER[r] for r in region], float)
    sp_rd = spearmanr(ro, day)
    sp_rs = spearmanr(ro, score)
    reg_day = {r: float(np.median(day[region == r])) for r in np.unique(region)}
    reg_score = {r: float(np.mean(score[region == r])) for r in np.unique(region)}
    print(f"[region] Kruskal day~region p={kw.pvalue:.3f} | "
          f"ordinal L<T<C Spearman(region,day)={sp_rd.statistic:+.3f} p={sp_rd.pvalue:.3f} | "
          f"(region,score)={sp_rs.statistic:+.3f}", flush=True)
    print(f"[region] median day {reg_day} | mean score {reg_score}", flush=True)

    # --- fully de-confounded refit: covar = [score, sex, region one-hot] ---
    reg_oh = pd.get_dummies(pd.Series(region), drop_first=True).to_numpy(float)
    covar = np.column_stack([score, sex, reg_oh])
    keep = genes != "Xist"
    pb2, genes2 = pb[:, keep], genes[keep]
    print(f"[refit] clock B'' covar=[score,sex,region({reg_oh.shape[1]}dummies)], "
          f"Xist dropped ({pb2.shape[1]} feats) ...", flush=True)
    predB = dc.loao_clock(pb2, day, covar=covar)
    dayB = dc.loao_target(day, covar)
    rhoB = spearmanr(predB, dayB).statistic
    r2B = 1 - np.sum((predB - dayB) ** 2) / np.sum((dayB - dayB.mean()) ** 2)
    print(f"[refit] fully-controlled clock Spearman={rhoB:+.3f} R2={r2B:+.3f}", flush=True)

    rng = np.random.default_rng(0)
    perm = np.empty(dc.N_PERM)
    for k in range(dc.N_PERM):
        yp = day[rng.permutation(n)]
        pk = dc.loao_clock(pb2, yp, covar=covar)
        perm[k] = spearmanr(pk, dc.loao_target(yp, covar)).statistic
        if (k + 1) % 10 == 0:
            print(f"[refit-perm] {k+1}/{dc.N_PERM}", flush=True)
    p_perm = (1 + np.sum(perm >= rhoB)) / (1 + dc.N_PERM)
    print(f"[refit] fully-controlled clock perm p={p_perm:.4f}", flush=True)

    res = {
        "n": n,
        "sex": {"n_female_xisthi": nf, "n_male_xistlo": nm, "xist_split_thr": round(thr, 4),
                "mannwhitney_day_sex_p": round(float(mw.pvalue), 4),
                "pointbiserial_r": round(float(pbi.statistic), 4),
                "pointbiserial_p": round(float(pbi.pvalue), 4),
                "spearman_xistraw_day": round(float(spearmanr(xi, day).statistic), 4)},
        "region": {"median_day": reg_day, "mean_score": reg_score,
                   "kruskal_day_region_p": round(float(kw.pvalue), 4),
                   "ordinal_LTC_spearman_day": round(float(sp_rd.statistic), 4),
                   "ordinal_LTC_spearman_day_p": round(float(sp_rd.pvalue), 4),
                   "ordinal_LTC_spearman_score": round(float(sp_rs.statistic), 4)},
        "fully_controlled_clock": {
            "covariates": "score + sex + region(one-hot)", "xist_dropped": True,
            "spearman": round(float(rhoB), 3), "r2_loao": round(float(r2B), 3),
            "perm_p": round(float(p_perm), 4),
            "original_clockB_spearman": 0.799},
        "verdict": ("Non-significant confound tests + a fully-controlled clock that retains "
                    "high LOAO Spearman with significant perm p => the duration axis is not "
                    "an artifact of sex, region, or severity in any combination."),
    }
    with open(os.path.join(OUT, "controls.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\n[done] -> {OUT}/controls.json", flush=True)


if __name__ == "__main__":
    main()
