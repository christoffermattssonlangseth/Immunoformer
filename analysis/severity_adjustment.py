"""Is the duration signal created by the severity adjustment? (follow-up to closing-block Task A)

Endothelial Col4a1/Col4a2 rise with severity and fall with duration at matched severity.
Severity is partly downstream of duration in EAE, so residualising it may adjust for a
mediator. Same ground rules as the closing block (cohorts separate, animal unit, LOAO, in-fold
transforms, 2000 animal bootstraps, paired bootstraps, zero-feature folds, no Global_niche).

  1  unadjusted block: endothelial Col4a1 / Col4a2 / mean vs day and vs severity
  2  stratify instead of adjusting: severity tertiles and fixed-width bands (score < 1, 1-2,
     >= 2); within-band rho, n, CI; combined estimate (inverse-variance Fisher z, bootstrap
     CI resampling animals within bands); severity-matched pairs (|dscore| <= PAIR_TOL):
     fraction where the longer-duration animal has lower Col4, sign test (pairs share
     animals, so the test is optimistic) plus an animal-bootstrap CI on the fraction
  3  severity variables (terminal score, cumulative score, peak score, days since onset):
     pairwise rho, rho with day; Col4 result adjusted for each in turn
  4  project-wide: clock with no severity adjustment, terminal (current), cumulative; each rho
     vs its own target; paired differences; selected-gene overlap (full-data fits)

    PYTHONPATH="$PWD:$PWD/scripts" python analysis/severity_adjustment.py
"""

from __future__ import annotations

import immunotransformer  # noqa: F401  (OpenMP guard — must precede numpy; see HANDOFF.md)

import itertools
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata, spearmanr, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clock_composition as CC  # noqa: E402
from closing_block import lognorm, partial_rank  # noqa: E402
from duration_clock import final_coefs, loao_target  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
OUT = "runs/closing_block/severity_adjustment"
COHORTS = {"RELAPSE REMITTING": "RR", "CHRONIC": "chronic"}
N_BOOT = 2000
PAIR_TOL = 0.25
FIXED_BANDS = [(-0.01, 0.99, "score < 1"), (0.99, 1.99, "1 <= score < 2"), (1.99, 9, "score >= 2")]
SEVERITY_VARS = {"terminal score": "score_sacrifice", "cumulative score": "cumulative_score",
                 "peak score": "peak_score", "days since onset": "disease_duration"}


def sp_(x, y):
    return float(spearmanr(x, y).statistic)


def boot_ci(f, *arrs):
    rng = np.random.default_rng(0)
    n = len(arrs[0])
    v = [f(*(a[i] for a in arrs)) for i in (rng.integers(0, n, n) for _ in range(N_BOOT))]
    v = np.array([x for x in v if np.isfinite(x)])
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]


def combined_z(groups):
    """Inverse-variance (n-3) weighted Fisher z across bands with n >= 4."""
    zs, ws = [], []
    for x, y in groups:
        if len(x) >= 4 and np.std(x) > 0 and np.std(y) > 0:
            r = np.clip(spearmanr(x, y).statistic, -0.999, 0.999)
            zs.append(np.arctanh(r)); ws.append(len(x) - 3)
    return float(np.tanh(np.average(zs, weights=ws))) if zs else np.nan


def stratified(col4, day, score, bands):
    out, groups = [], []
    for lo, hi, lab in bands:
        m = (score > lo) & (score <= hi)
        x, y = col4[m], day[m]
        row = {"band": lab, "n": int(m.sum())}
        if m.sum() >= 4 and np.std(x) > 0 and np.std(y) > 0:
            row["rho"] = sp_(x, y)
            row["ci"] = boot_ci(sp_, x, y)
            row["informative"] = bool(m.sum() >= 8)
        else:
            row["rho"] = None
            row["informative"] = False
        out.append(row)
        groups.append((x, y))
    comb = combined_z(groups)
    rng = np.random.default_rng(1)
    bs = []
    idx = [np.where((score > lo) & (score <= hi))[0] for lo, hi, _ in bands]
    for _ in range(N_BOOT):
        g = []
        for ii in idx:
            if len(ii):
                s = rng.choice(ii, len(ii))
                g.append((col4[s], day[s]))
        bs.append(combined_z(g))
    bs = np.array([b for b in bs if np.isfinite(b)])
    return {"bands": out, "combined_rho": comb,
            "combined_ci": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]}


def matched_pairs(col4, day, score, animals):
    pairs = [(i, j) for i, j in itertools.combinations(range(len(day)), 2)
             if abs(score[i] - score[j]) <= PAIR_TOL and day[i] != day[j]]
    if not pairs:
        return {"n_pairs": 0}
    hits = [(col4[i] < col4[j]) if day[i] > day[j] else (col4[j] < col4[i]) for i, j in pairs]
    ties = sum(col4[i] == col4[j] for i, j in pairs)
    k, n = int(sum(hits)), len(pairs)
    rng = np.random.default_rng(2)
    fr = []
    for _ in range(N_BOOT):
        s = rng.choice(len(day), len(day))
        pp = [(a, b) for a, b in itertools.combinations(s, 2)
              if a != b and abs(score[a] - score[b]) <= PAIR_TOL and day[a] != day[b]]
        if pp:
            fr.append(np.mean([(col4[a] < col4[b]) if day[a] > day[b] else (col4[b] < col4[a]) for a, b in pp]))
    used = sorted({animals[i] for p in pairs for i in p})
    return {"tolerance": PAIR_TOL, "n_pairs": n, "n_animals_in_pairs": len(used), "ties": int(ties),
            "frac_longer_duration_lower": k / n, "sign_test_p": float(binomtest(k, n, 0.5).pvalue),
            "animal_bootstrap_ci_frac": [float(np.quantile(fr, 0.025)), float(np.quantile(fr, 0.975))],
            "note": "pairs share animals; the sign test treats them as independent and is optimistic"}


def clock_variant(X, y, cov):
    pred, nnz = CC.loao_audit(X, y, cov)
    ref = loao_target(y, cov) if cov is not None else y
    return pred, ref, nnz


def paired2(pa, ra, pb, rb):
    """Each rho against its own target; bootstrap both on the same animal resamples."""
    rng = np.random.default_rng(0)
    n = len(pa)
    d = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        d.append(sp_(pa[i], ra[i]) - sp_(pb[i], rb[i]))
    d = np.array([x for x in d if np.isfinite(x)])
    return {"rho_A": sp_(pa, ra), "rho_B": sp_(pb, rb), "diff_mean": float(d.mean()),
            "diff_ci": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
            "frac_diff_le_0": float((d <= 0).mean())}


def main():
    os.makedirs(OUT, exist_ok=True)
    pb, info, meta = CC.load_rr()
    genes = pb.columns.to_numpy()
    traj = pd.read_csv(CC.TRAJ).set_index("sample_name")
    counts, n_cells, scats, ccats = CC.celltype_pseudobulk(None)
    sidx = {s: i for i, s in enumerate(scats)}
    j = list(ccats).index("Endothelial")
    res = {}
    for model, ck in COHORTS.items():
        coh = info[(info.model == model) & info.day.notna() & info.score.notna()]
        a_ = list(coh.index)
        E = pd.DataFrame(lognorm(np.vstack([counts[[sidx[s] for s in meta.loc[meta.sample_name == a, "meta_sample_id"]], j].sum(0)
                                            for a in a_])), index=a_, columns=genes)
        col4 = {"Col4a1": E["Col4a1"].to_numpy(), "Col4a2": E["Col4a2"].to_numpy(),
                "Col4a1+Col4a2 mean": ((E["Col4a1"] + E["Col4a2"]) / 2).to_numpy()}
        day, score = coh.day.to_numpy(), coh.score.to_numpy()
        R = {"n": len(a_)}
        # 1 unadjusted
        R["1_unadjusted"] = {g: {"rho_day": sp_(v, day), "ci_day": boot_ci(sp_, v, day),
                                 "rho_severity": sp_(v, score), "ci_severity": boot_ci(sp_, v, score)}
                             for g, v in col4.items()}
        R["1_day_vs_severity"] = {"rho": sp_(day, score), "ci": boot_ci(sp_, day, score)}
        # 2 stratified (on the combined mean)
        m = col4["Col4a1+Col4a2 mean"]
        q = np.quantile(rankdata(score, method="average"), [1 / 3, 2 / 3])
        rk = rankdata(score, method="average")
        tert = [(-np.inf, q[0], "tertile 1 (lowest)"), (q[0], q[1], "tertile 2"), (q[1], np.inf, "tertile 3 (highest)")]
        R["2_tertiles"] = stratified(m, day, rk, tert)
        R["2_tertiles"]["score_ranges"] = [[float(score[(rk > lo) & (rk <= hi)].min()), float(score[(rk > lo) & (rk <= hi)].max())]
                                          if ((rk > lo) & (rk <= hi)).any() else None for lo, hi, _ in tert]
        R["2_fixed_bands"] = stratified(m, day, score, FIXED_BANDS)
        R["2_matched_pairs"] = matched_pairs(m, day, score, a_)
        # 3 severity definitions
        sv = traj.loc[a_, list(SEVERITY_VARS.values())].astype(float)
        sv["day_of_sacrifice"] = day
        R["3_severity_correlations"] = sv.corr(method="spearman").round(3).to_dict()
        R["3_n_nonmissing"] = sv.notna().sum().to_dict()
        adj = {}
        for lab, col in SEVERITY_VARS.items():
            ok = sv[col].notna().to_numpy()
            x, y, z = m[ok], day[ok], sv[col].to_numpy()[ok]
            adj[lab] = {"n": int(ok.sum()), "rho_day_given": partial_rank(x, y, z),
                        "ci": boot_ci(lambda a, b, c: partial_rank(a, b, c), x, y, z)}
        R["3_col4_adjusted_by_each"] = adj
        # 4 project-wide clock sensitivity
        X = pb.loc[a_].to_numpy()
        cum = traj.loc[a_, "cumulative_score"].to_numpy(float)
        ok = np.isfinite(cum)
        variants = {"none": None, "terminal": score.reshape(-1, 1), "cumulative": cum.reshape(-1, 1)}
        fits = {}
        for k_, cov in variants.items():
            Xo, yo = X[ok], day[ok]
            co = cov[ok] if cov is not None else None
            p, r, nnz = clock_variant(Xo, yo, co)
            cf, _ = final_coefs(Xo, yo, genes, covar=co)
            fits[k_] = {"pred": p, "ref": r, "zero": int((nnz == 0).sum()), "genes": set(cf.gene)}
        R["4_clock"] = {k_: {"rho": sp_(f["pred"], f["ref"]), "ci": boot_ci(sp_, f["pred"], f["ref"]),
                             "zero_feature_folds": f["zero"], "n": int(ok.sum()), "n_nonzero_genes": len(f["genes"])}
                        for k_, f in fits.items()}
        for k_ in ("none", "cumulative"):
            R["4_clock"][f"{k_}_vs_terminal"] = paired2(fits[k_]["pred"], fits[k_]["ref"],
                                                        fits["terminal"]["pred"], fits["terminal"]["ref"])
        g = {k_: f["genes"] for k_, f in fits.items()}
        R["4_gene_overlap"] = {f"{a}&{b}": len(g[a] & g[b]) for a, b in itertools.combinations(g, 2)}
        R["4_gene_overlap"]["all_three"] = len(g["none"] & g["terminal"] & g["cumulative"])
        res[ck] = R
        print(f"[{ck}] done", flush=True)
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    print("[done]")


if __name__ == "__main__":
    main()
