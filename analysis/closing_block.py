"""Closing block — Task A (collagen IV standalone), Task B (non-lesion tissue control),
Task C (composition-only clock). Both cohorts, never pooled; animal = unit; LOAO;
day_of_sacrifice severity-residualised inside each training fold; ElasticNetCV duration-clock
pipeline unchanged; 2000-resample animal bootstraps; paired-difference bootstraps for every
comparison on the same animals; zero-feature folds in every fit. No Global_niche anywhere.

Coefficient-selection status of each feature set is stated in the results:
  A  Col4a1/Col4a2 — SELECTED from the fitted clock's coefficients (A4 is the null for that).
     A5 basement-membrane genes — annotation-defined (membership in the clock lists reported).
  B  full panel — not selected.
  C  cell-type proportions — not selected.

    PYTHONPATH="$PWD:$PWD/scripts" python analysis/closing_block.py
"""

from __future__ import annotations

import immunotransformer  # noqa: F401  (OpenMP guard — must precede numpy; see HANDOFF.md)

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clock_composition as CC  # noqa: E402
from duration_clock import ENET_KW, _topvar, final_coefs, loao_target  # noqa: E402
from lesion_clock import Clock  # noqa: E402
from paired_difference import paired  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
OUT = "runs/closing_block"
DATA = os.path.join(OUT, "data.npz")
COHORTS = {"RELAPSE REMITTING": "RR", "CHRONIC": "chronic"}
N_BOOT = 2000
MIN_CELLS_TYPE = 50
MIN_CELLS_COMPARTMENT = 200
BM_GENES = ["Col4a1", "Col4a2", "Lama4", "Lamb1", "Lamc1", "Nid1", "Nid2", "Hspg2", "Col4a3", "Col4a5"]
NON_CELLTYPES_L2 = {"ARTIFACT", "Doublet", "T_B_doublet", "Mixed", "B cell_LowQuality", "T cell_LowQuality"}
NON_CELLTYPES_L1 = {"Doublet", "T_B_doublet"}   # labels that are not cell types: kept in the
                                                 # denominator, excluded as features
ONLY_C = bool(os.environ.get("ONLY_C"))


def lognorm(c):
    return np.log1p(c / np.maximum(c.sum(-1, keepdims=True), 1) * 1e4)


def partial_rank(x, y, Z):
    rx, ry = rankdata(x), rankdata(y)
    A = np.column_stack([np.ones(len(rx))] + [rankdata(z) for z in np.atleast_2d(Z.T)])
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1]) if ex.std() > 0 and ey.std() > 0 else np.nan


def ci(f, *arrs):
    rng = np.random.default_rng(0)
    n = len(arrs[0])
    v = [f(*(a[i] for a in arrs)) for i in (rng.integers(0, n, n) for _ in range(N_BOOT))]
    v = np.array([x for x in v if np.isfinite(x)])
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]


def sp_(x, y):
    return float(spearmanr(x, y).statistic)


def stat_block(x, day, sev, extra=None):
    out = {"n": int(len(x)),
           "rho_day_raw": sp_(x, day), "ci_day_raw": ci(sp_, x, day),
           "rho_day_given_severity": partial_rank(x, day, sev),
           "ci_day_given_severity": ci(lambda a, b, c: partial_rank(a, b, c), x, day, sev),
           "rho_severity_raw": sp_(x, sev), "ci_severity_raw": ci(sp_, x, sev),
           "rho_severity_given_day": partial_rank(x, sev, day),
           "ci_severity_given_day": ci(lambda a, b, c: partial_rank(a, b, c), x, sev, day)}
    if extra is not None:
        Z = np.column_stack([sev, extra])
        out["rho_day_given_severity_and_depth"] = partial_rank(x, day, Z)
        out["ci_day_given_severity_and_depth"] = ci(lambda a, b, c: partial_rank(a, b, c), x, day, Z)
    return out


def loao_keep_audit(Xg, Xk, y, cov):
    """Clock with the top-variance filter on genes only; Xk columns always kept."""
    n = len(y)
    pred, nnz = np.full(n, np.nan), np.zeros(n, int)
    X = np.hstack([Xg, Xk]); ng = Xg.shape[1]
    for i in range(n):
        tr = np.ones(n, bool); tr[i] = False
        fx = LinearRegression().fit(cov[tr], X[tr]); fy = LinearRegression().fit(cov[tr], y[tr])
        Xtr, Xte = X[tr] - fx.predict(cov[tr]), X[i:i + 1] - fx.predict(cov[i:i + 1])
        sel = np.r_[_topvar(Xtr[:, :ng]), np.arange(ng, X.shape[1])]
        sc = StandardScaler().fit(Xtr[:, sel])
        m = ElasticNetCV(random_state=0, **ENET_KW).fit(sc.transform(Xtr[:, sel]), y[tr] - fy.predict(cov[tr]))
        pred[i] = m.predict(sc.transform(Xte[:, sel]))[0]; nnz[i] = int((m.coef_ != 0).sum())
    return pred, nnz


def loao_comp_resid(X, y, sev, comp):
    """C4: residualise genes (and day) on severity + in-fold PCA(80%) of CLR proportions."""
    n = len(y)
    clr = np.log(comp + 1e-4); clr = clr - clr.mean(1, keepdims=True)
    pred, nnz, ks = np.full(n, np.nan), np.zeros(n, int), []
    for i in range(n):
        tr = np.ones(n, bool); tr[i] = False
        sc = StandardScaler().fit(clr[tr]); p = PCA(random_state=0).fit(sc.transform(clr[tr]))
        k = int(np.searchsorted(np.cumsum(p.explained_variance_ratio_), 0.80) + 1); ks.append(k)
        cov = np.column_stack([sev, p.transform(sc.transform(clr))[:, :k]])
        fx = LinearRegression().fit(cov[tr], X[tr]); fy = LinearRegression().fit(cov[tr], y[tr])
        Xtr, Xte = X[tr] - fx.predict(cov[tr]), X[i:i + 1] - fx.predict(cov[i:i + 1])
        sel = _topvar(Xtr); s2 = StandardScaler().fit(Xtr[:, sel])
        m = ElasticNetCV(random_state=0, **ENET_KW).fit(s2.transform(Xtr[:, sel]), y[tr] - fy.predict(cov[tr]))
        pred[i] = m.predict(s2.transform(Xte[:, sel]))[0]; nnz[i] = int((m.coef_ != 0).sum())
    return pred, nnz, ks


def rho_pack(pred, ref, nnz):
    from immunotransformer.stats import bootstrap_spearman
    b = bootstrap_spearman(pred, ref, n_boot=N_BOOT, seed=0)
    return {"rho": float(b["rho"]), "ci": [float(b["ci_lo"]), float(b["ci_hi"])], "n": int(len(ref)),
            "zero_feature_folds": int((np.asarray(nnz) == 0).sum())}


def main():
    os.makedirs(OUT, exist_ok=True)
    pb, info, meta = CC.load_rr()
    genes = pb.columns.to_numpy()
    gi = {g: i for i, g in enumerate(genes)}
    d = np.load(DATA, allow_pickle=True)
    counts, n_cells, scats, ccats = CC.celltype_pseudobulk(None)
    sidx = {s: i for i, s in enumerate(scats)}
    cti = {c: i for i, c in enumerate(ccats)}
    pub = pd.read_csv(CC.OLD_CLOCK)
    refit = pd.read_csv("runs/clock_composition/clock_genes_baseline_refit.csv")
    coef_union = set(pub.gene) | set(refit.gene)
    cells = pd.read_parquet(os.path.join(OUT, "cells_qc.parquet"))
    res = {"coefficient_selected": {"A_Col4a1_Col4a2": True, "A5_genes_in_clock_lists": sorted(set(BM_GENES) & coef_union),
                                    "B": False, "C": False}}

    if ONLY_C:
        res = json.load(open(os.path.join(OUT, "results.json")))
    # =============================== TASK A
    A = {}
    pos_frac = pd.DataFrame(d["pos"] / np.maximum(d["n1"][:, None], 1), index=d["c1cats"].astype(str),
                            columns=d["genes"].astype(str))
    for model, ck in ({} if ONLY_C else COHORTS).items():
        coh = info[(info.model == model) & info.day.notna() & info.score.notna()]
        for ct in ("Endothelial", "VSMC"):
            j = cti[ct]
            per_c, per_n = {}, {}
            for a in coh.index:
                si = [sidx[s] for s in meta.loc[meta.sample_name == a, "meta_sample_id"]]
                per_c[a], per_n[a] = counts[si, j].sum(0), int(n_cells[si, j].sum())
            ncell = pd.Series(per_n)
            key = f"{ck}/{ct}"
            A[key] = {"cells_per_animal": {"median": float(ncell.median()), "min": int(ncell.min()),
                                           "max": int(ncell.max())},
                      "animals_below_50": ncell[ncell < MIN_CELLS_TYPE].to_dict()}
            Xall = pd.DataFrame(lognorm(np.vstack([per_c[a] for a in coh.index])), index=coh.index, columns=genes)
            qc = cells[(cells.celltype == ct) & cells.animal.isin(coh.index)].groupby("animal").agg(
                med_counts=("total_counts", "median"), med_genes=("n_genes", "median")).loc[coh.index]
            for variant, keep in (("excl_lt50", ncell[ncell >= MIN_CELLS_TYPE].index), ("all", coh.index)):
                X = Xall.loc[keep]
                day, sev = coh.loc[keep, "day"].to_numpy(), coh.loc[keep, "score"].to_numpy()
                vals = {"Col4a1": X["Col4a1"].to_numpy(), "Col4a2": X["Col4a2"].to_numpy(),
                        "Col4a1+Col4a2 mean": ((X["Col4a1"] + X["Col4a2"]) / 2).to_numpy()}
                # A6 technical
                tech = {"n_cells": ncell.loc[keep].to_numpy(float), "median_total_counts": qc.loc[keep, "med_counts"].to_numpy(),
                        "median_genes": qc.loc[keep, "med_genes"].to_numpy()}
                A6 = {k: {"rho_day": sp_(v, day), "ci": ci(sp_, v, day)} for k, v in tech.items()}
                trend = any(v["ci"][0] > 0 or v["ci"][1] < 0 for v in A6.values())
                A[key][variant] = {"n": int(len(keep)),
                                   "A2_A3": {g: stat_block(v, day, sev, tech["median_total_counts"] if trend else None)
                                             for g, v in vals.items()},
                                   "A6": A6, "A6_any_technical_trend": trend}
                if variant == "excl_lt50":
                    # A4 null: every gene expressed in >10% of this cell type's cells
                    expressed = [g for g in genes if pos_frac.loc[ct, g] > 0.10]
                    r_par = np.array([partial_rank(X[g].to_numpy(), day, sev) for g in expressed])
                    r_raw = np.array([sp_(X[g].to_numpy(), day) for g in expressed])
                    A4 = {"n_genes_expressed_gt10pct": len(expressed),
                          "statistic": "|rho(gene, day | severity)| across all expressed genes; raw reported too"}
                    for g in ("Col4a1", "Col4a2"):
                        if g not in expressed:
                            A4[g] = "not expressed in >10% of cells"; continue
                        k = expressed.index(g)
                        A4[g] = {"rho_partial": float(r_par[k]),
                                 "percentile_abs_partial": float((np.abs(r_par) < abs(r_par[k])).mean() * 100),
                                 "empirical_p_partial": float((np.abs(r_par) >= abs(r_par[k])).mean()),
                                 "rho_raw": float(r_raw[k]),
                                 "percentile_abs_raw": float((np.abs(r_raw) < abs(r_raw[k])).mean() * 100),
                                 "empirical_p_raw": float((np.abs(r_raw) >= abs(r_raw[k])).mean())}
                    A[key]["A4"] = A4
                    if ct == "Endothelial":
                        A[key]["A5"] = {g: {"frac_cells_pos": float(pos_frac.loc[ct, g]),
                                            "in_clock_coef_lists": g in coef_union,
                                            **stat_block(X[g].to_numpy(), day, sev)}
                                        for g in BM_GENES if g in X}
            print(f"[A] {key} done", flush=True)
    if not ONLY_C:
        res["A"] = A

    # =============================== TASK B
    B = {}
    comp, comp_n = d["comp"], d["comp_n"]
    csecs, cani = d["sections"].astype(str), d["section_animal"].astype(str)
    cnames = list(d["compartments"].astype(str))
    for model, ck in ({} if ONLY_C else COHORTS).items():
        coh = info[(info.model == model) & info.day.notna() & info.score.notna()]
        animals = list(coh.index)
        Xa = pb.loc[animals].to_numpy()
        y, sev = coh.day.to_numpy(), coh.score.to_numpy().reshape(-1, 1)
        ref = pd.Series(loao_target(y, sev), index=animals)
        preds = {c: {} for c in cnames}
        for i, a in enumerate(animals):
            tr = np.ones(len(animals), bool); tr[i] = False
            clk = Clock(Xa[tr], y[tr], sev[tr])                    # fold clock: never saw animal a
            for c in cnames:
                ci_ = cnames.index(c)
                ss = [k for k in np.where(cani == a)[0] if comp_n[k, ci_] >= MIN_CELLS_COMPARTMENT]
                if ss:
                    r, _ = clk.predict(lognorm(comp[ss, ci_].astype(float)), np.full(len(ss), sev[i, 0]))
                    preds[c][a] = float(np.mean(r))
        cell_tab = {c: pd.Series({a: int(comp_n[cani == a, cnames.index(c)].sum()) for a in animals}) for c in cnames}
        Bc = {"cells_per_animal": {c: {"median": float(v.median()), "min": int(v.min())} for c, v in cell_tab.items()},
              "animals_usable": {c: len(v) for c, v in preds.items()}}
        for c in cnames:
            p = pd.Series(preds[c])
            Bc[c] = rho_pack(p.to_numpy(), ref.loc[p.index].to_numpy(), [1] * len(p))
            Bc[c]["zero_feature_folds"] = "n/a (frozen fold clocks; see whole-tissue clock audit 0/n)"
        both = sorted(set(preds["lesion"]) & set(preds["nonlesion_m50"]))
        Bc["B3_nonlesion50_vs_lesion"] = paired("non-lesion (50 um) vs lesion", np.array([preds["nonlesion_m50"][a] for a in both]),
                                                np.array([preds["lesion"][a] for a in both]), ref.loc[both].to_numpy(),
                                                "non-lesion m50", "lesion")
        for m in ("nonlesion_m0", "nonlesion_m150"):
            bb = sorted(set(preds[m]) & set(preds["nonlesion_m50"]))
            Bc[f"B5_{m}_vs_m50"] = paired(f"{m} vs nonlesion_m50", np.array([preds[m][a] for a in bb]),
                                          np.array([preds["nonlesion_m50"][a] for a in bb]), ref.loc[bb].to_numpy(), m, "nonlesion_m50")
        # B4 de novo clock on non-lesion (50 um) animal pseudobulk
        nl = cnames.index("nonlesion_m50")
        usable = [a for a in animals if comp_n[cani == a, nl].sum() >= MIN_CELLS_COMPARTMENT]
        Xn = lognorm(np.vstack([comp[cani == a, nl].sum(0) for a in usable]).astype(float))
        yy, ss2 = coh.loc[usable, "day"].to_numpy(), coh.loc[usable, "score"].to_numpy().reshape(-1, 1)
        pn, nz = CC.loao_audit(Xn, yy, ss2)
        Bc["B4_denovo_nonlesion"] = rho_pack(pn, loao_target(yy, ss2), nz)
        cf, _ = final_coefs(Xn, yy, genes, covar=ss2)
        sh = cf.merge(pub, on="gene", suffixes=("_new", "_pub"))
        Bc["B4_denovo_nonlesion"].update({"n_nonzero": int(len(cf)), "overlap_with_published_135": int(len(sh)),
                                          "same_sign": int((np.sign(sh.coef_new) == np.sign(sh.coef_pub)).sum()),
                                          "top10": cf.head(10).round(3).to_dict("records")})
        B[ck] = Bc
        print(f"[B] {ck} done", flush=True)
    if not ONLY_C:
        res["B"] = B

    # =============================== TASK C
    C = {}
    l2, l2cats = d["l2"], d["c2cats"].astype(str)
    for model, ck in COHORTS.items():
        coh = info[(info.model == model) & info.day.notna() & info.score.notna()]
        animals = list(coh.index)
        y, sev = coh.day.to_numpy(), coh.score.to_numpy().reshape(-1, 1)
        ref = loao_target(y, sev)
        P1 = pd.DataFrame({ct: [n_cells[[sidx[s] for s in meta.loc[meta.sample_name == a, "meta_sample_id"]], j].sum()
                                for a in animals] for ct, j in cti.items()}, index=animals)
        P1 = P1.div(P1.sum(1), axis=0)[[c for c in cti if c not in NON_CELLTYPES_L1]]
        P2 = pd.DataFrame(np.vstack([l2[cani == a].sum(0) for a in animals]), index=animals, columns=l2cats)
        P2 = P2.div(P2.sum(1), axis=0)[[c for c in l2cats if c not in NON_CELLTYPES_L2]]
        Xe = pb.loc[animals].to_numpy()
        pe, ze = CC.loao_audit(Xe, y, sev)
        Cc = {"expression_clock": rho_pack(pe, ref, ze)}
        for lev, P in (("L1", P1), ("L2", P2)):
            pp, zz = CC.loao_audit(P.to_numpy(), y, sev)
            Cc[f"C1_{lev}"] = rho_pack(pp, ref, zz)
            Cc[f"C1_{lev}"]["n_features"] = int(P.shape[1])
            Cc[f"C2_{lev}_vs_expression"] = paired(f"composition {lev} vs expression", pp, pe, ref, f"composition {lev}", "expression")
            cf, _ = final_coefs(P.to_numpy(), y, P.columns.to_numpy(), covar=sev)
            Cc[f"C5_{lev}_nonzero"] = cf.round(4).to_dict("records")
        pj, zj = loao_keep_audit(Xe, P1.to_numpy(), y, sev)
        Cc["C3_expression_plus_L1"] = rho_pack(pj, ref, zj)
        Cc["C3_vs_expression"] = paired("expression + composition vs expression", pj, pe, ref, "expr+comp", "expression")
        pr, zr, ks = loao_comp_resid(Xe, y, sev, P1.to_numpy())
        Cc["C4_expression_resid_on_composition"] = rho_pack(pr, ref, zr)
        Cc["C4_expression_resid_on_composition"]["composition_pcs_per_fold"] = {"min": int(min(ks)), "max": int(max(ks))}
        Cc["C4_vs_expression"] = paired("expression | composition vs expression", pr, pe, ref, "expr|comp", "expression")
        C[ck] = Cc
        print(f"[C] {ck} done", flush=True)
    res["C"] = C
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    print("[done]")


if __name__ == "__main__":
    main()
