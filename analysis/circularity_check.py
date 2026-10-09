"""Circularity and dilution checks for gene-block results (matrix, circadian).

1a  membership audit: which block genes came from gene-family annotation and which were
    named because they were top coefficients of the fitted clock;
1b  CLEAN subset = annotation-defined block genes minus every gene with a non-zero
    coefficient in EITHER fitted clock list (published 135-gene list and the current
    baseline refit);
1c  clean-subset clock vs size-matched random and expression-matched null (same pipeline,
    same folds, N_SETS sets);
1d  calibration ceiling: the clock refit on its own top-|coef| genes (same size as the
    original block) — selection-inflated by design;
2a  direction of each clean gene vs day_of_sacrifice (whole tissue);
2b  clean-subset clock WITHIN cell type (astrocyte, fibroblast, endothelial,
    oligodendrocyte pseudobulk) vs a matched random null within that cell type;
2c  clean-subset clock with cell-type composition residualised inside each fold
    (in-fold PCA of CLR proportions, components to 80% variance; and immune fraction alone).
Pre-set reading (user): clean beats null AND survives within type -> remodelling is real;
beats null but collapses within type -> dilution; falls into null -> artefact of selection.

    N_JOBS=4 PYTHONPATH="$PWD:$PWD/scripts" python analysis/circularity_check.py
"""

from __future__ import annotations

import immunotransformer  # noqa: F401  (OpenMP guard — must precede numpy; see HANDOFF.md)

import json
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clock_composition as CC  # noqa: E402
import clock_selection as CS  # noqa: E402
from duration_clock import ENET_KW, _topvar, final_coefs, loao_target  # noqa: E402
from immunotransformer.stats import bootstrap_spearman  # noqa: E402

OUT = "runs/clock_composition/circularity"
N_SETS = int(os.environ.get("N_SETS", "500"))
N_SETS_WITHIN = int(os.environ.get("N_SETS_WITHIN", "200"))
N_JOBS = int(os.environ.get("N_JOBS", "4"))
WITHIN_TYPES = ["Astrocyte", "Fibroblast", "Endothelial", "Oligodendrocyte"]
IMMUNE = ["Myeloid", "T cell", "B cell", "DC", "NK/DC", "Neutrophil"]
COEF_NAMED_MATRIX = ["Col4a1", "Col4a2", "Thbs2", "Serpine2", "Eln", "Fbln2", "Ptx3", "Fmod"]


def rho_ci(X, y, sev):
    pred, nnz = CC.loao_audit(X, y, sev)
    b = bootstrap_spearman(pred, loao_target(y, sev), n_boot=2000, seed=0)
    return float(b["rho"]), [float(b["ci_lo"]), float(b["ci_hi"])], int((nnz == 0).sum())


def _r(X, y, sev, cols):
    pred, nnz = CC.loao_audit(X[:, cols], y, sev)
    return float(spearmanr(pred, loao_target(y, sev)).statistic)


def null_test(X, y, sev, genes, block, det_bins, n_sets, seed):
    gi = {g: i for i, g in enumerate(genes)}
    bidx = [gi[g] for g in block]
    obs, ci, zero = rho_ci(X[:, bidx], y, sev)
    pool = np.array([i for i, g in enumerate(genes) if g not in set(block)])
    rng = np.random.default_rng(seed)
    rand = [rng.choice(pool, len(block), replace=False) for _ in range(n_sets)]
    matched = []
    for _ in range(n_sets):
        s, used = [], set(bidx)
        for g in block:
            cand = [gi[c] for c in det_bins.index[det_bins == det_bins[g]] if gi[c] not in used]
            pick = int(rng.choice(cand)) if cand else int(rng.choice(pool))
            s.append(pick); used.add(pick)
        matched.append(np.array(s))
    out = {"n_genes": len(block), "rho": obs, "ci": ci, "zero_feature_folds": zero}
    for label, sets in (("random", rand), ("expression_matched", matched)):
        r = np.array(Parallel(n_jobs=N_JOBS)(delayed(_r)(X, y, sev, s) for s in sets))
        out[label] = {"mean": float(np.nanmean(r)), "q95": float(np.nanquantile(r, 0.95)),
                      "percentile": float((r < obs).mean() * 100),
                      "p": float((1 + (r >= obs).sum()) / (1 + len(r)))}
    return out


def clock_with_composition(X, y, sev, comp, mode):
    """LOAO clock with composition covariates fit inside each fold."""
    n = len(y)
    pred, nnz = np.full(n, np.nan), np.zeros(n, int)
    clr = np.log(comp + 1e-4)
    clr = clr - clr.mean(1, keepdims=True)
    for i in range(n):
        tr = np.ones(n, bool); tr[i] = False
        if mode == "pca":
            sc = StandardScaler().fit(clr[tr])
            p = PCA(random_state=0).fit(sc.transform(clr[tr]))
            k = int(np.searchsorted(np.cumsum(p.explained_variance_ratio_), 0.80) + 1)
            Z = p.transform(sc.transform(clr))[:, :k]
        else:
            Z = comp
        cov = np.column_stack([sev, Z])
        fx = LinearRegression().fit(cov[tr], X[tr]); fy = LinearRegression().fit(cov[tr], y[tr])
        Xtr, Xte = X[tr] - fx.predict(cov[tr]), X[i:i + 1] - fx.predict(cov[i:i + 1])
        ytr = y[tr] - fy.predict(cov[tr])
        sel = _topvar(Xtr)
        scl = StandardScaler().fit(Xtr[:, sel])
        m = ElasticNetCV(random_state=0, **ENET_KW).fit(scl.transform(Xtr[:, sel]), ytr)
        pred[i] = m.predict(scl.transform(Xte[:, sel]))[0]
        nnz[i] = int((m.coef_ != 0).sum())
    b = bootstrap_spearman(pred, loao_target(y, sev), n_boot=2000, seed=0)
    return float(b["rho"]), [float(b["ci_lo"]), float(b["ci_hi"])], int((nnz == 0).sum())


def main():
    os.makedirs(OUT, exist_ok=True)
    F = CS.feature_sets()
    rr, genes, X = F["rr"], F["genes"], F["X_all"]
    y, sev = rr.day.to_numpy(), rr.score.to_numpy().reshape(-1, 1)
    det = pd.read_csv("runs/clock_composition/gene_set_null/gene_detection_rr.csv", index_col=0).loc[genes]
    ebin = pd.qcut(X.mean(0), 10, labels=False, duplicates="drop")
    dbin = pd.qcut(det.frac_pos.rank(method="first"), 5, labels=False)
    bins = pd.Series(list(zip(ebin, dbin)), index=genes)

    pub = set(pd.read_csv(CC.OLD_CLOCK).gene)
    refit = pd.read_csv("runs/clock_composition/clock_genes_baseline_refit.csv")
    coef_genes = pub | set(refit.gene)
    matrix_all = CS.matrix_genes(genes)
    annotation = [g for g in matrix_all if g not in CS.MATRIX_NAMED or g in CS.PROTEOGLYCANS]
    import re
    pat = re.compile(r"^(Col\d+a\d+|Lam[abc]\d+|Fn1|Thbs\d+|Gpc\d+|Sdc\d+)$")
    by_annotation = [g for g in matrix_all if pat.match(g) or g in CS.PROTEOGLYCANS]
    named_only = [g for g in matrix_all if g in CS.MATRIX_NAMED and g not in by_annotation]
    clean_matrix = [g for g in by_annotation if g not in coef_genes]
    circ_all = [g for g in CC.CIRCADIAN if g in genes]
    clean_circ = [g for g in circ_all if g not in coef_genes]
    res = {"1a": {
        "matrix_total": len(matrix_all),
        "matrix_named_in_prompt_from_clock_coefficients": COEF_NAMED_MATRIX,
        "matrix_by_family_annotation": by_annotation,
        "matrix_only_via_named_list_not_annotation": named_only,
        "matrix_annotation_genes_with_nonzero_clock_coef": sorted(set(by_annotation) & coef_genes),
        "circadian_total": circ_all,
        "circadian_with_nonzero_clock_coef": sorted(set(circ_all) & coef_genes),
        "coef_lists": {"published_135": len(pub), "baseline_refit": int(len(refit)),
                       "union": len(coef_genes)}},
        "1b": {"clean_matrix": clean_matrix, "clean_matrix_size": len(clean_matrix),
               "clean_circadian": clean_circ, "clean_circadian_size": len(clean_circ)}}
    print(f"[1a/1b] matrix {len(matrix_all)} -> clean {len(clean_matrix)}; circadian "
          f"{len(circ_all)} -> clean {len(clean_circ)}", flush=True)

    res["1c_matrix_clean"] = null_test(X, y, sev, genes, clean_matrix, bins, N_SETS, 11)
    print(f"[1c matrix clean] {json.dumps({k: v for k, v in res['1c_matrix_clean'].items()})}", flush=True)
    res["1c_circadian_clean"] = null_test(X, y, sev, genes, clean_circ, bins, N_SETS, 12) \
        if len(clean_circ) >= 2 else {"skipped": f"only {len(clean_circ)} genes"}
    print(f"[1c circadian clean] {json.dumps(res['1c_circadian_clean'])}", flush=True)

    # 1d calibration ceiling: top-|coef| genes of the baseline refit, same size as the 58 block
    top = refit.reindex(refit.coef.abs().sort_values(ascending=False).index).gene.head(len(matrix_all))
    gi = {g: i for i, g in enumerate(genes)}
    r, ci, z = rho_ci(X[:, [gi[g] for g in top]], y, sev)
    res["1d_top_coef_ceiling"] = {"n_genes": int(len(top)), "rho": r, "ci": ci, "zero_feature_folds": z,
                                  "note": "selection used all animals: in-sample ceiling by design"}
    top_clean = refit.reindex(refit.coef.abs().sort_values(ascending=False).index).gene.head(len(clean_matrix))
    r2, ci2, z2 = rho_ci(X[:, [gi[g] for g in top_clean]], y, sev)
    res["1d_top_coef_ceiling_clean_size"] = {"n_genes": int(len(top_clean)), "rho": r2, "ci": ci2,
                                             "zero_feature_folds": z2}

    # 2a direction of clean genes
    dirs = []
    for g in clean_matrix:
        x = X[:, gi[g]]
        dirs.append({"gene": g, "rho_day": float(spearmanr(x, y).statistic),
                     "rho_day_partial_score": float(CC.np.corrcoef(
                         *(v - np.polyval(np.polyfit(sev.ravel(), v, 1), sev.ravel())
                           for v in (pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy())))[0, 1])})
    d = pd.DataFrame(dirs)
    res["2a"] = {"genes": dirs, "n_up": int((d.rho_day > 0).sum()), "n_down": int((d.rho_day < 0).sum()),
                 "n_up_partial": int((d.rho_day_partial_score > 0).sum()),
                 "n_down_partial": int((d.rho_day_partial_score < 0).sum())}

    # 2b within cell type
    res["2b"] = {}
    for ct in WITHIN_TYPES:
        e = F["task1"][ct]
        yy = rr.loc[e["animals"], "day"].to_numpy()
        ss = rr.loc[e["animals"], "score"].to_numpy().reshape(-1, 1)
        res["2b"][ct] = null_test(e["X"], yy, ss, genes, clean_matrix, bins, N_SETS_WITHIN, 20)
        res["2b"][ct]["n_animals"] = len(yy)
        print(f"[2b {ct}] {json.dumps(res['2b'][ct])}", flush=True)

    # 2c composition residualised (clean subset)
    comp = pd.DataFrame(F["cells"]).loc[rr.index]
    comp = comp.div(comp.sum(1), axis=0)
    Xc = X[:, [gi[g] for g in clean_matrix]]
    r_p, ci_p, z_p = clock_with_composition(Xc, y, sev, comp.to_numpy(), "pca")
    imm = comp[[c for c in IMMUNE if c in comp]].sum(1).to_numpy().reshape(-1, 1)
    r_i, ci_i, z_i = clock_with_composition(Xc, y, sev, imm, "raw")
    res["2c"] = {"composition_pca_80pct": {"rho": r_p, "ci": ci_p, "zero_feature_folds": z_p},
                 "immune_fraction_only": {"rho": r_i, "ci": ci_i, "zero_feature_folds": z_i},
                 "unadjusted": {"rho": res["1c_matrix_clean"]["rho"], "ci": res["1c_matrix_clean"]["ci"]}}
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    print(json.dumps({k: v for k, v in res.items() if k not in ("2a",)}, indent=1, default=str))


if __name__ == "__main__":
    main()
