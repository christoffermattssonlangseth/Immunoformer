"""Step-3 control for da_glia.py: does residualising ANY five same-shaped scores collapse the
clock? 20 replicates of random 'pseudo-DA' scores (random marker sets of the same sizes, in the
same cell types, scored with the same control-gene logic), residualised in-fold exactly as the
real DA scores. The real DA drop is DA-specific only if it falls outside this null.

    PYTHONPATH="$PWD:$PWD/scripts" python analysis/da_glia_step3_control.py
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clock_composition as CC  # noqa: E402
import clock_selection as CS  # noqa: E402
import da_glia as DG  # noqa: E402
from duration_clock import loao_target  # noqa: E402

N_REP = int(os.environ.get("N_REP", "20"))


def one(rep, X, y, sev, base_ref, tables, genes, overall, rr_index, da_genes):
    rng = np.random.default_rng(500 + rep)
    pool = np.array([g for g in genes if g not in da_genes])
    cols = []
    for state, (mk, ct) in DG.STATES.items():
        fake = list(rng.choice(pool, len([g for g in mk if g in set(genes)]), replace=False))
        ctrl = DG.control_genes(fake, overall, genes, seed=rep)
        gi = {g: i for i, g in enumerate(genes)}
        tot, n = tables[ct]
        s = (tot[:, [gi[g] for g in fake]].mean(1) - tot[:, [gi[g] for g in ctrl]].mean(1)) / np.maximum(n, 1)
        s = np.where(n >= 50, s, np.nan)
        s = np.where(np.isnan(s), np.nanmean(s), s)
        cols.append(s)
    cov = np.column_stack([sev] + cols)
    pred, nnz = CC.loao_audit(X, y, cov)
    return {"rep": rep, "rho_baseline_target": float(spearmanr(pred, base_ref).statistic),
            "rho_own_target": float(spearmanr(pred, loao_target(y, cov)).statistic),
            "zero_feature_folds": int((nnz == 0).sum())}


def main():
    sums, n_cells, scats, ccats, genes = DG.sum_logexpr()
    F = CS.feature_sets()
    pb, info, meta = CC.load_rr()
    rr = F["rr"]
    sidx = {s: i for i, s in enumerate(scats)}
    cti = {c: i for i, c in enumerate(ccats)}
    per = {a: [sidx[s] for s in meta.loc[meta.sample_name == a, "meta_sample_id"]] for a in info.index}
    overall = sum(sums[si].sum((0, 1)) for si in per.values()) / sum(n_cells[si].sum() for si in per.values())
    tables = {}
    for ct in {c for _, c in DG.STATES.values()}:
        j = cti[ct]
        tables[ct] = (np.vstack([sums[per[a], j].sum(0) for a in rr.index]),
                      np.array([n_cells[per[a], j].sum() for a in rr.index]))
    da_genes = {g for mk, _ in DG.STATES.values() for g in mk}
    X, y, sev = F["X_all"], rr.day.to_numpy(), rr.score.to_numpy().reshape(-1, 1)
    base_ref = loao_target(y, sev)
    out = Parallel(n_jobs=int(os.environ.get("N_JOBS", "4")))(
        delayed(one)(r, X, y, sev, base_ref, tables, genes, overall, rr.index, da_genes) for r in range(N_REP))
    d = pd.DataFrame(out)
    real = json.load(open(os.path.join(DG.OUT, "results.json")))["step3"]
    res = {"n_rep": N_REP, "replicates": out,
           "null_baseline_target": {"mean": float(d.rho_baseline_target.mean()),
                                    "q05": float(d.rho_baseline_target.quantile(0.05)),
                                    "min": float(d.rho_baseline_target.min())},
           "null_own_target": {"mean": float(d.rho_own_target.mean()),
                               "q05": float(d.rho_own_target.quantile(0.05)),
                               "min": float(d.rho_own_target.min())},
           "real_DA": {"rho_baseline_target": real["rho_A"], "rho_own_target": real["rho_vs_own_target"]},
           "real_below_all_null_baseline": bool(real["rho_A"] < d.rho_baseline_target.min()),
           "real_percentile_baseline": float((d.rho_baseline_target < real["rho_A"]).mean() * 100),
           "real_percentile_own": float((d.rho_own_target < real["rho_vs_own_target"]).mean() * 100)}
    with open(os.path.join(DG.OUT, "step3_random_covariate_control.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "replicates"}, indent=1))


if __name__ == "__main__":
    main()
