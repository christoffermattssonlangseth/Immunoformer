"""Selection-corrected tests for Task 1f (per-cell-type clocks) and Task 2 (spatial context),
plus the circadian/sex-excluded per-cell-type sweep (correction set A-C).

The headline numbers of 1f and Task 2 are MAXIMA of sweeps (best of 13 cell types, best of
9 spatial configs), so they are biased upward. This script provides:
  MODE=perm    A1/B2: permute day_of_sacrifice across RR animals, rerun the ENTIRE sweeps
               (13 cell types, all-cells, 9 spatial configs) and record every rho, giving
               the null of the maximum. Sharded: PERM_START/PERM_END.
  MODE=nested  A2/B3: choose the cell type / spatial config inside each LOAO training fold
               (inner 5-fold CV on training animals only), predict the held-out animal.
  MODE=c2      C1/C2/C4: per-cell-type sweep with circadian AND sex genes excluded before
               variance filtering, next to the unexcluded values; top-10 genes with the
               circadian share of coefficient mass.
  MODE=counts  A3: cells per animal per cell type.
  MODE=summary collect everything into report.md / selection.json.
The clock itself is unchanged (clock_composition.loao_audit == scripts/duration_clock
loao_clock, verified). Zero-feature folds are recorded everywhere.

    MODE=perm PERM_START=0 PERM_END=100 N_JOBS=6 PYTHONPATH="$PWD:$PWD/scripts" python analysis/clock_selection.py
"""

from __future__ import annotations

import immunotransformer  # noqa: F401  (OpenMP guard — must precede numpy; see HANDOFF.md)

import glob
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clock_composition as CC  # noqa: E402
import spatial_context as SC  # noqa: E402
from duration_clock import ENET_KW, _topvar, final_coefs, loao_target  # noqa: E402
from immunotransformer.stats import bootstrap_spearman  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
OUT = "runs/clock_composition"
PERM_DIR = os.path.join(OUT, "perm")
MODE = os.environ.get("MODE", "summary")
N_JOBS = int(os.environ.get("N_JOBS", "6"))
INNER_K = 5
CIRC_SEX = set(CC.CIRCADIAN) | set(CC.SEX_GENES)


# ---------------------------------------------------------------- feature sets

def feature_sets():
    pb, info, meta = CC.load_rr()
    rr = info[info.model == CC.RR]
    genes = pb.columns.to_numpy()
    clean_idx = np.array([g not in CIRC_SEX for g in genes])
    counts, n_cells, scats, ccats = CC.celltype_pseudobulk(None)
    sidx = {s: i for i, s in enumerate(scats)}
    task1, cells = {}, {}
    for j, ct in enumerate(ccats):
        per_n, per_c = {}, {}
        for a in rr.index:
            si = [sidx[s] for s in meta.loc[meta.sample_name == a, "meta_sample_id"]]
            per_n[a] = int(n_cells[si, j].sum())
            per_c[a] = counts[si, j].sum(0)
        cells[ct] = per_n
        ok = [a for a in rr.index if per_n[a] >= CC.MIN_CELLS]
        if len(ok) < CC.MIN_ANIMALS:
            continue
        X = CC._lognorm(np.vstack([per_c[a] for a in ok]))
        task1[ct] = {"animals": ok, "X": X, "X_clean": X[:, clean_idx]}
    d = SC.build(None)
    sec, cts, nc = d["sections"].astype(str), d["celltypes"].astype(str), d["n_cells"]
    s2 = {s: i for i, s in enumerate(sec)}
    pac = np.vstack([nc[[s2[s] for s in meta.loc[meta.sample_name == a, "meta_sample_id"]
                         if s in s2]].sum(0) for a in rr.index])
    keep = np.where((pac >= SC.MIN_CELLS).all(0))[0]
    X_pb = pb.loc[rr.index].to_numpy()
    task2 = {"own expression per cell type":
             SC.animal_matrix(d["own"], nc, sec, meta, rr.index, keep)}
    for k in SC.KS:
        for name, lab in (("nb", "neighbourhood mean"), ("disc", "|own - neighbourhood|")):
            F = SC.animal_matrix(d[f"{name}{k}"], nc, sec, meta, rr.index, keep)
            task2[f"k={k} {lab}"] = F
            task2[f"k={k} pseudobulk + {lab}"] = np.hstack([X_pb, F])
    return {"rr": rr, "genes": genes, "clean_genes": genes[clean_idx], "X_all": X_pb,
            "task1": task1, "task2": task2, "cells": cells}


def rho_of(X, y, sev):
    pred, nnz = CC.loao_audit(X, y, sev)
    return float(spearmanr(pred, loao_target(y, sev)).statistic), int((nnz == 0).sum())


# ---------------------------------------------------------------- A1/B2 permutation of max

def _perm_one(F, seed):
    rr = F["rr"]
    rng = np.random.default_rng(seed)
    yperm = pd.Series(rng.permutation(rr.day.to_numpy()), index=rr.index)
    sev_all = rr.score.to_numpy().reshape(-1, 1)
    out = {"seed": seed, "task1": {}, "task2": {}, "zero": {}}
    out["all_cells"], out["zero"]["all_cells"] = rho_of(F["X_all"], yperm.to_numpy(), sev_all)
    for ct, e in F["task1"].items():
        y = yperm.loc[e["animals"]].to_numpy()
        s = rr.loc[e["animals"], "score"].to_numpy().reshape(-1, 1)
        out["task1"][ct], out["zero"][ct] = rho_of(e["X"], y, s)
    for name, X in F["task2"].items():
        out["task2"][name], out["zero"][name] = rho_of(X, yperm.to_numpy(), sev_all)
    return out


def run_perm():
    F = feature_sets()
    a, b = int(os.environ.get("PERM_START", "0")), int(os.environ.get("PERM_END", "200"))
    os.makedirs(PERM_DIR, exist_ok=True)
    path = os.path.join(PERM_DIR, f"perm_{a}_{b}.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["seed"] for l in open(path)}
    seeds = [1000 + i for i in range(a, b) if 1000 + i not in done]
    t0 = time.time()
    with open(path, "a") as fh:
        for res in Parallel(n_jobs=N_JOBS, return_as="generator")(
                delayed(_perm_one)(F, s) for s in seeds):
            fh.write(json.dumps(res) + "\n"); fh.flush()
            print(f"  [perm] seed {res['seed']} best cell type {max(res['task1'].values()):+.3f} "
                  f"best spatial {max(res['task2'].values()):+.3f}  {time.time() - t0:.0f}s",
                  flush=True)


# ---------------------------------------------------------------- A2/B3 nested selection

def _clock_fit_predict(Xtr, ytr, ctr, Xte, cte):
    fx = LinearRegression().fit(ctr, Xtr)
    fy = LinearRegression().fit(ctr, ytr)
    Xtr_r, Xte_r = Xtr - fx.predict(ctr), Xte - fx.predict(cte)
    sel = _topvar(Xtr_r)
    sc = StandardScaler().fit(Xtr_r[:, sel])
    kw = {**ENET_KW, "cv": min(ENET_KW["cv"], len(ytr) - 1)}
    m = ElasticNetCV(random_state=0, **kw).fit(sc.transform(Xtr_r[:, sel]), ytr - fy.predict(ctr))
    return m.predict(sc.transform(Xte_r[:, sel])), int((m.coef_ != 0).sum())


def _inner_score(X, y, c):
    """Inner 5-fold CV rho on training animals only (fold-wise residual reference)."""
    pred, ref = np.empty(len(y)), np.empty(len(y))
    for tr, te in KFold(INNER_K, shuffle=True, random_state=0).split(X):
        pred[te], _ = _clock_fit_predict(X[tr], y[tr], c[tr], X[te], c[te])
        ref[te] = y[te] - LinearRegression().fit(c[tr], y[tr]).predict(c[te])
    return float(spearmanr(pred, ref).statistic)


def _nested_fold(held, cands, rr):
    """cands: {name: (animals, X)}. Pick by inner CV on training animals, predict held."""
    scores = {}
    for name, (animals, X) in cands.items():
        if held not in animals:
            continue
        tr = np.array([a != held for a in animals])
        y = rr.loc[animals, "day"].to_numpy()
        c = rr.loc[animals, "score"].to_numpy().reshape(-1, 1)
        scores[name] = _inner_score(X[tr], y[tr], c[tr])
    best = max(scores, key=scores.get)
    animals, X = cands[best]
    tr = np.array([a != held for a in animals])
    te = ~tr
    y = rr.loc[animals, "day"].to_numpy()
    c = rr.loc[animals, "score"].to_numpy().reshape(-1, 1)
    p, nnz = _clock_fit_predict(X[tr], y[tr], c[tr], X[te], c[te])
    return {"held_out": held, "chosen": best, "pred": float(p[0]), "nonzero": nnz,
            "inner_scores": scores}


def run_nested():
    F = feature_sets()
    rr = F["rr"]
    out = {}
    for label, cands in (
            ("task1_cell_type", {ct: (e["animals"], e["X"]) for ct, e in F["task1"].items()}),
            ("task2_spatial", {n: (list(rr.index), X) for n, X in F["task2"].items()})):
        t0 = time.time()
        rows = Parallel(n_jobs=N_JOBS)(delayed(_nested_fold)(h, cands, rr) for h in rr.index)
        df = pd.DataFrame(rows).set_index("held_out").loc[rr.index]
        y, sev = rr.day.to_numpy(), rr.score.to_numpy().reshape(-1, 1)
        ref = loao_target(y, sev)
        b = bootstrap_spearman(df.pred.to_numpy(), ref, n_boot=CC.N_BOOT, seed=0)
        out[label] = {"rho": float(b["rho"]), "ci": [float(b["ci_lo"]), float(b["ci_hi"])],
                      "n": len(rr), "zero_feature_folds": int((df.nonzero == 0).sum()),
                      "chosen_counts": df.chosen.value_counts().to_dict(),
                      "per_fold": df.reset_index().to_dict("records")}
        print(f"[nested {label}] rho {b['rho']:+.3f} [{b['ci_lo']:+.2f}, {b['ci_hi']:+.2f}] "
              f"chosen {out[label]['chosen_counts']}  {time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(OUT, "nested_selection.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)


# ---------------------------------------------------------------- C1/C2/C4

def run_c2():
    F = feature_sets()
    rr = F["rr"]
    circ = set(CC.CIRCADIAN)
    rows = []
    for ct, e in F["task1"].items():
        y = rr.loc[e["animals"], "day"].to_numpy()
        s = rr.loc[e["animals"], "score"].to_numpy().reshape(-1, 1)
        full, _ = CC.evaluate(f"{ct} (all genes)", e["X"], y, s)
        clean, _ = CC.evaluate(f"{ct} (circadian + sex excluded)", e["X_clean"], y, s)
        cf, _ = final_coefs(e["X"], y, F["genes"], covar=s)
        cc, _ = final_coefs(e["X_clean"], y, F["clean_genes"], covar=s)
        tot = cf.coef.abs().sum()
        rows.append({
            "cell_type": ct, "n": len(y), "rho_all": full["rho"], "ci_all": full["ci"],
            "zero_all": full["zero_feature_folds"], "rho_clean": clean["rho"],
            "ci_clean": clean["ci"], "zero_clean": clean["zero_feature_folds"],
            "circadian_coef_share": float(cf[cf.gene.isin(circ)].coef.abs().sum() / tot) if tot else 0,
            "sex_coef_share": float(cf[cf.gene.isin(CC.SEX_GENES)].coef.abs().sum() / tot) if tot else 0,
            "top10_all": [{"gene": g, "coef": round(float(c), 3), "circadian": g in circ}
                          for g, c in cf.head(10)[["gene", "coef"]].values],
            "top10_clean": [{"gene": g, "coef": round(float(c), 3)}
                            for g, c in cc.head(10)[["gene", "coef"]].values],
            "matrix_in_clean_nonzero": cc[cc.gene.isin(CC.BLOCKS["matrix/scar"])]
                .round(3).to_dict("records"),
            "matrix_share_clean": float(cc[cc.gene.isin(CC.BLOCKS["matrix/scar"])].coef.abs().sum()
                                        / cc.coef.abs().sum()) if len(cc) else 0,
            "n_nonzero_clean": int(len(cc))})
    with open(os.path.join(OUT, "c2_circadian_excluded.json"), "w") as fh:
        json.dump(rows, fh, indent=2, default=str)


MATRIX_NAMED = ["Col4a1", "Col4a2", "Thbs2", "Serpine2", "Eln", "Fbln2", "Ptx3", "Fmod"]
PROTEOGLYCANS = ["Acan", "Bcan", "Ncan", "Vcan", "Dcn", "Bgn", "Lum", "Fmod", "Prelp", "Ogn",
                 "Aspn", "Omd", "Kera", "Epyc", "Chad", "Podn", "Optc", "Nyx", "Hspg2", "Agrn",
                 "Prg4", "Srgn", "Cspg4", "Cspg5", "Spock1", "Spock2", "Spock3"]


def matrix_genes(genes):
    import re
    pat = re.compile(r"^(Col\d+a\d+|Lam[abc]\d+|Fn1|Thbs\d+|Gpc\d+|Sdc\d+)$")
    return sorted({g for g in genes if g in MATRIX_NAMED or g in PROTEOGLYCANS or pat.match(g)})


def run_c2m():
    """Matrix block excluded (before variance filtering), per cell type and all cells."""
    F = feature_sets()
    rr = F["rr"]
    mx = matrix_genes(F["genes"])
    keep = np.array([g not in mx for g in F["genes"]])
    rows = []
    y, s = rr.day.to_numpy(), rr.score.to_numpy().reshape(-1, 1)
    full, _ = CC.evaluate("all cells (all genes)", F["X_all"], y, s)
    nomx, _ = CC.evaluate("all cells (matrix excluded)", F["X_all"][:, keep], y, s, full)
    rows.append({"cell_type": "ALL CELLS", "n": len(y), "rho_all": full["rho"], "ci_all": full["ci"],
                 "zero_all": full["zero_feature_folds"], "rho_nomatrix": nomx["rho"],
                 "ci_nomatrix": nomx["ci"], "zero_nomatrix": nomx["zero_feature_folds"],
                 "fisher_p": nomx["fisher_p_vs_baseline"]})
    for ct, e in F["task1"].items():
        yy = rr.loc[e["animals"], "day"].to_numpy()
        ss = rr.loc[e["animals"], "score"].to_numpy().reshape(-1, 1)
        a, _ = CC.evaluate(f"{ct} (all genes)", e["X"], yy, ss)
        b, _ = CC.evaluate(f"{ct} (matrix excluded)", e["X"][:, keep], yy, ss, a)
        cf, _ = final_coefs(e["X"], yy, F["genes"], covar=ss)
        tot = cf.coef.abs().sum()
        rows.append({"cell_type": ct, "n": len(yy), "rho_all": a["rho"], "ci_all": a["ci"],
                     "zero_all": a["zero_feature_folds"], "rho_nomatrix": b["rho"],
                     "ci_nomatrix": b["ci"], "zero_nomatrix": b["zero_feature_folds"],
                     "fisher_p": b["fisher_p_vs_baseline"],
                     "matrix_coef_share_all_genes": float(cf[cf.gene.isin(mx)].coef.abs().sum() / tot)
                     if tot else 0})
    with open(os.path.join(OUT, "c2_matrix_excluded.json"), "w") as fh:
        json.dump({"excluded_matrix_genes": mx, "n_excluded": len(mx), "rows": rows}, fh,
                  indent=2, default=str)


def run_counts():
    F = feature_sets()
    rows = []
    for ct, per in F["cells"].items():
        v = np.array(list(per.values()))
        rows.append({"cell_type": ct, "median": float(np.median(v)), "min": int(v.min()),
                     "max": int(v.max()), "animals_ge_50": int((v >= CC.MIN_CELLS).sum()),
                     "in_sweep": ct in F["task1"]})
    pd.DataFrame(rows).sort_values("median", ascending=False).to_csv(
        os.path.join(OUT, "cells_per_animal_by_type.csv"), index=False)


if __name__ == "__main__":
    {"perm": run_perm, "nested": run_nested, "c2": run_c2, "c2m": run_c2m,
     "counts": run_counts}[MODE]()
