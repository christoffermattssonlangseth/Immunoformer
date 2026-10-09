"""Random gene-set null: is the circadian block special, or would ANY gene set do?

The clock is redundant across the panel (removing circadian, sex or matrix genes costs
<= 0.03), so "circadian genes alone reach rho 0.86" may be unremarkable. For each block —
circadian (genes on the panel), matrix (58) and sex (Xist; no Y genes on the panel) — the
unchanged duration clock (RR, day_of_sacrifice, severity-residualised LOAO) is refit on:
  (1a) N_SETS random gene sets of exactly the block's size, drawn from the panel minus the
       block itself;
  (1b) N_SETS expression-matched sets: each block gene is replaced by a random gene from the
       same (mean-expression decile x detection-rate quintile) bin, without replacement;
and the block's own rho is placed in each null (percentile, p = (1 + #null >= obs)/(1 + N)).
Detection rate = fraction of RR cells with >= 1 count (one pass over the raw counts).

Interpretation fixed in advance (by the user): circadian below the 95th percentile of the
null -> NOT special, the confound concern is closed on these grounds; circadian in the
extreme tail of the matched null -> special, the timing question stays open.

    N_JOBS=4 PYTHONPATH="$PWD:$PWD/scripts" python analysis/gene_set_null.py
"""

from __future__ import annotations

import immunotransformer  # noqa: F401  (OpenMP guard — must precede numpy; see HANDOFF.md)

import json
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clock_composition as CC  # noqa: E402
import clock_selection as CS  # noqa: E402
from duration_clock import loao_target  # noqa: E402

OUT = "runs/clock_composition/gene_set_null"
DET_CACHE = os.path.join(OUT, "gene_detection_rr.csv")
N_SETS = int(os.environ.get("N_SETS", "500"))
N_JOBS = int(os.environ.get("N_JOBS", "4"))


def gene_detection(rr_sections):
    if os.path.exists(DET_CACHE):
        return pd.read_csv(DET_CACHE, index_col=0)
    with h5py.File(CC.ATLAS, "r") as f:
        g = f["obs/meta_sample_id"]
        cats = np.array([c.decode() if isinstance(c, bytes) else c for c in g["categories"][:]])
        in_rr = np.isin(cats[g["codes"][:]], list(rr_sections))
        var = f["var"]
        genes = np.array([x.decode() if isinstance(x, bytes) else x
                          for x in var[var.attrs["_index"]][:]])
        C = f["layers/counts"]
        indptr = C["indptr"][:]
        pos, tot, n = np.zeros(len(genes)), np.zeros(len(genes)), 0
        for a in range(0, len(in_rr), 200_000):
            e = min(a + 200_000, len(in_rr))
            lo, hi = indptr[a], indptr[e]
            X = sp.csr_matrix((C["data"][lo:hi], C["indices"][lo:hi], indptr[a:e + 1] - lo),
                              shape=(e - a, len(genes)))[in_rr[a:e]]
            pos += np.asarray((X > 0).sum(0)).ravel()
            tot += np.asarray(X.sum(0)).ravel()
            n += X.shape[0]
    d = pd.DataFrame({"frac_pos": pos / n, "mean_counts": tot / n}, index=genes)
    os.makedirs(OUT, exist_ok=True)
    d.to_csv(DET_CACHE)
    return d


def _rho(X, y, sev, cols):
    pred, nnz = CC.loao_audit(X[:, cols], y, sev)
    return float(spearmanr(pred, loao_target(y, sev)).statistic), int((nnz == 0).sum())


def main():
    os.makedirs(OUT, exist_ok=True)
    F = CS.feature_sets()
    rr, genes, X = F["rr"], F["genes"], F["X_all"]
    pb, info, meta = CC.load_rr()
    y, sev = rr.day.to_numpy(), rr.score.to_numpy().reshape(-1, 1)
    det = gene_detection(set(meta.loc[meta.sample_name.isin(rr.index), "meta_sample_id"]))
    det = det.loc[genes]
    expr = X.mean(0)
    ebin = pd.qcut(expr, 10, labels=False, duplicates="drop")
    dbin = pd.qcut(det.frac_pos.rank(method="first"), 5, labels=False)
    bins = pd.Series(list(zip(ebin, dbin)), index=genes)
    gi = {g: i for i, g in enumerate(genes)}
    blocks = {"circadian": [g for g in CC.CIRCADIAN if g in gi],
              "matrix": CS.matrix_genes(genes),
              "sex": [g for g in CC.SEX_GENES if g in gi]}
    res = {"n_sets": N_SETS, "blocks": {}}
    for name, block in blocks.items():
        t0 = time.time()
        bidx = [gi[g] for g in block]
        obs, obs_zero = _rho(X, y, sev, bidx)
        pool = np.array([i for i in range(len(genes)) if genes[i] not in set(block)])
        rng = np.random.default_rng(len(block))
        rand_sets = [rng.choice(pool, len(block), replace=False) for _ in range(N_SETS)]
        matched_sets = []
        for _ in range(N_SETS):
            s, used = [], set(bidx)
            for g in block:
                cand = [gi[c] for c in bins.index[bins == bins[g]] if gi[c] not in used]
                pick = int(rng.choice(cand)) if cand else int(rng.choice(pool))
                s.append(pick); used.add(pick)
            matched_sets.append(np.array(s))
        out = {"genes": block, "size": len(block), "observed_rho": obs,
               "observed_zero_feature_folds": obs_zero,
               "block_mean_expr": float(expr[bidx].mean()),
               "block_median_frac_pos": float(det.frac_pos.iloc[bidx].median())}
        for label, sets in (("random", rand_sets), ("expression_matched", matched_sets)):
            vals = Parallel(n_jobs=N_JOBS)(delayed(_rho)(X, y, sev, s) for s in sets)
            r = np.array([v[0] for v in vals])
            z = np.array([v[1] for v in vals])
            out[label] = {
                "mean": float(np.nanmean(r)), "sd": float(np.nanstd(r)),
                "q05": float(np.nanquantile(r, 0.05)), "q50": float(np.nanquantile(r, 0.5)),
                "q95": float(np.nanquantile(r, 0.95)),
                "observed_percentile": float((r < obs).mean() * 100),
                "p": float((1 + (r >= obs).sum()) / (1 + len(r))),
                "sets_with_any_zero_feature_fold": int((z > 0).sum()),
                "median_zero_feature_folds": float(np.median(z)),
                "match_check_mean_expr": float(np.mean([expr[s].mean() for s in sets])),
                "match_check_median_frac_pos": float(np.median([det.frac_pos.iloc[s].median()
                                                                for s in sets])),
                "values": r.round(4).tolist()}
            print(f"[{name} / {label}] obs {obs:+.3f}  null mean {out[label]['mean']:+.3f}  "
                  f"q95 {out[label]['q95']:+.3f}  pct {out[label]['observed_percentile']:.1f}  "
                  f"p {out[label]['p']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        res["blocks"][name] = out
        with open(os.path.join(OUT, "results.json"), "w") as fh:
            json.dump(res, fh, indent=2)
    L = ["# Random gene-set null (same pipeline, same folds)", "",
         f"{N_SETS} sets per null. Matched = same mean-expression decile x detection quintile per gene.", "",
         "| block | genes | observed rho | null | mean | SD | 5th | 50th | 95th | observed percentile | p | sets with zero-feature folds |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, b in res["blocks"].items():
        for label in ("random", "expression_matched"):
            s = b[label]
            L.append(f"| {name} | {b['size']} | {b['observed_rho']:+.3f} | {label} | {s['mean']:+.3f} | "
                     f"{s['sd']:.3f} | {s['q05']:+.3f} | {s['q50']:+.3f} | {s['q95']:+.3f} | "
                     f"{s['observed_percentile']:.1f} | {s['p']:.3f} | {s['sets_with_any_zero_feature_fold']} |")
    c = res["blocks"]["circadian"]["expression_matched"]
    L += ["", "PRE-SET READING (circadian): " + (
        "below the 95th percentile of the expression-matched null -> the circadian block is NOT "
        "special; the confound concern is closed on these grounds." if
        res["blocks"]["circadian"]["observed_rho"] < c["q95"] else
        "in the extreme tail of the expression-matched null -> the circadian block IS special; "
        "the timing question stays open.")]
    with open(os.path.join(OUT, "report.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
