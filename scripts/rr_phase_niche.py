"""Phase 0, niche-resolved: does ANY cell type / spatial niche carry relapse PHASE
that animal-level pseudobulk averaged away? (docs/relapse-phase-model-design.md §7b)

Bulk TEST A was a clean null (peak vs remission, severity-residualized, AUC 0.51).
The one thing bulk can't rule out is phase that lives in a specific population. So we
repeat TEST A INSIDE each niche: build per-(animal x niche) pseudobulk, residualize
on clinical severity, and LOAO-classify peak vs remission from the residual molecular
structure of that niche alone.

  * Per-niche expression test: AUC + permutation null (screened: only niches with
    observed AUC > 0.62 get the expensive null). Bonferroni bar over #niches tested.
  * Niche COMPOSITION test: do cell-type/niche FRACTIONS separate peak vs remission
    beyond severity? (phase could be compositional, not transcriptional.)

Aggregation is cached to npz so the stats iterate without re-reading the 14GB file.

    python scripts/rr_phase_niche.py --cluster-col CellCharter_10
    python scripts/rr_phase_niche.py --cluster-col leiden_1
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.random import default_rng
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from immunotransformer.train import resolve_device  # noqa: F401  (OpenMP guard)

RRMAP2 = ("/Volumes/moldiassd/RRMAP2_xenium_adata/kmeans_separated/"
          "RRMAP2_xenium_all_samples.cellcharter.companion.ready.with_metadata.rerun.h5ad")
APEX = ["PEAK1", "PEAK2", "PEAK3"]
DESC = ["REMISSION1", "REMISSION2"]
MIN_CELLS = 50          # an animal needs >= this many cells in a niche to be used
MIN_PER_CLASS = 4       # need >= this many peak AND remission animals to test a niche


def residualize(pb, score):
    s = (score - score.mean()) / (score.std() + 1e-9)
    X = np.c_[np.ones_like(s), s]
    beta, *_ = np.linalg.lstsq(X, pb, rcond=None)
    return pb - X @ beta


def loao_auc(feat, y, C=0.5):
    n = len(y)
    proba = np.zeros(n)
    for i in range(n):
        tr = np.arange(n) != i
        if len(set(y[tr])) < 2:
            proba[i] = y[tr].mean()
            continue
        clf = LogisticRegression(C=C, max_iter=2000)
        clf.fit(feat[tr], y[tr])
        proba[i] = clf.predict_proba(feat[i:i + 1])[0, 1]
    return roc_auc_score(y, proba)


def perm_null(feat, y, obs, B, seed=0):
    rng = default_rng(seed)
    ge = 1
    for _ in range(B):
        ge += loao_auc(feat, rng.permutation(y)) >= obs
    return ge / (B + 1)


def build_or_load(cluster_col, cache):
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        print(f"[cache] {cache}")
        return (d["pb"], d["counts"], d["genes"].astype(str), d["animals"].astype(str),
                d["clusters"].astype(str), d["stage"].astype(str), d["score"].astype(float))
    import anndata as ad
    print(f"[load] {RRMAP2}")
    a = ad.read_h5ad(RRMAP2)
    a = a[a.obs["model"] == "RELAPSE REMITTING"].copy()
    o = a.obs
    print(f"       RR: {a.n_obs:,} cells x {a.n_vars:,} genes")
    animals = list(pd.unique(o["sample_name"]))
    clusters = [str(c) for c in pd.unique(o[cluster_col])]
    a_code = {x: i for i, x in enumerate(animals)}
    c_code = {x: i for i, x in enumerate(clusters)}
    ac = o["sample_name"].map(a_code).to_numpy()
    cc = o[cluster_col].astype(str).map(c_code).to_numpy()
    nA, nC = len(animals), len(clusters)
    grp = ac * nC + cc
    counts = a.layers["counts"]
    if not sp.issparse(counts):
        counts = sp.csr_matrix(counts)
    G = sp.csr_matrix((np.ones(a.n_obs), (grp, np.arange(a.n_obs))), shape=(nA * nC, a.n_obs))
    S = np.asarray((G @ counts).todense())                 # [nA*nC, genes]
    ncell = np.asarray(G.sum(1)).ravel()
    lib = S.sum(1, keepdims=True); lib[lib == 0] = 1.0
    pb = np.log1p(S / lib * 1e4).reshape(nA, nC, -1)        # [animal, cluster, gene]
    ncell = ncell.reshape(nA, nC)
    meta = o.drop_duplicates("sample_name").set_index("sample_name")
    stage = meta["stage"].astype(str).reindex(animals).to_numpy()
    score = pd.to_numeric(meta["score_sacrifice"], errors="coerce").reindex(animals).to_numpy(float)
    genes = np.array(a.var_names, dtype=str)
    np.savez(cache, pb=pb, counts=ncell, genes=genes, animals=np.array(animals),
             clusters=np.array(clusters), stage=stage, score=score)
    print(f"[cache] wrote {cache}")
    return pb, ncell, genes, np.array(animals), np.array(clusters), stage, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster-col", default="CellCharter_10")
    ap.add_argument("--n-pc", type=int, default=6)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--screen-auc", type=float, default=0.62)
    ap.add_argument("--out-dir", default="runs/rr_phase_niche")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cache = os.path.join(args.out_dir, f"agg_{args.cluster_col}.npz")

    pb, ncell, genes, animals, clusters, stage, score = build_or_load(args.cluster_col, cache)
    keep = ~np.isnan(score)
    pb, ncell, stage, score, animals = pb[keep], ncell[keep], stage[keep], score[keep], animals[keep]

    is_pr = np.isin(stage, APEX + DESC)
    y_all = np.isin(stage, APEX).astype(int)                # 1=peak, 0=remission
    print(f"[setup] {args.cluster_col}: {len(clusters)} niches | "
          f"{is_pr.sum()} peak/rem animals ({y_all[is_pr].sum()} peak / {(1-y_all[is_pr]).sum()} rem)")

    # ---- per-niche TEST A ----
    rows = []
    for ci, cl in enumerate(clusters):
        enough = ncell[:, ci] >= MIN_CELLS
        m = is_pr & enough
        y = y_all[m]
        if (y == 1).sum() < MIN_PER_CLASS or (y == 0).sum() < MIN_PER_CLASS:
            rows.append({"niche": cl, "n_peak": int((y == 1).sum()), "n_rem": int((y == 0).sum()),
                         "auc": None, "perm_p": None, "skipped": "too few animals w/ cells"})
            continue
        X = pb[m, ci, :]
        Xr = residualize(X, score[m])
        k = min(args.n_pc, Xr.shape[0] - 1)
        pcs = PCA(n_components=k, random_state=0).fit_transform(
            (Xr - Xr.mean(0)) / (Xr.std(0) + 1e-9))
        auc = loao_auc(pcs, y)
        rows.append({"niche": cl, "n_peak": int((y == 1).sum()), "n_rem": int((y == 0).sum()),
                     "auc": round(float(auc), 3), "perm_p": None, "n_pcs": k,
                     "_pcs": pcs, "_y": y})
    tested = [r for r in rows if r.get("auc") is not None]
    n_tested = len(tested)

    # permutation null only for screened-in niches (expensive)
    for r in tested:
        if r["auc"] >= args.screen_auc:
            r["perm_p"] = round(float(perm_null(r["_pcs"], r["_y"], r["auc"], args.B)), 4)
    for r in rows:
        r.pop("_pcs", None); r.pop("_y", None)

    bonf = 0.05 / max(1, n_tested)
    hits = [r for r in tested if r["perm_p"] is not None and r["perm_p"] < bonf]

    # ---- composition test: do niche FRACTIONS separate peak vs rem beyond severity? ----
    frac = ncell / ncell.sum(1, keepdims=True)
    mc = is_pr
    fr = residualize(frac[mc], score[mc])
    comp_auc = loao_auc((fr - fr.mean(0)) / (fr.std(0) + 1e-9), y_all[mc])
    comp_p = perm_null((fr - fr.mean(0)) / (fr.std(0) + 1e-9), y_all[mc], comp_auc, args.B)

    results = {
        "cluster_col": args.cluster_col, "n_niches": len(clusters), "n_tested": n_tested,
        "bonferroni_alpha": round(bonf, 4), "min_cells": MIN_CELLS,
        "per_niche": [{k: v for k, v in r.items()} for r in rows],
        "composition_test": {"residualized_auc": round(float(comp_auc), 3),
                             "perm_p": round(float(comp_p), 4)},
        "hits": [r["niche"] for r in hits],
    }

    # ---- report ----
    L = [f"\n=== Niche-resolved TEST A — {args.cluster_col} (peak vs remission, severity-residualized) ===",
         f"{len(clusters)} niches, {n_tested} testable (>= {MIN_CELLS} cells & >= {MIN_PER_CLASS}/class).",
         f"Bonferroni alpha = {bonf:.4f}. Perm null (B={args.B}) run only for AUC >= {args.screen_auc}.",
         "\n  niche            n_pk/n_rem   AUC     perm_p"]
    for r in sorted(tested, key=lambda x: -x["auc"]):
        pp = f"{r['perm_p']:.4f}" if r["perm_p"] is not None else "  -  "
        flag = "  <== HIT" if (r["perm_p"] is not None and r["perm_p"] < bonf) else ""
        L.append(f"  {r['niche']:<14} {r['n_peak']:>3}/{r['n_rem']:<3}     {r['auc']:.3f}   {pp}{flag}")
    skipped = [r for r in rows if r.get("auc") is None]
    if skipped:
        L.append(f"\n  skipped ({len(skipped)}): " + ", ".join(r["niche"] for r in skipped))
    L.append(f"\n-- Composition test (niche fractions, severity-residualized) --")
    L.append(f"   AUC = {comp_auc:.3f}  perm_p = {comp_p:.4f}")
    verdict = (f"PHASE FOUND in {len(hits)} niche(s): {results['hits']} — single-cell rescues the model angle"
               if hits else
               "NULL — no niche carries phase beyond severity. Bulk null confirmed at single-cell resolution.")
    L.append(f"\n=== NICHE GATE: {verdict} ===")
    report = "\n".join(L)
    print(report)

    with open(os.path.join(args.out_dir, f"report_{args.cluster_col}.txt"), "w") as fh:
        fh.write(report + "\n")
    with open(os.path.join(args.out_dir, f"results_{args.cluster_col}.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[done] -> {args.out_dir}/ (results_{args.cluster_col}.json)")


if __name__ == "__main__":
    main()
