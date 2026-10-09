"""TASK 2 — spatial context without predefined niches (BANKSY-style).

No niche labels and no clustering. For every RR cell, its k nearest spatial neighbours
within the same section (k = 10, 30; self excluded) define a neighbourhood expression
vector N_i = mean_{j in kNN(i)} x_j (x = per-cell log CP10k over all 5101 genes).
Two animal-level feature matrices per k, aggregated per cell type (Anno_L1_curated):
  NB   = mean over cells of type t of N_i          (what surrounds type-t cells)
  DISC = mean over cells of type t of |x_i - N_i|  (how unlike its surroundings a cell is;
         the signed version is a linear combination of own- and NB-means, so it is not used)
NB is computed exactly as (T A) X per section (T: type indicator, A: row-normalised kNN
adjacency) — no per-cell neighbourhood matrix is materialised; DISC is computed in chunks.
Cell types are kept if every RR animal has >= MIN_CELLS cells of that type, so no
imputation is needed. Each matrix — alone and concatenated with pseudobulk — is fed to the
unchanged duration clock (RR, day_of_sacrifice, severity-residualised LOAO), with the
zero-feature fold audit.

    PYTHONPATH="$PWD:$PWD/scripts" python analysis/spatial_context.py
"""

from __future__ import annotations

import immunotransformer  # noqa: F401  (OpenMP guard — must precede numpy; see HANDOFF.md)

import json
import os
import sys
import time
import warnings

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clock_composition as CC  # noqa: E402
from immunotransformer.stats import compare_rhos  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
OUT = "runs/spatial_context"
CACHE = os.path.join(OUT, "context_features.npz")
KS = [10, 30]
MIN_CELLS = 20
CHUNK = 4000


def _obs(f, k):
    g = f["obs"][k]
    if isinstance(g, h5py.Group):
        cats = np.array([c.decode() if isinstance(c, bytes) else c for c in g["categories"][:]])
        return g["codes"][:], cats
    return g[:], None


def build(rr_sections):
    """Per (section, cell type): n cells, sum of x, sum of N_k, sum of |x - N_k| for each k."""
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        return {k: d[k] for k in d.files}
    with h5py.File(CC.ATLAS, "r") as f:
        sc_, scats = _obs(f, "meta_sample_id")
        ct_, ccats = _obs(f, "Anno_L1_curated")
        xs, _ = _obs(f, "x_centroid")
        ys, _ = _obs(f, "y_centroid")
        C = f["layers/counts"]
        indptr = C["indptr"][:]
        ng = int(C.attrs["shape"][1])
        secs = [s for s in scats if s in rr_sections]
        nct = len(ccats)
        n_cells = np.zeros((len(secs), nct))
        own = np.zeros((len(secs), nct, ng), np.float32)
        nb = {k: np.zeros((len(secs), nct, ng), np.float32) for k in KS}
        disc = {k: np.zeros((len(secs), nct, ng), np.float32) for k in KS}
        t0 = time.time()
        for si, s in enumerate(secs):
            rows = np.where(sc_ == list(scats).index(s))[0]
            blocks = np.split(rows, np.where(np.diff(rows) != 1)[0] + 1)
            parts = []
            for b in blocks:
                lo, hi = indptr[b[0]], indptr[b[-1] + 1]
                parts.append(sp.csr_matrix((C["data"][lo:hi], C["indices"][lo:hi],
                                            indptr[b[0]:b[-1] + 2] - lo), shape=(len(b), ng)))
            X = sp.vstack(parts).tocsr().astype(np.float32)
            lib = np.asarray(X.sum(1)).ravel(); lib[lib == 0] = 1
            X = sp.diags((1e4 / lib).astype(np.float32)) @ X
            X.data = np.log1p(X.data)
            X = X.tocsr()
            coords = np.c_[xs[rows], ys[rows]]
            T = sp.csr_matrix((np.ones(len(rows)), (ct_[rows], np.arange(len(rows)))),
                              shape=(nct, len(rows)))
            n_cells[si] = np.asarray(T.sum(1)).ravel()
            own[si] = (T @ X).toarray()
            nn = NearestNeighbors(n_neighbors=max(KS) + 1).fit(coords)
            _, idx = nn.kneighbors(coords)
            for k in KS:
                nbr = idx[:, 1:k + 1]                       # self excluded
                A = sp.csr_matrix((np.full(nbr.size, 1.0 / k, np.float32),
                                   (np.repeat(np.arange(len(rows)), k), nbr.ravel())),
                                  shape=(len(rows), len(rows)))
                nb[k][si] = ((T @ A) @ X).toarray()
                acc = np.zeros((nct, ng), np.float32)
                for a in range(0, len(rows), CHUNK):
                    e = min(a + CHUNK, len(rows))
                    Nc = (A[a:e] @ X).toarray()
                    D = np.abs(X[a:e].toarray() - Nc)
                    acc += (T[:, a:e] @ D)
                disc[k][si] = acc
            print(f"  [context] section {si + 1}/{len(secs)} ({len(rows):,} cells) "
                  f"{time.time() - t0:.0f}s", flush=True)
    out = {"sections": np.array(secs), "celltypes": ccats, "n_cells": n_cells, "own": own,
           **{f"nb{k}": nb[k] for k in KS}, **{f"disc{k}": disc[k] for k in KS}}
    os.makedirs(OUT, exist_ok=True)
    np.savez(CACHE, **out)
    return out


def animal_matrix(F, n_cells, sections, meta, animals, keep_ct):
    """Per-animal, per-kept-cell-type mean (sum over sections / cells), flattened."""
    sidx = {s: i for i, s in enumerate(sections)}
    rows = []
    for a in animals:
        si = [sidx[s] for s in meta.loc[meta.sample_name == a, "meta_sample_id"] if s in sidx]
        tot = F[si][:, keep_ct].sum(0)
        n = n_cells[si][:, keep_ct].sum(0)[:, None]
        rows.append((tot / n).ravel())
    return np.vstack(rows)


def main():
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    pb, info, meta = CC.load_rr()
    rr = info[info.model == CC.RR]
    rr_sections = set(meta.loc[meta.sample_name.isin(rr.index), "meta_sample_id"])
    d = build(rr_sections)
    sections, ccats, n_cells = d["sections"].astype(str), d["celltypes"].astype(str), d["n_cells"]
    sidx = {s: i for i, s in enumerate(sections)}
    per_animal_ct = np.vstack([n_cells[[sidx[s] for s in meta.loc[meta.sample_name == a,
                                                                   "meta_sample_id"]
                                        if s in sidx]].sum(0) for a in rr.index])
    keep_ct = np.where((per_animal_ct >= MIN_CELLS).all(0))[0]
    y, sev = rr.day.to_numpy(), rr.score.to_numpy().reshape(-1, 1)
    X_pb = pb.loc[rr.index].to_numpy()
    res = {"n": len(rr), "k": KS, "cell_types_kept": ccats[keep_ct].tolist(),
           "cell_types_dropped": ccats[np.setdiff1d(np.arange(len(ccats)), keep_ct)].tolist(),
           "min_cells_per_animal": MIN_CELLS, "conditions": []}
    base, _ = CC.evaluate("baseline pseudobulk (all genes)", X_pb, y, sev)
    res["conditions"].append(base)
    own = animal_matrix(d["own"], n_cells, sections, meta, rr.index, keep_ct)
    r, _ = CC.evaluate("own expression per cell type (reference)", own, y, sev, base)
    res["conditions"].append(r)
    for k in KS:
        for name in ("nb", "disc"):
            F = animal_matrix(d[f"{name}{k}"], n_cells, sections, meta, rr.index, keep_ct)
            label = {"nb": "neighbourhood mean", "disc": "|own - neighbourhood|"}[name]
            r, _ = CC.evaluate(f"k={k} {label} per cell type", F, y, sev, base)
            res["conditions"].append(r)
            r, _ = CC.evaluate(f"k={k} pseudobulk + {label}", np.hstack([X_pb, F]), y, sev, base)
            res["conditions"].append(r)
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    L = ["# Task 2 — spatial context (machine table; see WRITEUP.md)", "",
         f"Cell types kept (>= {MIN_CELLS} cells in every RR animal): {res['cell_types_kept']}",
         f"Dropped: {res['cell_types_dropped']}", "",
         "| condition | n | features in | rho | 95% CI | Fisher z p vs baseline | zero-feature folds |",
         "|---|---|---|---|---|---|---|"] + [CC._fmt(r) for r in res["conditions"]]
    with open(os.path.join(OUT, "report.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
