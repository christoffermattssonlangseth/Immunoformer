"""Data pass for the closing block (tasks A, B, C). One pass over sections:
  * re-runs the lesionSegmenter port (analysis/lesion_segment.py, identical settings) and
    gives every cell the lesionSegmenter signed distance to the lesion edge (um; negative
    inside lesions);
  * sums raw counts per section for four compartments: lesion (distance <= 0) and
    non-lesion at margins 0 / 50 / 150 um (distance > margin);
  * counts, per Anno_L1_curated type, cells with >= 1 count of every gene (task A4 filter);
  * keeps per-cell total_counts / n_genes / cell types / section (task A6) and Anno_L2
    counts per section (task C).
Section C2_G3_Mid_1 excluded.

    PYTHONPATH="$PWD:$PWD/scripts" python analysis/closing_block_data.py
"""

from __future__ import annotations

import immunotransformer  # noqa: F401  (OpenMP guard — must precede numpy; see HANDOFF.md)

import os
import sys
import time

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clock_composition as CC  # noqa: E402
import lesion_segment as LS  # noqa: E402
from lesionseg.density import Grid, compute_density_maps  # noqa: E402
from lesionseg.lesion import detect_lesions  # noqa: E402

OUT = "runs/closing_block"
CACHE = os.path.join(OUT, "data.npz")
MARGINS = [0, 50, 150]
COMPARTMENTS = ["lesion"] + [f"nonlesion_m{m}" for m in MARGINS]


def signed_distance(x, y, pu1):
    x0, y0 = x.min(), y.min()
    cells = pd.DataFrame({"x_um": x - x0, "y_um": y - y0, "pu1_pos": pu1})
    grid = Grid.for_scene(int(np.ceil(cells.x_um.max())) + 1, int(np.ceil(cells.y_um.max())) + 1,
                          1.0, LS.BIN_UM)
    maps = compute_density_maps(cells, grid, sigma_um=LS.SIGMA_UM, tissue_source="cells",
                                tissue_params=LS.TISSUE)
    res = detect_lesions(maps, **LS.LESION)
    gx = np.clip((cells.x_um / LS.BIN_UM).astype(int), 0, grid.shape[1] - 1)
    gy = np.clip((cells.y_um / LS.BIN_UM).astype(int), 0, grid.shape[0] - 1)
    if res.lesion_labels.max() == 0:
        return np.full(len(x), np.inf)
    return res.signed_distance_um[gy, gx].astype(float)


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    with h5py.File(CC.ATLAS, "r") as f:
        sc_, scats = LS._obs(f, "meta_sample_id")
        an_, acats = LS._obs(f, "sample_name")
        c1_, c1cats = LS._obs(f, "Anno_L1_curated")
        c2_, c2cats = LS._obs(f, "Anno_L2")
        xs, _ = LS._obs(f, "x_centroid")
        ys, _ = LS._obs(f, "y_centroid")
        tc, _ = LS._obs(f, "total_counts")
        ngn, _ = LS._obs(f, "n_genes")
        var = f["var"]
        genes = np.array([x.decode() if isinstance(x, bytes) else x for x in var[var.attrs["_index"]][:]])
        C = f["layers/counts"]
        indptr = C["indptr"][:]
        secs = [s for s in scats if s not in CC.EXCLUDE_SECTIONS]
        comp = np.zeros((len(secs), len(COMPARTMENTS), len(genes)), np.float32)
        comp_n = np.zeros((len(secs), len(COMPARTMENTS)), np.int64)
        pos = np.zeros((len(c1cats), len(genes)))
        n1 = np.zeros(len(c1cats))
        l2 = np.zeros((len(secs), len(c2cats)), np.int64)
        sec_animal = []
        for k, s in enumerate(secs):
            rows = np.where(sc_ == list(scats).index(s))[0]
            blocks = np.split(rows, np.where(np.diff(rows) != 1)[0] + 1)
            X = sp.vstack([sp.csr_matrix((C["data"][indptr[b[0]]:indptr[b[-1] + 1]],
                                          C["indices"][indptr[b[0]]:indptr[b[-1] + 1]],
                                          indptr[b[0]:b[-1] + 2] - indptr[b[0]]),
                                         shape=(len(b), len(genes))) for b in blocks]).tocsr()
            pu1 = np.isin(c1cats[c1_[rows]], LS.PU1_LABELS)
            d = signed_distance(xs[rows], ys[rows], pu1)
            masks = [d <= 0] + [d > m for m in MARGINS]
            for j, m in enumerate(masks):
                comp[k, j] = np.asarray(X[m].sum(0)).ravel()
                comp_n[k, j] = int(m.sum())
            T = sp.csr_matrix((np.ones(len(rows)), (c1_[rows], np.arange(len(rows)))),
                              shape=(len(c1cats), len(rows)))
            pos += (T @ (X > 0).astype(float)).toarray()
            n1 += np.asarray(T.sum(1)).ravel()
            l2[k] = np.bincount(c2_[rows], minlength=len(c2cats))
            sec_animal.append(acats[an_[rows[0]]])
            print(f"  [{k + 1}/{len(secs)}] {s}: {len(rows):,} cells, lesion {masks[0].sum():,}, "
                  f"non-lesion(50) {masks[2].sum():,}  {time.time() - t0:.0f}s", flush=True)
        keep = ~np.isin(scats[sc_], list(CC.EXCLUDE_SECTIONS))
        cells = pd.DataFrame({"section": scats[sc_][keep], "animal": acats[an_][keep],
                              "celltype": c1cats[c1_][keep], "total_counts": tc[keep],
                              "n_genes": ngn[keep]})
    cells.to_parquet(os.path.join(OUT, "cells_qc.parquet"), index=False)
    np.savez(CACHE, comp=comp, comp_n=comp_n, compartments=np.array(COMPARTMENTS),
             sections=np.array(secs), section_animal=np.array(sec_animal), genes=genes,
             pos=pos, n1=n1, c1cats=c1cats, l2=l2, c2cats=c2cats)
    print(f"[done] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
