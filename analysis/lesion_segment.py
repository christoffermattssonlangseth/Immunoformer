"""Lesion segmentation on Xenium, ported from lesionSegmenter (no IHC images needed).

lesionSegmenter outlines EAE lesions on IHC whole-slide scans from (i) nuclei density and
(ii) Pu.1+ myeloid density. Re-engineered inputs for Xenium:
  * nuclei          -> Xenium cell centroids (every segmented cell carries a nucleus)
  * Pu.1+ cells     -> PU1_SOURCE="label" (primary): Anno_L1_curated Myeloid or DC — the
                       lineage Pu.1 IHC stains; label-based, so not fooled by spillover
                       transcripts (Spi1 alone flags 12% of non-immune cells).
                       PU1_SOURCE="markers" (sensitivity run): >= PU1_MIN_MARKERS of Spi1
                       (= Pu.1), Csf1r, Aif1, Cd68, Itgam detected (myeloid/DC sensitivity
                       0.74, non-immune false-positive rate 0.11).
Everything downstream calls lesionSegmenter's own functions with its config values
(configs/eae_dvp.yaml): density grid 10 um, sigma 40 um; score = Pu.1+ density; robust z
>= 2.5 vs the section's tissue bins; myeloid gates (Pu.1+ fraction >= 0.15 per bin, >= 0.2
per lesion); min area 5000 um^2; smoothing 20 um; holes filled; rim 50 / peri 150 um.
Tissue mask from cell positions (sigma 30 um, >= 100 cells/mm^2, min area 5e4 um^2).

Lesion unit = lesion mask + a 50 um perilesional margin (signed distance 0-50 um outside
the edge), each margin bin assigned to its nearest lesion. Per lesion: summed raw counts
(lesion pseudobulk), area, cells, Pu.1+ fraction, cell-type composition (Anno_L1_curated,
DESCRIPTION / immune-fraction control only). Section C2_G3_Mid_1 excluded.

    PYTHONPATH="$PWD:$PWD/scripts:$LESIONSEG" python analysis/lesion_segment.py
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
from scipy import ndimage as ndi

sys.path.insert(0, os.path.expanduser(
    "~/work/karolinska_institutet/projects/lesionSegmenter"))
from lesionseg.density import Grid, compute_density_maps  # noqa: E402
from lesionseg.lesion import detect_lesions  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clock_composition as CC  # noqa: E402

PU1_SOURCE = os.environ.get("PU1_SOURCE", "label")          # label | markers
PU1_LABELS = ["Myeloid", "DC"]
OUT = "runs/lesion_clock" if PU1_SOURCE == "label" else "runs/lesion_clock/sensitivity_markers"
CACHE = os.path.join(OUT, "lesions.npz")
PU1_MARKERS = ["Spi1", "Csf1r", "Aif1", "Cd68", "Itgam"]
PU1_MIN_MARKERS = 3
MARGIN_UM = 50.0
TISSUE = dict(sigma_um=30.0, min_cells_per_mm2=100.0, min_area_um2=5e4, hole_area_um2=2e5)
LESION = dict(score="pu1_density", threshold={"type": "zscore", "value": 2.5}, min_area_um2=5000.0,
              fill_holes=True, smooth_um=20.0, rim_width_um=50.0, peri_width_um=150.0,
              core_method="distance", min_pu1_fraction=0.15, min_lesion_pu1_fraction=0.2,
              dense_nonmyeloid_z=2.0)
BIN_UM, SIGMA_UM = 10.0, 40.0


def _obs(f, k):
    g = f["obs"][k]
    if isinstance(g, h5py.Group):
        return g["codes"][:], np.array([c.decode() if isinstance(c, bytes) else c
                                        for c in g["categories"][:]])
    return g[:], None


def segment_section(x, y, pu1):
    """Run lesionSegmenter on one section's cells. Returns per-cell lesion id (-1 = none),
    per-cell zone, and per-lesion area (um^2) of the lesion mask."""
    x0, y0 = x.min(), y.min()
    cells = pd.DataFrame({"x_um": x - x0, "y_um": y - y0, "pu1_pos": pu1})
    grid = Grid.for_scene(int(np.ceil(cells.x_um.max())) + 1, int(np.ceil(cells.y_um.max())) + 1,
                          1.0, BIN_UM)
    maps = compute_density_maps(cells, grid, sigma_um=SIGMA_UM, tissue_source="cells",
                                tissue_params=TISSUE)
    res = detect_lesions(maps, **LESION)
    lab = res.lesion_labels
    # perilesional margin: bins within MARGIN_UM outside the edge -> nearest lesion
    if lab.max() > 0:
        dist, (iy, ix) = ndi.distance_transform_edt(lab == 0, sampling=BIN_UM, return_indices=True)
        ext = np.where((lab == 0) & (dist <= MARGIN_UM) & maps.tissue, lab[iy, ix], 0)
        unit = np.where(lab > 0, lab, ext)
    else:
        unit = lab
    gx = np.clip((cells.x_um / BIN_UM).astype(int), 0, grid.shape[1] - 1)
    gy = np.clip((cells.y_um / BIN_UM).astype(int), 0, grid.shape[0] - 1)
    cell_lesion = unit[gy, gx].astype(int) - 1
    in_mask = lab[gy, gx] > 0
    areas = np.bincount(lab.ravel(), minlength=lab.max() + 1)[1:] * BIN_UM ** 2
    tissue_area = float(maps.tissue.sum() * BIN_UM ** 2)
    return cell_lesion, in_mask, areas, tissue_area


def main():
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    t0 = time.time()
    with h5py.File(CC.ATLAS, "r") as f:
        sc_, scats = _obs(f, "meta_sample_id")
        an_, acats = _obs(f, "sample_name")
        ct_, ccats = _obs(f, "Anno_L1_curated")
        xs, _ = _obs(f, "x_centroid")
        ys, _ = _obs(f, "y_centroid")
        var = f["var"]
        genes = np.array([x.decode() if isinstance(x, bytes) else x
                          for x in var[var.attrs["_index"]][:]])
        mk = [int(np.where(genes == g)[0][0]) for g in PU1_MARKERS]
        C = f["layers/counts"]
        indptr = C["indptr"][:]
        lesion_rows, lesion_counts, lesion_ct, section_rows = [], [], [], []
        for si, s in enumerate(scats):
            if s in CC.EXCLUDE_SECTIONS:
                continue
            rows = np.where(sc_ == si)[0]
            blocks = np.split(rows, np.where(np.diff(rows) != 1)[0] + 1)
            X = sp.vstack([sp.csr_matrix((C["data"][indptr[b[0]]:indptr[b[-1] + 1]],
                                          C["indices"][indptr[b[0]]:indptr[b[-1] + 1]],
                                          indptr[b[0]:b[-1] + 2] - indptr[b[0]]),
                                         shape=(len(b), len(genes))) for b in blocks]).tocsr()
            if PU1_SOURCE == "label":
                pu1 = np.isin(ccats[ct_[rows]], PU1_LABELS)
            else:
                pu1 = np.asarray((X[:, mk] > 0).sum(1)).ravel() >= PU1_MIN_MARKERS
            cl, in_mask, areas, tissue_area = segment_section(xs[rows], ys[rows], pu1)
            animal = acats[an_[rows[0]]]
            section_rows.append({"section": s, "animal": animal, "n_cells": len(rows),
                                 "n_pu1": int(pu1.sum()), "tissue_area_um2": tissue_area,
                                 "n_lesions": int(len(areas)),
                                 "lesion_area_um2": float(areas.sum())})
            for L in range(len(areas)):
                m = cl == L
                if m.sum() == 0:
                    continue
                lesion_rows.append({"section": s, "animal": animal, "lesion": L,
                                    "area_um2": float(areas[L]), "n_cells": int(m.sum()),
                                    "n_cells_mask": int((m & in_mask).sum()),
                                    "pu1_fraction": float(pu1[m].mean()),
                                    "cells_per_mm2": float((m & in_mask).sum() / (areas[L] / 1e6))})
                lesion_counts.append(np.asarray(X[m].sum(0)).ravel().astype(np.float32))
                lesion_ct.append(np.bincount(ct_[rows][m], minlength=len(ccats)))
            print(f"  [{si + 1}/{len(scats)}] {s} ({animal}): {len(rows):,} cells, "
                  f"{len(areas)} lesions, {time.time() - t0:.0f}s", flush=True)
    les = pd.DataFrame(lesion_rows)
    np.savez(CACHE, counts=np.vstack(lesion_counts) if lesion_counts else np.zeros((0, len(genes))),
             celltype_counts=np.vstack(lesion_ct) if lesion_ct else np.zeros((0, len(ccats))),
             genes=genes, celltypes=ccats)
    les.to_csv(os.path.join(OUT, "lesions.csv"), index=False)
    pd.DataFrame(section_rows).to_csv(os.path.join(OUT, "sections.csv"), index=False)
    with open(os.path.join(OUT, "segmentation_params.json"), "w") as fh:
        json.dump({"pu1_source": PU1_SOURCE, "pu1_labels": PU1_LABELS, "pu1_markers": PU1_MARKERS, "pu1_min_markers": PU1_MIN_MARKERS,
                   "margin_um": MARGIN_UM, "bin_um": BIN_UM, "sigma_um": SIGMA_UM,
                   "tissue": TISSUE, "lesion": LESION}, fh, indent=2)
    print(f"[done] {len(les)} lesions in {les.section.nunique() if len(les) else 0} sections")


if __name__ == "__main__":
    main()
