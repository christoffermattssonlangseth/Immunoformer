"""Generate a small synthetic Xenium-like AnnData to smoke-test the pipeline.

Injects a genuine timepoint signal (a subset of genes ramps with stage, in a
stage-dependent fraction of cells) so a working model should reach rho > 0 on
held-out animals. This is a stand-in ONLY — replace with your real .h5ad.

    python scripts/make_synthetic.py --out data/synthetic.h5ad
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import anndata as ad
import pandas as pd


STAGES = ["control", "pre_onset", "onset", "peak", "late_1", "late_2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic.h5ad")
    ap.add_argument("--genes", type=int, default=300)
    ap.add_argument("--animals-per-stage", type=int, default=4)
    ap.add_argument("--sections-per-animal", type=int, default=2)
    ap.add_argument("--cells-per-section", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    # separate stream for coordinates, so X is unchanged from earlier versions
    rng_xy = np.random.default_rng(args.seed + 1000)
    n_sig = max(5, args.genes // 20)          # signal genes
    sig = np.arange(n_sig)

    rows_X, xy_rows, animal_col, section_col, stage_col, region_col = [], [], [], [], [], []
    regions = ["lumbar", "thoracic", "cervical"]

    for s_idx, stage in enumerate(STAGES):
        frac = s_idx / (len(STAGES) - 1)       # 0..1 disease severity
        for a in range(args.animals_per_stage):
            animal = f"{stage}_m{a}"
            for sec in range(args.sections_per_animal):
                section = f"{animal}_s{sec}"
                n = args.cells_per_section
                base = rng.poisson(1.0, size=(n, args.genes)).astype(np.float32)
                # a stage-dependent fraction of cells upregulate signal genes
                n_aff = int(frac * n * 0.5)
                # coordinates (um): background cells uniform over the section; affected
                # cells cluster around two lesion centres, so neighbourhoods carry signal
                xy = rng_xy.uniform(0, 1000, size=(n, 2))
                if n_aff:
                    aff = rng.choice(n, n_aff, replace=False)
                    base[np.ix_(aff, sig)] += rng.poisson(
                        3.0 + 5.0 * frac, size=(n_aff, n_sig))
                    centres = rng_xy.uniform(200, 800, size=(2, 2))
                    xy[aff] = (centres[rng_xy.integers(0, 2, n_aff)]
                               + rng_xy.normal(0, 60, size=(n_aff, 2)))
                rows_X.append(base)
                xy_rows.append(xy.astype(np.float32))
                animal_col += [animal] * n
                section_col += [section] * n
                stage_col += [stage] * n
                region_col += list(rng.choice(regions, n))

    X = np.vstack(rows_X)
    obs = pd.DataFrame({
        "animal_id": animal_col,
        "section_id": section_col,
        "timepoint": pd.Categorical(stage_col, categories=STAGES, ordered=True),
        "region": region_col,
        "model": "synthetic_eae",
    })
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(args.genes)])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = np.vstack(xy_rows)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    adata.write_h5ad(args.out)
    print(f"[ok] wrote {adata.n_obs} cells x {adata.n_vars} genes -> {args.out}")
    print(f"     {obs.section_id.nunique()} sections, {obs.animal_id.nunique()} animals")


if __name__ == "__main__":
    main()
