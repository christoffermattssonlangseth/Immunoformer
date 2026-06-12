"""Inspect obs metadata for all three datasets before running the experiment.

Run this first to verify column names, label vocabularies, cell counts, and
animal/section counts so you can fill in the correct values in the YAML configs.

python scripts/inspect_labels.py \
    --rrmap2   data/rrmap2.h5ad \
    --optic    data/optic_nerve.h5ad \
    --mtdna    data/mtdna_dsb.h5ad
"""

from __future__ import annotations

import argparse
import anndata as ad
import pandas as pd
import numpy as np


def inspect(path: str, name: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {name}: {path}")
    print(f"{'=' * 60}")

    adata = ad.read_h5ad(path)
    print(f"  Shape: {adata.n_obs:,} cells  x  {adata.n_vars:,} genes")
    print(f"  Layers: {list(adata.layers.keys())}")
    print(f"  X dtype: {adata.X.dtype}  (sparse={hasattr(adata.X, 'toarray')})")
    print(f"\n  obs columns: {list(adata.obs.columns)}")

    for col in adata.obs.columns:
        n_unique = adata.obs[col].nunique()
        if n_unique <= 30:
            vals = sorted(adata.obs[col].dropna().unique())
            print(f"\n  [{col}]  ({n_unique} unique)")
            for v in vals:
                n = (adata.obs[col] == v).sum()
                print(f"    {str(v):<30} n={n:>7,}")
        else:
            print(f"\n  [{col}]  ({n_unique} unique values — too many to list)")

    # Summary: cells per animal per section for common key candidates
    print()
    for col in ["sample_name", "animal_id", "mouse_id"]:
        if col in adata.obs.columns:
            n_animals = adata.obs[col].nunique()
            print(f"  Candidate animal key '{col}': {n_animals} unique values")

    for col in ["meta_sample_id", "section_id", "sample_id"]:
        if col in adata.obs.columns:
            n_sec = adata.obs[col].nunique()
            med_cells = adata.obs.groupby(col, observed=True).size().median()
            print(
                f"  Candidate section key '{col}': {n_sec} sections, "
                f"median {med_cells:.0f} cells/section"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rrmap2",  default=None, help="Path to RRMAP2 h5ad")
    ap.add_argument("--optic",   default=None, help="Path to optic nerve h5ad")
    ap.add_argument("--mtdna",   default=None, help="Path to mtDNA-DSB h5ad")
    args = ap.parse_args()

    if not any([args.rrmap2, args.optic, args.mtdna]):
        ap.print_help()
        return

    if args.rrmap2:
        inspect(args.rrmap2, "RRMAP2 (relapse-remitting EAE, spinal cord)")
    if args.optic:
        inspect(args.optic, "Optic nerve EAE")
    if args.mtdna:
        inspect(args.mtdna, "mtDNA-DSB (inside-out model)")

    print("\n[done] Use the values above to fill in the label_order lists in configs/")


if __name__ == "__main__":
    main()
