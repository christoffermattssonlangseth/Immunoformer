"""Light .h5ad reading with h5py: counts matrix, a few obs columns, spatial coords.

`anndata.read_h5ad` on the RRMAP2 atlas materialises X, every layer and obsm at once
(tens of GB). Self-supervised pretraining needs one count matrix (kept sparse), the
animal/section keys and the coordinates, so read just those, straight from the file.
Each element is decoded with anndata's own `read_elem`, so every on-disk encoding
(sparse or dense matrices, categorical / nullable / plain obs columns) is handled.
"""
from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp


try:  # anndata >= 0.11
    from anndata.io import read_elem
except ImportError:  # anndata 0.10
    from anndata.experimental import read_elem


def read_obs_column(f: h5py.File, key: str) -> np.ndarray:
    return np.asarray(read_elem(f["obs"][key]), dtype=object)


def obs_names(f: h5py.File) -> np.ndarray:
    return np.asarray(read_elem(f["obs"][f["obs"].attrs.get("_index", "_index")]), dtype=object)


def var_names(f: h5py.File) -> np.ndarray:
    return np.asarray(read_elem(f["var"][f["var"].attrs.get("_index", "_index")]), dtype=object)


def read_matrix(f: h5py.File, layer: str | None) -> sp.csr_matrix:
    """Cells x genes as float32 CSR. `layer=None` reads X."""
    M = read_elem(f["X"] if layer is None else f["layers"][layer])
    M = sp.csr_matrix(M) if not sp.issparse(M) else M.tocsr()
    return M.astype(np.float32, copy=False)


def read_cells(path: str, *, layer: str | None, animal_key: str, section_key: str,
               spatial_key: str = "spatial", extra_obs: tuple[str, ...] = ()):
    """Returns (counts CSR, genes, obs DataFrame, xy[N, 2] float32)."""
    with h5py.File(path, "r") as f:
        if spatial_key not in f.get("obsm", {}):
            raise KeyError(f"obsm['{spatial_key}'] not in {path}; pretraining needs coordinates")
        cols = {"animal": animal_key, "section": section_key, **{k: k for k in extra_obs}}
        obs = pd.DataFrame({name: read_obs_column(f, key) for name, key in cols.items()},
                           index=obs_names(f))
        xy = np.asarray(f["obsm"][spatial_key][:, :2], dtype=np.float32)
        genes = var_names(f)
        X = read_matrix(f, layer)
    obs["animal"] = obs["animal"].astype(str)
    obs["section"] = obs["section"].astype(str)
    return X, genes, obs, xy
