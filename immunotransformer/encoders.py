"""Pluggable per-cell encoders.

Stage 1 keeps the encoder *frozen* and cheap: we turn each cell's expression
vector into a fixed embedding once, then only the MIL head trains. Swapping in a
real foundation model (Nicheformer / scGPT) later means implementing the same
`fit` / `transform` interface in `FrozenFMEncoder` — nothing else changes.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


class CellEncoder:
    """Interface: fit on a training cell matrix, transform any cell matrix."""
    out_dim: int

    def fit(self, X: np.ndarray) -> "CellEncoder":
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class IdentityEncoder(CellEncoder):
    """Use the (normalized) expression vector directly. out_dim = n_genes."""

    def fit(self, X: np.ndarray) -> "IdentityEncoder":
        self.out_dim = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(X, dtype=np.float32)


class PCAEncoder(CellEncoder):
    """Linear PCA fit on training cells. Cheap, strong baseline, deterministic."""

    def __init__(self, n_components: int = 64, seed: int = 0):
        self.n_components = n_components
        self.seed = seed

    def fit(self, X: np.ndarray) -> "PCAEncoder":
        n_comp = min(self.n_components, X.shape[1], X.shape[0])
        self.pca = PCA(n_components=n_comp, random_state=self.seed)
        self.pca.fit(X)
        self.out_dim = n_comp
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(X).astype(np.float32)


class FrozenFMEncoder(CellEncoder):
    """Placeholder for a frozen foundation-model encoder (Nicheformer/scGPT).

    Implement `fit` (usually a no-op — the FM is already pretrained) and
    `transform` (map the 5101-gene panel to the FM's vocabulary, forward through
    the frozen model, return the cell embedding). Kept as a stub so the rest of
    the pipeline is FM-ready without pulling the heavy dependency now.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X: np.ndarray) -> "FrozenFMEncoder":
        raise NotImplementedError(
            "Wire up Nicheformer/scGPT here. See docs/ 'Backbone decision'."
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def build_encoder(name: str, *, pca_dim: int = 64, seed: int = 0) -> CellEncoder:
    name = name.lower()
    if name == "identity":
        return IdentityEncoder()
    if name == "pca":
        return PCAEncoder(n_components=pca_dim, seed=seed)
    if name == "fm":
        return FrozenFMEncoder()
    raise ValueError(f"Unknown encoder '{name}' (identity|pca|fm)")
