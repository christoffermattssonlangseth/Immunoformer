"""AnnData -> section bags.

One *bag* = all cells of one section, carrying a single section/animal-level
label. We normalize counts, optionally HVG-filter, fit a frozen encoder on the
TRAIN cells only (no leakage), and serve bags of encoded cells to the MIL model.

The train/val split is over ANIMALS, not sections or cells — this is the single
most important guard against inflated metrics in this dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scanpy as sc
import anndata as ad
import torch
from torch.utils.data import Dataset

from .config import Config
from .encoders import build_encoder, CellEncoder


@dataclass
class Bag:
    section_id: str
    animal_id: str
    label: int
    cell_idx: np.ndarray   # row indices into the (encoded) cell matrix


def _to_dense(X) -> np.ndarray:
    return np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)


def normalize_expression(adata: ad.AnnData, layer: Optional[str]) -> np.ndarray:
    """Library-size normalize + log1p. Operates on a copy of counts."""
    counts = adata.layers[layer] if layer else adata.X
    tmp = ad.AnnData(X=counts.copy())
    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)
    return _to_dense(tmp.X).astype(np.float32)


def build_bags(adata: ad.AnnData, cfg: Config) -> list[Bag]:
    obs = adata.obs
    o = cfg.obs
    label_map = {name: i for i, name in enumerate(o.label_order)}

    sec_key = o.section_id if o.section_id in obs.columns else o.animal_id
    if o.label not in obs.columns:
        raise KeyError(f"label column '{o.label}' not in adata.obs")

    bags: list[Bag] = []
    for sec, idx in obs.groupby(sec_key, observed=True).indices.items():
        idx = np.asarray(idx)
        if len(idx) < cfg.data.min_cells_per_bag:
            continue
        sub = obs.iloc[idx]
        lab_raw = sub[o.label].iloc[0]
        if lab_raw not in label_map:
            continue  # label not in the declared ordinal order -> skip
        animal = sub[o.animal_id].iloc[0]
        bags.append(Bag(str(sec), str(animal), label_map[lab_raw], idx))
    if not bags:
        raise RuntimeError("No bags built — check obs schema / label_order.")
    return bags


def split_animals(bags: list[Bag], val_fraction: float, seed: int):
    """Group split: every animal is entirely in train OR val."""
    animals = sorted({b.animal_id for b in bags})
    rng = np.random.default_rng(seed)
    rng.shuffle(animals)
    n_val = max(1, round(len(animals) * val_fraction))
    val_animals = set(animals[:n_val])
    train = [b for b in bags if b.animal_id not in val_animals]
    val = [b for b in bags if b.animal_id in val_animals]
    return train, val, val_animals


def fit_encoder_on_train(
    X: np.ndarray, train_bags: list[Bag], cfg: Config
) -> tuple[CellEncoder, np.ndarray]:
    """Fit the frozen encoder on TRAIN cells only, then encode all cells."""
    train_rows = np.concatenate([b.cell_idx for b in train_bags])
    enc = build_encoder(cfg.data.encoder, pca_dim=cfg.data.pca_dim, seed=cfg.data.seed)
    enc.fit(X[train_rows])
    X_enc = enc.transform(X)
    return enc, X_enc


class BagDataset(Dataset):
    """Yields (cells[N, D] float tensor, label int) for one section.

    Cells are randomly subsampled to `max_cells_per_bag` each access, so every
    epoch sees a different view of large sections (a cheap augmentation).
    """

    def __init__(self, bags: list[Bag], X_enc: np.ndarray, cfg: Config, train: bool):
        self.bags = bags
        self.X = X_enc
        self.cap = cfg.data.max_cells_per_bag
        self.train = train
        self._rng = np.random.default_rng(cfg.data.seed + (1 if train else 2))

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, i: int):
        b = self.bags[i]
        idx = b.cell_idx
        if len(idx) > self.cap:
            # random subsample in train; deterministic head slice in val
            if self.train:
                idx = self._rng.choice(idx, self.cap, replace=False)
            else:
                idx = idx[: self.cap]
        cells = torch.from_numpy(self.X[idx])           # [N, D]
        return cells, b.label, b.section_id


def collate_single(batch):
    """batch_size is always 1 bag (variable N). Returns the single item."""
    cells, label, sec = batch[0]
    return cells, torch.tensor(label, dtype=torch.long), sec
