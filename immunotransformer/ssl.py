"""Self-supervised cell + niche pretraining on unlabelled Xenium cells.

Why: supervised models here are capped by the number of animals (~30 per cohort), not
cells. Pretraining uses every cell (1.38M) and no disease label, so that cap does not
apply; the animal-level clock on top stays a small linear model (baseline ladder arm 7).

Model (one forward = one centre cell + its k spatial nearest neighbours, same section):
  cell encoder    MLP, expression -> z_cell (d)
  niche encoder   small transformer over [centre token, k neighbour tokens]; each token is
                  cell-encoder(expression) + MLP(relative xy / pos_scale); the centre
                  token's output is z_niche (d)
Objectives:
  masked expression  hide a random fraction of the centre cell's genes, reconstruct them
                     from z_niche -> the niche encoder must use the neighbours
  niche prediction   predict the neighbours' mean expression from z_cell alone -> the
                     per-cell embedding is pushed to carry what its surroundings look like
Both are label-free. Expression is library-size normalised on the full panel, log1p, then
z-scored per gene with statistics from the training cells.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree


# ---------------------------------------------------------------- preprocessing

def normalize_log(X: sp.csr_matrix, target_sum: float = 1e4) -> sp.csr_matrix:
    """Per-cell library-size normalisation + log1p, on the full panel, staying sparse."""
    X = X.tocsr().astype(np.float32, copy=True)   # one copy; the input is left untouched
    lib = np.asarray(X.sum(1)).ravel()
    lib[lib == 0] = 1.0
    X.data *= np.repeat((target_sum / lib).astype(np.float32), np.diff(X.indptr))
    np.log1p(X.data, out=X.data)
    return X


def _subsample(rows: np.ndarray, n: int, rng) -> np.ndarray:
    return rows if len(rows) <= n else rng.choice(rows, n, replace=False)


def gene_stats(X: sp.csr_matrix, rows: np.ndarray, n: int = 100_000, seed: int = 0):
    """Per-gene mean and std of normalised expression on a subsample of `rows`."""
    S = X[_subsample(rows, n, np.random.default_rng(seed))]
    mean = np.asarray(S.mean(0)).ravel()
    sq = np.asarray(S.multiply(S).mean(0)).ravel()
    std = np.sqrt(np.maximum(sq - mean ** 2, 0))
    return mean.astype(np.float32), np.maximum(std, 1e-3).astype(np.float32)


def top_variance_genes(X: sp.csr_matrix, rows: np.ndarray, n_genes: int, seed: int = 0):
    """Unsupervised gene filter (no labels). n_genes <= 0 keeps every gene."""
    if n_genes <= 0 or n_genes >= X.shape[1]:
        return np.arange(X.shape[1])
    _, std = gene_stats(X, rows, seed=seed)
    return np.sort(np.argsort(-std)[:n_genes])


def knn_by_section(xy: np.ndarray, sections: np.ndarray, k: int) -> np.ndarray:
    """k nearest neighbours of every cell within its own section (self excluded).
    Sections with <= k cells repeat their farthest available neighbour as padding."""
    nbr = np.empty((len(xy), k), dtype=np.int64)
    for sec in np.unique(sections):
        rows = np.flatnonzero(sections == sec)
        kk = min(k + 1, len(rows))
        _, j = cKDTree(xy[rows]).query(xy[rows], k=kk)
        j = np.asarray(j).reshape(len(rows), kk)[:, 1:]
        if j.shape[1] == 0:          # single-cell section: the cell is its own neighbour
            j = np.zeros((len(rows), 1), dtype=np.int64)
        if j.shape[1] < k:
            j = np.hstack([j, np.repeat(j[:, -1:], k - j.shape[1], axis=1)])
        nbr[rows] = rows[j]
    return nbr


# ---------------------------------------------------------------- model

@dataclass
class SSLConfig:
    k: int = 16                 # spatial neighbours per centre cell
    dim: int = 64               # embedding size (z_cell and z_niche)
    hidden: int = 512
    layers: int = 2             # niche transformer depth
    heads: int = 4
    dropout: float = 0.1
    mask_frac: float = 0.4      # fraction of the centre cell's genes hidden
    w_niche: float = 0.5        # weight of the niche-prediction loss
    batch_size: int = 256
    steps: int = 20_000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0


class NicheSSL(nn.Module):
    def __init__(self, n_genes: int, cfg: SSLConfig):
        super().__init__()
        d, h = cfg.dim, cfg.hidden
        self.cell_enc = nn.Sequential(
            nn.Linear(n_genes, h), nn.GELU(), nn.LayerNorm(h), nn.Dropout(cfg.dropout),
            nn.Linear(h, d))
        self.pos = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.centre_tag = nn.Parameter(torch.zeros(d))
        layer = nn.TransformerEncoderLayer(d, cfg.heads, 2 * d, cfg.dropout,
                                           batch_first=True, norm_first=True)
        self.niche_enc = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.recon = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, n_genes))
        self.niche_head = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, n_genes))

    def niche(self, centre, neigh, rel):
        """centre [B, G], neigh [B, k, G], rel [B, k, 2] -> z_niche [B, d]."""
        B, k, G = neigh.shape
        tok_c = self.cell_enc(centre) + self.centre_tag + self.pos(torch.zeros_like(rel[:, 0]))
        tok_n = self.cell_enc(neigh.reshape(B * k, G)).reshape(B, k, -1) + self.pos(rel)
        h = self.niche_enc(torch.cat([tok_c[:, None], tok_n], 1))
        return self.norm(h[:, 0])

    def losses(self, centre, neigh, rel, mask_frac: float, w_niche: float):
        mask = torch.rand_like(centre) < mask_frac
        z_n = self.niche(centre.masked_fill(mask, 0.0), neigh, rel)
        l_rec = F.mse_loss(self.recon(z_n)[mask], centre[mask])
        z_c = self.cell_enc(centre)
        l_nic = F.mse_loss(self.niche_head(z_c), neigh.mean(1))
        return l_rec + w_niche * l_nic, {"masked_recon": l_rec.item(), "niche_pred": l_nic.item()}


# ---------------------------------------------------------------- batches

class CellBatcher:
    """Builds dense, z-scored (centre, neighbours, relative xy) tensors from the sparse matrix."""

    def __init__(self, X: sp.csr_matrix, nbr: np.ndarray, xy: np.ndarray,
                 mean: np.ndarray, std: np.ndarray, pos_scale: float, device):
        self.X, self.nbr, self.xy = X, nbr, xy
        self.mean = torch.from_numpy(mean).to(device)
        self.std = torch.from_numpy(std).to(device)
        self.pos_scale, self.device = pos_scale, device

    def __call__(self, centres: np.ndarray):
        B, k = len(centres), self.nbr.shape[1]
        nb = self.nbr[centres]
        rows = np.concatenate([centres, nb.ravel()])
        D = torch.from_numpy(self.X[rows].toarray()).to(self.device)
        D = (D - self.mean) / self.std
        rel = (self.xy[nb] - self.xy[centres][:, None]) / self.pos_scale
        rel = torch.from_numpy(rel.astype(np.float32)).to(self.device)
        return D[:B], D[B:].reshape(B, k, -1), rel


def pos_scale_from(xy: np.ndarray, nbr: np.ndarray, n: int = 50_000, seed: int = 0) -> float:
    """Median distance to the k-th neighbour: relative positions become unit-free."""
    c = _subsample(np.arange(len(xy)), n, np.random.default_rng(seed))
    d = np.linalg.norm(xy[nbr[c, -1]] - xy[c], axis=1)
    return float(np.median(d[d > 0])) if np.any(d > 0) else 1.0


# ---------------------------------------------------------------- train / embed

def pretrain(model: NicheSSL, batcher: CellBatcher, train_rows: np.ndarray,
             val_rows: np.ndarray, cfg: SSLConfig, log_every: int = 200, log=print):
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warm = max(1, cfg.steps // 50)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm)
        * 0.5 * (1 + math.cos(math.pi * min(s, cfg.steps) / cfg.steps)))
    val_fixed = _subsample(val_rows, 4096, np.random.default_rng(cfg.seed + 1))
    history, t0 = [], time.time()
    if len(val_fixed):  # untrained reference point
        val = evaluate_loss(model, batcher, val_fixed, cfg)
        history.append({"step": 0, **{f"val_{k}": v for k, v in val.items()}})
    for step in range(1, cfg.steps + 1):
        model.train()
        c = rng.choice(train_rows, cfg.batch_size, replace=len(train_rows) < cfg.batch_size)
        loss, parts = model.losses(*batcher(c), cfg.mask_frac, cfg.w_niche)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % log_every == 0 or step == cfg.steps:
            val = evaluate_loss(model, batcher, val_fixed, cfg) if len(val_fixed) else {}
            rec = {"step": step, "train_loss": loss.item(), **parts,
                   **{f"val_{k}": v for k, v in val.items()}, "sec": round(time.time() - t0, 1)}
            history.append(rec)
            log("  " + "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                                 for k, v in rec.items()))
    return history


@torch.no_grad()
def evaluate_loss(model, batcher, rows, cfg: SSLConfig, bs: int = 512):
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(cfg.seed + 2)   # same mask every call
    tot, n = {"masked_recon": 0.0, "niche_pred": 0.0}, 0
    for i in range(0, len(rows), bs):
        centre, neigh, rel = batcher(rows[i:i + bs])
        mask = (torch.rand(centre.shape, generator=g) < cfg.mask_frac).to(centre.device)
        z_n = model.niche(centre.masked_fill(mask, 0.0), neigh, rel)
        tot["masked_recon"] += F.mse_loss(model.recon(z_n)[mask], centre[mask]).item() * len(rel)
        tot["niche_pred"] += F.mse_loss(model.niche_head(model.cell_enc(centre)),
                                        neigh.mean(1)).item() * len(rel)
        n += len(rel)
    return {k: v / n for k, v in tot.items()}


@torch.no_grad()
def embed(model: NicheSSL, batcher: CellBatcher, rows: np.ndarray, bs: int = 256):
    """(z_cell, z_niche) for `rows`, float32 [n, d] each. No masking at inference."""
    model.eval()
    zc, zn = [], []
    for i in range(0, len(rows), bs):
        centre, neigh, rel = batcher(rows[i:i + bs])
        zc.append(model.cell_enc(centre).cpu().numpy())
        zn.append(model.niche(centre, neigh, rel).cpu().numpy())
    return np.vstack(zc), np.vstack(zn)


def group_means(Z: np.ndarray, groups: np.ndarray):
    """Mean of Z per group -> (sorted unique groups, means [n_groups, d], counts)."""
    keys, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    sums = np.zeros((len(keys), Z.shape[1]), dtype=np.float64)
    np.add.at(sums, inv, Z)
    return keys, (sums / counts[:, None]).astype(np.float32), counts


def config_dict(cfg: SSLConfig) -> dict:
    return asdict(cfg)
