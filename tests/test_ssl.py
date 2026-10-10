"""Fast checks for the self-supervised pretraining pieces (CPU, a few seconds)."""
import immunotransformer  # noqa: F401  (OpenMP guard — must precede numpy)

import numpy as np
import scipy.sparse as sp
import torch

from immunotransformer import ssl
from immunotransformer.stats import paired_bootstrap_delta


def _toy(n_sec=3, n=80, g=20, seed=0):
    rng = np.random.default_rng(seed)
    X = sp.csr_matrix(rng.poisson(1.0, size=(n_sec * n, g)).astype(np.float32))
    xy = rng.uniform(0, 100, size=(n_sec * n, 2)).astype(np.float32)
    sections = np.repeat([f"s{i}" for i in range(n_sec)], n)
    return X, xy, sections


def test_knn_stays_within_section_and_excludes_self():
    _, xy, sections = _toy()
    nbr = ssl.knn_by_section(xy, sections, k=8)
    assert nbr.shape == (len(xy), 8)
    assert (sections[nbr] == sections[:, None]).all()
    assert not (nbr == np.arange(len(xy))[:, None]).any()


def test_knn_pads_small_sections():
    xy = np.array([[0, 0], [1, 0], [5, 5]], dtype=np.float32)
    nbr = ssl.knn_by_section(xy, np.array(["a", "a", "b"]), k=4)
    assert (nbr[0] == 1).all() and (nbr[1] == 0).all() and (nbr[2] == 2).all()


def test_normalize_log_matches_dense():
    X, _, _ = _toy()
    D = X.toarray()
    ref = np.log1p(D / D.sum(1, keepdims=True) * 1e4)
    assert np.allclose(ssl.normalize_log(X).toarray(), ref, atol=1e-5)


def test_pretrain_and_embed_shapes():
    X, xy, sections = _toy()
    X = ssl.normalize_log(X)
    cfg = ssl.SSLConfig(k=4, dim=8, hidden=16, steps=5, batch_size=16)
    rows = np.arange(X.shape[0])
    mean, std = ssl.gene_stats(X, rows)
    nbr = ssl.knn_by_section(xy, sections, cfg.k)
    b = ssl.CellBatcher(X, nbr, xy, mean, std, ssl.pos_scale_from(xy, nbr), torch.device("cpu"))
    model = ssl.NicheSSL(X.shape[1], cfg)
    hist = ssl.pretrain(model, b, rows[:200], rows[200:], cfg, log_every=5, log=lambda *_: None)
    assert hist[0]["step"] == 0 and hist[-1]["step"] == 5
    zc, zn = ssl.embed(model, b, rows)
    assert zc.shape == zn.shape == (len(rows), 8) and np.isfinite(zc).all()
    keys, means, counts = ssl.group_means(zc, sections)
    assert list(keys) == ["s0", "s1", "s2"] and counts.sum() == len(rows)
    assert np.allclose(means[0], zc[sections == "s0"].mean(0), atol=1e-5)


def test_paired_bootstrap_delta():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=40)
    good, noisy = ref + 0.1 * rng.normal(size=40), ref + 2.0 * rng.normal(size=40)
    d = paired_bootstrap_delta(good, noisy, ref, n_boot=300)
    assert d["diff"] > 0 and d["ci_lo"] > 0 and d["frac_le_0"] < 0.05
    same = paired_bootstrap_delta(good, good, ref, n_boot=50)
    assert same["diff"] == 0 and same["ci_lo"] == same["ci_hi"] == 0
