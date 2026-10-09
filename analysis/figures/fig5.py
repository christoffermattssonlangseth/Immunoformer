"""Figure 5 — spatial illustration (ILLUSTRATIVE, not quantitative).
Endothelial cells in situ in severity-matched early/late RR pairs, lumbar section each, coloured
by per-cell Col4a1+Col4a2 (log CP10k) on a shared colour and physical scale; other cells grey.
Pairs picked by rule from the 111 matched pairs of Fig. 4d (|delta score| <= 0.25):
  a  high severity: exact score match (2.5), largest day gap
  b  low severity: exact score match (1.0), largest day gap
  c  counterexample: largest Col4 increase early->late among pairs >= 10 days apart

    PYTHONPATH="$PWD:$PWD/scripts:$PWD/analysis" python analysis/figures/fig5.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from common import INK, LIGHT, MUTED, animals, letter, plt, save  # noqa: E402

PAIRS = [("a", "high severity, goes down", "RR_P1_1", "RR_P3_1"),
         ("b", "low severity, goes down", "RR_OS2_1", "RR_R2_5"),
         ("c", "counterexample (22% of pairs go up)", "RR_P1_1", "RR_P2_3")]
REGION = "L"


def load_section(f, sec, sc_, scats, genes, gidx):
    rows = np.where(sc_ == list(scats).index(sec))[0]
    C = f["layers/counts"]
    ip = C["indptr"]
    blocks = np.split(rows, np.where(np.diff(rows) != 1)[0] + 1)
    X = sp.vstack([sp.csr_matrix((C["data"][ip[b[0]]:ip[b[-1] + 1]], C["indices"][ip[b[0]]:ip[b[-1] + 1]],
                                  ip[b[0]:b[-1] + 2] - ip[b[0]]), shape=(len(b), len(genes))) for b in blocks]).tocsr()
    lib = np.asarray(X.sum(1)).ravel()
    col4 = np.log1p(np.asarray(X[:, gidx].sum(1)).ravel() / np.maximum(lib, 1) * 1e4)
    return rows, col4


def main():
    import lesion_segment as LS
    A = animals().set_index("animal")
    d = np.load("runs/rr_within_relapse/pseudobulk_all67.npz", allow_pickle=True)
    meta = pd.DataFrame(d["section_meta"].tolist(), columns=d["section_meta_cols"])
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 10.5))
    fig.subplots_adjust(right=0.84, wspace=0.08, hspace=0.3)
    vmax = None
    with h5py.File(LS.CC.ATLAS, "r") as f:
        sc_, scats = LS._obs(f, "meta_sample_id")
        c1_, c1cats = LS._obs(f, "Anno_L1_curated")
        xs, _ = LS._obs(f, "x_centroid")
        ys, _ = LS._obs(f, "y_centroid")
        var = f["var"]
        genes = np.array([x.decode() if isinstance(x, bytes) else x for x in var[var.attrs["_index"]][:]])
        gidx = [int(np.where(genes == g)[0][0]) for g in ("Col4a1", "Col4a2")]
        cache, data = {}, []
        for pl, desc, early, late in PAIRS:
            for animal, lab in ((early, "early"), (late, "late")):
                if animal not in cache:
                    sec = meta[(meta.sample_name == animal) & (meta.region == REGION)].meta_sample_id.iloc[0]
                    rows, col4 = load_section(f, sec, sc_, scats, genes, gidx)
                    endo = c1cats[c1_[rows]] == "Endothelial"
                    cache[animal] = (sec, xs[rows], ys[rows], endo, col4)
                data.append((animal, lab, *cache[animal]))
        vmax = np.quantile(np.concatenate([c[e] for *_, e, c in data]), 0.98)
    span = max(max(x.max() - x.min(), y.max() - y.min()) for _, _, _, x, y, _, _ in data) + 200
    for k, (animal, lab, sec, x, y, endo, col4) in enumerate(data):
        ax = axes.flat[k]
        ax.scatter(x[~endo], y[~endo], s=0.15, color=LIGHT, rasterized=True)
        o = np.argsort(col4[endo])
        sc = ax.scatter(x[endo][o], y[endo][o], c=col4[endo][o], s=1.6, cmap="cividis", vmin=0, vmax=vmax,
                        rasterized=True)
        cx, cy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
        ax.set_xlim(cx - span / 2, cx + span / 2); ax.set_ylim(cy - span / 2, cy + span / 2)
        ax.set_aspect("equal"); ax.axis("off")
        x0, y0 = cx - span / 2 + 50, cy - span / 2 + 60
        ax.plot([x0, x0 + 500], [y0, y0], color=INK, lw=1.5)
        ax.text(x0 + 250, y0 - 40, "500 µm", ha="center", va="top", fontsize=6)
        r = A.loc[animal]
        ax.set_title(f"{lab}: {animal}, day {int(r.day)}, score {r.score:g}\n{sec} (lumbar), "
                     f"{endo.sum():,} endothelial cells, mean Col4 {col4[endo].mean():.2f}", loc="left", fontsize=6.5)
        if k % 2 == 0:
            pl, desc = PAIRS[k // 2][:2]
            letter(ax, pl)
            ax.text(-0.02, 1.2, desc, transform=ax.transAxes, fontsize=7, fontweight="bold", color=INK)
    cb = fig.colorbar(sc, cax=fig.add_axes([0.86, 0.4, 0.015, 0.2]))
    cb.set_label("Col4a1+Col4a2 per endothelial cell (log CP10k)", fontsize=6)
    fig.text(0.01, 0.0, "Same colour and physical scale in all panels. ILLUSTRATIVE ONLY: one section per animal; pairs chosen by fixed rules (see script header) "
             "from the 111 severity-matched pairs of Fig. 4d.\n"
             "Grey = all other cells. Not a quantitative comparison — see Fig. 4 for animal-level statistics.",
             fontsize=6, color=MUTED)
    save(fig, "fig5_in_situ_illustration")


if __name__ == "__main__":
    main()
