"""Endothelial Col4 image gallery (ILLUSTRATIVE, not quantitative).
G1  every RR animal, one lumbar section each, ordered by day; endothelial cells coloured by
    per-cell Col4a1+Col4a2 (log CP10k), shared colour and physical scale; other cells grey.
G2  close-ups (600 um windows) for the Fig. 5 pairs; window centred, by rule, on the densest
    endothelial neighbourhood of each section.

    PYTHONPATH="$PWD:$PWD/scripts:$PWD/analysis" python analysis/figures/endo_gallery.py
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
from scipy.spatial import cKDTree  # noqa: E402
from common import INK, LIGHT, MUTED, animals, letter, plt, save  # noqa: E402
from fig5 import PAIRS, REGION, load_section  # noqa: E402

WIN = 600


def load_all(names):
    import lesion_segment as LS
    d = np.load("runs/rr_within_relapse/pseudobulk_all67.npz", allow_pickle=True)
    meta = pd.DataFrame(d["section_meta"].tolist(), columns=d["section_meta_cols"])
    out = {}
    with h5py.File(LS.CC.ATLAS, "r") as f:
        sc_, scats = LS._obs(f, "meta_sample_id")
        c1_, c1cats = LS._obs(f, "Anno_L1_curated")
        xs, _ = LS._obs(f, "x_centroid")
        ys, _ = LS._obs(f, "y_centroid")
        var = f["var"]
        genes = np.array([x.decode() if isinstance(x, bytes) else x for x in var[var.attrs["_index"]][:]])
        gidx = [int(np.where(genes == g)[0][0]) for g in ("Col4a1", "Col4a2")]
        for a in names:
            m = meta[(meta.sample_name == a) & (meta.region == REGION)]
            if m.empty:
                continue
            sec = m.meta_sample_id.iloc[0]
            rows, col4 = load_section(f, sec, sc_, scats, genes, gidx)
            out[a] = (sec, xs[rows], ys[rows], c1cats[c1_[rows]] == "Endothelial", col4)
    return out


def draw(ax, x, y, endo, col4, vmax, s_bg, s_en, lim=None):
    ax.scatter(x[~endo], y[~endo], s=s_bg, color=LIGHT, rasterized=True, lw=0)
    o = np.argsort(col4[endo])
    sc = ax.scatter(x[endo][o], y[endo][o], c=col4[endo][o], s=s_en, cmap="cividis", vmin=0, vmax=vmax,
                    rasterized=True, lw=0)
    if lim is not None:
        ax.set_xlim(lim[0]); ax.set_ylim(lim[1])
    ax.set_aspect("equal"); ax.axis("off")
    return sc


def scalebar(ax, x0, y0, length, label, fs=5):
    ax.plot([x0, x0 + length], [y0, y0], color=INK, lw=1.2)
    ax.text(x0 + length / 2, y0 - 0.02 * length * 4, label, ha="center", va="top", fontsize=fs)


def main():
    A = animals().set_index("animal")
    rr = A[A.cohort == "RR"].sort_values(["condition", "day", "score"], ascending=[False, True, True])
    S = load_all(list(rr.index))
    vmax = np.quantile(np.concatenate([c[e] for _, _, _, e, c in S.values()]), 0.98)
    span = max(max(x.max() - x.min(), y.max() - y.min()) for _, x, y, _, _ in S.values()) + 100
    # G1
    names = [a for a in rr.index if a in S]
    nc = 5; nr = int(np.ceil(len(names) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(7.4, 1.75 * nr + 0.6), gridspec_kw={"hspace": 0.4, "wspace": 0.05})
    for ax in axes.flat:
        ax.axis("off")
    for ax, a in zip(axes.flat, names):
        sec, x, y, endo, col4 = S[a]
        cx, cy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
        sc = draw(ax, x, y, endo, col4, vmax, 0.04, 0.5,
                  ((cx - span / 2, cx + span / 2), (cy - span / 2, cy + span / 2)))
        r = A.loc[a]
        ctl = " (CFA)" if r.condition == "CONTROL" else ""
        ax.set_title(f"{a}{ctl}\nday {int(r.day)}, score {r.score:g}\nmean Col4 {col4[endo].mean():.2f}", fontsize=5.5,
                     loc="left", color=MUTED if ctl else INK)
    a0 = axes.flat[0]
    scalebar(a0, a0.get_xlim()[0] + 40, a0.get_ylim()[0] + 40, 500, "500 µm", fs=4.5)
    cax = fig.add_axes([0.3, 0.06, 0.4, 0.007])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label("Col4a1+Col4a2 per endothelial cell (log CP10k)", fontsize=6); cb.ax.tick_params(labelsize=5)
    fig.text(0.01, 0.015, f"All {len(names)} RR animals with a lumbar section, ordered by day of sacrifice (EAE, then CFA "
             "controls). One section per animal; same colour and physical scale.\n'mean Col4' = mean over endothelial cells "
             "in that section. ILLUSTRATIVE — animal-level statistics in Fig. 4.", fontsize=5.5, color=MUTED, wrap=True)
    save(fig, "supp_S5_endothelial_gallery_RR")
    # G2
    fig, axes = plt.subplots(len(PAIRS), 4, figsize=(7.4, 2.1 * len(PAIRS)),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.08, "width_ratios": [1, 1, 1, 1]})
    for i, (pl, desc, early, late) in enumerate(PAIRS):
        for j, a in enumerate((early, late)):
            sec, x, y, endo, col4 = S[a]
            pe = np.c_[x[endo], y[endo]]
            dens = np.array([len(v) for v in cKDTree(pe).query_ball_point(pe, 150)])
            cx, cy = pe[np.argmax(dens)]
            cx = np.clip(cx, x.min() + WIN / 2, x.max() - WIN / 2); cy = np.clip(cy, y.min() + WIN / 2, y.max() - WIN / 2)
            box = ((cx - WIN / 2, cx + WIN / 2), (cy - WIN / 2, cy + WIN / 2))
            ov, zm = axes[i, 2 * j], axes[i, 2 * j + 1]
            ccx, ccy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
            draw(ov, x, y, endo, col4, vmax, 0.04, 0.6, ((ccx - span / 2, ccx + span / 2), (ccy - span / 2, ccy + span / 2)))
            ov.add_patch(plt.Rectangle((box[0][0], box[1][0]), WIN, WIN, fill=False, ec=INK, lw=0.8))
            inb = (x > box[0][0]) & (x < box[0][1]) & (y > box[1][0]) & (y < box[1][1])
            sc = draw(zm, x[inb], y[inb], endo[inb], col4[inb], vmax, 2.5, 14, box)
            for s in ("left", "right", "top", "bottom"):
                zm.spines[s].set_visible(True); zm.spines[s].set_color(INK); zm.spines[s].set_linewidth(0.8)
            zm.axis("on"); zm.set_xticks([]); zm.set_yticks([])
            scalebar(zm, box[0][0] + 25, box[1][0] + 30, 100, "100 µm")
            r = A.loc[a]
            ov.set_title(f"{'early' if j == 0 else 'late'}: {a}\nday {int(r.day)}, score {r.score:g}", fontsize=6, loc="left")
            zm.set_title(f"{int(endo[inb].sum())} endothelial cells in window\nmean {col4[inb & endo].mean():.2f} "
                         f"(section {col4[endo].mean():.2f})", fontsize=5.8, loc="left")
        letter(axes[i, 0], pl)
        axes[i, 0].text(0.25, 1.32, desc, transform=axes[i, 0].transAxes, fontsize=6.5, fontweight="bold")
    cb = fig.colorbar(sc, cax=fig.add_axes([0.3, 0.07, 0.4, 0.008]), orientation="horizontal")
    cb.set_label("Col4a1+Col4a2 per endothelial cell (log CP10k)", fontsize=6); cb.ax.tick_params(labelsize=5)
    fig.text(0.01, 0.005, f"Pairs as in Fig. 5. Box = {WIN} µm window centred on the section's densest endothelial "
             "neighbourhood (most endothelial cells within 150 µm) — chosen by rule, not by eye. Grey = other cells. "
             "ILLUSTRATIVE.", fontsize=5.5, color=MUTED)
    save(fig, "supp_S6_endothelial_closeups")


if __name__ == "__main__":
    main()
