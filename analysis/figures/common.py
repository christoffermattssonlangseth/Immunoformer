"""Shared style and helpers for the paper figures (analysis/figures/fig*.py).

Conventions: one point per animal; n and rho [95% CI] on every correlation panel (2000-resample
animal bootstrap); RR blue / chronic orange (colour-blind safe, no red-green); panel letters
top left; vector PDF + 300 dpi PNG to docs/figures/.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402

FIG_DIR = "docs/figures"
DATA = os.path.join(FIG_DIR, "data")
COL = {"RR": "#2a78d6", "chronic": "#eb6834"}
INK, MUTED, LIGHT = "#1a1a1a", "#5a5a5a", "#d9d9d9"
N_BOOT = 2000

plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42, "font.family": "sans-serif",
})


def animals():
    return pd.read_csv(os.path.join(DATA, "animals.csv"))


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def boot_rho(x, y, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(x), len(x))
        if np.std(x[i]) > 0 and np.std(y[i]) > 0:
            v.append(spearmanr(x[i], y[i]).statistic)
    return float(spearmanr(x, y).statistic), float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))


def rho_label(x, y, prefix="ρ"):
    r, lo, hi = boot_rho(x, y)
    return f"{prefix} {r:+.2f} [{lo:+.2f}, {hi:+.2f}], n = {len(x)}"


def partial_rank(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    A = np.c_[np.ones(len(rz)), rz]
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def letter(ax, s):
    ax.text(-0.18, 1.06, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


def corner(ax, text, loc="upper left"):
    x, ha = (0.03, "left") if "left" in loc else (0.97, "right")
    y, va = (0.97, "top") if "upper" in loc else (0.03, "bottom")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=6.3, color=INK)


def forest(ax, labels, est, lo, hi, colors=None, ref=0.0, xlabel="", annotate=True):
    yy = np.arange(len(labels))[::-1]
    for k, (e, l_, h) in enumerate(zip(est, lo, hi)):
        c = colors[k] if colors else INK
        ax.plot([l_, h], [yy[k]] * 2, color=c, lw=1.6, solid_capstyle="round")
        ax.plot(e, yy[k], "o", color=c, ms=4.5, mec="white", mew=0.8)
        if annotate:
            ax.text(1.02, yy[k], f"{e:+.2f} [{l_:+.2f}, {h:+.2f}]", transform=ax.get_yaxis_transform(),
                    va="center", fontsize=6, color=INK)
    if ref is not None:
        ax.axvline(ref, color=LIGHT, lw=0.8, zorder=0)
    ax.set_yticks(yy, labels)
    ax.set_xlabel(xlabel)
    ax.tick_params(axis="y", length=0)


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")
