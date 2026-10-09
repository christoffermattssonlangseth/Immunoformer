"""Figure 3 — two timescales.
3a DA-glia scores vs day (quadratic fits, peaks marked); 3b clock-predicted day vs day on the
same animals and x-axis; 3c DA scores vs severity. RR and chronic columns.

    PYTHONPATH="$PWD" python analysis/figures/fig3.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import COL, MUTED, animals, boot_rho, corner, letter, plt, rho_label, save  # noqa: E402

STATES = ["DA-MOL2", "DA-MOL5/6", "DA-OPC/COP", "DA-Astro", "DA-MiGL"]
SCOL = {"DA-MOL2": "#1baf7a", "DA-MOL5/6": "#008300", "DA-OPC/COP": "#eda100", "DA-Astro": "#e87ba4",
        "DA-MiGL": "#4a3aa7"}
MODEL = {"RR": "RELAPSE REMITTING", "chronic": "CHRONIC"}


def main():
    D = pd.read_csv("runs/da_glia_vs_clock/da_scores_by_animal.csv", index_col=0)
    A = animals().set_index("animal")
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.4), gridspec_kw={"hspace": 0.55, "wspace": 0.35})
    for j, ck in enumerate(("RR", "chronic")):
        d = D[(D.model == MODEL[ck])].copy()
        z = (d[STATES] - d[STATES].mean()) / d[STATES].std()
        e = d.condition == "EAE"
        # 3a
        ax = axes[0, j]
        for s in STATES:
            x, y = d.loc[e, "day"].to_numpy(), z.loc[e, s].to_numpy()
            ax.scatter(x, y, s=7, color=SCOL[s], alpha=0.75, edgecolor="none")
            q = np.polyfit(x, y, 2)
            tt = np.linspace(x.min(), x.max(), 80)
            ax.plot(tt, np.polyval(q, tt), color=SCOL[s], lw=1.2, label=s)
            vx = -q[1] / (2 * q[0]) if q[0] < 0 else np.nan
            if np.isfinite(vx) and x.min() < vx < x.max():
                ax.plot(vx, np.polyval(q, vx), marker="v", color=SCOL[s], ms=5, mec="white", mew=0.6)
        ax.set_ylabel("DA score (z within cohort)")
        ax.set_title(f"{ck}: DA-glia states (EAE animals, n = {int(e.sum())})\n▼ = fitted peak inside the sampled range",
                     loc="left")
        if j == 1:
            ax.legend(frameon=False, fontsize=5.8, loc="upper right", ncol=1)
        # 3b
        ax = axes[1, j]
        a = A[(A.cohort == ck) & (A.condition == "EAE")]
        ax.scatter(a.day, a.clock_pred_day, s=14, color=COL[ck], edgecolor="white", lw=0.4)
        lim = [min(a.day.min(), a.clock_pred_day.min()) - 3, max(a.day.max(), a.clock_pred_day.max()) + 3]
        ax.plot(lim, lim, color=MUTED, lw=0.7, ls="--")
        ax.set_ylabel("clock-predicted day (LOAO)")
        ax.set_title(f"{ck}: duration clock on the same animals", loc="left")
        corner(ax, rho_label(a.clock_pred_day, a.day), loc="upper left")
        for ax_ in (axes[0, j], axes[1, j]):
            ax_.set_xlim(d.loc[e, "day"].min() - 2, d.loc[e, "day"].max() + 2)
            ax_.set_xlabel("day of sacrifice")
        # 3c
        ax = axes[2, j]
        txt = []
        for s in STATES:
            ax.scatter(d.score, z[s], s=7, color=SCOL[s], alpha=0.75, edgecolor="none")
            r, lo, hi = boot_rho(z[s], d.score)
            txt.append(f"{s} ρ {r:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        ax.set_xlabel("clinical score at sacrifice")
        ax.set_ylabel("DA score (z within cohort)")
        ax.set_title(f"{ck}: DA states track current severity (all animals, n = {len(d)})", loc="left")
        corner(ax, "\n".join(txt), loc="lower right")
        if j == 0:
            letter(axes[0, 0], "a"); letter(axes[1, 0], "b"); letter(axes[2, 0], "c")
    fig.text(0.01, 0.03, "Points = animals. DA scores (Kukanja et al. 2024, Fig. 4B markers) per animal in their cell "
             "type, z-scored within cohort for display. Lines = quadratic fits. CIs = 95% bootstrap over animals.",
             fontsize=6, color=MUTED)
    save(fig, "fig3_two_timescales")


if __name__ == "__main__":
    main()
