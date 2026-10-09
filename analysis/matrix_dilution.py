"""Collagen IV sign check: matrix loss, dilution by infiltrating cells, or too sparse to say?

The clock's largest negative weight is Col4a2, while the MS literature reports collagen IV
PROTEIN accumulating in active / chronic active lesions. Before any matrix interpretation:
  1. trajectory of every matrix gene on the panel vs day_of_sacrifice (RR, n = 33),
     (a) whole-section pseudobulk and (b) WITHIN each cell type (Anno_L1_curated;
     animals with >= MIN_CELLS cells of that type) — (b) removes composition effects;
  2. immune cell fraction per animal vs day (all immune, and infiltrating-only, since
     `Myeloid` mixes resident microglia with infiltrating macrophages);
  3. detection: fraction of cells with >= 1 count and mean counts per cell, per gene and
     cell type (RR cells), from the raw counts.
Reported rho are Spearman, raw and partial on score_sacrifice (the clock target is
severity-residualised), with 95% animal-bootstrap CIs.

RESOLUTION RULES (fixed before running), for each focus gene:
  * "too sparsely detected": mean counts/cell < 0.05 AND < 5% of cells positive in every
    cell type;
  * "dilution": whole-tissue rho CI excludes 0 (negative) AND in the highest-expressing
    cell type(s) the within-type rho CI includes 0;
  * "genuine within-type decline": within-type rho CI below 0 in the highest-expressing
    cell type;
  * otherwise "no clear trend".

    PYTHONPATH="$PWD:$PWD/scripts" python analysis/matrix_dilution.py
"""

from __future__ import annotations

import immunotransformer  # noqa: F401  (OpenMP guard — must precede numpy; see HANDOFF.md)

import json
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clock_composition as CC  # noqa: E402
import clock_selection as CS  # noqa: E402

OUT = "runs/clock_composition/matrix_dilution"
FOCUS = ["Col4a1", "Col4a2", "Lama1", "Lama2", "Lama4", "Lama5", "Fn1", "Hspg2", "Thbs2",
         "Serpine2", "Eln", "Fbln2", "Ptx3", "Fmod"]
IMMUNE = ["Myeloid", "T cell", "B cell", "DC", "NK/DC", "Neutrophil"]
INFILTRATING = ["T cell", "B cell", "DC", "NK/DC", "Neutrophil"]
MIN_CELLS = 50
N_BOOT = 2000


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    A = np.c_[np.ones(len(rz)), rz]
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1]) if ex.std() > 0 and ey.std() > 0 else np.nan


def rho_ci(x, y, z=None, seed=0):
    rng = np.random.default_rng(seed)
    f = (lambda a, b, c: partial_spearman(a, b, c)) if z is not None else \
        (lambda a, b, c: float(spearmanr(a, b).statistic))
    obs = f(x, y, z)
    bs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(x), len(x))
        if np.std(x[i]) > 0:
            bs.append(f(x[i], y[i], None if z is None else z[i]))
    bs = np.array([b for b in bs if np.isfinite(b)])
    return obs, [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]


def detection(genes_idx, rr_sections):
    """Cells with >=1 count and summed counts per (cell type, gene), RR cells only."""
    with h5py.File(CC.ATLAS, "r") as f:
        obs = f["obs"]

        def codes(k):
            g = obs[k]
            return g["codes"][:], np.array([c.decode() if isinstance(c, bytes) else c
                                            for c in g["categories"][:]])
        sc_, scats = codes("meta_sample_id")
        ct_, ccats = codes("Anno_L1_curated")
        in_rr = np.isin(scats[sc_], list(rr_sections))
        C = f["layers/counts"]
        indptr = C["indptr"][:]
        ng = int(C.attrs["shape"][1])
        pos = np.zeros((len(ccats), len(genes_idx)))
        tot = np.zeros((len(ccats), len(genes_idx)))
        ncell = np.zeros(len(ccats))
        t0 = time.time()
        for a in range(0, len(sc_), 200_000):
            e = min(a + 200_000, len(sc_))
            lo, hi = indptr[a], indptr[e]
            X = sp.csr_matrix((C["data"][lo:hi], C["indices"][lo:hi], indptr[a:e + 1] - lo),
                              shape=(e - a, ng))[:, genes_idx]
            m = in_rr[a:e]
            T = sp.csr_matrix((m.astype(float), (ct_[a:e], np.arange(e - a))),
                              shape=(len(ccats), e - a))
            pos += (T @ (X > 0).astype(float)).toarray()
            tot += (T @ X).toarray()
            ncell += np.asarray(T.sum(1)).ravel()
            print(f"  [detect] {e:,}/{len(sc_):,} {time.time() - t0:.0f}s", flush=True)
    return pos, tot, ncell, ccats


def main():
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    F = CS.feature_sets()
    rr, genes = F["rr"], F["genes"]
    pb, info, meta = CC.load_rr()
    mx = CS.matrix_genes(genes)
    focus = [g for g in FOCUS if g in genes]
    gidx = [list(genes).index(g) for g in mx]
    day, sev = rr.day.to_numpy(), rr.score.to_numpy()
    res = {"matrix_genes": mx, "focus_on_panel": focus,
           "focus_missing": [g for g in FOCUS if g not in genes]}

    # (a) whole-section pseudobulk
    whole = []
    for g in mx:
        x = pb.loc[rr.index, g].to_numpy()
        r, ci = rho_ci(x, day)
        rp, cip = rho_ci(x, day, sev)
        whole.append({"gene": g, "rho": r, "ci": ci, "rho_partial_score": rp, "ci_partial": cip})
    res["whole"] = whole

    # (b) within cell type
    counts, n_cells, scats, ccats = CC.celltype_pseudobulk(None)
    sidx = {s: i for i, s in enumerate(scats)}
    within = []
    for j, ct in enumerate(ccats):
        per_c, per_n = {}, {}
        for a in rr.index:
            si = [sidx[s] for s in meta.loc[meta.sample_name == a, "meta_sample_id"]]
            per_c[a], per_n[a] = counts[si, j].sum(0), n_cells[si, j].sum()
        ok = [a for a in rr.index if per_n[a] >= MIN_CELLS]
        if len(ok) < 15:
            continue
        X = CC._lognorm(np.vstack([per_c[a] for a in ok]))
        d_, s_ = rr.loc[ok, "day"].to_numpy(), rr.loc[ok, "score"].to_numpy()
        for g, gi in zip(mx, gidx):
            x = X[:, gi]
            if np.std(x) == 0:
                continue
            r, ci = rho_ci(x, d_)
            rp, cip = rho_ci(x, d_, s_)
            within.append({"gene": g, "cell_type": ct, "n": len(ok), "rho": r, "ci": ci,
                           "rho_partial_score": rp, "ci_partial": cip})
    res["within"] = within

    # 2. immune fraction
    ctn = pd.DataFrame({ct: [n_cells[[sidx[s] for s in meta.loc[meta.sample_name == a,
                                                                 "meta_sample_id"]], j].sum()
                             for a in rr.index] for j, ct in enumerate(ccats)}, index=rr.index)
    frac = ctn.div(ctn.sum(1), axis=0)
    imm = {}
    for name, cols in (("all immune", IMMUNE), ("infiltrating (no Myeloid)", INFILTRATING),
                       ("Myeloid (microglia + macrophages)", ["Myeloid"])):
        v = frac[[c for c in cols if c in frac]].sum(1).to_numpy()
        r, ci = rho_ci(v, day)
        rp, cip = rho_ci(v, day, sev)
        imm[name] = {"median_fraction": float(np.median(v)), "range": [float(v.min()), float(v.max())],
                     "rho_day": r, "ci": ci, "rho_day_partial_score": rp, "ci_partial": cip}
    res["immune_fraction"] = imm
    frac.assign(day=day).to_csv(os.path.join(OUT, "celltype_fraction_by_animal.csv"))

    # 3. detection
    rr_sections = set(meta.loc[meta.sample_name.isin(rr.index), "meta_sample_id"])
    pos, tot, ncell, dcats = detection(gidx, rr_sections)
    det = []
    for j, ct in enumerate(dcats):
        if ncell[j] < 1000:
            continue
        for k, g in enumerate(mx):
            det.append({"gene": g, "cell_type": ct, "n_cells": int(ncell[j]),
                        "frac_pos": float(pos[j, k] / ncell[j]), "mean_counts": float(tot[j, k] / ncell[j])})
    det = pd.DataFrame(det)
    det.to_csv(os.path.join(OUT, "detection_by_celltype.csv"), index=False)
    overall = (det.assign(c=det.mean_counts * det.n_cells, p=det.frac_pos * det.n_cells)
               .groupby("gene")[["c", "p", "n_cells"]].sum())
    overall = pd.DataFrame({"mean_counts_all_cells": overall.c / overall.n_cells,
                            "frac_pos_all_cells": overall.p / overall.n_cells})
    res["detection_overall"] = overall.round(4).to_dict(orient="index")

    # 4. resolution per focus gene
    W, B = pd.DataFrame(within), pd.DataFrame(whole).set_index("gene")
    resolution = {}
    for g in focus:
        d = det[det.gene == g].sort_values("mean_counts", ascending=False)
        top = d.head(2)
        sparse = bool((d.mean_counts < 0.05).all() and (d.frac_pos < 0.05).all())
        wb = B.loc[g]
        whole_neg = wb.ci[1] < 0
        wt = W[(W.gene == g) & W.cell_type.isin(top.cell_type)]
        within_neg = bool(len(wt) and (wt.ci.str[1] < 0).any())
        within_null = bool(len(wt) and (wt.ci.str[0] <= 0).all() and (wt.ci.str[1] >= 0).all())
        if sparse:
            verdict = "too sparsely detected to say"
        elif within_neg:
            verdict = "genuine within-cell-type decline"
        elif whole_neg and within_null:
            verdict = "composition / dilution"
        else:
            verdict = "no clear trend"
        resolution[g] = {"verdict": verdict, "top_expressing": top[["cell_type", "frac_pos",
                                                                    "mean_counts"]].round(3).to_dict("records"),
                         "whole_rho": wb.rho, "whole_ci": wb.ci,
                         "within_top": wt[["cell_type", "n", "rho", "ci"]].round(3).to_dict("records")}
    res["resolution"] = resolution
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    _plot(res, focus)
    _report(res, focus)


def _plot(res, focus):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    W = pd.DataFrame(res["within"])
    W = W[W.gene.isin(focus)]
    piv = W.pivot(index="gene", columns="cell_type", values="rho").reindex(focus)
    whole = pd.DataFrame(res["whole"]).set_index("gene").rho.reindex(focus)
    M = pd.concat([whole.rename("WHOLE TISSUE"), piv], axis=1)
    fig, ax = plt.subplots(figsize=(1 + 0.55 * M.shape[1], 0.45 * len(M) + 1.5))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    im = ax.imshow(M.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(M.shape[1]), M.columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(M)), M.index, fontsize=7)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=5.5)
    ax.axvline(0.5, color="#0b0b0b", lw=1)
    ax.set_title("Spearman ρ with day_of_sacrifice (RR): whole tissue vs within cell type",
                 fontsize=9, loc="left")
    fig.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures", "matrix_within_vs_whole.png"), dpi=160)
    plt.close(fig)


def _report(res, focus):
    W = pd.DataFrame(res["within"])
    L = ["# Matrix sign check — machine tables", "",
         f"Focus genes on panel: {focus}; not on panel: {res['focus_missing']}", "",
         "## (a) whole-tissue pseudobulk vs day (RR, n = 33)", "",
         "| gene | rho | 95% CI | rho partial on score | 95% CI |", "|---|---|---|---|---|"]
    for w in res["whole"]:
        if w["gene"] in focus:
            L.append(f"| {w['gene']} | {w['rho']:+.2f} | [{w['ci'][0]:+.2f}, {w['ci'][1]:+.2f}] | "
                     f"{w['rho_partial_score']:+.2f} | [{w['ci_partial'][0]:+.2f}, {w['ci_partial'][1]:+.2f}] |")
    L += ["", "## immune fraction vs day", ""]
    for k, v in res["immune_fraction"].items():
        L.append(f"- {k}: median {v['median_fraction']:.1%} (range {v['range'][0]:.1%}-{v['range'][1]:.1%}); "
                 f"rho with day {v['rho_day']:+.2f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]; partial on score "
                 f"{v['rho_day_partial_score']:+.2f} [{v['ci_partial'][0]:+.2f}, {v['ci_partial'][1]:+.2f}]")
    L += ["", "## resolution per focus gene", ""]
    for g, v in res["resolution"].items():
        top = "; ".join(f"{t['cell_type']} {t['frac_pos']:.1%} pos, {t['mean_counts']:.3f}/cell"
                        for t in v["top_expressing"])
        wt = "; ".join(f"{t['cell_type']} {t['rho']:+.2f} [{t['ci'][0]:+.2f}, {t['ci'][1]:+.2f}]"
                       for t in v["within_top"])
        L.append(f"- **{g}: {v['verdict']}** — whole {v['whole_rho']:+.2f} "
                 f"[{v['whole_ci'][0]:+.2f}, {v['whole_ci'][1]:+.2f}]; top expressing: {top}; "
                 f"within those: {wt}")
    L += ["", "Full within-type table: results.json -> within; detection: detection_by_celltype.csv"]
    with open(os.path.join(OUT, "report.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
