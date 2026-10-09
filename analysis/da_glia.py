"""DA-glia (Kukanja et al., Cell 2024) vs the duration clock.

Kukanja et al. report disease-associated (DA) glia induced independently of lesions and
dynamically induced AND RESOLVED (peaking at peak EAE). Something that resolves cannot be a
duration clock, so: is the clock the DA response, or a second, slower layer?

Markers: Kukanja et al. 2024, Figure 4B ("selected DA-glia marker genes"); the full lists
are in the paper's supplementary tables, which are not available here. Per state, the genes
high in the DA row relative to its homeostatic counterpart:
  DA-MOL2    Apod, Klk6, Serpina3n, Serpina3h, C4b               -> Oligodendrocyte
  DA-MOL5/6  Il12rb1, Irgm1, Igtp, C4b                          -> Oligodendrocyte
  DA-OPC/COP Col20a1, Serpina3n, Irgm1, Igtp, C4b, B2m, H2-D1   -> OPC
  DA-Astro   Mt1, Serping1, Serpina3n, Serpina3h, Irgm1, Igtp, C4b -> Astrocyte
  DA-MiGL    B2m, H2-D1, Cd74, H2-Aa, Fcgr2b                    -> Myeloid
Per-cell score = mean log-CP10k of markers minus mean of expression-matched control genes
(scanpy score_genes logic: 25 expression bins, 50 controls per marker, seed 0). The score is
linear in per-cell log expression, so animal x cell-type means are computed exactly from a
per-(section, cell type) sum of per-cell log-CP10k (one pass over the raw counts, cached).

  1  DA scores per animal per cell type
  2  correlation with (a) score_sacrifice, (b) day raw, (c) day partial on severity — RR and
     chronic separately, 95% animal-bootstrap CIs
  3  clock (RR, unchanged pipeline) with all DA scores residualised inside each fold next to
     severity; paired difference vs baseline on the same target and animals
  4  overlap between DA markers and the matrix gene set; matrix clock without DA genes and
     DA-gene clock without matrix genes
  5  time course per DA score, RR and chronic, animals shown; rise-and-fall vs monotonic

    PYTHONPATH="$PWD:$PWD/scripts" python analysis/da_glia.py
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
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clock_composition as CC  # noqa: E402
import clock_selection as CS  # noqa: E402
from duration_clock import ENET_KW, _topvar, loao_target  # noqa: E402
from immunotransformer.stats import bootstrap_spearman  # noqa: E402
from paired_difference import paired  # noqa: E402

OUT = "runs/da_glia_vs_clock"
CACHE = os.path.join(OUT, "sum_logexpr_section_celltype.npz")
STATES = {
    "DA-MOL2": (["Apod", "Klk6", "Serpina3n", "Serpina3h", "C4b"], "Oligodendrocyte"),
    "DA-MOL5/6": (["Il12rb1", "Irgm1", "Igtp", "C4b"], "Oligodendrocyte"),
    "DA-OPC/COP": (["Col20a1", "Serpina3n", "Irgm1", "Igtp", "C4b", "B2m", "H2-D1"], "OPC"),
    "DA-Astro": (["Mt1", "Serping1", "Serpina3n", "Serpina3h", "Irgm1", "Igtp", "C4b"], "Astrocyte"),
    "DA-MiGL": (["B2m", "H2-D1", "Cd74", "H2-Aa", "Fcgr2b"], "Myeloid"),
}
COHORTS = {"RELAPSE REMITTING": "RR", "CHRONIC": "chronic"}
N_BOOT = 2000


def sum_logexpr():
    """Per (section, Anno_L1_curated): sum over cells of log1p(CP10k) for every gene."""
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        return d["sums"], d["n_cells"], d["sections"].astype(str), d["celltypes"].astype(str), d["genes"].astype(str)
    with h5py.File(CC.ATLAS, "r") as f:
        def codes(k):
            g = f["obs"][k]
            return g["codes"][:], np.array([c.decode() if isinstance(c, bytes) else c for c in g["categories"][:]])
        sc_, scats = codes("meta_sample_id")
        ct_, ccats = codes("Anno_L1_curated")
        var = f["var"]
        genes = np.array([x.decode() if isinstance(x, bytes) else x for x in var[var.attrs["_index"]][:]])
        nct, ng = len(ccats), len(genes)
        grp = sc_.astype(np.int64) * nct + ct_
        C = f["layers/counts"]
        indptr = C["indptr"][:]
        sums = np.zeros((len(scats) * nct, ng))
        t0 = time.time()
        for a in range(0, len(grp), 100_000):
            e = min(a + 100_000, len(grp))
            lo, hi = indptr[a], indptr[e]
            X = sp.csr_matrix((C["data"][lo:hi], C["indices"][lo:hi], indptr[a:e + 1] - lo),
                              shape=(e - a, ng)).astype(np.float64)
            lib = np.asarray(X.sum(1)).ravel(); lib[lib == 0] = 1
            X = sp.diags(1e4 / lib) @ X
            X.data = np.log1p(X.data)
            G = sp.csr_matrix((np.ones(e - a), (grp[a:e], np.arange(e - a))), shape=(len(scats) * nct, e - a))
            sums += (G @ X).toarray()
            print(f"  [log-sum] {e:,}/{len(grp):,} {time.time() - t0:.0f}s", flush=True)
        n_cells = np.bincount(grp, minlength=len(scats) * nct)
    sums = sums.reshape(len(scats), nct, ng)
    n_cells = n_cells.reshape(len(scats), nct)
    os.makedirs(OUT, exist_ok=True)
    np.savez(CACHE, sums=sums, n_cells=n_cells, sections=scats, celltypes=ccats, genes=genes)
    return sums, n_cells, scats, ccats, genes


def control_genes(markers, mean_expr, genes, n_bins=25, ctrl_size=50, seed=0):
    """scanpy score_genes control selection: expression bins over all genes, ctrl_size per marker."""
    rng = np.random.default_rng(seed)
    order = pd.Series(mean_expr, index=genes).rank(method="min")
    bins = pd.cut(order, n_bins, labels=False)
    ctrl = set()
    for g in markers:
        pool = [x for x in genes[bins == bins[g]] if x not in markers]
        ctrl |= set(rng.choice(pool, min(ctrl_size, len(pool)), replace=False))
    return sorted(ctrl)


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    A = np.c_[np.ones(len(rz)), rz]
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def boot(f, *arrs, seed=0):
    rng = np.random.default_rng(seed)
    n = len(arrs[0])
    v = [f(*(a[i] for a in arrs)) for i in (rng.integers(0, n, n) for _ in range(N_BOOT))]
    v = np.array([x for x in v if np.isfinite(x)])
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]


def main():
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    sums, n_cells, scats, ccats, genes = sum_logexpr()
    F = CS.feature_sets()
    pb, info, meta = CC.load_rr()
    traj = pd.read_csv(CC.TRAJ).set_index("sample_name")
    gi = {g: i for i, g in enumerate(genes)}
    sidx = {s: i for i, s in enumerate(scats)}
    cti = {c: i for i, c in enumerate(ccats)}
    # animal x cell type mean of per-cell log expression
    animals = info.index
    mean_log = {}
    for a in animals:
        si = [sidx[s] for s in meta.loc[meta.sample_name == a, "meta_sample_id"]]
        mean_log[a] = (sums[si].sum(0), n_cells[si].sum(0))
    overall = sum(v[0].sum(0) for v in mean_log.values()) / sum(v[1].sum() for v in mean_log.values())
    res = {"markers_source": "Kukanja et al. 2024 Cell, Figure 4B", "states": {}}
    scores = pd.DataFrame(index=animals)
    for state, (mk, ct) in STATES.items():
        on = [g for g in mk if g in gi]
        ctrl = control_genes(on, overall, genes)
        j = cti[ct]
        s = {}
        for a in animals:
            tot, n = mean_log[a]
            s[a] = (tot[j, [gi[g] for g in on]].mean() - tot[j, [gi[g] for g in ctrl]].mean()) / n[j] \
                if n[j] >= 50 else np.nan
        scores[state] = pd.Series(s)
        res["states"][state] = {"markers_on_panel": on, "missing": [g for g in mk if g not in gi],
                                "cell_type": ct, "n_control_genes": len(ctrl)}
    scores = scores.join(info[["model", "condition", "stage", "day", "score"]])
    scores.to_csv(os.path.join(OUT, "da_scores_by_animal.csv"))

    # ---- 2 severity vs duration
    res["step2"] = {}
    for model, key in COHORTS.items():
        d = scores[(scores.model == model)]
        res["step2"][key] = {}
        for state in STATES:
            x = d[[state, "day", "score"]].dropna()
            a_ = float(spearmanr(x[state], x.score).statistic)
            b_ = float(spearmanr(x[state], x.day).statistic)
            c_ = partial_spearman(x[state].to_numpy(), x.day.to_numpy(), x.score.to_numpy())
            res["step2"][key][state] = {
                "n": int(len(x)), "a_rho_score": a_,
                "a_ci": boot(lambda u, v: spearmanr(u, v).statistic, x[state].to_numpy(), x.score.to_numpy()),
                "b_rho_day_raw": b_,
                "b_ci": boot(lambda u, v: spearmanr(u, v).statistic, x[state].to_numpy(), x.day.to_numpy()),
                "c_rho_day_partial_score": c_,
                "c_ci": boot(partial_spearman, x[state].to_numpy(), x.day.to_numpy(), x.score.to_numpy())}

    # ---- 3 clock with DA scores residualised in-fold (RR)
    rr = F["rr"]
    X = F["X_all"]
    y, sev = rr.day.to_numpy(), rr.score.to_numpy().reshape(-1, 1)
    ref = loao_target(y, sev)
    D = scores.loc[rr.index, list(STATES)]
    D = D.fillna(D.mean())
    base = (pd.read_csv("runs/baseline_ladder/loao_predictions_arms1-5.csv")
            .query("target == 'day_of_sacrifice' and arm == 'arm2'").set_index("sample_name").loc[rr.index, "pred"].to_numpy())
    cov = np.column_stack([sev, D.to_numpy()])
    pred, nnz = CC.loao_audit(X, y, cov)          # covariates fit inside each fold by loao_audit
    r3 = paired("clock + DA-glia scores residualised vs baseline", pred, base, ref,
                "clock | severity + 5 DA scores", "clock | severity")
    r3["zero_feature_folds"] = int((nnz == 0).sum())
    r3["rho_vs_own_target"] = float(spearmanr(pred, loao_target(y, cov)).statistic)
    res["step3"] = r3
    res["step3"]["note"] = ("covariates residualised from X and day inside each LOAO training fold "
                            "(loao_audit == loao_clock mechanics); paired comparison on the "
                            "baseline target day|severity, same 33 animals")

    # ---- 4 overlap with matrix set
    da_genes = sorted({g for mk, _ in STATES.values() for g in mk if g in gi})
    mx = CS.matrix_genes(genes)
    ov = sorted(set(da_genes) & set(mx))
    res["step4"] = {"da_genes": da_genes, "matrix_n": len(mx), "overlap": ov}
    for label, block, drop in (("matrix without DA genes", mx, ov), ("DA genes without matrix genes", da_genes, ov)):
        cols = [gi[g] for g in block if g not in drop]
        p, nz = CC.loao_audit(X[:, cols], y, sev)
        b = bootstrap_spearman(p, ref, n_boot=N_BOOT, seed=0)
        res["step4"][label] = {"n_genes": len(cols), "rho": float(b["rho"]),
                               "ci": [float(b["ci_lo"]), float(b["ci_hi"])], "zero_feature_folds": int((nz == 0).sum())}
    p, nz = CC.loao_audit(X[:, [gi[g] for g in da_genes]], y, sev)
    b = bootstrap_spearman(p, ref, n_boot=N_BOOT, seed=0)
    res["step4"]["DA genes (all)"] = {"n_genes": len(da_genes), "rho": float(b["rho"]),
                                      "ci": [float(b["ci_lo"]), float(b["ci_hi"])], "zero_feature_folds": int((nz == 0).sum())}

    # ---- 5 time course: rise-and-fall vs monotonic (quadratic vs linear in day, animals)
    res["step5"] = {}
    for model, key in COHORTS.items():
        d = scores[(scores.model == model) & (scores.condition == "EAE")]
        res["step5"][key] = {}
        for state in STATES:
            x = d[[state, "day"]].dropna()
            t, v = x.day.to_numpy(), x[state].to_numpy()
            q = np.polyfit(t, v, 2)
            vertex = -q[1] / (2 * q[0]) if q[0] != 0 else np.nan
            peak_inside = bool(q[0] < 0 and t.min() < vertex < t.max())
            late = x[x.day >= np.quantile(t, 0.75)][state].median()
            mid = x[(x.day >= np.quantile(t, 0.25)) & (x.day < np.quantile(t, 0.75))][state].max()
            res["step5"][key][state] = {"n": int(len(x)), "rho_day": float(spearmanr(t, v).statistic),
                                        "quadratic_a": float(q[0]), "vertex_day": float(vertex),
                                        "rise_and_fall": peak_inside,
                                        "late_quartile_median": float(late), "mid_max": float(mid)}
    _plot(scores)
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    print(json.dumps(res, indent=1, default=str))


def _plot(scores):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
    COL = {"RELAPSE REMITTING": "#2a78d6", "CHRONIC": "#eb6834"}
    fig, axes = plt.subplots(2, len(STATES), figsize=(3 * len(STATES), 5.6), sharex="row")
    fig.patch.set_facecolor(SURFACE)
    for i, (model, c) in enumerate(COL.items()):
        d = scores[scores.model == model]
        for j, state in enumerate(STATES):
            ax = axes[i, j]
            ax.set_facecolor(SURFACE)
            e, k = d[d.condition == "EAE"], d[d.condition != "EAE"]
            ax.scatter(e.day, e[state], s=22, color=c, edgecolor=SURFACE, lw=0.8, zorder=3)
            ax.scatter(k.day, k[state], s=22, color=MUTED, marker="x", zorder=3)
            x = e[["day", state]].dropna().sort_values("day")
            if len(x) > 4:
                q = np.polyfit(x.day, x[state], 2)
                tt = np.linspace(x.day.min(), x.day.max(), 50)
                ax.plot(tt, np.polyval(q, tt), color=c, lw=1.5)
            ax.set_title(f"{'RR' if 'RELAPSE' in model else 'chronic'} · {state}", loc="left", fontsize=8, color=INK)
            for sp_ in ("top", "right"):
                ax.spines[sp_].set_visible(False)
            ax.tick_params(colors=MUTED, labelsize=7)
            if i == 1:
                ax.set_xlabel("day of sacrifice", fontsize=7, color=MUTED)
        axes[i, 0].set_ylabel("DA score (per-animal mean)", fontsize=7, color=MUTED)
    fig.text(0.01, 0.005, "dots = EAE animals; x = controls; line = quadratic fit (a downturn inside the "
             "range = rise-and-fall)", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(OUT, "figures", "da_scores_vs_day.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
