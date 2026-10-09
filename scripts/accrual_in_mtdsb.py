"""Does the RRMAP2 time/ratchet program rise in the mtDNA-DSB model? (WORKORDER Task 2)

mtDNA-DSB = oligodendrocyte-intrinsic mitochondrial DNA damage with essentially no lymphoid
compartment. If the RRMAP2 accrual/ratchet genes rise in mtDSB oligodendrocytes, that
program can arise without an immune trigger; the acute set (Hal, Arg1, Chil3) is the
contrast and should not rise if the two are separable.

Framing after Task 4 (2026-10-08): the ratchet program failed its pre-registered
dissociation in RRMAP2 and tracks day_of_sacrifice — read it as the RRMAP2 TIME axis, not
as an established separate accrual program. Prior expectation (user): little signal in
this dataset. A null is reported as a null.

Design: 12 animals (sample_id = animal = section), condition control vs mtDSB x age 21/60,
n = 3 per cell. Program scores per cell with scanpy.tl.score_genes (matched control
genes), within oligodendrocyte-lineage cells (Oligodendrocytes, Immature Oligodendrocytes,
DA-Oligodendrocytes) and separately within microglia; averaged per animal. Exact Mann-
Whitney U per age stratum — with 3 vs 3 the smallest attainable two-sided p is 0.10, so
no within-age test can reach 0.05. Effect sizes (rank-biserial, Hedges g) are the result;
an age-pooled 6 vs 6 test is reported with its caveat. Age is aliased with slide/run in
this dataset: the condition contrast is within-slide and clean, any age contrast is not.

    MTDSB_H5AD=... python scripts/accrual_in_mtdsb.py      (run where the data is mounted)
"""

from __future__ import annotations

import immunotransformer  # noqa: F401  (OpenMP guard — must precede numpy; see HANDOFF.md)

import json
import os
import time

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import mannwhitneyu

H5AD = os.environ.get(
    "MTDSB_H5AD",
    "/Volumes/moldiassd/oligo-mtDSB/data/"
    "mtDNA_DSB_5k_clustered_annotation_with_rbd_2_cytetype_brain_novae4.companion.ready.h5ad")
CLOCK_GENES = "runs/duration_clock/clock_genes.csv"
OUT = "runs/accrual_in_mtdsb"
RATCHET = ["Gpnmb", "Plin4", "Fcrls", "Igf2", "Fmod", "Pmp22", "Ptgds"]
ACUTE = ["Hal", "Arg1", "Chil3"]
N_CLOCK = 20
COMPARTMENTS = {
    "oligodendrocyte_lineage": ["Oligodendrocytes", "Immature Oligodendrocytes",
                                "DA-Oligodendrocytes"],
    "microglia": ["Microglia"],
}
CHUNK_ROWS = 100_000


def _col(obs, k):
    g = obs[k]
    if isinstance(g, h5py.Group):
        cats = [c.decode() if isinstance(c, bytes) else c for c in g["categories"][:]]
        return pd.Categorical.from_codes(g["codes"][:], cats)
    return g[:]


def load(cell_classes):
    with h5py.File(H5AD, "r") as f:
        obs = f["obs"]
        meta = pd.DataFrame({k: pd.Series(_col(obs, k)).astype(str)
                             for k in ["sample_id", "condition", "age", "cell_class_updated",
                                       "run"]})
        var = f["var"]
        genes = np.array([x.decode() if isinstance(x, bytes) else x
                          for x in var[var.attrs["_index"]][:]])
        keep = np.where(meta.cell_class_updated.isin(cell_classes))[0]
        C = f["layers/counts"]
        indptr = C["indptr"][:]
        parts, t0 = [], time.time()
        for a in range(0, len(meta), CHUNK_ROWS):
            e = min(a + CHUNK_ROWS, len(meta))
            sel = keep[(keep >= a) & (keep < e)]
            if not len(sel):
                continue
            lo, hi = indptr[a], indptr[e]
            X = sp.csr_matrix((C["data"][lo:hi], C["indices"][lo:hi], indptr[a:e + 1] - lo),
                              shape=(e - a, len(genes)))
            parts.append(X[sel - a])
            print(f"  [load] {e:,}/{len(meta):,} rows, kept {sum(p.shape[0] for p in parts):,}"
                  f"  {time.time() - t0:.0f}s", flush=True)
    a = ad.AnnData(X=sp.vstack(parts).tocsr().astype(np.float32),
                   obs=meta.iloc[keep].reset_index(drop=True))
    a.var_names = genes
    return a, meta


def programs(genes):
    cg = pd.read_csv(CLOCK_GENES)
    clock = cg[cg.coef > 0].sort_values("coef", ascending=False).gene.head(N_CLOCK).tolist()
    accrual = list(dict.fromkeys(clock + RATCHET))
    p = {"accrual (clock top-20 + ratchet)": accrual, "ratchet only": RATCHET, "acute": ACUTE}
    surv = {k: {"defined": len(v), "on_panel": [g for g in v if g in genes],
                "missing": [g for g in v if g not in genes]} for k, v in p.items()}
    return {k: v["on_panel"] for k, v in surv.items()}, surv


def effect(a, b):
    a, b = np.asarray(a), np.asarray(b)
    u, p = mannwhitneyu(b, a, alternative="two-sided", method="exact")
    rb = 2 * u / (len(a) * len(b)) - 1                     # rank-biserial, + = mtDSB higher
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                / (len(a) + len(b) - 2))
    g = (b.mean() - a.mean()) / s * (1 - 3 / (4 * (len(a) + len(b)) - 9)) if s > 0 else np.nan
    return {"n_control": len(a), "n_mtdsb": len(b), "mean_control": float(a.mean()),
            "mean_mtdsb": float(b.mean()), "diff": float(b.mean() - a.mean()),
            "U": float(u), "p_exact": float(p), "rank_biserial": float(rb), "hedges_g": float(g)}


def main():
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    all_classes = sorted({c for v in COMPARTMENTS.values() for c in v})
    adata, meta = load(all_classes)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    progs, survival = programs(set(adata.var_names))
    animals = meta.drop_duplicates("sample_id").set_index("sample_id")[["condition", "age", "run"]]
    animals["slide"] = animals.run.str.split("__").str[1]          # e.g. 0060539

    per_animal, tests, detection = [], {}, {}
    for comp, classes in COMPARTMENTS.items():
        sub = adata[adata.obs.cell_class_updated.isin(classes)].copy()
        gl = sorted({g for v in progs.values() for g in v})
        det = pd.DataFrame((sub[:, gl].X > 0).toarray(), columns=gl)
        det["condition"] = sub.obs.condition.to_numpy()
        detection[comp] = det.groupby("condition").mean().T.round(4).to_dict()
        for name, genes in progs.items():
            sc.tl.score_genes(sub, genes, score_name="s", random_state=0)
            m = sub.obs.groupby("sample_id").s.mean()
            n = sub.obs.groupby("sample_id").size()
            for sid, v in m.items():
                per_animal.append({"compartment": comp, "program": name, "sample_id": sid,
                                   "score": float(v), "n_cells": int(n[sid]),
                                   **animals.loc[sid, ["condition", "age", "slide"]].to_dict()})
    pa = pd.DataFrame(per_animal)
    pa.to_csv(os.path.join(OUT, "per_animal_scores.csv"), index=False)
    for (comp, name), g in pa.groupby(["compartment", "program"]):
        t = {}
        for age, h in list(g.groupby("age")) + [("pooled", g)]:
            t[str(age)] = effect(h[h.condition == "control"].score, h[h.condition == "mtDSB"].score)
        tests[f"{comp} | {name}"] = t

    crosstab = pd.crosstab(animals.age, animals.slide)
    results = {"h5ad": H5AD, "program_survival": survival, "tests": tests,
               "detection_fraction": detection,
               "age_x_run": crosstab.to_dict(), "animals": animals.reset_index().to_dict("records")}
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    _plot(pa)
    _report(results, crosstab)


def _plot(pa):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
    COL = {"control": "#86b6ef", "mtDSB": "#eb6834"}
    progs = ["accrual (clock top-20 + ratchet)", "acute"]
    comps = list(COMPARTMENTS)
    fig, axes = plt.subplots(len(comps), len(progs), figsize=(9, 3.4 * len(comps)), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for i, comp in enumerate(comps):
        for j, prog in enumerate(progs):
            ax = axes[i, j]
            ax.set_facecolor(SURFACE)
            g = pa[(pa.compartment == comp) & (pa.program == prog)]
            for k, (age, cond) in enumerate([(a, c) for a in ("21", "60")
                                             for c in ("control", "mtDSB")]):
                v = g[(g.age == age) & (g.condition == cond)].score
                ax.scatter(np.full(len(v), k) + np.linspace(-0.08, 0.08, len(v)), v, s=40,
                           color=COL[cond], edgecolor=SURFACE, lw=1.2, zorder=3)
                ax.plot([k - 0.2, k + 0.2], [v.mean()] * 2, color=INK, lw=1.5)
            ax.set_xticks(range(4), ["21 ctrl", "21 mtDSB", "60 ctrl", "60 mtDSB"], fontsize=8)
            ax.set_title(f"{comp.replace('_', ' ')} — {prog}", loc="left", fontsize=8, color=INK)
            ax.set_ylabel("program score (per-animal mean)", fontsize=7, color=MUTED)
            for sp_ in ("top", "right"):
                ax.spines[sp_].set_visible(False)
            ax.tick_params(colors=MUTED, labelsize=7)
    fig.text(0.01, 0.005, "one dot per animal (n = 3 per group); bar = group mean · age is "
             "aliased with slide: compare within age only", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(OUT, "figures", "program_scores_by_animal.png"), dpi=160)
    plt.close(fig)


def _report(r, crosstab):
    L = ["ACCRUAL (TIME/RATCHET) PROGRAM IN THE mtDNA-DSB MODEL", "=" * 60, "",
         "THIS IS A 12-ANIMAL DESIGN: 3 control vs 3 mtDSB per age group. With 3 vs 3 the",
         "smallest attainable exact two-sided Mann-Whitney p is 0.10 — no within-age test can",
         "reach 0.05. Effect sizes (rank-biserial r, Hedges g) are the result.",
         "Framing (Task 4): the ratchet program tracks RRMAP2 day_of_sacrifice and failed its",
         "dissociation test; it is read here as the RRMAP2 time axis, not a proven separate",
         "accrual program.", "",
         "CONFOUND: age is aliased with slide/run in this dataset —",
         "  " + crosstab.to_string().replace("\n", "\n  "),
         "  The condition contrast is within-slide and clean; any age contrast is not.", "",
         "PROGRAM GENES ON THE PANEL"]
    for k, v in r["program_survival"].items():
        L.append(f"  {k}: {len(v['on_panel'])}/{v['defined']} on panel"
                 + (f" (missing {v['missing']})" if v["missing"] else ""))
    L += ["", "DETECTION — fraction of cells with >= 1 count (control / mtDSB)"]
    for comp, d in r["detection_fraction"].items():
        acute = ", ".join(f"{g} {d['control'][g]:.3f}/{d['mtDSB'][g]:.3f}" for g in ACUTE)
        L.append(f"  {comp}: acute genes {acute}")
    L += ["  Acute genes are near-undetected in these cell types, so acute program scores are",
          "  ~0 and the acute 'contrast' is uninformative here rather than a negative.",
          "", f"  {'compartment | program':52s} {'stratum':>7s} {'ctrl':>7s} {'mtDSB':>7s} "
              f"{'diff':>7s} {'r_rb':>6s} {'g':>6s} {'p_exact':>8s}"]
    for key, t in r["tests"].items():
        for st, e in t.items():
            L.append(f"  {key:52s} {st:>7s} {e['mean_control']:+7.3f} {e['mean_mtdsb']:+7.3f} "
                     f"{e['diff']:+7.3f} {e['rank_biserial']:+6.2f} {e['hedges_g']:+6.2f} "
                     f"{e['p_exact']:8.3f}")
    acc = r["tests"]["oligodendrocyte_lineage | accrual (clock top-20 + ratchet)"]
    acu = r["tests"]["oligodendrocyte_lineage | acute"]
    rises = lambda t: all(t[a]["rank_biserial"] > 0.5 for a in ("21", "60"))
    L += ["", "ANSWER",
          f"  Accrual program in mtDSB oligodendrocyte lineage: "
          f"{'RISES in both age groups' if rises(acc) else 'does NOT consistently rise'} "
          f"(rank-biserial 21d {acc['21']['rank_biserial']:+.2f}, 60d {acc['60']['rank_biserial']:+.2f}; "
          f"pooled 6 vs 6 p = {acc['pooled']['p_exact']:.3f}).",
          f"  Acute program: {'RISES in both age groups' if rises(acu) else 'does NOT consistently rise'} "
          f"(rank-biserial 21d {acu['21']['rank_biserial']:+.2f}, 60d {acu['60']['rank_biserial']:+.2f}; "
          f"pooled p = {acu['pooled']['p_exact']:.3f}).",
          "  Pooled tests mix the age/slide strata; read them as a summary, not a clean test.",
          "", "figures/: program_scores_by_animal.png; per-animal table: per_animal_scores.csv"]
    report = "\n".join(L)
    print(report)
    with open(os.path.join(OUT, "report.txt"), "w") as fh:
        fh.write(report + "\n")


if __name__ == "__main__":
    main()
