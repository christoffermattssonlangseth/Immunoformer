"""Is duration accrual COMPOSITIONAL (cell-number) or CELL-INTRINSIC (per-cell)?

A pseudobulk gene can rise with disease duration for two very different reasons:

  * COMPOSITIONAL — the cell type that expresses it becomes more abundant (TLS B/T cells,
    foamy macrophages, fibroblast-like cells), with no change in per-cell expression.
  * CELL-INTRINSIC — individual cells of a fixed type upregulate the gene.

The duration clock and the gene decomposition both run on animal-level pseudobulk, which
cannot tell these apart. This script does, using the cached per-(animal x leiden-cluster x
gene) means + cell counts (runs/rr_phase_niche/agg_leiden_1.npz) labelled to cell types
(runs/rr_region_gradient/cluster_labels.json) — NO h5ad reload.

For every gene, write the pseudobulk as a sum over cell types c:
    pb(a) = Σ_c f(a,c) · e(a,c)        f = cell-type fraction, e = per-cell-type mean
With f = f̄ + δf and e = ē + δe this splits additively (shift-share / Oaxaca):
    pb(a) − const = Σ_c f̄(c) δe(a,c)   [INTRINSIC]
                  + Σ_c ē(c) δf(a,c)   [COMPOSITIONAL]
                  + Σ_c δf(a,c) δe(a,c) [interaction]
Covariance is linear, so the severity-removed day trend decomposes exactly:
    Cov(pb, day|score) = Cov_intrinsic + Cov_compositional + Cov_interaction
and each gene's accrual gets a compositional share vs an intrinsic share.

    PYTHONPATH="$PWD" python scripts/duration_composition_intrinsic.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

AGG = "runs/rr_phase_niche/agg_leiden_1.npz"
LABELS = "runs/rr_region_gradient/cluster_labels.json"
META = "runs/duration_clock/rr_meta.csv"
CLOCK_GENES = "runs/duration_clock/clock_genes.csv"
DECOMP = "runs/duration_gene_decomposition/gene_day_decomposition.csv"
OUT_DIR = "runs/duration_composition_intrinsic"


def partial_corr_vec(X, y, z):
    """Partial corr of each column of X with y, controlling for z."""
    def zc(a):
        a = a - a.mean(0)
        s = a.std(0)
        return a / np.where(s == 0, 1, s)
    Xz, yz, zz = zc(X), zc(y), zc(z)
    r_xy = (Xz * yz[:, None]).mean(0)
    r_xz = (Xz * zz[:, None]).mean(0)
    r_yz = float((yz * zz).mean())
    denom = np.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))
    denom = np.where(denom < 1e-9, np.nan, denom)
    return np.clip(np.nan_to_num((r_xy - r_xz * r_yz) / denom), -0.999, 0.999)


def main():
    os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)
    d = np.load(AGG, allow_pickle=True)
    cpb, counts = d["pb"], d["counts"]                 # (A,C0,G), (A,C0)
    genes, animals = d["genes"].astype(str), d["animals"].astype(str)
    clusters = d["clusters"].astype(str)
    labels = json.load(open(LABELS))
    meta = pd.read_csv(META).set_index("sample_name")
    day = meta.loc[animals, "day_of_sacrifice"].to_numpy(float)
    score = meta.loc[animals, "score_sacrifice"].to_numpy(float)
    A, _, G = cpb.shape

    # ---- collapse leiden clusters -> labelled cell types (count-weighted) ----
    names = sorted(set(labels[c] for c in clusters))
    C = len(names)
    tot = counts.sum(1)                                # cells per animal
    frac = np.zeros((A, C))                            # f(a, ct)
    e = np.zeros((A, C, G))                            # per-cell-type mean expression
    present = np.zeros((A, C), bool)
    for k, nm in enumerate(names):
        idx = [j for j, c in enumerate(clusters) if labels[c] == nm]
        w = counts[:, idx]                             # (A, k)
        ctot = w.sum(1)                                # cells of this type per animal
        frac[:, k] = ctot / tot
        wn = np.where(ctot[:, None] > 0, w / np.where(ctot[:, None] == 0, 1, ctot[:, None]), 0)
        e[:, k, :] = np.einsum("ak,akg->ag", wn, cpb[:, idx, :])
        present[:, k] = ctot > 0
    print(f"[load] {A} animals x {C} cell types x {G} genes | day "
          f"{day.min():.0f}-{day.max():.0f}")

    # cohort references; impute missing cell-type expression with cohort mean (δe=0)
    fbar = frac.mean(0)                                # f̄(ct)
    ebar = np.zeros((C, G))
    for k in range(C):
        m = present[:, k]
        ebar[k] = e[m, k, :].mean(0) if m.any() else 0.0
    e_imp = e.copy()
    for k in range(C):
        e_imp[~present[:, k], k, :] = ebar[k]

    # ---- component pseudobulks ----
    pb_total = np.einsum("ac,acg->ag", frac, e_imp)
    pb_comp = np.einsum("ac,cg->ag", frac, ebar)       # composition varies, expr fixed
    pb_intr = np.einsum("c,acg->ag", fbar, e_imp)      # expr varies, composition fixed

    # ---- severity-removed day axis: residualize day on score ----
    b = np.polyfit(score, day, 1)
    day_r = day - np.polyval(b, score)
    day_r -= day_r.mean()

    def cov_day(X):
        return ((X - X.mean(0)) * day_r[:, None]).mean(0)
    cov_tot = cov_day(pb_total)
    cov_comp = cov_day(pb_comp)
    cov_intr = cov_day(pb_intr)
    cov_inter = cov_tot - cov_comp - cov_intr

    # per-gene shares (only meaningful where the gene actually trends with day)
    eps = 1e-9
    denom = np.where(np.abs(cov_tot) < eps, np.nan, cov_tot)
    comp_share = cov_comp / denom
    intr_share = cov_intr / denom
    inter_share = cov_inter / denom

    # partial day-corr of each component, for a correlation-scale readout
    r_tot = partial_corr_vec(pb_total, day, score)
    r_comp = partial_corr_vec(pb_comp, day, score)
    r_intr = partial_corr_vec(pb_intr, day, score)

    df = pd.DataFrame({
        "gene": genes, "cov_total": cov_tot, "cov_comp": cov_comp,
        "cov_intr": cov_intr, "cov_inter": cov_inter,
        "comp_share": comp_share, "intr_share": intr_share, "inter_share": inter_share,
        "r_day_total": r_tot, "r_day_comp": r_comp, "r_day_intr": r_intr,
    })

    # ---- which cell types expand / shrink with duration ----
    frac_trend = {nm: float(partial_corr_vec(frac[:, [k]], day, score)[0])
                  for k, nm in enumerate(names)}

    # ---- focus: genes that genuinely ACCRUE with day (severity removed) ----
    # The full accrual set (r_day>=0.30) gives a robust median share; per-gene shares are
    # only stable where the gene trends strongly, so classification + example lists use a
    # tighter cut (r_day>=0.40) — below that, a tiny cov_total inflates the interaction term
    # and shares run outside [0,1].
    ACC_FULL, ACC_STRONG = 0.30, 0.40
    full = df[(df.r_day_total >= ACC_FULL) & (df.cov_total > 0)].copy()
    sub = df[(df.r_day_total >= ACC_STRONG) & (df.cov_total > 0)].copy()
    sub["driver"] = np.where(sub.comp_share >= 0.60, "compositional",
                             np.where(sub.intr_share >= 0.60, "intrinsic", "mixed"))

    n_comp = int((sub.driver == "compositional").sum())
    n_intr = int((sub.driver == "intrinsic").sum())
    n_mixed = int((sub.driver == "mixed").sum())
    results = {
        "n_animals": A, "n_cell_types": C, "cell_types": names,
        "n_accrual_genes_full": int(len(full)),
        "n_accrual_genes_classified": int(len(sub)),
        "median_intrinsic_share": float(np.nanmedian(full.intr_share)),
        "median_compositional_share": float(np.nanmedian(full.comp_share)),
        "median_interaction_share": float(np.nanmedian(full.inter_share)),
        "driver_counts": {"compositional": n_comp, "intrinsic": n_intr, "mixed": n_mixed},
        "cell_type_fraction_day_trend": frac_trend,
        "expanding_with_duration": sorted(
            [(k, v) for k, v in frac_trend.items() if v > 0], key=lambda t: -t[1]),
        "shrinking_with_duration": sorted(
            [(k, v) for k, v in frac_trend.items() if v < 0], key=lambda t: t[1]),
        "top_compositional_accrual": sub[sub.driver == "compositional"]
            .sort_values("comp_share", ascending=False)
            .head(20)[["gene", "comp_share", "intr_share", "r_day_total"]].to_dict("records"),
        "top_intrinsic_accrual": sub[sub.driver == "intrinsic"]
            .sort_values("intr_share", ascending=False)
            .head(20)[["gene", "intr_share", "comp_share", "r_day_total"]].to_dict("records"),
        "note": ("Shares are fractions of the severity-removed day covariance "
                 "Cov(pb,day|score) = Cov_comp + Cov_intr + Cov_inter, computed only over "
                 "genes that actually accrue (positive day trend). Cell types are 25 leiden "
                 "clusters collapsed to labelled lineages."),
    }
    with open(os.path.join(OUT_DIR, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    df.sort_values("cov_total", ascending=False).to_csv(
        os.path.join(OUT_DIR, "gene_composition_intrinsic.csv"), index=False)

    _plot(frac_trend, sub, OUT_DIR)
    _report(results, sub)
    print(f"[done] -> {OUT_DIR}/")


def _plot(frac_trend, sub, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.8, 5.6))

    # panel A: cell-type fraction trend with duration
    items = sorted(frac_trend.items(), key=lambda t: t[1])
    nm = [k for k, _ in items]
    vv = [v for _, v in items]
    cols = ["#2c6fbb" if v < 0 else "#c0392b" for v in vv]
    a1.barh(nm, vv, color=cols, edgecolor="k", linewidth=.4)
    a1.axvline(0, color="k", lw=.8)
    for y, v in enumerate(vv):
        a1.text(v + (0.01 if v >= 0 else -0.01), y, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9, fontweight="bold")
    a1.set_xlim(-0.6, 0.65)
    a1.set_xlabel("r(cell-type fraction, day | score)")
    a1.set_title("Which compartments expand vs shrink with duration\n"
                 "(severity removed)", fontsize=11)
    a1.spines[["top", "right"]].set_visible(False)

    # panel B: intrinsic vs compositional day-covariance for accruing genes
    cmap = {"compositional": "#e8590c", "intrinsic": "#2f9e44", "mixed": "#999999"}
    for drv, c in cmap.items():
        s = sub[sub.driver == drv]
        a2.scatter(s.cov_intr, s.cov_comp, s=22, c=c, alpha=.75, edgecolor="k",
                   linewidth=.2, label=f"{drv} (n={len(s)})", zorder=2)
    lim = max(sub.cov_intr.abs().max(), sub.cov_comp.abs().max()) * 1.08
    a2.plot([0, lim], [0, lim], "--", color="grey", lw=.8, zorder=1)
    a2.axhline(0, color="#bbb", lw=.6)
    a2.axvline(0, color="#bbb", lw=.6)
    lab = pd.concat([
        sub[sub.driver == "compositional"].sort_values("cov_comp", ascending=False).head(8),
        sub[sub.driver == "intrinsic"].sort_values("cov_intr", ascending=False).head(8),
    ])
    for r in lab.itertuples():
        a2.annotate(r.gene, (r.cov_intr, r.cov_comp), fontsize=7.2,
                    xytext=(3, 2), textcoords="offset points")
    a2.set_xlabel("intrinsic day-covariance  (per-cell upregulation)")
    a2.set_ylabel("compositional day-covariance  (cell-number)")
    a2.set_title("Per accruing gene: cell-intrinsic vs compositional\n"
                 "(above the diagonal = composition-dominated)", fontsize=11)
    a2.legend(fontsize=8.5, frameon=False, loc="upper left")
    a2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figures", "composition_intrinsic.png"),
                dpi=135, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "figures", "composition_intrinsic.pdf"),
                bbox_inches="tight")
    plt.close(fig)


def _report(results, sub):
    L = [f"\n=== Compositional vs cell-intrinsic accrual (n={results['n_animals']}, "
         f"{results['n_cell_types']} cell types) ==="]
    L.append(f"Accrual genes (full set, r_day>=0.30): {results['n_accrual_genes_full']}  |  "
             f"classified (r_day>=0.40): {results['n_accrual_genes_classified']}")
    L.append(f"median shares -> intrinsic {results['median_intrinsic_share']:+.2f}  "
             f"compositional {results['median_compositional_share']:+.2f}  "
             f"interaction {results['median_interaction_share']:+.2f}")
    dc = results["driver_counts"]
    L.append(f"driver: intrinsic={dc['intrinsic']}  compositional={dc['compositional']}  "
             f"mixed={dc['mixed']}")
    L.append("\n-- Cell-type fraction trend with duration (severity removed) --")
    for nm, v in sorted(results["cell_type_fraction_day_trend"].items(), key=lambda t: -t[1]):
        arrow = "expands" if v > 0 else "shrinks"
        L.append(f"   {nm:16s} r={v:+.2f}  {arrow}")
    L.append("\n-- Top COMPOSITIONAL accrual genes (cell-number-driven) --")
    for r in results["top_compositional_accrual"][:12]:
        L.append(f"   {r['gene']:<14} comp_share={r['comp_share']:+.2f}  "
                 f"intr_share={r['intr_share']:+.2f}  r_day={r['r_day_total']:+.2f}")
    L.append("\n-- Top INTRINSIC accrual genes (per-cell upregulation) --")
    for r in results["top_intrinsic_accrual"][:12]:
        L.append(f"   {r['gene']:<14} intr_share={r['intr_share']:+.2f}  "
                 f"comp_share={r['comp_share']:+.2f}  r_day={r['r_day_total']:+.2f}")
    report = "\n".join(L)
    print(report)
    with open(os.path.join(OUT_DIR, "report.txt"), "w") as fh:
        fh.write(report + "\n")


if __name__ == "__main__":
    main()
