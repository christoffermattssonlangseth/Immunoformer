"""Three myeloid states of the RRMAP2 EAE lesion as a recruitment->activation->resolution axis.

The relapse atlas (docs/rrmap2-relapse-atlas.md) describes a two-program cycle
(acute-reversible oscillators that reset each relapse; a cumulative ratchet that
accrues) on a concentric lesion (core -> margin -> astro rim). Within the
inflammatory-myeloid compartment (leiden_1) three states recur:

  c5  = acute glycolytic / M2 activated  (Arg1/Chil3/Hal/Acod1 — the oscillator)
  c3  = repair / resident-like           (Mrc1/Igf1/Timp1/C6 — persistent baseline)
  c18 = antigen-presenting / IFN         (Cd74/H2-Ab1/Ciita/Cxcl10/Cxcl9/Ccr2/Plac8)

Microglia (c0/c20: P2ry12/Tmem119) are SEPARATE and ratchet monotonically.

This script asks whether c18 -> c5 -> c3 is a coherent
recruitment(peripheral) -> activation(core) -> resolution(persistent) axis:
  1. marker identity from per-cluster pseudobulk (states x markers heatmap);
  2. cycle dynamics — per-stage share of each state within myeloid and as % all
     cells; the c5<->c3 swing; each state's loading on the acute-oscillator
     (amplitude) vs cumulative-ratchet (floor_drift) axes;
  3. radial position — signed distance to lesion edge (re-using the
     lesion_radial DBSCAN/convex-hull segmentation) per state.

All from caches (no 14GB reload). Cross-sectional, severity-confounded — the
temporal ORDERING is INFERRED from a population snapshot, not observed.

    PYTHONPATH="$PWD" python scripts/myeloid_states.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from scipy.spatial import ConvexHull
from scipy.stats import mannwhitneyu
from sklearn.cluster import DBSCAN

from immunotransformer.train import resolve_device  # noqa: F401  (OpenMP guard)

OUT = "runs/myeloid_states"
PB = "runs/rr_phase_niche/agg_leiden_1.npz"
REGION = "runs/rr_region_gradient/agg_region.npz"
CELLS = "runs/lesion_radial/cells.npz"
OSC = "runs/rr_cycle_oscillation/gene_oscillation_metrics.csv"

# --- myeloid states + reference clusters ---
MYELOID = {"5": "c5_acute_glycolytic", "3": "c3_repair_resident",
           "18": "c18_antigen_IFN"}
MICROGLIA = {"0": "c0_microglia", "20": "c20_microglia"}
RADIAL_REF = {"18": "c18_antigen_IFN", "5": "c5_acute_glycolytic",
              "3": "c3_repair_resident", "13": "c13_T_NK", "9": "c9_astro"}

# Curated markers grouped by hypothesized function.
MARKERS = {
    "recruitment":   ["Ccr2", "Plac8", "Ly6c2", "Vcan", "Ms4a7"],
    "antigen_pres":  ["Ciita", "Cd74", "H2-Ab1", "H2-Aa", "H2-Eb1"],
    "interferon":    ["Cxcl10", "Cxcl9", "Isg15", "Ifit3", "Gbp2", "Irf7"],
    "activation_glyc": ["Arg1", "Acod1", "Hal", "Chil3", "Hk2", "Slc2a1", "Ldha"],
    "repair":        ["Mrc1", "Igf1", "Timp1", "Cd163", "C6", "Pf4"],
    "microglia_home": ["P2ry12", "Tmem119", "Cx3cr1", "Hexb"],
}

# --- relapse cycle ordering (atlas): pseudo-trajectory across animals ---
# cycle_rank groups stages onto the peak<->remission oscillation.
STAGE_ORDER = ["PLP CFA", "ONSET1", "ONSET2", "PEAK1", "REMISSION1",
               "PEAK2", "REMISSION2", "PEAK3", "MONOPHASIC"]
PEAK_STAGES = {"PEAK1", "PEAK2", "PEAK3"}
REM_STAGES = {"REMISSION1", "REMISSION2"}

# --- lesion segmentation (identical to scripts/lesion_radial.py) ---
INFL = {"5", "3", "18", "13", "21"}
EPS, MIN_SAMPLES, MIN_LESION = 60.0, 10, 20
BIN_LO, BIN_HI = -80.0, 160.0


# ---------------------------------------------------------------------------
# 1. MARKER IDENTITY
# ---------------------------------------------------------------------------
def marker_identity():
    d = np.load(PB, allow_pickle=True)
    pb, genes, clusters = d["pb"], list(d["genes"]), list(d["clusters"])
    counts = d["counts"]  # [animal, cluster] cell counts
    gi = {g: i for i, g in enumerate(genes)}
    ci = {c: i for i, c in enumerate(clusters)}

    # count-weighted mean expression per cluster across animals -> [cluster, gene]
    # pb is per-animal mean expression; weight by cells so big animals dominate fairly.
    w = counts / (counts.sum(0, keepdims=True) + 1e-9)        # [animal, cluster]
    clus_expr = np.einsum("acg,ac->cg", pb, w)                # [cluster, gene]

    # z-score each gene across the four states of interest for the heatmap
    states = ["18", "5", "3", "0", "20"]                      # ordering for display
    state_lbl = {**MYELOID, **MICROGLIA}
    rows, table = [], {}
    flat_markers = [(grp, g) for grp, gs in MARKERS.items() for g in gs if g in gi]
    M = np.array([[clus_expr[ci[s], gi[g]] for s in states] for grp, g in flat_markers])
    # z across the displayed states (row-wise)
    Mz = (M - M.mean(1, keepdims=True)) / (M.std(1, keepdims=True) + 1e-9)

    for (grp, g), raw, z in zip(flat_markers, M, Mz):
        table[g] = {"group": grp,
                    **{state_lbl[s]: float(r) for s, r in zip(states, raw)}}
        rows.append({"group": grp, "gene": g,
                     **{s: float(v) for s, v in zip(states, z)}})
    heat = pd.DataFrame(rows)
    raw_table = pd.DataFrame(
        [{"gene": g, "group": v["group"], **{k: v[k] for k in v if k != "group"}}
         for g, v in table.items()])
    return heat, raw_table, states, [state_lbl[s] for s in states]


# ---------------------------------------------------------------------------
# 2. CYCLE DYNAMICS
# ---------------------------------------------------------------------------
def cycle_dynamics():
    d = np.load(PB, allow_pickle=True)
    clusters = list(d["clusters"])
    counts = d["counts"]                                       # [animal, cluster]
    stage = np.array([str(s) for s in d["stage"]])
    animals = list(d["animals"])
    score = d["score"]
    ci = {c: i for i, c in enumerate(clusters)}

    myel_cols = [ci[c] for c in MYELOID]
    micro_cols = [ci[c] for c in MICROGLIA]

    # per-animal compartment counts
    n_total = counts.sum(1)
    n_myel = counts[:, myel_cols].sum(1)

    rows = []
    for a in range(len(animals)):
        r = {"animal": animals[a], "stage": stage[a], "score": float(score[a]),
             "n_total": float(n_total[a]), "n_myel": float(n_myel[a])}
        for c, name in MYELOID.items():
            n = counts[a, ci[c]]
            r[f"{name}_of_myel"] = float(n / n_myel[a]) if n_myel[a] > 0 else np.nan
            r[f"{name}_of_all"] = float(n / n_total[a]) if n_total[a] > 0 else np.nan
        for c, name in MICROGLIA.items():
            r[f"{name}_of_all"] = float(counts[a, ci[c]] / n_total[a]) if n_total[a] > 0 else np.nan
        rows.append(r)
    A = pd.DataFrame(rows)

    # per-stage means (animal-level), ordered along the cycle
    A["stage"] = pd.Categorical(A["stage"], categories=STAGE_ORDER, ordered=True)
    share_cols = [c for c in A.columns if c.endswith("_of_myel") or c.endswith("_of_all")]
    by_stage = A.groupby("stage", observed=True)[share_cols + ["score"]].mean()
    n_by_stage = A.groupby("stage", observed=True).size().rename("n_animals")

    # peak vs remission contrast for the c5<->c3 swing
    pk = A[A["stage"].isin(PEAK_STAGES)]
    rm = A[A["stage"].isin(REM_STAGES)]
    swing = {}
    for c, name in MYELOID.items():
        col = f"{name}_of_myel"
        u, p = mannwhitneyu(pk[col].dropna(), rm[col].dropna(), alternative="two-sided")
        swing[name] = {"peak_mean_of_myel": float(pk[col].mean()),
                       "rem_mean_of_myel": float(rm[col].mean()),
                       "peak_minus_rem": float(pk[col].mean() - rm[col].mean()),
                       "mwu_p": float(p),
                       "peak_mean_of_all": float(pk[f"{name}_of_all"].mean()),
                       "rem_mean_of_all": float(rm[f"{name}_of_all"].mean())}
    return A, by_stage, n_by_stage, swing


# ---------------------------------------------------------------------------
# 2b. STATE LOADING ON OSCILLATOR vs RATCHET AXES
# ---------------------------------------------------------------------------
def state_axis_loadings():
    """For each myeloid/microglia state, take its top marker genes and report their
    mean amplitude (acute-oscillator) and floor_drift (ratchet) from the cycle
    oscillation metrics. Uses count-weighted cluster expression to pick the genes
    most specific to each state."""
    osc = pd.read_csv(OSC).set_index("gene")
    d = np.load(PB, allow_pickle=True)
    pb, genes, clusters, counts = d["pb"], list(d["genes"]), list(d["clusters"]), d["counts"]
    gi = {g: i for i, g in enumerate(genes)}
    ci = {c: i for i, c in enumerate(clusters)}
    w = counts / (counts.sum(0, keepdims=True) + 1e-9)
    clus_expr = np.einsum("acg,ac->cg", pb, w)                # [cluster, gene]

    states = {**MYELOID, **MICROGLIA}
    # specificity score: cluster expr minus mean over other clusters
    other = clus_expr.mean(0, keepdims=True)
    spec = clus_expr - other                                  # [cluster, gene]

    loadings = {}
    for c, name in states.items():
        s = spec[ci[c]]
        top_idx = np.argsort(s)[::-1][:30]
        top_genes = [genes[i] for i in top_idx]
        common = [g for g in top_genes if g in osc.index]
        sub = osc.loc[common]
        loadings[name] = {
            "n_top_genes_in_osc": int(len(common)),
            "mean_amplitude": float(sub["amplitude"].mean()),
            "mean_floor_drift": float(sub["floor_drift"].mean()),
            "frac_acute_oscillating": float((sub["class"] == "acute_oscillating").mean()),
            "top_markers": top_genes[:12],
        }
    return loadings, osc


# ---------------------------------------------------------------------------
# 3. RADIAL POSITION (signed distance to lesion edge)
# ---------------------------------------------------------------------------
def signed_dist_to_hull(pts, hp):
    path = MplPath(hp)
    inside = path.contains_points(pts)
    a = hp
    b = np.roll(hp, -1, axis=0)
    ab = b - a
    ab2 = (ab ** 2).sum(1) + 1e-12
    d = np.full(len(pts), np.inf)
    for e in range(len(a)):
        ap = pts - a[e]
        t = np.clip((ap @ ab[e]) / ab2[e], 0, 1)
        proj = a[e] + t[:, None] * ab[e]
        d = np.minimum(d, np.hypot(*(pts - proj).T))
    return np.where(inside, -d, d)


def radial_position():
    d = np.load(CELLS, allow_pickle=True)
    xy = d["xy"]
    sec = d["sec"].astype(str)
    leiden = d["leiden"].astype(str)
    meta = d["meta"].item()

    is_infl = np.isin(leiden, list(INFL))
    sections = pd.unique(sec)

    # per-cell signed distance (NaN if not within any lesion's bbox+margin)
    sd_all = np.full(len(xy), np.nan)
    n_lesions = 0
    for s in sections:
        m = meta.get(s)
        if m is None:
            continue
        idx = np.where(sec == s)[0]
        sub_infl = idx[is_infl[idx]]
        if len(sub_infl) < MIN_LESION:
            continue
        lab = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit_predict(xy[sub_infl])
        for L in set(lab) - {-1}:
            mem = sub_infl[lab == L]
            if len(mem) < MIN_LESION:
                continue
            try:
                hull = ConvexHull(xy[mem])
            except Exception:
                continue
            hp = xy[mem][hull.vertices]
            n_lesions += 1
            lo, hi = hp.min(0) - 160, hp.max(0) + 160
            cand = idx[(xy[idx, 0] >= lo[0]) & (xy[idx, 0] <= hi[0]) &
                       (xy[idx, 1] >= lo[1]) & (xy[idx, 1] <= hi[1])]
            if len(cand) < 5:
                continue
            sd = signed_dist_to_hull(xy[cand], hp)
            keep = (sd >= BIN_LO) & (sd < BIN_HI)
            cand, sd = cand[keep], sd[keep]
            # a cell can fall in multiple lesions' bboxes; keep the most-core (min) value
            cur = sd_all[cand]
            take = np.isnan(cur) | (sd < cur)
            sd_all[cand[take]] = sd[take]

    # collect signed distances per reference cluster + animal id (for animal-level stats)
    animal_of = np.array([meta.get(s, {}).get("animal", "NA") for s in sec])
    stage_of = np.array([meta.get(s, {}).get("stage", "NA") for s in sec])

    per_state = {}
    for c, name in RADIAL_REF.items():
        sel = (leiden == c) & ~np.isnan(sd_all)
        vals = sd_all[sel]
        per_state[name] = {
            "n_cells": int(sel.sum()),
            "mean_sd": float(np.mean(vals)) if len(vals) else np.nan,
            "median_sd": float(np.median(vals)) if len(vals) else np.nan,
            "frac_core": float(np.mean(vals < 0)) if len(vals) else np.nan,     # inside hull
            "frac_margin": float(np.mean((vals >= 0) & (vals < 40))) if len(vals) else np.nan,
            "q25": float(np.percentile(vals, 25)) if len(vals) else np.nan,
            "q75": float(np.percentile(vals, 75)) if len(vals) else np.nan,
        }

    # animal-level mean signed distance per state, then paired Wilcoxon-style contrasts
    rows = []
    for c, name in RADIAL_REF.items():
        sel = (leiden == c) & ~np.isnan(sd_all)
        df = pd.DataFrame({"animal": animal_of[sel], "sd": sd_all[sel]})
        am = df.groupby("animal")["sd"].mean()
        for an, v in am.items():
            rows.append({"state": name, "animal": an, "mean_sd": float(v)})
    animal_sd = pd.DataFrame(rows)

    # pairwise: is c18 more peripheral than c5? (positive = more peripheral)
    contrasts = {}
    piv = animal_sd.pivot_table(index="animal", columns="state", values="mean_sd")
    pairs = [("c18_antigen_IFN", "c5_acute_glycolytic"),
             ("c18_antigen_IFN", "c3_repair_resident"),
             ("c5_acute_glycolytic", "c3_repair_resident"),
             ("c13_T_NK", "c5_acute_glycolytic"),
             ("c9_astro", "c5_acute_glycolytic")]
    from scipy.stats import wilcoxon
    for a, b in pairs:
        if a in piv and b in piv:
            sub = piv[[a, b]].dropna()
            if len(sub) >= 6:
                diff = sub[a] - sub[b]
                try:
                    w, p = wilcoxon(sub[a], sub[b])
                except Exception:
                    w, p = np.nan, np.nan
                contrasts[f"{a}_minus_{b}"] = {
                    "n_animals": int(len(sub)),
                    "mean_diff_um": float(diff.mean()),
                    "wilcoxon_p": float(p) if p == p else None,
                }

    # raw per-cell signed distances for violin plotting (subsample)
    rng = np.random.default_rng(0)
    viol = {}
    for c, name in RADIAL_REF.items():
        sel = (leiden == c) & ~np.isnan(sd_all)
        vals = sd_all[sel]
        if len(vals) > 8000:
            vals = rng.choice(vals, 8000, replace=False)
        viol[name] = vals
    return per_state, contrasts, animal_sd, viol, n_lesions


# ---------------------------------------------------------------------------
# FIGURES
# ---------------------------------------------------------------------------
def figures(heat, states, state_labels, by_stage, viol, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig_dir = os.path.join(out, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # --- (1) marker heatmap: states x markers ---
    M = heat[states].to_numpy().T                             # [state, marker]
    fig, ax = plt.subplots(figsize=(max(10, 0.32 * len(heat)), 3.6))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-1.5, vmax=1.5)
    ax.set_yticks(range(len(states)))
    ax.set_yticklabels(state_labels)
    ax.set_xticks(range(len(heat)))
    ax.set_xticklabels(heat["gene"], rotation=90, fontsize=7)
    # group separators
    grp = heat["group"].to_numpy()
    for i in range(1, len(grp)):
        if grp[i] != grp[i - 1]:
            ax.axvline(i - 0.5, c="k", lw=0.6)
    ax.set_title("Myeloid-state marker identity (row z-scored across states)")
    fig.colorbar(im, ax=ax, fraction=0.018, pad=0.01, label="z")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "marker_heatmap.png"), dpi=140, bbox_inches="tight")
    fig.savefig(os.path.join(fig_dir, "marker_heatmap.pdf"), bbox_inches="tight")
    plt.close(fig)

    # --- (2) state-share-vs-cycle lines ---
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.6))
    x = list(range(len(by_stage.index)))
    colors = {"c5_acute_glycolytic": "#c0392b", "c3_repair_resident": "#2f9e44",
              "c18_antigen_IFN": "#1f6feb"}
    for c, name in MYELOID.items():
        axA.plot(x, by_stage[f"{name}_of_myel"], "-o", lw=2, label=name, color=colors[name])
    axA.set_xticks(x)
    axA.set_xticklabels(by_stage.index, rotation=45, ha="right", fontsize=8)
    axA.set_ylabel("share of myeloid compartment")
    axA.set_title("Myeloid state share across the relapse cycle")
    axA.legend(fontsize=8, frameon=False)
    axA.spines[["top", "right"]].set_visible(False)
    ax2 = axA.twinx()
    ax2.plot(x, by_stage["score"], "--", color="gray", lw=1.2, label="score")
    ax2.set_ylabel("clinical score", color="gray")
    for c, name in MYELOID.items():
        axB.plot(x, by_stage[f"{name}_of_all"], "-o", lw=2, label=name, color=colors[name])
    axB.set_xticks(x)
    axB.set_xticklabels(by_stage.index, rotation=45, ha="right", fontsize=8)
    axB.set_ylabel("share of ALL cells")
    axB.set_title("Myeloid state as % of all cells")
    axB.legend(fontsize=8, frameon=False)
    axB.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "state_share_vs_cycle.png"), dpi=140, bbox_inches="tight")
    fig.savefig(os.path.join(fig_dir, "state_share_vs_cycle.pdf"), bbox_inches="tight")
    plt.close(fig)

    # --- (3) radial position violin per state ---
    order = ["c18_antigen_IFN", "c5_acute_glycolytic", "c3_repair_resident",
             "c13_T_NK", "c9_astro"]
    data = [viol[k] for k in order if k in viol]
    labs = [k for k in order if k in viol]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    parts = ax.violinplot(data, showmedians=True, widths=0.85)
    for i, b in enumerate(parts["bodies"]):
        b.set_alpha(0.7)
    ax.axhline(0, c="k", lw=0.8, ls="--")
    ax.set_xticks(range(1, len(labs) + 1))
    ax.set_xticklabels(labs, rotation=20, ha="right")
    ax.set_ylabel("signed distance to lesion edge (µm)\n← core    margin / rim →")
    ax.set_title("Radial position of states (per cell)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "radial_position.png"), dpi=140, bbox_inches="tight")
    fig.savefig(os.path.join(fig_dir, "radial_position.pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)

    print("[1] marker identity ...")
    heat, raw_table, states, state_labels = marker_identity()

    print("[2] cycle dynamics ...")
    A, by_stage, n_by_stage, swing = cycle_dynamics()

    print("[2b] state axis loadings (oscillator vs ratchet) ...")
    loadings, osc = state_axis_loadings()

    print("[3] radial position (segmenting lesions) ...")
    per_state, contrasts, animal_sd, viol, n_lesions = radial_position()

    print("[fig] figures ...")
    figures(heat, states, state_labels, by_stage, viol, OUT)

    # --- save tables ---
    raw_table.to_csv(os.path.join(OUT, "marker_table.csv"), index=False)
    by_stage.to_csv(os.path.join(OUT, "state_share_by_stage.csv"))
    animal_sd.to_csv(os.path.join(OUT, "radial_animal_sd.csv"), index=False)

    results = {
        "n_lesions": int(n_lesions),
        "cycle_swing_c5_c3_c18": swing,
        "axis_loadings": loadings,
        "radial_per_state": per_state,
        "radial_contrasts": contrasts,
        "n_animals_by_stage": {str(k): int(v) for k, v in n_by_stage.items()},
    }
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    # --- report ---
    L = ["\n=== THREE MYELOID STATES: recruitment -> activation -> resolution ===",
         "(RRMAP2 RR cohort, cross-sectional / severity-confounded — ordering INFERRED)\n",
         "-- 1. MARKER IDENTITY (count-weighted cluster mean expression) --"]
    show = ["Ccr2", "Plac8", "Ly6c2", "Ciita", "Cd74", "H2-Ab1", "Cxcl10", "Gbp2",
            "Arg1", "Acod1", "Hal", "Chil3", "Mrc1", "Igf1", "Timp1", "C6",
            "P2ry12", "Tmem119"]
    rt = raw_table.set_index("gene")
    hdr = f"  {'gene':<9}" + "".join(f"{s:>22}" for s in state_labels)
    L.append(hdr)
    for g in show:
        if g in rt.index:
            r = rt.loc[g]
            L.append(f"  {g:<9}" + "".join(f"{r[s]:>22.3f}" for s in state_labels))

    L.append("\n-- 2. CYCLE DYNAMICS: c5<->c3 swing within myeloid compartment --")
    for name, sw in swing.items():
        L.append(f"  {name:<22} peak={sw['peak_mean_of_myel']:.3f}  "
                 f"rem={sw['rem_mean_of_myel']:.3f}  "
                 f"peak-rem={sw['peak_minus_rem']:+.3f}  MWU p={sw['mwu_p']:.3g}")
    L.append("  (% of all cells)")
    for name, sw in swing.items():
        L.append(f"  {name:<22} peak={sw['peak_mean_of_all']:.4f}  rem={sw['rem_mean_of_all']:.4f}")

    L.append("\n-- 2b. STATE LOADING on oscillator (amplitude) vs ratchet (floor_drift) --")
    L.append(f"  {'state':<22}{'amplitude':>11}{'floor_drift':>13}{'frac_acute':>12}")
    for name, ld in loadings.items():
        L.append(f"  {name:<22}{ld['mean_amplitude']:>11.3f}{ld['mean_floor_drift']:>13.3f}"
                 f"{ld['frac_acute_oscillating']:>12.2f}")

    L.append(f"\n-- 3. RADIAL POSITION ({n_lesions} lesions; signed dist µm, neg=core) --")
    L.append(f"  {'state':<22}{'n_cells':>9}{'mean_sd':>9}{'median':>8}{'frac_core':>10}{'frac_margin':>12}")
    for name, ps in per_state.items():
        L.append(f"  {name:<22}{ps['n_cells']:>9}{ps['mean_sd']:>9.1f}{ps['median_sd']:>8.1f}"
                 f"{ps['frac_core']:>10.2f}{ps['frac_margin']:>12.2f}")
    L.append("  pairwise animal-level contrasts (mean diff µm; + = first more peripheral):")
    for k, v in contrasts.items():
        L.append(f"    {k:<46} {v['mean_diff_um']:+7.1f}  Wilcoxon p={v['wilcoxon_p']}  (n={v['n_animals']})")

    report = "\n".join(L)
    print(report)
    with open(os.path.join(OUT, "report.txt"), "w") as fh:
        fh.write(report + "\n")
    print(f"\n[done] -> {OUT}/ (results.json, report.txt, *.csv, figures/)")


if __name__ == "__main__":
    main()
