"""RRMAP2 atlas — LESION-FIELD MORPHOMETRY: is there spatial ARCHITECTURE beyond severity?

Every prior null in this project (bulk, 10 niches, 25 cell types, composition;
peak-vs-remission, direction-of-travel) was on EXPRESSION or COMPOSITION — never on
lesion GEOMETRY. This script tests the last untested layer: the shape, size, number
and confluence of inflammatory lesion fields, and asks whether any morphometric
feature survives residualization on clinical severity (score_sacrifice) and on the
total inflammatory burden. If yes, there is structure beyond severity; if null, it is
a clean negative on the geometry layer.

Pipeline:
  1. Per section (meta_sample_id, the one-tissue-section unit; coords meaningful only
     within), take inflammatory cells (leiden_1 in INFL set) and segment LESIONS as
     connected components of the precomputed intra-section 6-NN spatial graph
     (obsp['spatial_connectivities'], ~77.5um radius, 0 cross-section edges),
     restricted to inflammatory cells. A lesion = a connected inflammatory field with
     >= MIN_LESION_CELLS cells.
  2. Per-lesion morphometry: cell count, convex-hull area, density, compactness
     (4*pi*area/perimeter^2), inflammatory purity, lymphocyte (T/B) fraction,
     nearest-lesion distance. Per-SECTION summary: lesion count, lesion burden
     (fraction of cells in lesions), mean/max lesion size, CONFLUENCE (largest-lesion
     fraction + Gini of lesion sizes), fragmentation (lesions per 1000 infl cells).
  3. Aggregate section -> animal (animal = unit of inference; 33 animals).
  4. THE NOVEL TEST — morphology beyond severity: residualize each section-level
     morphometric on score_sacrifice and on total infl fraction (OLS), aggregate the
     residuals to animals, then test (a) peak vs remission, (b) region L->T->C trend,
     via permutation nulls (animal-label shuffles) with honest small-n caveats.
  5. Figure: 2-3 representative sections colored by lesion segmentation + scatter of
     confluence/size vs severity.

    PYTHONPATH="$PWD" python scripts/lesion_morphometry.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import spearmanr

from immunotransformer.train import resolve_device  # noqa: F401  (OpenMP guard)

RRMAP2 = ("/Volumes/moldiassd/RRMAP2_xenium_adata/kmeans_separated/"
          "RRMAP2_xenium_all_samples.cellcharter.companion.ready.with_metadata.rerun.h5ad")
OUT_DIR = "runs/lesion_morphometry"
CELL_CACHE = os.path.join(OUT_DIR, "infl_cells.npz")   # heavy per-cell extraction
CLUSTER_COL = "leiden_1"

# Inflammatory cell-type sets (leiden_1; from cluster_labels.json / atlas).
INFL_MYELOID = ["5", "3", "18"]   # 5 = Hal-high lesion state, 3 & 18 other infl-myeloid
LYMPHO = ["13", "21"]             # 13 T/NK, 21 B/plasma
INFL_SET = INFL_MYELOID + LYMPHO  # full inflammatory field

MIN_LESION_CELLS = 20             # a lesion = connected infl field >= 20 cells
REGION_RANK = {"L": 0, "T": 1, "C": 2}

# Stage -> phase grouping (PEAK ~ score 2.6, REMISSION ~ 0.9).
PEAK_STAGES = {"PEAK1", "PEAK2", "PEAK3"}
REM_STAGES = {"REMISSION1", "REMISSION2"}

# Morphometric features tested for beyond-severity structure (section level).
# (mean_purity is constant 1.0 by construction of the infl-only graph -> not tested.)
MORPH_FEATURES = [
    "lesion_count", "lesion_burden", "mean_lesion_size", "max_lesion_size",
    "largest_lesion_frac", "gini_lesion_size", "fragmentation",
    "mean_density", "mean_compactness", "mean_lympho_frac",
    "mean_nn_dist",
]


# ----------------------------------------------------------------------------- utils
def bh_fdr(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = p[o] * n / (np.arange(n) + 1)
    q[o] = np.minimum.accumulate(q[o][::-1])[::-1]
    return np.clip(q, 0, 1)


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return np.nan
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def ols_resid(y, X):
    """Residualize y on design X (columns already include an intercept)."""
    y = np.asarray(y, float)
    keep = np.isfinite(y) & np.isfinite(X).all(1)
    res = np.full_like(y, np.nan)
    if keep.sum() < X.shape[1] + 1:
        return res
    beta, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
    res[keep] = y[keep] - X[keep] @ beta
    return res


# ----------------------------------------------------------- heavy cell extraction
def build_or_load_cells():
    """Extract, per RR cell: section code, xy, is-infl, infl-subtype, animal/meta.

    Also slice the intra-section infl-only spatial graph once. Cached to disk because
    this touches the 14GB load.
    """
    if os.path.exists(CELL_CACHE):
        d = np.load(CELL_CACHE, allow_pickle=True)
        print(f"[cache] {CELL_CACHE}")
        meta = pd.read_parquet(os.path.join(OUT_DIR, "section_meta.parquet"))
        return d, meta

    import anndata as ad
    print(f"[load] {RRMAP2}")
    a = ad.read_h5ad(RRMAP2)
    a = a[a.obs["model"] == "RELAPSE REMITTING"].copy()
    o = a.obs
    print(f"       RR: {a.n_obs:,} cells")

    lab = o[CLUSTER_COL].astype(str).to_numpy()
    is_infl = np.isin(lab, INFL_SET)
    is_lympho = np.isin(lab, LYMPHO)
    xy = np.asarray(a.obsm["spatial"], float)

    # section codes
    sections = list(pd.unique(o["meta_sample_id"].astype(str)))
    s_code = {s: i for i, s in enumerate(sections)}
    sc = o["meta_sample_id"].astype(str).map(s_code).to_numpy()

    # per-section metadata (constant within section): animal, stage, region, score
    meta = (o.assign(_sec=o["meta_sample_id"].astype(str))
            .groupby("_sec", observed=True)
            .agg(sample_name=("sample_name", "first"),
                 stage=("stage", "first"),
                 region=("region", "first"),
                 score=("score_sacrifice", "first"),
                 n_cells=("sample_name", "size"))
            .reset_index().rename(columns={"_sec": "meta_sample_id"}))
    meta["sec_code"] = meta["meta_sample_id"].map(s_code)
    meta = meta.sort_values("sec_code").reset_index(drop=True)

    # intra-section infl-only spatial graph: keep edges between two infl cells.
    G = a.obsp["spatial_connectivities"].tocsr()
    infl_idx = np.where(is_infl)[0]
    Gi = G[infl_idx][:, infl_idx]            # subgraph on infl cells only
    Gi = sp.csr_matrix(Gi)
    Gi.data[:] = 1.0
    Gi = Gi.maximum(Gi.T)                    # symmetrize
    # RESTRICT to within-section: the global precomputed graph has 14.6% CROSS-section
    # edges (sections share one coordinate frame; median 7.9um links between overlaid
    # sections). Drop any edge joining two different meta_sample_id sections, else
    # lesions merge across sections. (The earlier "0 cross edges" claim was wrong.)
    infl_sec_local = sc[infl_idx]
    Gc = Gi.tocoo()
    keep = infl_sec_local[Gc.row] == infl_sec_local[Gc.col]
    Gi = sp.csr_matrix((Gc.data[keep], (Gc.row[keep], Gc.col[keep])), shape=Gi.shape)

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(
        CELL_CACHE,
        sc=sc, xy=xy, is_infl=is_infl, is_lympho=is_lympho,
        infl_idx=infl_idx,
        gi_data=Gi.data, gi_indices=Gi.indices, gi_indptr=Gi.indptr, gi_shape=Gi.shape,
        sections=np.array(sections),
    )
    meta.to_parquet(os.path.join(OUT_DIR, "section_meta.parquet"))
    print(f"[cache] wrote {CELL_CACHE}")
    d = np.load(CELL_CACHE, allow_pickle=True)
    return d, meta


# --------------------------------------------------------------- lesion segmentation
def segment_lesions(d, meta):
    """Connected components of the infl-only intra-section graph -> lesions."""
    from scipy.sparse.csgraph import connected_components

    sc = d["sc"]
    xy = d["xy"]
    is_lympho = d["is_lympho"]
    infl_idx = d["infl_idx"]
    Gi = sp.csr_matrix((d["gi_data"], d["gi_indices"], d["gi_indptr"]),
                       shape=tuple(d["gi_shape"]))

    n_comp, comp = connected_components(Gi, directed=False)
    # component -> section (infl cells in one component all share a section: 0 cross
    # edges, verified). Map each infl cell to its global section.
    infl_sec = sc[infl_idx]
    infl_xy = xy[infl_idx]
    infl_lympho = is_lympho[infl_idx]

    lesion_rows = []
    # group infl cells by component
    order = np.argsort(comp, kind="stable")
    comp_s = comp[order]
    bnd = np.where(np.diff(comp_s) != 0)[0] + 1
    starts = np.concatenate([[0], bnd])
    ends = np.concatenate([bnd, [len(comp_s)]])
    for st, en in zip(starts, ends):
        members = order[st:en]
        n = en - st
        if n < MIN_LESION_CELLS:
            continue
        sec = int(infl_sec[members[0]])
        pts = infl_xy[members]
        area, perim = hull_area_perim(pts)
        density = n / area if area and area > 0 else np.nan
        compact = (4 * np.pi * area / perim ** 2) if (area and perim and perim > 0) else np.nan
        lympho_frac = float(infl_lympho[members].mean())
        cx, cy = pts.mean(0)
        lesion_rows.append({
            "sec_code": sec, "n_cells": int(n), "area": area, "density": density,
            "compactness": compact, "lympho_frac": lympho_frac, "cx": cx, "cy": cy,
        })

    les = pd.DataFrame(lesion_rows)
    # nearest-lesion distance (within section, centroid-to-centroid)
    les["nn_dist"] = np.nan
    for sec, g in les.groupby("sec_code"):
        if len(g) < 2:
            continue
        c = g[["cx", "cy"]].to_numpy()
        D = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(D, np.inf)
        les.loc[g.index, "nn_dist"] = D.min(1)
    return les, comp, infl_idx


def hull_area_perim(pts):
    if len(pts) < 3:
        return np.nan, np.nan
    try:
        h = ConvexHull(pts)
        return float(h.volume), float(h.area)   # 2D: volume=area, area=perimeter
    except (QhullError, ValueError):
        return np.nan, np.nan


# ----------------------------------------------------------- section-level summary
def section_summary(les, meta, d):
    """One row per section: lesion stats + total infl fraction + purity."""
    sc = d["sc"]
    is_infl = d["is_infl"]
    n_sec = len(meta)
    # total cells & infl cells per section
    tot = np.bincount(sc, minlength=n_sec).astype(float)
    infl_tot = np.bincount(sc[is_infl], minlength=n_sec).astype(float)

    rows = []
    grp = {sec: g for sec, g in les.groupby("sec_code")}
    for _, m in meta.iterrows():
        sec = int(m["sec_code"])
        g = grp.get(sec)
        ncell = tot[sec]
        infl_frac = infl_tot[sec] / ncell if ncell > 0 else np.nan
        row = {"meta_sample_id": m["meta_sample_id"], "sec_code": sec,
               "sample_name": m["sample_name"], "stage": m["stage"],
               "region": m["region"], "score": m["score"], "n_cells": ncell,
               "infl_frac": infl_frac}
        if g is None or len(g) == 0:
            row.update({f: 0.0 if f in ("lesion_count", "lesion_burden") else np.nan
                        for f in MORPH_FEATURES})
            row["lesion_count"] = 0
            row["lesion_burden"] = 0.0
        else:
            sizes = g["n_cells"].to_numpy()
            in_lesion = sizes.sum()
            row["lesion_count"] = int(len(g))
            row["lesion_burden"] = float(in_lesion / ncell) if ncell > 0 else np.nan
            row["mean_lesion_size"] = float(sizes.mean())
            row["max_lesion_size"] = float(sizes.max())
            row["largest_lesion_frac"] = float(sizes.max() / in_lesion)
            row["gini_lesion_size"] = gini(sizes)
            row["fragmentation"] = float(len(g) / (infl_tot[sec] / 1000.0)) \
                if infl_tot[sec] > 0 else np.nan
            row["mean_density"] = float(np.nanmean(g["density"]))
            row["mean_compactness"] = float(np.nanmean(g["compactness"]))
            row["mean_lympho_frac"] = float(np.nanmean(g["lympho_frac"]))
            # purity: fraction of in-lesion cells that are infl (=1 by construction of
            # the infl-only graph, but kept for clarity / alpha-shape variants)
            row["mean_purity"] = 1.0
            row["mean_nn_dist"] = float(np.nanmean(g["nn_dist"]))
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------- beyond-severity test
def beyond_severity(sec_df, n_perm=10000, seed=0):
    """Residualize each section feature on score (+ infl_frac), aggregate to animal,
    test peak-vs-remission and region L->T->C trend with animal-shuffle permutation."""
    rng = np.random.default_rng(seed)
    df = sec_df.copy()

    # designs: residualize on severity only, and on severity + burden
    n = len(df)
    one = np.ones(n)
    X_sev = np.column_stack([one, df["score"].to_numpy()])
    X_sevburden = np.column_stack([one, df["score"].to_numpy(), df["infl_frac"].to_numpy()])

    results = {"peak_vs_rem": [], "region_trend": []}

    for feat in MORPH_FEATURES:
        y = df[feat].to_numpy(float)
        for resid_on, X in [("severity", X_sev), ("severity+burden", X_sevburden)]:
            r = ols_resid(y, X)
            tmp = df.assign(resid=r)

            # ---- aggregate section residuals to ANIMAL (unit of inference) ----
            an = (tmp.dropna(subset=["resid"])
                  .groupby("sample_name", observed=True)
                  .agg(resid=("resid", "mean"),
                       stage=("stage", "first"))
                  .reset_index())
            # animal phase: an animal's sections share a stage (terminal) -> 1 phase
            an["phase"] = an["stage"].map(
                lambda s: "PEAK" if s in PEAK_STAGES else ("REM" if s in REM_STAGES else None))

            # (a) peak vs remission, animal level
            pk = an.loc[an["phase"] == "PEAK", "resid"].to_numpy()
            rm = an.loc[an["phase"] == "REM", "resid"].to_numpy()
            if len(pk) >= 2 and len(rm) >= 2:
                obs = pk.mean() - rm.mean()
                pooled = np.concatenate([pk, rm])
                k = len(pk)
                perm = np.empty(n_perm)
                for i in range(n_perm):
                    s = rng.permutation(pooled)
                    perm[i] = s[:k].mean() - s[k:].mean()
                p = (np.sum(np.abs(perm) >= abs(obs)) + 1) / (n_perm + 1)
                results["peak_vs_rem"].append(
                    {"feature": feat, "resid_on": resid_on, "n_peak": len(pk),
                     "n_rem": len(rm), "peak_minus_rem": round(float(obs), 4),
                     "perm_p": round(float(p), 4)})

            # (b) region L->T->C trend — within-animal then across animals.
            # Use section residuals directly with region rank, animal-level slope.
            slopes = []
            for animal, gg in tmp.dropna(subset=["resid"]).groupby("sample_name", observed=True):
                gg = gg.assign(rk=gg["region"].map(REGION_RANK))
                gg = gg.dropna(subset=["rk"])
                if gg["rk"].nunique() < 2:
                    continue
                x = gg["rk"].to_numpy(float)
                yy = gg["resid"].to_numpy(float)
                xc = x - x.mean()
                if (xc ** 2).sum() == 0:
                    continue
                slopes.append(float((xc * (yy - yy.mean())).sum() / (xc ** 2).sum()))
            slopes = np.array(slopes)
            if len(slopes) >= 4:
                obs = slopes.mean()
                # permutation: sign-flip per-animal slopes (null = symmetric about 0)
                perm = np.empty(n_perm)
                for i in range(n_perm):
                    perm[i] = (slopes * rng.choice([-1, 1], len(slopes))).mean()
                p = (np.sum(np.abs(perm) >= abs(obs)) + 1) / (n_perm + 1)
                results["region_trend"].append(
                    {"feature": feat, "resid_on": resid_on, "n_animals": len(slopes),
                     "mean_slope_LtoC": round(float(obs), 4), "perm_p": round(float(p), 4)})

    # BH-FDR within each test family
    for fam in results:
        if results[fam]:
            ps = [r["perm_p"] for r in results[fam]]
            qs = bh_fdr(ps)
            for r, q in zip(results[fam], qs):
                r["q"] = round(float(q), 4)
    return results


# ------------------------------------------------------------------------- raw stats
def raw_descriptives(sec_df):
    """Animal-level descriptives by phase and by region (NOT residualized)."""
    df = sec_df.copy()
    df["phase"] = df["stage"].map(
        lambda s: "PEAK" if s in PEAK_STAGES else ("REM" if s in REM_STAGES else "OTHER"))
    feats = ["lesion_count", "lesion_burden", "mean_lesion_size", "max_lesion_size",
             "largest_lesion_frac", "gini_lesion_size", "fragmentation", "infl_frac"]
    # section -> animal mean, then group
    an = df.groupby(["sample_name", "phase", "region"], observed=True)[feats].mean().reset_index()
    by_phase = an.groupby("phase", observed=True)[feats].mean().round(3)
    by_region = an.groupby("region", observed=True)[feats].mean().round(3)

    # raw severity correlation (animal level, one value per animal)
    an_animal = df.groupby("sample_name", observed=True).agg(
        score=("score", "first"),
        **{f: (f, "mean") for f in feats}).reset_index()
    sev_corr = {}
    for f in feats:
        rho, p = spearmanr(an_animal["score"], an_animal[f], nan_policy="omit")
        sev_corr[f] = {"spearman_rho": round(float(rho), 3), "p": round(float(p), 4)}
    return by_phase, by_region, sev_corr, an_animal


# ------------------------------------------------------------------------- figures
def make_figure(les, sec_df, d, meta):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    sc = d["sc"]
    xy = d["xy"]
    is_infl = d["is_infl"]

    # pick representative sections: lumbar PEAK (max burden) vs cervical REMISSION
    df = sec_df.copy()
    df["phase"] = df["stage"].map(
        lambda s: "PEAK" if s in PEAK_STAGES else ("REM" if s in REM_STAGES else "OTHER"))
    lum_peak = df[(df["region"] == "L") & (df["phase"] == "PEAK")].sort_values(
        "lesion_burden", ascending=False)
    cerv_rem = df[(df["region"] == "C") & (df["phase"] == "REM")].sort_values(
        "lesion_burden", ascending=False)
    thor_peak = df[(df["region"] == "T") & (df["phase"] == "PEAK")].sort_values(
        "lesion_burden", ascending=False)
    panels = []
    for name, cand in [("lumbar PEAK", lum_peak), ("thoracic PEAK", thor_peak),
                       ("cervical REMISSION", cerv_rem)]:
        if len(cand):
            panels.append((name, int(cand.iloc[0]["sec_code"])))

    # map: for each section, draw all cells gray, infl cells colored by lesion id
    from scipy.sparse.csgraph import connected_components
    Gi = sp.csr_matrix((d["gi_data"], d["gi_indices"], d["gi_indptr"]),
                       shape=tuple(d["gi_shape"]))
    _, comp = connected_components(Gi, directed=False)
    infl_idx = d["infl_idx"]
    # lesion components meeting size threshold
    sizes = np.bincount(comp)
    big = set(np.where(sizes >= MIN_LESION_CELLS)[0])

    n_panel = len(panels)
    fig, axes = plt.subplots(1, n_panel + 1, figsize=(4.2 * (n_panel + 1), 4.3))
    if n_panel == 0:
        axes = [axes]
    for ax, (name, sec) in zip(axes[:n_panel], panels):
        cell_mask = sc == sec
        ax.scatter(xy[cell_mask, 0], xy[cell_mask, 1], s=0.6, c="#e3e5e8", linewidths=0)
        # infl cells in this section
        sel = np.where(cell_mask & is_infl)[0]
        # position of each infl cell in infl_idx ordering
        pos = np.searchsorted(infl_idx, sel)
        comp_sel = comp[pos]
        in_lesion = np.array([c in big for c in comp_sel])
        # color confluent lesions
        uc = pd.factorize(comp_sel[in_lesion])[0]
        ax.scatter(xy[sel[in_lesion], 0], xy[sel[in_lesion], 1], s=2.5,
                   c=uc, cmap="tab20", linewidths=0)
        # scattered (sub-threshold) infl cells in light red
        ax.scatter(xy[sel[~in_lesion], 0], xy[sel[~in_lesion], 1], s=1.2,
                   c="#f1a9a0", linewidths=0)
        srow = df[df["sec_code"] == sec].iloc[0]
        ax.set_title(f"{name}\n{srow['meta_sample_id']}\n"
                     f"burden={srow['lesion_burden']:.2f} "
                     f"n_lesion={int(srow['lesion_count'])} "
                     f"largest={srow['largest_lesion_frac']:.2f}", fontsize=8)
        ax.set_aspect("equal")
        ax.axis("off")

    # final panel: confluence / burden vs severity (animal level)
    ax = axes[-1]
    an = df.groupby("sample_name", observed=True).agg(
        score=("score", "first"),
        burden=("lesion_burden", "mean"),
        largest=("largest_lesion_frac", "mean")).reset_index()
    sccat = ax.scatter(an["score"], an["burden"], c=an["largest"], cmap="viridis",
                       s=40, edgecolors="k", linewidths=0.4)
    ax.set_xlabel("score_sacrifice (severity)")
    ax.set_ylabel("lesion burden (animal mean)")
    ax.set_title("burden & confluence vs severity\n(color = largest-lesion frac)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.colorbar(sccat, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, "figures", "lesion_morphometry.png"),
                dpi=140, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "figures", "lesion_morphometry.pdf"),
                bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------ main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    d, meta = build_or_load_cells()
    print(f"[setup] {len(meta)} RR sections, "
          f"{meta['sample_name'].nunique()} animals")

    les, comp, infl_idx = segment_lesions(d, meta)
    print(f"[segment] {len(les)} lesions (>= {MIN_LESION_CELLS} cells) "
          f"across {les['sec_code'].nunique()} sections")
    les.to_parquet(os.path.join(OUT_DIR, "lesions.parquet"))

    sec_df = section_summary(les, meta, d)
    sec_df.to_csv(os.path.join(OUT_DIR, "section_morphometry.csv"), index=False)

    by_phase, by_region, sev_corr, an_animal = raw_descriptives(sec_df)
    an_animal.to_csv(os.path.join(OUT_DIR, "animal_morphometry.csv"), index=False)

    bs = beyond_severity(sec_df)

    # ---- assemble results ----
    results = {
        "params": {"infl_set": INFL_SET, "min_lesion_cells": MIN_LESION_CELLS,
                   "n_sections": int(len(meta)),
                   "n_animals": int(meta["sample_name"].nunique()),
                   "n_lesions": int(len(les))},
        "by_phase": by_phase.reset_index().to_dict("records"),
        "by_region": by_region.reset_index().to_dict("records"),
        "raw_severity_corr": sev_corr,
        "beyond_severity_peak_vs_rem": bs["peak_vs_rem"],
        "beyond_severity_region_trend": bs["region_trend"],
    }
    with open(os.path.join(OUT_DIR, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=float)

    # ---- report ----
    L = ["\n=== RRMAP2 LESION-FIELD MORPHOMETRY (beyond-severity test) ===",
         f"{results['params']['n_lesions']} lesions, "
         f"{results['params']['n_sections']} sections, "
         f"{results['params']['n_animals']} animals "
         f"(infl set leiden_1={INFL_SET}, min {MIN_LESION_CELLS} cells)",
         "\n-- Animal-level descriptives BY PHASE --",
         by_phase.to_string(),
         "\n-- Animal-level descriptives BY REGION (L->T->C) --",
         by_region.to_string(),
         "\n-- RAW severity correlation (animal level, Spearman) --"]
    for f, v in sev_corr.items():
        L.append(f"   {f:<22} rho={v['spearman_rho']:+.3f}  p={v['p']:.3g}")

    L.append("\n-- BEYOND-SEVERITY: peak vs remission (residual, animal-level, perm) --")
    L.append("   feature                resid_on          p_minus_r   perm_p   q")
    for r in sorted(bs["peak_vs_rem"], key=lambda x: x["perm_p"]):
        L.append(f"   {r['feature']:<22} {r['resid_on']:<16} "
                 f"{r['peak_minus_rem']:+8.3f}   {r['perm_p']:.3f}   {r.get('q', float('nan')):.3f}")
    n_sig_pr = sum(1 for r in bs["peak_vs_rem"] if r.get("q", 1) < 0.05)

    L.append("\n-- BEYOND-SEVERITY: region L->T->C trend (residual, animal slopes, perm) --")
    L.append("   feature                resid_on          slope_LtoC  perm_p   q")
    for r in sorted(bs["region_trend"], key=lambda x: x["perm_p"]):
        L.append(f"   {r['feature']:<22} {r['resid_on']:<16} "
                 f"{r['mean_slope_LtoC']:+8.4f}   {r['perm_p']:.3f}   {r.get('q', float('nan')):.3f}")
    n_sig_rt = sum(1 for r in bs["region_trend"] if r.get("q", 1) < 0.05)

    L.append(f"\n-- VERDICT --")
    L.append(f"   peak-vs-rem residual tests surviving q<0.05: {n_sig_pr}")
    L.append(f"   region-trend residual tests surviving q<0.05: {n_sig_rt}")
    report = "\n".join(L)
    print(report)
    with open(os.path.join(OUT_DIR, "report.txt"), "w") as fh:
        fh.write(report + "\n")

    make_figure(les, sec_df, d, meta)
    print(f"\n[done] -> {OUT_DIR}/ (results.json, section_morphometry.csv, "
          f"animal_morphometry.csv, lesions.parquet, report.txt, figures/)")


if __name__ == "__main__":
    main()
