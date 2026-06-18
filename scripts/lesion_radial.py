"""Lesion core-rim RADIAL profiling — the last untested sub-layer of "beyond severity".

Whole-lesion geometry was null beyond severity (scripts/lesion_morphometry.py). This
asks whether the INTERNAL radial organization of lesions (myeloid core -> astrocyte/
complement rim -> spared parenchyma) carries severity-independent structure.

Method: per section (meta_sample_id), segment lesions = DBSCAN(eps=60µm) on
inflammatory cells (leiden_1 {5,3,18 myeloid, 13 T/NK, 21 B}), >=20 cells. For each
lesion take the convex hull; assign every nearby cell a SIGNED distance to the hull
boundary (negative inside=core, positive outside=parenchyma). Build radial profiles
of curated gene programs and of cell-type composition. Then the beyond-severity test:
per-animal radial-organization metrics (myeloid core-rim polarization, astrocyte-rim
index), residualized on score_sacrifice and lesion size, tested peak-vs-remission and
region L->T->C with permutation nulls.

    PYTHONPATH="$PWD" python scripts/lesion_radial.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from scipy.spatial import ConvexHull
from scipy.stats import mannwhitneyu, ttest_1samp
from sklearn.cluster import DBSCAN

from immunotransformer.train import resolve_device  # noqa: F401  (OpenMP guard)

RRMAP2 = ("/Volumes/moldiassd/RRMAP2_xenium_adata/kmeans_separated/"
          "RRMAP2_xenium_all_samples.cellcharter.companion.ready.with_metadata.rerun.h5ad")
OUT = "runs/lesion_radial"
CACHE = os.path.join(OUT, "cells.npz")
INFL = {"5", "3", "18", "13", "21"}
EPS, MIN_SAMPLES, MIN_LESION = 60.0, 10, 20
BIN_EDGES = np.array([-80, -40, -20, -10, 0, 10, 20, 40, 70, 110, 160], float)

PROGRAMS = {
    "myeloid_core": ["Hal", "Arg1", "Chil3", "Acod1", "Cd68"],
    "lymphoid": ["Cd3e", "Cd8a", "Cd2", "Igkc"],
    "complement": ["C3", "C6", "C4b"],
    "astrocyte": ["Gfap", "Serpina3n", "Vim"],
    "myelin_parenchyma": ["Plp1", "Mbp", "Mog"],
}
ALL_GENES = sorted({g for gs in PROGRAMS.values() for g in gs})


def build_or_load():
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        print(f"[cache] {CACHE}")
        return (d["xy"], d["sec"].astype(str), d["leiden"].astype(str), d["expr"],
                d["genes"].astype(str), d["meta"].item())
    import anndata as ad
    from scipy.sparse import issparse
    print(f"[load] {RRMAP2}")
    a = ad.read_h5ad(RRMAP2)
    a = a[a.obs["model"] == "RELAPSE REMITTING"].copy()
    o = a.obs
    print(f"       RR: {a.n_obs:,} cells")
    xy = np.asarray(a.obsm["spatial"], float)
    sec = o["meta_sample_id"].astype(str).to_numpy()
    leiden = o["leiden_1"].astype(str).to_numpy()
    present = [g for g in ALL_GENES if g in a.var_names]
    X = a[:, present].X
    expr = np.asarray(X.todense()) if issparse(X) else np.asarray(X)
    meta = (o.drop_duplicates("meta_sample_id").set_index("meta_sample_id")
            [["stage", "region", "score_sacrifice", "sample_name"]])
    meta = {s: {"stage": str(meta.loc[s, "stage"]), "region": str(meta.loc[s, "region"]),
                "score": float(meta.loc[s, "score_sacrifice"]),
                "animal": str(meta.loc[s, "sample_name"])} for s in meta.index}
    os.makedirs(OUT, exist_ok=True)
    np.savez(CACHE, xy=xy, sec=sec, leiden=leiden, expr=expr,
             genes=np.array(present), meta=np.array(meta, dtype=object))
    print(f"[cache] wrote {CACHE}")
    return xy, sec, leiden, expr, np.array(present), meta


def signed_dist_to_hull(pts, hull_pts):
    """Signed distance of pts to convex hull boundary (neg inside). Vectorised."""
    hp = hull_pts
    path = MplPath(hp)
    inside = path.contains_points(pts)
    # min distance to hull edges
    a = hp
    b = np.roll(hp, -1, axis=0)
    ab = b - a                                              # [E,2]
    ab2 = (ab ** 2).sum(1) + 1e-12
    d = np.full(len(pts), np.inf)
    for e in range(len(a)):
        ap = pts - a[e]
        t = np.clip((ap @ ab[e]) / ab2[e], 0, 1)
        proj = a[e] + t[:, None] * ab[e]
        d = np.minimum(d, np.hypot(*(pts - proj).T))
    return np.where(inside, -d, d)


def main():
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    xy, sec, leiden, expr, genes, meta = build_or_load()
    gi = {g: i for i, g in enumerate(genes)}
    # z-score expression across all cells, then program scores
    z = (expr - expr.mean(0)) / (expr.std(0) + 1e-9)
    prog_score = {p: z[:, [gi[g] for g in gs if g in gi]].mean(1) for p, gs in PROGRAMS.items()}
    is_infl = np.isin(leiden, list(INFL))
    is_astro = leiden == "9"  # reactive astro niche (label from atlas); also c8
    is_astro |= leiden == "8"

    nbin = len(BIN_EDGES) - 1
    # accumulate radial profiles: sum & count per (program, bin), split by phase
    prof = {ph: {p: np.zeros(nbin) for p in PROGRAMS} for ph in ["PEAK", "REMISSION", "all"]}
    cnt = {ph: np.zeros(nbin) for ph in ["PEAK", "REMISSION", "all"]}
    comp = {"infl": np.zeros(nbin), "astro": np.zeros(nbin), "total": np.zeros(nbin)}

    per_animal = {}                                        # animal -> list of metric dicts
    n_lesions = 0
    sections = pd.unique(sec)
    for s in sections:
        m = meta.get(s)
        if m is None or np.isnan(m["score"]):
            continue
        idx = np.where(sec == s)[0]
        sub_infl = idx[is_infl[idx]]
        if len(sub_infl) < MIN_LESION:
            continue
        lab = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit_predict(xy[sub_infl])
        phase = "PEAK" if m["stage"].startswith("PEAK") else (
            "REMISSION" if m["stage"].startswith("REMISSION") else None)
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
            # candidate cells: section cells within bbox+margin of hull
            lo, hi = hp.min(0) - 160, hp.max(0) + 160
            cand = idx[(xy[idx, 0] >= lo[0]) & (xy[idx, 0] <= hi[0]) &
                       (xy[idx, 1] >= lo[1]) & (xy[idx, 1] <= hi[1])]
            if len(cand) < 5:
                continue
            sd = signed_dist_to_hull(xy[cand], hp)
            keep = (sd >= BIN_EDGES[0]) & (sd < BIN_EDGES[-1])
            cand, sd = cand[keep], sd[keep]
            b = np.digitize(sd, BIN_EDGES) - 1
            for bb in range(nbin):
                sel = cand[b == bb]
                if len(sel) == 0:
                    continue
                cnt["all"][bb] += len(sel)
                comp["total"][bb] += len(sel)
                comp["infl"][bb] += is_infl[sel].sum()
                comp["astro"][bb] += is_astro[sel].sum()
                for p in PROGRAMS:
                    prof["all"][p][bb] += prog_score[p][sel].sum()
                if phase:
                    cnt[phase][bb] += len(sel)
                    for p in PROGRAMS:
                        prof[phase][p][bb] += prog_score[p][sel].sum()
            # per-lesion radial-organization metrics
            core = sd < -10
            rim = (sd >= 0) & (sd < 40)
            if core.sum() >= 5 and rim.sum() >= 5:
                met = {
                    "myeloid_core_rim": float(prog_score["myeloid_core"][cand[core]].mean()
                                              - prog_score["myeloid_core"][cand[rim]].mean()),
                    "astro_rim_core": float(prog_score["astrocyte"][cand[rim]].mean()
                                            - prog_score["astrocyte"][cand[core]].mean()),
                    "complement_rim_core": float(prog_score["complement"][cand[rim]].mean()
                                                 - prog_score["complement"][cand[core]].mean()),
                    "size": int(len(mem)),
                }
                per_animal.setdefault(m["animal"], []).append(met)

    # finalize profiles (mean per bin)
    def prof_df(ph):
        c = np.where(cnt[ph] == 0, 1, cnt[ph])
        return {p: (prof[ph][p] / c).tolist() for p in PROGRAMS}
    comp_frac = {k: (comp[k] / np.where(comp["total"] == 0, 1, comp["total"])).tolist()
                 for k in ["infl", "astro"]}
    bin_centers = ((BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2).tolist()

    # ---- beyond-severity test: animal-level radial metrics, residualize severity+size ----
    rows = []
    for an, lst in per_animal.items():
        if not lst:
            continue
        df = pd.DataFrame(lst)
        # find this animal's stage/region/score from any of its sections
        sm = next(meta[s] for s in meta if meta[s]["animal"] == an)
        rows.append({"animal": an, "stage": sm["stage"], "region": sm["region"],
                     "score": sm["score"], "n_lesions": len(lst),
                     **{k: float(df[k].mean()) for k in ["myeloid_core_rim", "astro_rim_core",
                                                         "complement_rim_core", "size"]}})
    A = pd.DataFrame(rows)
    beyond = {"n_animals": len(A)}
    if len(A) >= 8:
        def residualize(y, covs):
            X = np.c_[np.ones(len(y)), covs]
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            return y - X @ beta
        A = A[A["score"].notna()].copy()
        covs = np.c_[(A["score"] - A["score"].mean()), np.log(A["size"] + 1)]
        peak = A["stage"].str.startswith("PEAK").to_numpy()
        rem = A["stage"].str.startswith("REMISSION").to_numpy()
        rrank = A["region"].map({"L": 0, "T": 1, "C": 2}).to_numpy(float)
        rng = np.random.default_rng(0)
        for metric in ["myeloid_core_rim", "astro_rim_core", "complement_rim_core"]:
            resid = residualize(A[metric].to_numpy(float), covs)
            res = {"raw_mean": float(A[metric].mean())}
            if peak.sum() >= 3 and rem.sum() >= 3:
                obs = abs(resid[peak].mean() - resid[rem].mean())
                null = [abs(rp[peak].mean() - rp[rem].mean())
                        for rp in (rng.permutation(resid) for _ in range(2000))]
                res["peak_vs_rem_resid_diff"] = float(resid[peak].mean() - resid[rem].mean())
                res["peak_vs_rem_perm_p"] = float((1 + sum(n >= obs for n in null)) / 2001)
            mreg = ~np.isnan(rrank)
            if mreg.sum() >= 6:
                rc = rrank[mreg] - rrank[mreg].mean()
                slope = (rc * (resid[mreg] - resid[mreg].mean())).sum() / (rc ** 2).sum()
                null = [((rc * (rng.permutation(resid[mreg]) - resid[mreg].mean())).sum() / (rc ** 2).sum())
                        for _ in range(2000)]
                res["region_slope_resid"] = float(slope)
                res["region_perm_p"] = float((1 + sum(abs(n) >= abs(slope) for n in null)) / 2001)
            beyond[metric] = res

    results = {
        "n_lesions": n_lesions, "n_animals_with_metrics": len(per_animal),
        "bin_centers_um": bin_centers,
        "radial_programs_all": prof_df("all"),
        "radial_programs_peak": prof_df("PEAK"),
        "radial_programs_remission": prof_df("REMISSION"),
        "radial_composition": comp_frac,
        "beyond_severity": beyond,
    }
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    # ---- report ----
    L = [f"\n=== Lesion core-rim radial profiling ({n_lesions} lesions, {len(A) if len(A) else 0} animals) ===",
         "Radial program scores (z) by signed distance-to-edge (neg=core, 0=edge, pos=parenchyma):",
         "  bin(µm): " + " ".join(f"{c:>6.0f}" for c in bin_centers)]
    for p in PROGRAMS:
        L.append(f"  {p:<18} " + " ".join(f"{v:>6.2f}" for v in prof_df("all")[p]))
    L.append("  " + "-" * 50)
    L.append("  %inflammatory " + " ".join(f"{v:>6.2f}" for v in comp_frac["infl"]))
    L.append("  %astrocyte    " + " ".join(f"{v:>6.2f}" for v in comp_frac["astro"]))
    L.append("\n-- BEYOND-SEVERITY test (residualized on score + lesion size) --")
    for k, v in beyond.items():
        if isinstance(v, dict):
            pp = v.get("peak_vs_rem_perm_p"); rp = v.get("region_perm_p")
            L.append(f"  {k:<20} peak-rem p={pp if pp is None else round(pp,3)}  "
                     f"region p={rp if rp is None else round(rp,3)}")
    any_sig = any(isinstance(v, dict) and
                  (((v.get("peak_vs_rem_perm_p") or 1) < 0.05) or ((v.get("region_perm_p") or 1) < 0.05))
                  for v in beyond.values())
    L.append(f"\n=== VERDICT: {'SOME radial signal beyond severity — investigate' if any_sig else 'NULL — radial organization is also explained by severity'} ===")
    report = "\n".join(L)
    print(report)
    with open(os.path.join(OUT, "report.txt"), "w") as fh:
        fh.write(report + "\n")

    _plot(bin_centers, prof_df, comp_frac, OUT)
    print(f"\n[done] -> {OUT}/ (results.json, report.txt, figures/)")


def _plot(bin_centers, prof_df, comp_frac, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))
    pa = prof_df("all")
    for p, vals in pa.items():
        axA.plot(bin_centers, vals, "-o", lw=2, label=p)
    axA.axvline(0, c="k", lw=.8, ls="--")
    axA.set_xlabel("signed distance to lesion edge (µm)   ← core    parenchyma →")
    axA.set_ylabel("program score (z)")
    axA.set_title("Radial organization of lesions")
    axA.legend(fontsize=8, frameon=False)
    axA.spines[["top", "right"]].set_visible(False)
    axB.plot(bin_centers, comp_frac["infl"], "-o", lw=2, c="#c0392b", label="inflammatory")
    axB.plot(bin_centers, comp_frac["astro"], "-o", lw=2, c="#2f9e44", label="astrocyte")
    axB.axvline(0, c="k", lw=.8, ls="--")
    axB.set_xlabel("signed distance to lesion edge (µm)")
    axB.set_ylabel("cell fraction")
    axB.set_title("Composition vs radial distance")
    axB.legend(fontsize=8, frameon=False)
    axB.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figures", "radial_profile.png"), dpi=130, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "figures", "radial_profile.pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
