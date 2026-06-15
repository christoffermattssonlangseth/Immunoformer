"""Within-CHRONIC disease trajectory — descriptive, animal-level (NOT a trained model).

The companion within-RR scripts (`rr_within_relapse.py`, `rr_cycle_oscillation.py`)
map the relapsing-remitting cycle. This is the parallel pass for the CHRONIC model
(34 animals, B6/MOG strain). RR-vs-chronic direct level comparison is DEAD (100%
strain+slide confounded), so we stay strictly WITHIN the chronic cohort, where stage
is spread over 9 slides and both chronic run_dates (20250110, 20260506) carry the
severity range — so run_date can be a covariate / robustness check.

Chronic has its OWN severity-graded course (mean score_sacrifice in parens):
    control/CFA & NONSYMPTOM (0.0) < OS1 (0.47) < MILD30 (1.0)/MILD16 (1.3)
      < SEVERE30 (2.37)/SEVERE16 (2.50) < PEAK1 (2.99)
The 16/30 suffix = day of sacrifice (~d27-29 vs ~d41-50): a severity x duration grid
that RR lacks. Chronic is L/T regions only (~no cervical).

Five analyses, all ANIMAL-LEVEL (the unit of independence), n small -> exploratory:

  1. Pseudobulk (cached like rr_within_relapse, so re-runs skip the 14GB reload).
  2. Severity trajectory: rank-regress each gene on score_sacrifice (Spearman,
     BH-FDR); report top up/down + key programs. Robustness: partial Spearman
     controlling for run_date, and per-batch sign agreement.
  3. NOVEL — day16 vs day30 at MATCHED severity: MILD16 vs MILD30 and SEVERE16 vs
     SEVERE30 (same clinical score, different time-since-onset). Descriptive POC,
     n=2-5/group; CAVEAT day-suffix is confounded with run_date in this cohort.
  4. CONSERVATION: Spearman of the chronic severity-trajectory gene rho against the
     RR relapse-cycle gene rho (mono_rho from rr_cycle_oscillation). Correlation of
     within-model slopes cancels the strain offset -> the legitimate cross-model
     question: is the disease program conserved across models/strains?
  5. Cell-type composition along the chronic severity axis (which niches expand).

    python scripts/chronic_trajectory.py
    python scripts/chronic_trajectory.py --cluster-col CellCharter_10 --top 25
"""

from __future__ import annotations

import argparse
import json
import os

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.stats import mannwhitneyu, rankdata, spearmanr
from scipy.stats import t as tdist

from immunotransformer.train import resolve_device  # noqa: F401  (imports package OpenMP guard)

RRMAP2 = (
    "/Volumes/moldiassd/RRMAP2_xenium_adata/kmeans_separated/"
    "RRMAP2_xenium_all_samples.cellcharter.companion.ready.with_metadata.rerun.h5ad"
)
RR_CYCLE_METRICS = "runs/rr_cycle_oscillation/gene_oscillation_metrics.csv"
CLUSTER_LABELS = "runs/rr_region_gradient/cluster_labels.json"

# day16 vs day30 matched-severity contrasts (same clinical score, different duration)
DAY_CONTRASTS = [("MILD16", "MILD30"), ("SEVERE16", "SEVERE30")]

# key biological programs to read out explicitly (genes present in the 5101-gene panel
# are kept; missing ones are silently dropped at report time)
PROGRAMS = {
    "inflammation_complement": ["C1qa", "C1qb", "C1qc", "C3", "C4b", "C6", "Cxcl10",
                                "Ccl2", "Ccl5", "Il1b", "Tnf", "Gbp2"],
    "MHC_antigen": ["Cd74", "H2-Aa", "H2-Ab1", "H2-K1", "H2-D1", "B2m", "Tap1", "Cd14"],
    "DAM_myeloid": ["Gpnmb", "Cd68", "Itgax", "Apoe", "Trem2", "Cst7", "Lpl", "Arg1",
                    "Chil3", "Fcrls", "Tmem119", "P2ry12", "Hal"],
    "cholesterol_myelin": ["Hmgcr", "Msmo1", "Idi1", "Ldlr", "Lss", "Hsd17b7",
                           "Mbp", "Plp1", "Mog", "Mag", "Cnp"],
}


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    q[order] = (p[order] * n) / (np.arange(n) + 1)
    q[order] = np.minimum.accumulate(q[order][::-1])[::-1]
    return np.clip(q, 0, 1)


def pseudobulk(counts, codes, n_groups):
    """Sum counts rows by integer group code -> [n_groups, n_genes], then CP10k+log1p."""
    g = counts.shape[1]
    pb = np.zeros((n_groups, g), dtype=np.float64)
    for k in range(n_groups):
        rows = counts[codes == k]
        s = np.asarray(rows.sum(axis=0)).ravel() if issparse(rows) else rows.sum(axis=0)
        pb[k] = s
    libsize = pb.sum(axis=1, keepdims=True)
    libsize[libsize == 0] = 1.0
    return np.log1p(pb / libsize * 1e4)  # log CP10k


def spearman_trend(M, names, x, mask):
    """Spearman of each column of M against continuous x, over `mask` rows. BH-FDR."""
    xv = x[mask]
    rows = []
    for j in range(M.shape[1]):
        col = M[mask, j]
        if np.ptp(col) == 0:
            continue
        rho, p = spearmanr(col, xv)
        if not np.isnan(rho):
            rows.append((names[j], float(rho), float(p)))
    df = pd.DataFrame(rows, columns=["name", "rho", "p"])
    df["q"] = bh_fdr(df["p"].to_numpy()) if len(df) else []
    return df.sort_values("rho")


def partial_trend(pb, names, mask, score, covar):
    """Partial Spearman of each gene vs severity, controlling for a covariate (run_date).

    Removes the linear (rank) effect of `covar` from both gene and severity, so a
    surviving partial_rho means a severity signal not explained by the covariate.
    Returns (df sorted by partial_rho, r_score_vs_covar, n_used).
    """
    keep = mask & ~np.isnan(score) & ~np.isnan(covar)
    sub = pb[keep]
    m = sub.shape[0]
    rs, rc = rankdata(score[keep]), rankdata(covar[keep])

    def zc(v):
        v = v - v.mean(); s = v.std()
        return v / s if s > 0 else np.zeros_like(v)

    zs, zc_ = zc(rs), zc(rc)
    r_sc = float(zs @ zc_ / m)
    gr = np.apply_along_axis(rankdata, 0, sub)
    zx = (gr - gr.mean(0)) / (gr.std(0) + 1e-12)
    r_xs = (zx.T @ zs) / m
    r_xc = (zx.T @ zc_) / m
    denom = np.sqrt(np.clip((1 - r_xc ** 2) * (1 - r_sc ** 2), 1e-12, None))
    pr = (r_xs - r_xc * r_sc) / denom
    df_ = m - 3
    tstat = pr * np.sqrt(df_ / np.clip(1 - pr ** 2, 1e-12, None))
    pval = 2 * tdist.sf(np.abs(tstat), df_)
    out = pd.DataFrame({"name": names, "partial_rho": pr, "raw_p": pval})
    out = out[sub.std(0) > 0].copy()
    out["q"] = bh_fdr(out["raw_p"].to_numpy())
    return out.sort_values("partial_rho"), r_sc, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", default=RRMAP2)
    ap.add_argument("--cluster-col", default="leiden_1")
    ap.add_argument("--layer", default="counts")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out-dir", default="runs/chronic_trajectory")
    ap.add_argument("--cache", default="runs/chronic_trajectory/pseudobulk.npz")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out_dir, "figures"), exist_ok=True)

    # ---------------------------------------------------------------- load / cache
    if os.path.exists(args.cache):
        print(f"[cache] {args.cache}")
        d = np.load(args.cache, allow_pickle=True)
        pb = d["pb"]
        genes = d["genes"].astype(str)
        animals = list(d["animals"].astype(str))
        a_stage = pd.Series(d["stage"].astype(str), index=animals)
        a_score = pd.Series(d["score"].astype(float), index=animals)
        a_rundate = pd.Series(d["run_date"].astype(str), index=animals)
        comp = pd.DataFrame(d["comp"], index=animals, columns=d["clusters"].astype(str)) \
            if "comp" in d and d["comp"].size else None
        clusters = list(d["clusters"].astype(str)) if comp is not None else []
    else:
        print(f"[load] {args.h5ad}")
        adata = ad.read_h5ad(args.h5ad)
        adata = adata[adata.obs["model"] == "CHRONIC"].copy()
        print(f"       CHRONIC subset: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
        o = adata.obs
        counts = adata.layers[args.layer] if args.layer else adata.X
        genes = np.array(adata.var_names, dtype=str)

        animals = list(pd.unique(o["sample_name"]))
        a_code = {a: i for i, a in enumerate(animals)}
        codes = o["sample_name"].map(a_code).to_numpy()
        a_meta = o.drop_duplicates("sample_name").set_index("sample_name")
        a_stage = a_meta["stage"].astype(str).reindex(animals)
        a_score = pd.to_numeric(a_meta["score_sacrifice"], errors="coerce").reindex(animals)
        a_rundate = a_meta["run_date"].astype(str).reindex(animals)

        print("[pseudobulk] summing counts per animal ...")
        pb = pseudobulk(counts, codes, len(animals))

        have_clusters = args.cluster_col in o.columns
        if have_clusters:
            comp = pd.crosstab(o["sample_name"], o[args.cluster_col].astype(str))
            comp = comp.reindex(animals).fillna(0)
            comp = comp.div(comp.sum(axis=1), axis=0)
            clusters = list(comp.columns)
        else:
            print(f"[warn] cluster col '{args.cluster_col}' missing — skipping composition")
            comp, clusters = None, []

        np.savez(
            args.cache, pb=pb, genes=genes, animals=np.array(animals),
            stage=a_stage.to_numpy().astype(str), score=a_score.to_numpy(float),
            run_date=a_rundate.to_numpy().astype(str),
            comp=(comp.to_numpy() if comp is not None else np.array([])),
            clusters=(np.array(clusters) if clusters else np.array([])),
        )
        print(f"[cache] wrote {args.cache}")

    score = a_score.to_numpy(float)
    stage = a_stage.to_numpy().astype(str)
    rundate = a_rundate.to_numpy().astype(str)
    # numeric run_date code for the partial-Spearman covariate
    rd_levels = {v: i for i, v in enumerate(sorted(set(rundate)))}
    rd_code = np.array([rd_levels[v] for v in rundate], float)

    print(f"[chronic] {len(animals)} animals; run_dates {sorted(set(rundate))}")
    cluster_labels = {}
    if os.path.exists(CLUSTER_LABELS):
        cluster_labels = json.load(open(CLUSTER_LABELS))

    results = {
        "n_chronic_animals": len(animals),
        "cluster_col": args.cluster_col,
        "run_dates": {v: int((rundate == v).sum()) for v in sorted(set(rundate))},
        "stage_counts": {s: int((stage == s).sum()) for s in sorted(set(stage))},
    }

    # ============================================================ 2. severity trajectory
    fin = ~np.isnan(score)
    gene_traj = spearman_trend(pb, genes, score, fin)
    results["severity_trajectory"] = {
        "n_animals": int(fin.sum()),
        "n_sig_q05": int((gene_traj["q"] < 0.05).sum()),
        "increasing": gene_traj.tail(args.top)[::-1].to_dict("records"),
        "decreasing": gene_traj.head(args.top).to_dict("records"),
    }

    # program-level readout
    gt = gene_traj.set_index("name")
    prog_summary = {}
    for prog, glist in PROGRAMS.items():
        present = [g for g in glist if g in gt.index]
        rows = [{"gene": g, "rho": float(gt.loc[g, "rho"]), "q": float(gt.loc[g, "q"])}
                for g in present]
        prog_summary[prog] = {
            "mean_rho": float(np.mean([r["rho"] for r in rows])) if rows else None,
            "n_genes": len(rows),
            "genes": sorted(rows, key=lambda r: -r["rho"]),
        }
    results["severity_trajectory"]["programs"] = prog_summary

    # robustness: partial Spearman controlling for run_date + per-batch sign agreement
    gene_adj, r_score_rd, n_adj = partial_trend(pb, genes, fin, score, rd_code)
    ga = gene_adj.set_index("name")
    raw_sig = set(gene_traj.loc[gene_traj["q"] < 0.05, "name"])
    adj_sig = set(gene_adj.loc[gene_adj["q"] < 0.05, "name"])

    # per-batch independent severity rho (sign agreement)
    batch_rho = {}
    for v in sorted(set(rundate)):
        bm = fin & (rundate == v)
        if bm.sum() >= 4 and len(set(score[bm])) > 2:
            bt = spearman_trend(pb, genes, score, bm).set_index("name")["rho"]
            batch_rho[v] = bt
    batches = list(batch_rho.keys())
    sign_agree = None
    if len(batches) == 2:
        common = batch_rho[batches[0]].index.intersection(batch_rho[batches[1]].index)
        b0 = batch_rho[batches[0]].reindex(common)
        b1 = batch_rho[batches[1]].reindex(common)
        ok = b0.notna() & b1.notna()
        rho_bb, p_bb = spearmanr(b0[ok], b1[ok])
        frac_same = float((np.sign(b0[ok]) == np.sign(b1[ok])).mean())
        sign_agree = {
            "batch_a": batches[0], "batch_b": batches[1],
            "n_genes": int(ok.sum()),
            "spearman_rho_between_batch_slopes": float(rho_bb),
            "p": float(p_bb),
            "frac_same_sign": frac_same,
        }
    results["severity_trajectory"]["robustness"] = {
        "score_vs_run_date_rho": r_score_rd,
        "n_animals_partial": n_adj,
        "n_sig_raw_q05": len(raw_sig),
        "n_sig_partial_runDate_q05": len(adj_sig),
        "n_survive_run_date_adjustment": len(raw_sig & adj_sig),
        "between_batch_slope_agreement": sign_agree,
        "increasing_runDate_adjusted": gene_adj.tail(args.top)[::-1].to_dict("records"),
        "decreasing_runDate_adjusted": gene_adj.head(args.top).to_dict("records"),
    }

    # ============================================================ 3. day16 vs day30 (NOVEL)
    day_results = []
    for s16, s30 in DAY_CONTRASTS:
        m16 = (stage == s16)
        m30 = (stage == s30)
        n16, n30 = int(m16.sum()), int(m30.sum())
        entry = {
            "contrast": f"{s16}_vs_{s30}", "n_day16": n16, "n_day30": n30,
            "score_day16": float(np.nanmean(score[m16])) if n16 else None,
            "score_day30": float(np.nanmean(score[m30])) if n30 else None,
            "run_date_day16": sorted(set(rundate[m16])),
            "run_date_day30": sorted(set(rundate[m30])),
        }
        if n16 >= 2 and n30 >= 2:
            diff = pb[m16].mean(0) - pb[m30].mean(0)   # day16 - day30
            rows = []
            for j in range(pb.shape[1]):
                xa, xb = pb[m16, j], pb[m30, j]
                if np.ptp(np.concatenate([xa, xb])) == 0:
                    continue
                try:
                    _, p = mannwhitneyu(xa, xb, alternative="two-sided")
                except ValueError:
                    continue
                rows.append((genes[j], float(diff[j]), float(p)))
            df = pd.DataFrame(rows, columns=["name", "day16_minus_day30", "p"])
            df["q"] = bh_fdr(df["p"].to_numpy()) if len(df) else []
            df = df.sort_values("day16_minus_day30")
            entry["n_sig_q05"] = int((df["q"] < 0.05).sum())
            entry["up_in_day16_earlier"] = df.tail(args.top)[::-1].to_dict("records")
            entry["up_in_day30_later"] = df.head(args.top).to_dict("records")
            # program means of the day16-day30 effect
            entry["program_mean_day16_minus_day30"] = {
                prog: float(np.mean([diff[list(genes).index(g)] for g in glist if g in set(genes)]))
                for prog, glist in PROGRAMS.items()
            }
        day_results.append(entry)
    results["day16_vs_day30"] = {
        "note": ("Same clinical severity, different time-since-onset (d16~d27-29 vs "
                 "d30~d41-50). CAVEAT: in this cohort the day suffix is confounded "
                 "with run_date (MILD16/SEVERE16 on 20260506; MILD30/SEVERE30 on "
                 "20250110), so 'earlier vs later' cannot be cleanly separated from "
                 "batch. Descriptive POC, n=2-5/group."),
        "contrasts": day_results,
    }

    # ============================================================ 4. conservation vs RR
    conservation = {"note": "RR-vs-chronic level comparison is dead (strain+slide). "
                            "Correlating within-model slopes cancels the strain offset."}
    if os.path.exists(RR_CYCLE_METRICS):
        rr = pd.read_csv(RR_CYCLE_METRICS).set_index("gene")
        chr_rho = gene_traj.set_index("name")["rho"]
        common = chr_rho.index.intersection(rr.index)
        c = chr_rho.reindex(common)
        # RR relapse-cycle monotonic trend
        r_mono = rr["mono_rho"].reindex(common)
        ok = c.notna() & r_mono.notna()
        rho_cons, p_cons = spearmanr(c[ok], r_mono[ok])
        conservation["chronic_severity_vs_rr_cycle_mono_rho"] = {
            "n_genes": int(ok.sum()),
            "spearman_rho": float(rho_cons),
            "p": float(p_cons),
        }
        # also vs RR oscillation amplitude (acute program)
        r_amp = rr["amplitude"].reindex(common)
        ok2 = c.notna() & r_amp.notna()
        rho_amp, p_amp = spearmanr(c[ok2], r_amp[ok2])
        conservation["chronic_severity_vs_rr_amplitude"] = {
            "n_genes": int(ok2.sum()),
            "spearman_rho": float(rho_amp),
            "p": float(p_amp),
        }
        # genes conserved-up and conserved-down in BOTH models
        joint = pd.DataFrame({"chr": c, "rr_mono": r_mono}).dropna()
        conservation["conserved_up_both"] = (
            joint[(joint["chr"] > 0) & (joint["rr_mono"] > 0)]
            .assign(score=lambda d: d["chr"] + d["rr_mono"])
            .sort_values("score", ascending=False).head(args.top)
            .reset_index().rename(columns={"index": "gene"})
            [["gene", "chr", "rr_mono"]].to_dict("records"))
        conservation["conserved_down_both"] = (
            joint[(joint["chr"] < 0) & (joint["rr_mono"] < 0)]
            .assign(score=lambda d: d["chr"] + d["rr_mono"])
            .sort_values("score").head(args.top)
            .reset_index().rename(columns={"index": "gene"})
            [["gene", "chr", "rr_mono"]].to_dict("records"))
    else:
        print(f"[warn] {RR_CYCLE_METRICS} missing — skipping conservation")
    results["conservation_vs_rr"] = conservation

    # ============================================================ 5. composition along severity
    if clusters and comp is not None:
        clu_traj = spearman_trend(comp.to_numpy(), np.array(clusters), score, fin)
        clu_traj["lineage"] = clu_traj["name"].map(lambda c: cluster_labels.get(str(c), "?"))
        results["composition_severity"] = {
            "n_animals": int(fin.sum()),
            "clusters": clu_traj.sort_values("rho", ascending=False).to_dict("records"),
        }

    # ============================================================ report
    L = []
    L.append(f"\n=== WITHIN-CHRONIC disease trajectory (n={len(animals)} chronic animals, animal-level) ===")
    L.append("B6/MOG strain. Descriptive; stage spread over 9 slides, both run_dates carry severity.")
    L.append("RR-vs-chronic level comparison is DEAD (100% strain+slide confounded) — stay within chronic.")
    L.append("Stages (n animals): " + "  ".join(
        f"{s}={results['stage_counts'][s]}" for s in sorted(results['stage_counts'])))
    L.append(f"run_dates (animals): {results['run_dates']}")

    st = results["severity_trajectory"]
    L.append(f"\n-- 2. Severity trajectory (Spearman vs score_sacrifice, n={st['n_animals']}) --")
    L.append(f"   genes at BH-q<0.05: {st['n_sig_q05']}")
    L.append("   INCREASING with severity:")
    for r in st["increasing"][:12]:
        L.append(f"     {r['name']:<16} rho={r['rho']:+.3f}  q={r['q']:.3g}")
    L.append("   DECREASING with severity:")
    for r in st["decreasing"][:12]:
        L.append(f"     {r['name']:<16} rho={r['rho']:+.3f}  q={r['q']:.3g}")
    L.append("   program-level mean rho (key disease axes):")
    for prog, d in st["programs"].items():
        if d["mean_rho"] is not None:
            L.append(f"     {prog:<24} mean_rho={d['mean_rho']:+.3f}  (n={d['n_genes']} genes)")

    rob = st["robustness"]
    L.append(f"\n-- 2b. Robustness across the two chronic batches --")
    L.append(f"   score vs run_date rho = {rob['score_vs_run_date_rho']:+.3f} "
             f"(low = severity not aliased to batch)")
    L.append(f"   genes q<0.05: raw {rob['n_sig_raw_q05']} -> run_date-partial "
             f"{rob['n_sig_partial_runDate_q05']} ({rob['n_survive_run_date_adjustment']} survive)")
    if rob["between_batch_slope_agreement"]:
        ba = rob["between_batch_slope_agreement"]
        L.append(f"   independent per-batch severity slopes ({ba['batch_a']} vs {ba['batch_b']}): "
                 f"Spearman={ba['spearman_rho_between_batch_slopes']:+.3f} "
                 f"(p={ba['p']:.2g}), {ba['frac_same_sign']*100:.0f}% same sign "
                 f"over {ba['n_genes']} genes")

    L.append(f"\n-- 3. NOVEL: day16 vs day30 at MATCHED severity (POC, small n) --")
    L.append("   CAVEAT: day suffix is confounded with run_date here — read as exploratory.")
    for e in results["day16_vs_day30"]["contrasts"]:
        L.append(f"   {e['contrast']}: n={e['n_day16']} (score~{e['score_day16']}) vs "
                 f"n={e['n_day30']} (score~{e['score_day30']})")
        if "n_sig_q05" in e:
            L.append(f"     genes q<0.05: {e['n_sig_q05']}")
            L.append("     UP in day16 (earlier / more active):")
            for r in e["up_in_day16_earlier"][:6]:
                L.append(f"       {r['name']:<14} d16-d30={r['day16_minus_day30']:+.3f}  q={r['q']:.3g}")
            L.append("     UP in day30 (later / chronic accrual):")
            for r in e["up_in_day30_later"][:6]:
                L.append(f"       {r['name']:<14} d16-d30={r['day16_minus_day30']:+.3f}  q={r['q']:.3g}")
            L.append("     program mean (day16 - day30): " + "  ".join(
                f"{p.split('_')[0]}={v:+.2f}" for p, v in e["program_mean_day16_minus_day30"].items()))

    cons = results["conservation_vs_rr"]
    L.append(f"\n-- 4. CONSERVATION: chronic severity trajectory vs RR relapse cycle --")
    if "chronic_severity_vs_rr_cycle_mono_rho" in cons:
        cc = cons["chronic_severity_vs_rr_cycle_mono_rho"]
        L.append(f"   Spearman(chronic severity rho, RR cycle mono_rho) = {cc['spearman_rho']:+.3f} "
                 f"(p={cc['p']:.2g}, {cc['n_genes']} genes)")
        ca = cons["chronic_severity_vs_rr_amplitude"]
        L.append(f"   Spearman(chronic severity rho, RR peak-rem amplitude) = {ca['spearman_rho']:+.3f} "
                 f"(p={ca['p']:.2g})")
        L.append("   -> positive = the disease program is CONSERVED across model/strain "
                 "(slope correlation cancels strain offset).")
        L.append("   top conserved-UP in both: " + ", ".join(r["gene"] for r in cons["conserved_up_both"][:12]))
        L.append("   top conserved-DOWN in both: " + ", ".join(r["gene"] for r in cons["conserved_down_both"][:12]))

    if "composition_severity" in results:
        L.append(f"\n-- 5. Cell-type composition along chronic severity ({args.cluster_col}) --")
        ct = sorted(results["composition_severity"]["clusters"], key=lambda r: -abs(r["rho"]))[:10]
        for r in ct:
            L.append(f"     cluster {r['name']:<4} [{r['lineage']:<14}] rho={r['rho']:+.3f}  q={r['q']:.3g}")

    L.append("\n-- HONEST CAVEATS --")
    L.append("   * Cross-sectional: 1 stage/animal, terminal. Trajectory reconstructed across mice.")
    L.append("   * Small per-stage n (2-6). Animal-level stats; individual stage means noisy.")
    L.append("   * day16/day30 contrasts confounded with run_date in this cohort (see note).")
    L.append("   * Chronic is L/T regions only (~no cervical), so no region-spared axis here.")
    report = "\n".join(L)
    print(report)

    with open(os.path.join(args.out_dir, "report.txt"), "w") as fh:
        fh.write(report + "\n")
    with open(os.path.join(args.out_dir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    _plot(pb, genes, stage, score, gene_traj, results, args.out_dir)
    print(f"\n[done] -> {args.out_dir}/  (report.txt, results.json, pseudobulk.npz, figures/)")


def _plot(pb, genes, stage, score, gene_traj, results, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # severity-ordered stages for the heatmap x-axis
    stage_score = {s: np.nanmean(score[stage == s]) for s in set(stage)}
    order = [s for s in sorted(stage_score, key=lambda s: stage_score[s])
             if (stage == s).sum() > 0]

    # (1) heatmap: top severity-trending genes x severity-ordered stages
    top = pd.concat([gene_traj.head(15), gene_traj.tail(15)])
    gidx = {g: i for i, g in enumerate(genes)}
    rows = [gidx[n] for n in top["name"]]
    M = np.full((len(rows), len(order)), np.nan)
    for cj, st in enumerate(order):
        m = stage == st
        if m.sum():
            M[:, cj] = pb[np.ix_(m, rows)].mean(axis=0)
    M = (M - np.nanmean(M, axis=1, keepdims=True)) / (np.nanstd(M, axis=1, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(figsize=(max(5, len(order) * 0.9), 8))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{s}\n({stage_score[s]:.1f})" for s in order], rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(list(top["name"]), fontsize=7)
    ax.set_title("Top chronic severity-trending genes (z-scored stage means)\nx-axis: stages by mean score_sacrifice")
    fig.colorbar(im, ax=ax, fraction=0.025, label="z")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figures", "severity_trajectory_heatmap.pdf"), bbox_inches="tight")
    plt.close(fig)

    # (2) conservation scatter: chronic severity rho vs RR cycle mono_rho
    cons = results.get("conservation_vs_rr", {})
    if "chronic_severity_vs_rr_cycle_mono_rho" in cons and os.path.exists(RR_CYCLE_METRICS):
        rr = pd.read_csv(RR_CYCLE_METRICS).set_index("gene")
        chr_rho = gene_traj.set_index("name")["rho"]
        common = chr_rho.index.intersection(rr.index)
        x = chr_rho.reindex(common)
        y = rr["mono_rho"].reindex(common)
        ok = x.notna() & y.notna()
        fig, ax = plt.subplots(figsize=(5.8, 5.8))
        ax.scatter(x[ok], y[ok], s=5, c="#C0C0C0")
        label = ["Hal", "C6", "Fcrls", "Igf1", "Gpnmb", "Hmgcr", "Msmo1", "Idi1",
                 "Cd74", "Arg1", "Chil3", "Apoe", "C1qa", "Plp1", "Mbp"]
        for g in label:
            if g in x.index and ok.get(g, False):
                ax.scatter(x[g], y[g], s=24, c="#C44E52", zorder=3)
                ax.annotate(g, (x[g], y[g]), fontsize=7)
        rho = cons["chronic_severity_vs_rr_cycle_mono_rho"]["spearman_rho"]
        ax.axhline(0, c="gray", lw=0.5); ax.axvline(0, c="gray", lw=0.5)
        ax.set_xlabel("chronic severity trajectory rho")
        ax.set_ylabel("RR relapse-cycle mono_rho")
        ax.set_title(f"Disease-program conservation across models\nSpearman = {rho:+.3f}")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "figures", "conservation_chronic_vs_rr.pdf"), bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
