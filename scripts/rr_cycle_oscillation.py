"""Within-RR relapse cycle — OSCILLATION vs RATCHET decomposition (temporal axis).

The companion script `rr_within_relapse.py` orders the RR stages along the cycle
and finds genes that trend MONOTONICALLY (Spearman vs cycle rank). But the relapse
course is not monotonic — it oscillates:

    PLP CFA -> ONSET1 -> PEAK1 -> REMISSION1 -> ONSET2 -> PEAK2 -> REMISSION2 -> PEAK3

This script separates two temporal behaviours a gene can have along that axis:

  * OSCILLATING / ACUTE  — high at every PEAK, falls back at every REMISSION,
    and (critically) its remission FLOOR does NOT rise across cycles. It resets.
    Reads as a marker of acute relapse activity (reversible).
  * RATCHET / CUMULATIVE — its remission floor RISES cycle over cycle (REM2 > REM1)
    and/or its peaks climb (PEAK3 > PEAK1). Damage that survives remission.

Why the distinction matters: peaks are also the high-clinical-severity timepoints
(score_sacrifice peaks ~2.5, remissions ~0.8), so raw peak>remission largely just
tracks acute severity. The severity-INDEPENDENT signature of true accumulation is a
rising remission floor — a gene that is still elevated once the animal has clinically
recovered. That is what tells cumulative damage apart from acute reactivity.

Metrics per gene (animal-level pseudobulk, log CP10k):
  amplitude    = mean(PEAK animals)      - mean(REMISSION animals)   # oscillation size
  floor_drift  = mean(REMISSION2)        - mean(REMISSION1)          # cumulative residue
  peak_drift   = mean(PEAK3)             - mean(PEAK1)               # peak climb
  mono_rho     = Spearman(gene, cycle rank)                          # monotonic trend
  amp p/q      = Mann-Whitney peaks vs remissions, BH-FDR

Reuses the cached pseudobulk from rr_within_relapse.py (no 14GB reload).

    python scripts/rr_cycle_oscillation.py
    python scripts/rr_cycle_oscillation.py --genes Hal,C6,Fcrls,Igf1,Gpnmb
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

CACHE = "runs/rr_within_relapse/pseudobulk.npz"
# ONSET2 sits right after ONSET1 (per experimental staging), NOT as the onset of
# the 2nd attack. Peaks/remissions are what define the oscillation, so this order
# only affects mono_rho and the plot x-axis, not the amplitude/floor_drift metrics.
CYCLE_ORDER = ["PLP CFA", "ONSET1", "ONSET2", "PEAK1", "REMISSION1",
               "PEAK2", "REMISSION2", "PEAK3"]
PEAKS = ["PEAK1", "PEAK2", "PEAK3"]
REMISSIONS = ["REMISSION1", "REMISSION2"]


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    q[order] = (p[order] * n) / (np.arange(n) + 1)
    q[order] = np.minimum.accumulate(q[order][::-1])[::-1]
    return np.clip(q, 0, 1)


def stage_means(pb, stage, gene_idx):
    """Per-stage mean expression for one gene column, over the ordered cycle."""
    out = {}
    for s in CYCLE_ORDER:
        m = stage == s
        out[s] = float(pb[m, gene_idx].mean()) if m.sum() else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--genes", default="Hal",
                    help="comma-separated genes for the per-stage profile readout")
    ap.add_argument("--out-dir", default="runs/rr_cycle_oscillation")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out_dir, "figures"), exist_ok=True)

    if not os.path.exists(args.cache):
        raise SystemExit(
            f"cache {args.cache} not found — run scripts/rr_within_relapse.py first "
            "to build the pseudobulk cache.")
    d = np.load(args.cache, allow_pickle=True)
    pb, genes, stage = d["pb"], d["genes"].astype(str), d["stage"].astype(str)
    gi = {g: i for i, g in enumerate(genes)}
    n_by_stage = {s: int((stage == s).sum()) for s in CYCLE_ORDER}
    print(f"[load] {args.cache}: {pb.shape[0]} RR animals x {pb.shape[1]} genes")
    print("[stages] " + "  ".join(f"{s}:{n_by_stage[s]}" for s in CYCLE_ORDER))

    peak_mask = np.isin(stage, PEAKS)
    rem_mask = np.isin(stage, REMISSIONS)
    rank_map = {s: i for i, s in enumerate(CYCLE_ORDER)}
    traj_mask = np.array([s in rank_map for s in stage])
    ranks = np.array([rank_map.get(s, -1) for s in stage], float)

    rem1 = stage == "REMISSION1"
    rem2 = stage == "REMISSION2"
    pk1, pk3 = stage == "PEAK1", stage == "PEAK3"

    # ---- per-gene metrics (vectorised where cheap, loop for the stats) ----
    peak_mean = pb[peak_mask].mean(0)
    rem_mean = pb[rem_mask].mean(0)
    amplitude = peak_mean - rem_mean
    floor_drift = pb[rem2].mean(0) - pb[rem1].mean(0)
    peak_drift = pb[pk3].mean(0) - pb[pk1].mean(0)

    rows = []
    rr = ranks[traj_mask]
    for j in range(pb.shape[1]):
        xp, xr = pb[peak_mask, j], pb[rem_mask, j]
        if np.ptp(np.concatenate([xp, xr])) == 0:
            p = 1.0
        else:
            try:
                _, p = mannwhitneyu(xp, xr, alternative="two-sided")
            except ValueError:
                p = 1.0
        xt = pb[traj_mask, j]
        rho = spearmanr(xt, rr)[0] if np.ptp(xt) > 0 else 0.0
        rows.append((genes[j], amplitude[j], floor_drift[j], peak_drift[j],
                     float(rho) if not np.isnan(rho) else 0.0, p))
    df = pd.DataFrame(rows, columns=["gene", "amplitude", "floor_drift",
                                     "peak_drift", "mono_rho", "amp_p"])
    df["amp_q"] = bh_fdr(df["amp_p"].to_numpy())

    # ---- classify temporal behaviour ----
    # ACUTE/oscillating: big positive amplitude, flat floor (resets at remission).
    # RATCHET/cumulative: floor rises across cycles (survives remission).
    amp_hi = df["amplitude"] > 0.3
    floor_not_rising = df["floor_drift"] < 0.15   # resets (flat OR drops) at remission
    floor_up = df["floor_drift"] > 0.2
    df["class"] = "other"
    df.loc[amp_hi & floor_not_rising, "class"] = "acute_oscillating"
    df.loc[floor_up, "class"] = "ratchet_cumulative"
    df.loc[amp_hi & floor_up, "class"] = "acute+ratchet"

    results = {
        "n_by_stage": n_by_stage,
        "n_peak_animals": int(peak_mask.sum()),
        "n_remission_animals": int(rem_mask.sum()),
        "note": ("Peaks are also high-severity (score~2.5) vs remissions (~0.8); raw "
                 "amplitude tracks acute severity. floor_drift (REM2-REM1) is the "
                 "severity-independent accumulation signature."),
    }

    # top acute oscillators (reset) and top ratchet genes
    acute = df[df["class"] == "acute_oscillating"].sort_values("amplitude", ascending=False)
    ratchet = df[df["floor_drift"] > 0].sort_values("floor_drift", ascending=False)
    results["top_acute_oscillating"] = acute.head(args.top)[
        ["gene", "amplitude", "floor_drift", "peak_drift", "amp_q"]].to_dict("records")
    results["top_ratchet_cumulative"] = ratchet.head(args.top)[
        ["gene", "amplitude", "floor_drift", "peak_drift", "mono_rho"]].to_dict("records")

    # ---- per-stage profile for requested genes (incl. Hal) ----
    want = [g.strip() for g in args.genes.split(",") if g.strip() in gi]
    profiles = {}
    for g in want:
        sm = stage_means(pb, stage, gi[g])
        r = df.set_index("gene").loc[g]
        profiles[g] = {
            "per_stage": sm,
            "amplitude": float(r["amplitude"]),
            "floor_drift": float(r["floor_drift"]),
            "peak_drift": float(r["peak_drift"]),
            "mono_rho": float(r["mono_rho"]),
            "amp_q": float(r["amp_q"]),
            "class": str(r["class"]),
        }
    results["gene_profiles"] = profiles

    # ---- co-oscillators of Hal: genes whose 8-stage profile correlates with Hal's ----
    if "Hal" in gi:
        present = [s for s in CYCLE_ORDER if n_by_stage[s] > 0]
        sm_mat = np.array([[pb[stage == s, j].mean() for s in present]
                           for j in range(pb.shape[1])])  # [g, stages]
        hal_prof = sm_mat[gi["Hal"]]
        hz = (hal_prof - hal_prof.mean())
        smz = sm_mat - sm_mat.mean(1, keepdims=True)
        denom = (np.linalg.norm(smz, axis=1) * np.linalg.norm(hz) + 1e-12)
        cocorr = (smz @ hz) / denom
        co = pd.DataFrame({"gene": genes, "profile_corr_with_Hal": cocorr,
                           "amplitude": amplitude, "floor_drift": floor_drift})
        co = co[co["gene"] != "Hal"].sort_values("profile_corr_with_Hal", ascending=False)
        results["hal_cooscillators"] = co.head(args.top).to_dict("records")

    # ---- report ----
    L = []
    L.append(f"\n=== RR relapse-cycle OSCILLATION vs RATCHET (n={pb.shape[0]} RR animals) ===")
    L.append("Stages (n): " + "  ".join(f"{s}={n_by_stage[s]}" for s in CYCLE_ORDER))
    L.append(f"Peak animals={int(peak_mask.sum())}  Remission animals={int(rem_mask.sum())}")
    L.append("\nCAVEAT: peaks are high-severity (score~2.5) vs remissions (~0.8), so raw")
    L.append("amplitude (peak-remission) largely tracks ACUTE severity. The severity-")
    L.append("independent accumulation signal is floor_drift = REM2 - REM1 (does the gene")
    L.append("stay elevated once the animal has clinically recovered?).")

    for g in want:
        pr = profiles[g]
        L.append(f"\n-- {g}: temporal profile (log CP10k per stage) [class={pr['class']}] --")
        L.append("   " + "  ".join(
            f"{s.replace('REMISSION','REM').replace('PLP CFA','CFA')}={pr['per_stage'][s]:+.2f}"
            for s in CYCLE_ORDER if not np.isnan(pr['per_stage'][s])))
        L.append(f"   amplitude(peak-rem)={pr['amplitude']:+.3f} (q={pr['amp_q']:.2g})  "
                 f"floor_drift(REM2-REM1)={pr['floor_drift']:+.3f}  "
                 f"peak_drift(PK3-PK1)={pr['peak_drift']:+.3f}  mono_rho={pr['mono_rho']:+.3f}")
        verdict = ("RESETS each cycle -> acute/reversible relapse marker"
                   if pr["floor_drift"] < 0.15 and pr["amplitude"] > 0.3
                   else "floor rises -> cumulative/ratchet"
                   if pr["floor_drift"] > 0.2 else "weak/flat oscillation")
        L.append(f"   => {verdict}")

    L.append(f"\n-- Top ACUTE OSCILLATING genes (big peak>remission, floor resets) --")
    for r in results["top_acute_oscillating"][:12]:
        L.append(f"     {r['gene']:<14} amp={r['amplitude']:+.3f}  floor_drift={r['floor_drift']:+.3f}  "
                 f"peak_drift={r['peak_drift']:+.3f}  q={r['amp_q']:.2g}")
    L.append(f"\n-- Top RATCHET / CUMULATIVE genes (remission floor RISES across cycles) --")
    for r in results["top_ratchet_cumulative"][:12]:
        L.append(f"     {r['gene']:<14} floor_drift={r['floor_drift']:+.3f}  amp={r['amplitude']:+.3f}  "
                 f"mono_rho={r['mono_rho']:+.3f}")
    if "hal_cooscillators" in results:
        L.append(f"\n-- Hal CO-OSCILLATORS (genes whose stage profile tracks Hal's) --")
        for r in results["hal_cooscillators"][:15]:
            L.append(f"     {r['gene']:<14} profile_corr={r['profile_corr_with_Hal']:+.3f}  "
                     f"amp={r['amplitude']:+.3f}  floor_drift={r['floor_drift']:+.3f}")
    report = "\n".join(L)
    print(report)

    with open(os.path.join(args.out_dir, "report.txt"), "w") as fh:
        fh.write(report + "\n")
    with open(os.path.join(args.out_dir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    df.sort_values("amplitude", ascending=False).to_csv(
        os.path.join(args.out_dir, "gene_oscillation_metrics.csv"), index=False)

    _plot(pb, genes, gi, stage, n_by_stage, want, df, args.out_dir)
    print(f"\n[done] -> {args.out_dir}/ (report.txt, results.json, "
          "gene_oscillation_metrics.csv, figures/)")


def _plot(pb, genes, gi, stage, n_by_stage, want, df, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    present = [s for s in CYCLE_ORDER if n_by_stage[s] > 0]
    xpos = range(len(present))
    is_peak = [s in PEAKS for s in present]

    # (1) line profiles for the requested genes + clinical relapse shading
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for s, pk in zip(xpos, is_peak):
        if pk:
            ax.axvspan(s - 0.4, s + 0.4, color="#F2C9C9", alpha=0.5, zorder=0)
    for g in want:
        prof = [pb[stage == s, gi[g]].mean() for s in present]
        ax.plot(xpos, prof, "-o", lw=2, label=g)
    ax.set_xticks(list(xpos))
    ax.set_xticklabels([s.replace("REMISSION", "REM").replace("PLP CFA", "CFA") for s in present],
                       rotation=40, ha="right")
    ax.set_ylabel("pseudobulk log CP10k")
    ax.set_title("Relapse-cycle temporal profile (peaks shaded)")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figures", "cycle_profile.pdf"), bbox_inches="tight")
    plt.close(fig)

    # (2) amplitude vs floor_drift scatter — acute (right, y~0) vs ratchet (up)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(df["amplitude"], df["floor_drift"], s=6, c="#C0C0C0")
    hi = df[(df["amplitude"] > 0.4) | (df["floor_drift"].abs() > 0.3)]
    label_genes = set(want) | set(["C6", "Fcrls", "Igf1", "Gpnmb", "Cd68", "Arg1",
                                    "Igkc", "Cd109", "Chil3"])
    for _, r in df[df["gene"].isin(label_genes)].iterrows():
        ax.scatter(r["amplitude"], r["floor_drift"], s=22, c="#C44E52", zorder=3)
        ax.annotate(r["gene"], (r["amplitude"], r["floor_drift"]), fontsize=7)
    ax.axhline(0, c="gray", lw=0.6)
    ax.axhline(0.2, c="#4C72B0", lw=0.6, ls="--")
    ax.axvline(0.3, c="#4C72B0", lw=0.6, ls="--")
    ax.set_xlabel("amplitude  (peak - remission)  -> acute oscillation")
    ax.set_ylabel("floor_drift  (REM2 - REM1)  -> cumulative ratchet")
    ax.set_title("Acute-reversible (lower-right) vs cumulative (upper) genes")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figures", "amplitude_vs_floordrift.pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
