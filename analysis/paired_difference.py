"""Paired comparison of clocks on the SAME animals (decisive test for B3 and 1f).

Comparing two rho values by overlapping marginal CIs is not valid when both are computed
on the same animals from correlated predictions. Here, per comparison:
  * bootstrap over animals (2000): on each resample recompute BOTH rho values against the
    same residualised target and record rho_A - rho_B; report mean difference, 95% CI and
    the fraction of resamples with difference <= 0;
  * Wilcoxon signed-rank on per-animal absolute errors |pred - target| (paired).
Comparisons:
  B3   nested-selection spatial (Task 2)        vs pseudobulk baseline (ladder arm 2)
  A2   nested-selection cell type (Task 1f)     vs pseudobulk baseline
  1f   each per-cell-type clock                 vs all-cells clock on the SAME animals
Pre-registered reading (set before running, by the user): if the B3 difference CI
excludes zero, spatial context gives a real but small gain and a spage2vec scoping run is
warranted; if it includes zero, learned spatial context does not beat pseudobulk and the
spage2vec direction is shelved.

    PYTHONPATH="$PWD:$PWD/scripts" python analysis/paired_difference.py
"""

from __future__ import annotations

import immunotransformer  # noqa: F401  (OpenMP guard — must precede numpy; see HANDOFF.md)

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clock_composition as CC  # noqa: E402
import clock_selection as CS  # noqa: E402
from duration_clock import loao_target  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
OUT = "runs/clock_composition"
N_BOOT = 2000


def paired(name, pa, pb_, ref, la, lb):
    rng = np.random.default_rng(0)
    n = len(ref)
    diffs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        ra, rb = spearmanr(pa[i], ref[i]).statistic, spearmanr(pb_[i], ref[i]).statistic
        if np.isfinite(ra) and np.isfinite(rb):
            diffs.append(ra - rb)
    diffs = np.array(diffs)
    ea, eb = np.abs(pa - ref), np.abs(pb_ - ref)
    w = wilcoxon(ea, eb)
    r = {"comparison": name, "A": la, "B": lb, "n": n,
         "rho_A": float(spearmanr(pa, ref).statistic), "rho_B": float(spearmanr(pb_, ref).statistic),
         "diff_mean": float(diffs.mean()),
         "diff_ci": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))],
         "frac_diff_le_0": float((diffs <= 0).mean()),
         "median_abs_err_A": float(np.median(ea)), "median_abs_err_B": float(np.median(eb)),
         "wilcoxon_p": float(w.pvalue)}
    r["ci_excludes_zero"] = bool(r["diff_ci"][0] > 0 or r["diff_ci"][1] < 0)
    print(f"  {name:52s} {r['rho_A']:+.3f} vs {r['rho_B']:+.3f}  diff {r['diff_mean']:+.3f} "
          f"[{r['diff_ci'][0]:+.3f}, {r['diff_ci'][1]:+.3f}]  P(diff<=0) {r['frac_diff_le_0']:.2f}  "
          f"Wilcoxon p {r['wilcoxon_p']:.3f}", flush=True)
    return r


def main():
    F = CS.feature_sets()
    rr = F["rr"]
    y, sev = rr.day.to_numpy(), rr.score.to_numpy().reshape(-1, 1)
    ref = loao_target(y, sev)
    lad = pd.read_csv("runs/baseline_ladder/loao_predictions_arms1-5.csv")
    base = (lad[(lad.target == "day_of_sacrifice") & (lad.arm == "arm2")]
            .set_index("sample_name").loc[rr.index, "pred"].to_numpy())
    nested = json.load(open(os.path.join(OUT, "nested_selection.json")))
    out = []

    def nested_pred(label):
        df = pd.DataFrame(nested[label]["per_fold"])
        df = df.set_index("held_out" if "held_out" in df else "index").loc[rr.index]
        return df.pred.to_numpy()
    out.append(paired("B3 nested spatial vs pseudobulk", nested_pred("task2_spatial"), base, ref,
                      "nested-selection spatial (Task 2)", "pseudobulk (arm 2)"))
    out.append(paired("A2 nested cell type vs pseudobulk", nested_pred("task1_cell_type"), base,
                      ref, "nested-selection cell type (Task 1f)", "pseudobulk (arm 2)"))
    for ct, e in F["task1"].items():
        animals = e["animals"]
        yy = rr.loc[animals, "day"].to_numpy()
        ss = rr.loc[animals, "score"].to_numpy().reshape(-1, 1)
        rf = loao_target(yy, ss)
        pct, _ = CC.loao_audit(e["X"], yy, ss)
        if len(animals) == len(rr):
            pall = base
        else:
            pall, _ = CC.loao_audit(F["X_all"][[list(rr.index).index(a) for a in animals]], yy, ss)
        out.append(paired(f"1f {ct} vs all cells (same {len(animals)} animals)", pct, pall, rf,
                          f"{ct} pseudobulk", "all-cells pseudobulk"))
    with open(os.path.join(OUT, "paired_difference.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    L = ["# Paired comparisons on the same animals (decisive test)", "",
         "Bootstrap over animals (2000): both rho recomputed on each resample; difference = A - B.",
         "Wilcoxon signed-rank on per-animal |prediction - residualised day|.", "",
         "| comparison | n | rho A | rho B | mean diff | 95% CI of diff | P(diff <= 0) | Wilcoxon p |",
         "|---|---|---|---|---|---|---|---|"]
    for r in out:
        L.append(f"| {r['comparison']} | {r['n']} | {r['rho_A']:+.3f} | {r['rho_B']:+.3f} | "
                 f"{r['diff_mean']:+.3f} | [{r['diff_ci'][0]:+.3f}, {r['diff_ci'][1]:+.3f}] | "
                 f"{r['frac_diff_le_0']:.2f} | {r['wilcoxon_p']:.3f} |")
    b3 = out[0]
    L += ["", "PRE-REGISTERED READING (B3): " + (
        "difference CI EXCLUDES zero -> spatial context gives a real but small gain; a spage2vec "
        "scoping run is warranted." if b3["ci_excludes_zero"] else
        "difference CI INCLUDES zero -> learned spatial context does not beat pseudobulk; the "
        "spage2vec direction is shelved.")]
    with open(os.path.join(OUT, "paired_difference.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
