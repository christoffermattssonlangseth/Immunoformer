"""Wrap-up block items 1 and 2.

1  Severity-matched null for collagen IV. For each cohort and for endothelial cells and
   VSMCs: rho(gene, severity) and rho(gene, day | severity) for every gene in the A4 null set
   (expressed in > 10% of that type's cells). Comparison set = genes whose rho(gene, severity)
   is within +/- 0.10 of Col4a1's (resp. Col4a2's) value; widened to +/- 0.15 if < 100 genes.
   Rank of Col4a1 / Col4a2 by |rho(gene, day | severity)| and by |rho(gene, day)| (raw)
   within that set: percentile and empirical p. Col4a1/Col4a2 were selected from the fitted
   clock's coefficients; this rank is the selection-aware null.
2  Composition follow-up: three-type composition clock (fibroblast, endothelial, astrocyte
   proportions) vs the 17-type C1 clock (paired bootstrap); raw and severity-adjusted rho of
   each proportion with day; endothelial proportion vs within-endothelial Col4 mean.

    PYTHONPATH="$PWD:$PWD/scripts" python analysis/wrapup_stats.py
"""

from __future__ import annotations

import immunotransformer  # noqa: F401  (OpenMP guard — must precede numpy; see HANDOFF.md)

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clock_composition as CC  # noqa: E402
from closing_block import ci, lognorm, partial_rank, sp_, rho_pack, NON_CELLTYPES_L1  # noqa: E402
from duration_clock import loao_target  # noqa: E402
from paired_difference import paired  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
OUT = "runs/closing_block/wrapup"
COHORTS = {"RELAPSE REMITTING": "RR", "CHRONIC": "chronic"}
THREE = ["Fibroblast", "Endothelial", "Astrocyte"]


def per_type_matrix(ct, animals, meta, counts, cti, sidx, genes):
    j = cti[ct]
    c = np.vstack([counts[[sidx[s] for s in meta.loc[meta.sample_name == a, "meta_sample_id"]], j].sum(0)
                   for a in animals])
    return pd.DataFrame(lognorm(c), index=animals, columns=genes)


def main():
    os.makedirs(OUT, exist_ok=True)
    pb, info, meta = CC.load_rr()
    genes = pb.columns.to_numpy()
    counts, n_cells, scats, ccats = CC.celltype_pseudobulk(None)
    sidx = {s: i for i, s in enumerate(scats)}
    cti = {c: i for i, c in enumerate(ccats)}
    d = np.load("runs/closing_block/data.npz", allow_pickle=True)
    pos_frac = pd.DataFrame(d["pos"] / np.maximum(d["n1"][:, None], 1), index=d["c1cats"].astype(str),
                            columns=d["genes"].astype(str))
    res = {"1": {}, "2": {}}
    for model, ck in COHORTS.items():
        coh = info[(info.model == model) & info.day.notna() & info.score.notna()]
        a_ = list(coh.index)
        day, sev = coh.day.to_numpy(), coh.score.to_numpy()
        # ---------------- 1
        for ct in ("Endothelial", "VSMC"):
            X = per_type_matrix(ct, a_, meta, counts, cti, sidx, genes)
            expressed = [g for g in genes if pos_frac.loc[ct, g] > 0.10]
            tab = pd.DataFrame({"rho_sev": [sp_(X[g].to_numpy(), sev) for g in expressed],
                                "rho_day_given_sev": [partial_rank(X[g].to_numpy(), day, sev) for g in expressed],
                                "rho_day_raw": [sp_(X[g].to_numpy(), day) for g in expressed]},
                               index=expressed)
            tab.to_csv(os.path.join(OUT, f"gene_stats_{ck}_{ct}.csv"))
            out = {"n_expressed": len(expressed)}
            for g in ("Col4a1", "Col4a2"):
                v = tab.loc[g]
                tol = 0.10
                band = tab[(tab.rho_sev - v.rho_sev).abs() <= tol]
                if len(band) < 100:
                    tol = 0.15
                    band = tab[(tab.rho_sev - v.rho_sev).abs() <= tol]
                rk = lambda col: {"percentile": float((band[col].abs() < abs(v[col])).mean() * 100),
                                  "empirical_p": float((band[col].abs() >= abs(v[col])).mean())}
                out[g] = {"rho_sev": float(v.rho_sev), "rho_day_given_sev": float(v.rho_day_given_sev),
                          "rho_day_raw": float(v.rho_day_raw), "tolerance": tol, "set_size": int(len(band)),
                          "rank_adjusted": rk("rho_day_given_sev"), "rank_raw": rk("rho_day_raw"),
                          "whole_set_raw_percentile": float((tab.rho_day_raw.abs() < abs(v.rho_day_raw)).mean() * 100),
                          "whole_set_adjusted_percentile": float((tab.rho_day_given_sev.abs() < abs(v.rho_day_given_sev)).mean() * 100)}
            res["1"][f"{ck}/{ct}"] = out
            print(f"[1] {ck}/{ct}: " + json.dumps({g: out[g] for g in ("Col4a1", "Col4a2")}), flush=True)
        # ---------------- 2
        P = pd.DataFrame({c: [n_cells[[sidx[s] for s in meta.loc[meta.sample_name == a, "meta_sample_id"]], j].sum()
                              for a in a_] for c, j in cti.items()}, index=a_)
        P = P.div(P.sum(1), axis=0)
        P17 = P[[c for c in cti if c not in NON_CELLTYPES_L1]]
        y, s2 = day, sev.reshape(-1, 1)
        ref = loao_target(y, s2)
        p17, z17 = CC.loao_audit(P17.to_numpy(), y, s2)
        p3, z3 = CC.loao_audit(P[THREE].to_numpy(), y, s2)
        endo = per_type_matrix("Endothelial", a_, meta, counts, cti, sidx, genes)
        col4 = ((endo["Col4a1"] + endo["Col4a2"]) / 2).to_numpy()
        ep = P["Endothelial"].to_numpy()
        res["2"][ck] = {
            "three_type_clock": rho_pack(p3, ref, z3), "seventeen_type_clock": rho_pack(p17, ref, z17),
            "three_vs_seventeen": paired("3-type vs 17-type composition clock", p3, p17, ref, "3-type", "17-type"),
            "proportions": {c: {"rho_day_raw": sp_(P[c].to_numpy(), day), "ci_raw": ci(sp_, P[c].to_numpy(), day),
                                "rho_day_given_sev": partial_rank(P[c].to_numpy(), day, sev),
                                "ci_given_sev": ci(lambda a, b, c_: partial_rank(a, b, c_), P[c].to_numpy(), day, sev),
                                "median": float(P[c].median())} for c in THREE},
            "endo_proportion_vs_col4": {
                "rho_raw": sp_(ep, col4), "ci_raw": ci(sp_, ep, col4),
                "rho_given_day_and_sev": partial_rank(ep, col4, np.column_stack([day, sev])),
                "ci_given_day_and_sev": ci(lambda a, b, c_: partial_rank(a, b, c_), ep, col4, np.column_stack([day, sev]))}}
        print(f"[2] {ck} done", flush=True)
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print("[done]")


if __name__ == "__main__":
    main()
