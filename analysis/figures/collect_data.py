"""Collect per-animal tables for the paper figures into docs/figures/data/.

Nothing new is analysed here. Values that earlier runs computed but did not save per animal
(clock LOAO predictions, compartment predictions, endothelial Col4, cell-type proportions,
selected-gene sets under each severity adjustment) are regenerated with the SAME code and
settings, and each regenerated set is checked against the rho already reported in runs/.

    PYTHONPATH="$PWD:$PWD/scripts:$PWD/analysis" python analysis/figures/collect_data.py
"""

from __future__ import annotations

import immunotransformer  # noqa: F401  (OpenMP guard — must precede numpy; see HANDOFF.md)

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
import clock_composition as CC  # noqa: E402
from closing_block import lognorm, NON_CELLTYPES_L1  # noqa: E402
from duration_clock import final_coefs, loao_target  # noqa: E402
from lesion_clock import Clock  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
OUT = "docs/figures/data"
COHORTS = {"RELAPSE REMITTING": "RR", "CHRONIC": "chronic"}


def check(label, got, expected, tol=0.005):
    ok = abs(got - expected) <= tol
    print(f"  [check] {label}: regenerated {got:+.3f} vs reported {expected:+.3f} -> {'OK' if ok else 'MISMATCH'}")
    if not ok:
        raise SystemExit(f"regenerated value does not match the reported one: {label}")


def main():
    os.makedirs(OUT, exist_ok=True)
    pb, info, meta = CC.load_rr()
    genes = pb.columns.to_numpy()
    traj = pd.read_csv(CC.TRAJ).set_index("sample_name")
    counts, n_cells, scats, ccats = CC.celltype_pseudobulk(None)
    sidx = {s: i for i, s in enumerate(scats)}
    cti = {c: i for i, c in enumerate(ccats)}
    d = np.load("runs/closing_block/data.npz", allow_pickle=True)
    comp, comp_n = d["comp"], d["comp_n"]
    cani = d["section_animal"].astype(str)
    cnames = list(d["compartments"].astype(str))
    cb = json.load(open("runs/closing_block/results.json"))
    sev_res = json.load(open("runs/closing_block/severity_adjustment/results.json"))
    rows, genesets = [], {}
    for model, ck in COHORTS.items():
        coh = info[(info.model == model) & info.day.notna() & info.score.notna()]
        a_ = list(coh.index)
        X = pb.loc[a_].to_numpy()
        y, sev = coh.day.to_numpy(), coh.score.to_numpy().reshape(-1, 1)
        ref = loao_target(y, sev)
        pr, pdays, cpred = np.empty(len(a_)), np.empty(len(a_)), {c: {} for c in ("lesion", "nonlesion_m50")}
        for i, a in enumerate(a_):
            tr = np.ones(len(a_), bool); tr[i] = False
            clk = Clock(X[tr], y[tr], sev[tr])
            r, dd = clk.predict(X[i:i + 1], sev[i])
            pr[i], pdays[i] = r[0], dd[0]
            for c in cpred:
                ci_ = cnames.index(c)
                ss = [k for k in np.where(cani == a)[0] if comp_n[k, ci_] >= 200]
                if ss:
                    rr_, _ = clk.predict(lognorm(comp[ss, ci_].astype(float)), np.full(len(ss), sev[i, 0]))
                    cpred[c][a] = float(np.mean(rr_))
        check(f"{ck} clock rho", spearmanr(pr, ref).statistic, sev_res[ck]["4_clock"]["terminal"]["rho"])
        for c, key in (("lesion", "lesion"), ("nonlesion_m50", "nonlesion_m50")):
            p = pd.Series(cpred[c])
            check(f"{ck} {c} compartment rho", spearmanr(p, pd.Series(ref, index=a_).loc[p.index]).statistic,
                  cb["B"][ck][key]["rho"])
        # endothelial Col4 and proportions
        j = cti["Endothelial"]
        E = pd.DataFrame(lognorm(np.vstack([counts[[sidx[s] for s in meta.loc[meta.sample_name == a, "meta_sample_id"]], j].sum(0)
                                            for a in a_])), index=a_, columns=genes)
        P = pd.DataFrame({c: [n_cells[[sidx[s] for s in meta.loc[meta.sample_name == a, "meta_sample_id"]], k].sum() for a in a_]
                          for c, k in cti.items()}, index=a_)
        P = P.div(P.sum(1), axis=0)
        for i, a in enumerate(a_):
            rows.append({"cohort": ck, "animal": a, "stage": traj.loc[a, "stage"], "condition": traj.loc[a, "condition"],
                         "day": y[i], "score": sev[i, 0], "run_date": info.loc[a, "run_date"],
                         "cumulative_score": traj.loc[a, "cumulative_score"], "peak_score": traj.loc[a, "peak_score"],
                         "clock_target_resid": ref[i], "clock_pred_resid": pr[i], "clock_pred_day": pdays[i],
                         "pred_lesion": cpred["lesion"].get(a, np.nan), "pred_nonlesion_m50": cpred["nonlesion_m50"].get(a, np.nan),
                         "endo_Col4a1": E.loc[a, "Col4a1"], "endo_Col4a2": E.loc[a, "Col4a2"],
                         "endo_Col4_mean": (E.loc[a, "Col4a1"] + E.loc[a, "Col4a2"]) / 2,
                         **{f"prop_{c}": P.loc[a, c] for c in ("Fibroblast", "Astrocyte", "Endothelial")}})
        # gene sets under three severity adjustments (S4)
        cum = traj.loc[a_, "cumulative_score"].to_numpy(float)
        ok = np.isfinite(cum)
        for k_, cov in (("none", None), ("terminal", sev), ("cumulative", cum.reshape(-1, 1))):
            cf, _ = final_coefs(X[ok], y[ok], genes, covar=None if cov is None else cov[ok])
            genesets[f"{ck}/{k_}"] = sorted(cf.gene)
        check(f"{ck} gene-set overlap none&terminal", len(set(genesets[f"{ck}/none"]) & set(genesets[f"{ck}/terminal"])),
              sev_res[ck]["4_gene_overlap"]["none&terminal"], tol=0)
        print(f"[{ck}] collected", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "animals.csv"), index=False)
    json.dump(genesets, open(os.path.join(OUT, "severity_variant_genesets.json"), "w"), indent=1)
    print("[done]")


if __name__ == "__main__":
    main()
