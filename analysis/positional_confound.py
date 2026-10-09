"""Positional confound: is the clock partly reading WHERE along the cord a section was cut?

Hoxa10 is the top astrocyte-clock gene; Hox genes encode rostrocaudal position, which
varies continuously within each gross region (C/T/L).
  2a  positional genes on the panel (Hox*, Cdx*, Meis1/2) — coefficients and total |coef|
      share in the all-cells and astrocyte clocks;
  2b  continuous section position = PC1 of standardised Hox section pseudobulk, oriented
      so lumbar is high; check it orders cervical < thoracic < lumbar;
  2c  clock refit with the animal's cell-weighted mean section position residualised out
      inside each fold (PCA refit on the training animals' sections only), next to
      severity; all cells and astrocytes; paired difference vs the baseline on the SAME
      target (day | score) and animals;
  2d  clock refit with all positional genes excluded from features (before the variance
      filter); same paired comparison;
  2e  position (animal mean and within each region) vs day_of_sacrifice and stage.

    PYTHONPATH="$PWD:$PWD/scripts" python analysis/positional_confound.py
"""

from __future__ import annotations

import immunotransformer  # noqa: F401  (OpenMP guard — must precede numpy; see HANDOFF.md)

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clock_composition as CC  # noqa: E402
import clock_selection as CS  # noqa: E402
from duration_clock import ENET_KW, _topvar, final_coefs, loao_target  # noqa: E402
from paired_difference import paired  # noqa: E402

OUT = "runs/clock_composition/positional"
REGION_ORDER = {"C": 0, "T": 1, "L": 2}


def positional_genes(genes):
    return [g for g in genes if g.startswith("Hox") or g.startswith("Cdx") or g in ("Meis1", "Meis2")]


def section_table(rr_index):
    d = np.load(CC.PB, allow_pickle=True)
    meta = pd.DataFrame(d["section_meta"].tolist(), columns=d["section_meta_cols"])
    keep = (~meta.meta_sample_id.isin(CC.EXCLUDE_SECTIONS) & meta.sample_name.isin(rr_index)).to_numpy()
    meta = meta[keep].reset_index(drop=True)
    X = pd.DataFrame(CC._lognorm(d["section_counts"][keep]), columns=d["genes"].astype(str))
    meta["n_cells"] = meta.n_cells.astype(int)
    return meta, X


def fit_position(meta, Xsec, hox, train_animals):
    """PCA(1) on standardised Hox section expression of TRAINING animals' sections;
    oriented so lumbar > cervical on the training sections; returns all-section scores."""
    tr = meta.sample_name.isin(train_animals).to_numpy()
    sc = StandardScaler().fit(Xsec.loc[tr, hox])
    pca = PCA(n_components=1, random_state=0).fit(sc.transform(Xsec.loc[tr, hox]))
    s = pca.transform(sc.transform(Xsec[hox]))[:, 0]
    if s[tr & (meta.region == "L").to_numpy()].mean() < s[tr & (meta.region == "C").to_numpy()].mean():
        s = -s
    return s, float(pca.explained_variance_ratio_[0])


def animal_position(meta, s, animals):
    m = meta.assign(s=s)
    return np.array([np.average(m.loc[m.sample_name == a, "s"], weights=m.loc[m.sample_name == a, "n_cells"])
                     for a in animals])


def clock_with_position(X, y, sev, animals, meta, Xsec, hox):
    """LOAO clock (same mechanics as loao_clock) with an in-fold position covariate."""
    n = len(y)
    pred, nnz = np.full(n, np.nan), np.zeros(n, int)
    for i in range(n):
        tr = np.ones(n, bool); tr[i] = False
        s, _ = fit_position(meta, Xsec, hox, [a for a, t in zip(animals, tr) if t])
        pos = animal_position(meta, s, animals)
        cov = np.column_stack([sev, pos])
        fx = LinearRegression().fit(cov[tr], X[tr]); fy = LinearRegression().fit(cov[tr], y[tr])
        Xtr, Xte = X[tr] - fx.predict(cov[tr]), X[i:i + 1] - fx.predict(cov[i:i + 1])
        ytr = y[tr] - fy.predict(cov[tr])
        sel = _topvar(Xtr)
        scl = StandardScaler().fit(Xtr[:, sel])
        m = ElasticNetCV(random_state=0, **ENET_KW).fit(scl.transform(Xtr[:, sel]), ytr)
        pred[i] = m.predict(scl.transform(Xte[:, sel]))[0]
        nnz[i] = int((m.coef_ != 0).sum())
    return pred, nnz


def main():
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    F = CS.feature_sets()
    rr, genes = F["rr"], F["genes"]
    animals = list(rr.index)
    y, sev = rr.day.to_numpy(), rr.score.to_numpy().reshape(-1, 1)
    ref = loao_target(y, sev)
    meta, Xsec = section_table(rr.index)
    pos_genes = positional_genes(genes)
    hox = [g for g in pos_genes if g.startswith("Hox")]
    astro = F["task1"]["Astrocyte"]
    assert astro["animals"] == animals
    res = {"positional_genes_on_panel": pos_genes, "n_hox": len(hox)}

    # 2a coefficients
    for label, Xf in (("all_cells", F["X_all"]), ("astrocyte", astro["X"])):
        cf, _ = final_coefs(Xf, y, genes, covar=sev)
        tot = cf.coef.abs().sum()
        p = cf[cf.gene.isin(pos_genes)]
        res[f"2a_{label}"] = {"positional_nonzero": p.round(3).to_dict("records"),
                              "positional_coef_share": float(p.coef.abs().sum() / tot),
                              "n_nonzero_total": int(len(cf))}

    # 2b position check (all RR sections, fit on all — descriptive only)
    s_all, ve = fit_position(meta, Xsec, hox, animals)
    meta["position"] = s_all
    reg = meta.region.map(REGION_ORDER)
    pairs = {f"{a}<{b}": float(np.mean([x < z for x in meta.loc[meta.region == a, "position"]
                                       for z in meta.loc[meta.region == b, "position"]]))
             for a, b in (("C", "T"), ("T", "L"), ("C", "L"))}
    res["2b"] = {"pc1_variance_explained": ve,
                 "median_by_region": meta.groupby("region").position.median().round(3).to_dict(),
                 "spearman_with_region_order": float(spearmanr(meta.position, reg).statistic),
                 "fraction_correctly_ordered_pairs": pairs}
    meta[["meta_sample_id", "sample_name", "region", "n_cells", "position"]].to_csv(
        os.path.join(OUT, "section_position.csv"), index=False)

    # baselines (predictions on the same target, same animals)
    base_all, _ = CC.loao_audit(F["X_all"], y, sev)
    base_ast, _ = CC.loao_audit(astro["X"], y, sev)
    comps = []
    for label, Xf, base in (("all cells", F["X_all"], base_all), ("astrocyte", astro["X"], base_ast)):
        p, nnz = clock_with_position(Xf, y, sev, animals, meta, Xsec, hox)
        r = paired(f"2c {label}: + position covariate vs baseline", p, base, ref,
                   f"{label} + position", f"{label} baseline")
        r["zero_feature_folds"] = int((nnz == 0).sum())
        comps.append(r)
        keep = np.array([g not in set(pos_genes) for g in genes])
        p2, nnz2 = CC.loao_audit(Xf[:, keep], y, sev)
        r = paired(f"2d {label}: positional genes excluded vs baseline", p2, base, ref,
                   f"{label} no positional genes", f"{label} baseline")
        r["zero_feature_folds"] = int((nnz2 == 0).sum())
        comps.append(r)
    res["comparisons"] = comps

    # 2e position vs day / stage
    apos = animal_position(meta, s_all, animals)
    e = {"animal_position_vs_day": float(spearmanr(apos, y).statistic),
         "animal_position_vs_score": float(spearmanr(apos, sev.ravel()).statistic),
         "animal_position_kruskal_stage_p": float(kruskal(*[apos[rr.stage.to_numpy() == s]
                                                           for s in rr.stage.unique()
                                                           if (rr.stage == s).sum() > 1]).pvalue)}
    for rg in ("C", "T", "L"):
        m = meta[meta.region == rg].groupby("sample_name").position.mean()
        m = m[m.index.isin(animals)]
        e[f"within_{rg}_position_vs_day"] = float(spearmanr(m, rr.loc[m.index, "day"]).statistic)
        e[f"within_{rg}_n"] = int(len(m))
    rng = np.random.default_rng(0)
    bs = [spearmanr(apos[i], y[i]).statistic for i in (rng.integers(0, len(y), len(y)) for _ in range(2000))]
    e["animal_position_vs_day_ci"] = [float(np.nanquantile(bs, 0.025)), float(np.nanquantile(bs, 0.975))]
    e["by_stage"] = pd.DataFrame({"pos": apos, "stage": rr.stage.to_numpy(), "day": y}).groupby("stage") \
        .agg(n=("pos", "size"), position_median=("pos", "median"), day=("day", "median")) \
        .sort_values("day").round(3).to_dict(orient="index")
    res["2e"] = e
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=str)

    L = ["# Positional confound — machine tables", "",
         f"Positional genes on panel ({len(pos_genes)}): {pos_genes}", ""]
    for label in ("all_cells", "astrocyte"):
        a = res[f"2a_{label}"]
        L.append(f"2a {label}: positional |coef| share {a['positional_coef_share']:.1%} of "
                 f"{a['n_nonzero_total']} non-zero genes; non-zero positional: "
                 + (", ".join(f"{g['gene']} ({g['coef']:+.2f})" for g in a["positional_nonzero"]) or "none"))
    b = res["2b"]
    L += ["", f"2b Hox PC1 ({b['pc1_variance_explained']:.0%} of Hox variance): median by region "
          f"{b['median_by_region']}; Spearman with C<T<L order {b['spearman_with_region_order']:+.2f}; "
          f"correctly ordered pairs {b['fraction_correctly_ordered_pairs']}", "",
          "| comparison | n | rho A | rho B | mean diff | 95% CI | P(diff<=0) | Wilcoxon p | zero-feature folds |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in comps:
        L.append(f"| {r['comparison']} | {r['n']} | {r['rho_A']:+.3f} | {r['rho_B']:+.3f} | {r['diff_mean']:+.3f} | "
                 f"[{r['diff_ci'][0]:+.3f}, {r['diff_ci'][1]:+.3f}] | {r['frac_diff_le_0']:.2f} | "
                 f"{r['wilcoxon_p']:.3f} | {r['zero_feature_folds']}/{r['n']} |")
    L += ["", f"2e: {json.dumps({k: v for k, v in e.items() if k != 'by_stage'})}",
          f"2e by stage: {json.dumps(e['by_stage'])}"]
    with open(os.path.join(OUT, "report.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
