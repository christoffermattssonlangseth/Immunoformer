"""TASK 1 — what is the duration clock made of?

The RR duration clock (scripts/duration_clock.py loao_clock: LOAO, ElasticNetCV on the
top-1000-variance log-CP10k pseudobulk, X and day residualised on score_sacrifice inside
each fold) reaches rho ~0.83. How much of that is disease time, and how much time-of-day
(circadian genes), sex (Xist), or something else?

Nothing about the architecture or CV changes between conditions — only (a) which genes
enter the feature matrix (exclusions applied BEFORE the top-variance filter) and (b) which
covariates are residualised inside each fold. Every fit reports its zero-feature folds
(folds where the elastic net kept no feature, so the prediction is the training mean of the
residualised target).

Section C2_G3_Mid_1 is excluded (Task 0 reassignment rule; chronic, does not touch RR).

    PYTHONPATH="$PWD:$PWD/scripts" python analysis/clock_composition.py
"""

from __future__ import annotations

import immunotransformer  # noqa: F401  (OpenMP guard — must precede numpy; see HANDOFF.md)

import json
import os
import sys
import time
import warnings

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from duration_clock import ENET_KW, _topvar, final_coefs, loao_clock, loao_target  # noqa: E402
from immunotransformer.stats import bootstrap_spearman, compare_rhos  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

ATLAS = os.environ.get(
    "RRMAP2_H5AD",
    os.path.expanduser(
        "~/Downloads/RRMAP2_xenium_all_samples.cellcharter.companion.ready.with_metadata."
        "rerun.with_AnnoL1Curated_with_Region_Anno2to4Updated.h5ad"),
)
PB = "runs/rr_within_relapse/pseudobulk_all67.npz"
TRAJ = "runs/trajectory_features/animal_trajectory.csv"
OLD_CLOCK = "runs/duration_clock/clock_genes.csv"
OUT = "runs/clock_composition"
CT_CACHE = os.path.join(OUT, "pseudobulk_section_celltype.npz")
EXCLUDE_SECTIONS = {"C2_G3_Mid_1"}
RR, CHR = "RELAPSE REMITTING", "CHRONIC"
N_BOOT = 2000
MIN_CELLS, MIN_ANIMALS = 50, 25

CIRCADIAN = ["Nr1d1", "Nr1d2", "Dbp", "Bhlhe40", "Bhlhe41", "Per1", "Per2", "Per3", "Cry1",
             "Cry2", "Arntl", "Arntl2", "Clock", "Npas2", "Nfil3", "Tef", "Hlf", "Ciart", "Rora",
             "Rorb", "Rorc"]
CIRC_SCORE = ["Nr1d1", "Dbp", "Bhlhe40", "Per1", "Hlf"]
SEX_GENES = ["Xist", "Tsix", "Ddx3y", "Uty", "Eif2s3y", "Kdm5d"]
BLOCKS = {
    "matrix/scar": ["Col4a2", "Thbs2", "Serpine2", "Col4a1", "Eln", "Ptx3", "Fbln2", "Fmod"],
    "circadian": CIRCADIAN,
    "sex": SEX_GENES,
    "lymphoid": ["Cxcl13", "Mzb1", "Jchain", "Tcf7"],
    "lipid": ["Srebf1", "Lpl", "Hmgcr", "Hsd17b7"],
    "tissue loss": ["Mog", "Gad1"],
    "microglial identity": ["Tmem173", "Ccr2", "Tmem119"],
}


# ---------------------------------------------------------------- data

def _lognorm(c):
    return np.log1p(c / np.maximum(c.sum(1, keepdims=True), 1) * 1e4)


def load_rr():
    d = np.load(PB, allow_pickle=True)
    meta = pd.DataFrame(d["section_meta"].tolist(), columns=d["section_meta_cols"])
    keep = ~meta.meta_sample_id.isin(EXCLUDE_SECTIONS).to_numpy()
    meta, counts = meta[keep].reset_index(drop=True), d["section_counts"][keep]
    genes = d["genes"].astype(str)
    animals = meta.sample_name.drop_duplicates().to_numpy()
    pb = pd.DataFrame(_lognorm(np.vstack([counts[(meta.sample_name == a).to_numpy()].sum(0)
                                          for a in animals])), index=animals, columns=genes)
    traj = pd.read_csv(TRAJ).set_index("sample_name").loc[animals]
    info = pd.DataFrame({"model": meta.drop_duplicates("sample_name").set_index("sample_name")
                         .loc[animals, "model"].to_numpy(),
                         "run_date": meta.drop_duplicates("sample_name").set_index("sample_name")
                         .loc[animals, "run_date"].to_numpy(),
                         "stage": traj.stage.to_numpy(), "sex": traj.sex.to_numpy(),
                         "condition": traj.condition.to_numpy(),
                         "day": traj.day_of_sacrifice.to_numpy(float),
                         "score": traj.score_sacrifice.to_numpy(float),
                         "cumulative_score": traj.cumulative_score.to_numpy(float)},
                        index=animals)
    return pb, info, meta


def celltype_pseudobulk(sections):
    """Summed counts per (section, Anno_L1_curated) from layers/counts, cached."""
    if os.path.exists(CT_CACHE):
        d = np.load(CT_CACHE, allow_pickle=True)
        return d["counts"], d["n_cells"], d["sections"].astype(str), d["celltypes"].astype(str)
    with h5py.File(ATLAS, "r") as f:
        obs = f["obs"]

        def codes(k):
            g = obs[k]
            cats = np.array([c.decode() if isinstance(c, bytes) else c for c in g["categories"][:]])
            return g["codes"][:], cats
        sc_, scats = codes("meta_sample_id")
        cc_, ccats = codes("Anno_L1_curated")
        nct = len(ccats)
        grp = sc_.astype(np.int64) * nct + cc_
        ngroups = len(scats) * nct
        C = f["layers/counts"]
        indptr = C["indptr"][:]
        ng = C.attrs["shape"][1] if "shape" in C.attrs else f["var"][f["var"].attrs["_index"]].shape[0]
        sums = np.zeros((ngroups, ng))
        t0 = time.time()
        for a in range(0, len(grp), 100_000):
            e = min(a + 100_000, len(grp))
            lo, hi = indptr[a], indptr[e]
            X = sp.csr_matrix((C["data"][lo:hi], C["indices"][lo:hi], indptr[a:e + 1] - lo),
                              shape=(e - a, ng))
            G = sp.csr_matrix((np.ones(e - a), (grp[a:e], np.arange(e - a))), shape=(ngroups, e - a))
            sums += (G @ X).toarray()
            print(f"  [celltype pb] {e:,}/{len(grp):,}  {time.time() - t0:.0f}s", flush=True)
        n_cells = np.bincount(grp, minlength=ngroups)
    counts = sums.reshape(len(scats), nct, ng)
    n_cells = n_cells.reshape(len(scats), nct)
    os.makedirs(OUT, exist_ok=True)
    np.savez(CT_CACHE, counts=counts, n_cells=n_cells, sections=scats, celltypes=ccats)
    return counts, n_cells, scats, ccats


# ---------------------------------------------------------------- clock with audit

def loao_audit(X, y, covar=None, seed=0):
    """loao_clock's exact fold loop, additionally returning the number of non-zero
    coefficients per fold (0 = zero-feature fold)."""
    n = len(y)
    pred, nnz = np.full(n, np.nan), np.zeros(n, int)
    for i in range(n):
        tr = np.ones(n, bool); tr[i] = False
        Xtr, Xte, ytr = X[tr], X[i:i + 1], y[tr]
        if covar is not None:
            ctr, cte = covar[tr], covar[i:i + 1]
            fx = LinearRegression().fit(ctr, Xtr)
            fy = LinearRegression().fit(ctr, ytr)
            Xtr, Xte = Xtr - fx.predict(ctr), Xte - fx.predict(cte)
            ytr = ytr - fy.predict(ctr)
        sel = _topvar(Xtr)
        Xtr, Xte = Xtr[:, sel], Xte[:, sel]
        sc = StandardScaler().fit(Xtr)
        m = ElasticNetCV(random_state=seed, **ENET_KW).fit(sc.transform(Xtr), ytr)
        pred[i] = m.predict(sc.transform(Xte))[0]
        nnz[i] = int((m.coef_ != 0).sum())
    return pred, nnz


def evaluate(name, X, y, covar, base=None, genes=None, extra=None):
    pred, nnz = loao_audit(X, y, covar)
    ref = loao_target(y, covar) if covar is not None else y
    b = bootstrap_spearman(pred, ref, n_boot=N_BOOT, seed=0)
    r = {"condition": name, "n": int(len(y)), "n_features_in": int(X.shape[1]),
         "rho": float(b["rho"]), "ci": [float(b["ci_lo"]), float(b["ci_hi"])],
         "r2": float(1 - np.sum((pred - ref) ** 2) / np.sum((ref - ref.mean()) ** 2)),
         "zero_feature_folds": int((nnz == 0).sum()), "median_nonzero": float(np.median(nnz))}
    if base is not None:
        r["fisher_p_vs_baseline"] = compare_rhos(r["rho"], r["n"], base["rho"], base["n"])["p_value"]
    if extra:
        r.update(extra)
    print(f"  {name:48s} rho {r['rho']:+.3f} [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}]  "
          f"zero-feature folds {r['zero_feature_folds']}/{r['n']}", flush=True)
    return r, pred


def top_genes(X, y, covar, genes, k):
    coefs, _ = final_coefs(X, y, np.asarray(genes), covar=covar)
    return coefs.head(k), coefs


def block_mass(coefs):
    tot = coefs.coef.abs().sum()
    out = {b: float(coefs[coefs.gene.isin(g)].coef.abs().sum() / tot) if tot else 0.0
           for b, g in BLOCKS.items()}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


# ---------------------------------------------------------------- main

def main():
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    pb, info, meta = load_rr()
    genes = pb.columns.to_numpy()
    rr = info[info.model == RR]
    X_all = pb.loc[rr.index].to_numpy()
    y, sev = rr.day.to_numpy(), rr.score.to_numpy().reshape(-1, 1)
    res = {"cohort": "RR", "n": len(rr)}

    # sanity: the audit loop reproduces loao_clock exactly
    p_ref = loao_clock(X_all, y, covar=sev)
    print("[check] audit loop vs loao_clock ...", flush=True)
    base, p_base = evaluate("baseline (all genes)", X_all, y, sev)
    res["audit_matches_loao_clock"] = bool(np.allclose(p_ref, p_base, atol=1e-8))
    rows = [base]

    # 1a time of day
    res["time_of_day"] = {
        "recorded": False,
        "searched": ["spreadsheets/*AllScore*.xlsx, *AllWeight*.xlsx (all columns, 1 sheet each)",
                     "data/rrmap2_animal_meta.csv", "atlas obs + uns (RRMAP2 h5ad)",
                     "RRMAP2/README.md, RRMap/README.md, RRMap/misc/LOG/logbook.md", "docs/"],
        "note": ("docs/rrmap2-duration-clock.md states 'sacrifice time-of-day is constant across "
                 "animals' (added in commit fba9936, 2026-06-18) without a source; no file in the "
                 "repo or sibling repos records it.")}
    circ_present = [g for g in CIRCADIAN if g in genes]
    cs = pb.loc[rr.index, [g for g in CIRC_SCORE if g in genes]]
    circ_score = ((cs - cs.mean()) / cs.std()).mean(1)
    res["circadian_score_vs_day"] = float(spearmanr(circ_score, y).statistic)
    res["circadian_score_vs_score"] = float(spearmanr(circ_score, sev.ravel()).statistic)

    # 1b / 1c
    no_circ = [g for g in genes if g not in CIRCADIAN]
    r, _ = evaluate("1b without circadian genes", pb.loc[rr.index, no_circ].to_numpy(), y, sev, base)
    rows.append(r)
    r, _ = evaluate("1c circadian genes only", pb.loc[rr.index, circ_present].to_numpy(), y, sev, base)
    rows.append(r)
    chrn = info[(info.model == CHR) & (info.condition == "EAE") & info.cumulative_score.notna()]
    cov_c = chrn[["score", "day"]].to_numpy()
    cbase, _ = evaluate("chronic cumulative_score, all genes (= ladder arm 2)",
                        pb.loc[chrn.index].to_numpy(), chrn.cumulative_score.to_numpy(), cov_c)
    r, _ = evaluate("1c chronic cumulative_score, circadian only",
                    pb.loc[chrn.index, circ_present].to_numpy(), chrn.cumulative_score.to_numpy(),
                    cov_c, cbase)
    rows += [cbase, r]

    # 1d sex
    sex_present = [g for g in SEX_GENES if g in genes]
    male = (rr.sex == "M").astype(float).to_numpy()
    xist = pb.loc[rr.index, "Xist"] if "Xist" in genes else None
    bins = pd.cut(rr.day, [0, 20, 35, 60], labels=["<=20", "21-35", ">35"])
    res["sex"] = {
        "source": "metadata field `sex` (sheet + atlas); cross-checked against Xist pseudobulk",
        "sex_genes_on_panel": sex_present,
        # agreement = Xist separates the metadata sexes with no overlap (min F > max M)
        "xist_agrees_with_metadata": bool(xist[rr.sex == "F"].min() > xist[rr.sex == "M"].max())
        if xist is not None else None,
        "xist_range_F": [float(xist[rr.sex == "F"].min()), float(xist[rr.sex == "F"].max())]
        if xist is not None else None,
        "xist_range_M": [float(xist[rr.sex == "M"].min()), float(xist[rr.sex == "M"].max())]
        if xist is not None else None,
        "sex_x_day_bin": pd.crosstab(rr.sex, bins).to_dict(orient="index"),
        "sex_x_stage": pd.crosstab(rr.stage, rr.sex).to_dict(orient="index"),
        "rho_sex_day": float(spearmanr(male, y).statistic)}
    no_sex = [g for g in genes if g not in SEX_GENES]
    r, _ = evaluate("1d(i) sex genes dropped", pb.loc[rr.index, no_sex].to_numpy(), y, sev, base)
    rows.append(r)
    r, _ = evaluate("1d(ii) sex residualised (covariate)", X_all, y, np.column_stack([sev, male]),
                    base)
    rows.append(r)

    # 1e circadian + sex excluded
    clean = [g for g in genes if g not in CIRCADIAN and g not in SEX_GENES]
    Xc = pb.loc[rr.index, clean].to_numpy()
    r, _ = evaluate("1e circadian AND sex genes excluded", Xc, y, sev, base)
    rows.append(r)
    top30, coefs_clean = top_genes(Xc, y, sev, clean, 30)
    _, coefs_base = top_genes(X_all, y, sev, genes, 30)
    old = pd.read_csv(OLD_CLOCK)
    shared = coefs_clean.merge(old, on="gene", suffixes=("_new", "_old"))
    res["1e"] = {
        "top30": top30.round(4).to_dict("records"),
        "n_nonzero": int(len(coefs_clean)),
        "block_mass_clean": block_mass(coefs_clean),
        "block_mass_baseline_refit": block_mass(coefs_base),
        "block_mass_published_135": block_mass(old),
        "first_matrix_gene_rank": int(coefs_clean.reset_index(drop=True).index[
            coefs_clean.gene.isin(BLOCKS["matrix/scar"])].min() + 1)
        if coefs_clean.gene.isin(BLOCKS["matrix/scar"]).any() else None,
        "overlap_with_published_135": int(len(shared)),
        "same_sign": int((np.sign(shared.coef_new) == np.sign(shared.coef_old)).sum()),
        "baseline_refit_n_nonzero": int(len(coefs_base)),
        "overlap_baseline_refit_with_published": int(coefs_base.gene.isin(old.gene).sum())}
    coefs_clean.to_csv(os.path.join(OUT, "clock_genes_no_circadian_no_sex.csv"), index=False)
    coefs_base.to_csv(os.path.join(OUT, "clock_genes_baseline_refit.csv"), index=False)

    # 1f per cell type
    counts, n_cells, scats, ccats = celltype_pseudobulk(set(meta.meta_sample_id))
    sec_idx = {s: i for i, s in enumerate(scats)}
    ct_rows = []
    for j, ct in enumerate(ccats):
        per_animal_cells, per_animal_counts = {}, {}
        for a in rr.index:
            si = [sec_idx[s] for s in meta.loc[meta.sample_name == a, "meta_sample_id"]]
            per_animal_cells[a] = int(n_cells[si, j].sum())
            per_animal_counts[a] = counts[si, j].sum(0)
        ok = [a for a in rr.index if per_animal_cells[a] >= MIN_CELLS]
        entry = {"cell_type": ct, "animals_passing": len(ok),
                 "median_cells_per_animal": float(np.median(list(per_animal_cells.values())))}
        if len(ok) < MIN_ANIMALS:
            entry["skipped"] = f"only {len(ok)} animals with >= {MIN_CELLS} cells"
            ct_rows.append(entry)
            print(f"  [1f] {ct}: skipped ({len(ok)} animals)", flush=True)
            continue
        Xct = _lognorm(np.vstack([per_animal_counts[a] for a in ok]))
        sub = rr.loc[ok]
        yy, ss = sub.day.to_numpy(), sub.score.to_numpy().reshape(-1, 1)
        r, _ = evaluate(f"1f {ct}", Xct, yy, ss)
        ref_all, _ = evaluate(f"     all cells, same {len(ok)} animals", pb.loc[ok].to_numpy(), yy, ss)
        top10, _ = top_genes(Xct, yy, ss, genes, 10)
        entry.update({**r, "all_cells_same_animals": ref_all,
                      "fisher_p_vs_all_cells_same_animals":
                          compare_rhos(r["rho"], r["n"], ref_all["rho"], r["n"])["p_value"],
                      "top10": top10.round(4).to_dict("records")})
        ct_rows.append(entry)
    res["per_cell_type"] = sorted(ct_rows, key=lambda e: -e.get("rho", -9))
    res["conditions"] = rows
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    _report(res)


def _fmt(r):
    p = r.get("fisher_p_vs_baseline")
    return (f"| {r['condition']} | {r['n']} | {r['n_features_in']} | {r['rho']:+.3f} | "
            f"[{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}] | {'—' if p is None else f'{p:.3f}'} | "
            f"{r['zero_feature_folds']}/{r['n']} |")


def _report(res):
    L = ["# Task 1 — report (machine-generated tables; see WRITEUP.md for the reading)", "",
         f"Audit loop reproduces loao_clock exactly: {res['audit_matches_loao_clock']}", "",
         "| condition | n | genes in | rho | 95% CI | Fisher z p vs baseline | zero-feature folds |",
         "|---|---|---|---|---|---|---|"]
    L += [_fmt(r) for r in res["conditions"]]
    L += ["", f"circadian score (mean z of {CIRC_SCORE}) vs day: rho {res['circadian_score_vs_day']:+.2f}; "
          f"vs score_sacrifice: {res['circadian_score_vs_score']:+.2f}", "",
          f"sex: {json.dumps(res['sex'], default=str)}", "",
          f"1e: {json.dumps({k: v for k, v in res['1e'].items() if k != 'top30'}, default=str)}",
          "1e top 30: " + ", ".join(f"{g['gene']} ({g['coef']:+.2f})" for g in res["1e"]["top30"]),
          "", "## 1f per cell type (ranked)", "",
          "| cell type | animals | rho | 95% CI | all cells, same animals | p vs all cells | zero-feature folds | top genes |",
          "|---|---|---|---|---|---|---|---|"]
    for e in res["per_cell_type"]:
        if "skipped" in e:
            L.append(f"| {e['cell_type']} | {e['animals_passing']} | skipped | | | | | |")
            continue
        a = e["all_cells_same_animals"]
        L.append(f"| {e['cell_type']} | {e['n']} | {e['rho']:+.3f} | [{e['ci'][0]:+.2f}, {e['ci'][1]:+.2f}] | "
                 f"{a['rho']:+.3f} | {e['fisher_p_vs_all_cells_same_animals']:.3f} | "
                 f"{e['zero_feature_folds']}/{e['n']} | "
                 + ", ".join(f"{g['gene']} ({g['coef']:+.2f})" for g in e["top10"]) + " |")
    with open(os.path.join(OUT, "report.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
