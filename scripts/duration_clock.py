"""The DURATION CLOCK — a multivariate molecular clock of accumulated EAE damage.

The relapse-PHASE/direction model was abandoned (unidentifiable: terminal cross-
sectional data + severity dominates). But DURATION is identifiable: `day_of_sacrifice`
is a real continuous temporal label, and within RR the relapse cycle DECOUPLES it from
severity (peaks recur at the same severity but later days; day~score collinearity
rho~0.14) and it is strain-clean (all SJL/PLP). Univariately, 844 genes track day
beyond severity (scripts/duration_axis.py). This builds the multivariate version: a
supervised clock that predicts time-since-induction.

This is the legitimate "temporal ML" for this dataset. There are NO per-animal
sequences (each mouse is one terminal timepoint), so sequence models (LSTM/Mamba/
Neural-ODE) do not apply. Instead we regress onto a de-confounded temporal LABEL.

Cohort: RR only (chronic's day aliases run_date/batch -> confounded).

Two clocks, leave-one-animal-out (LOAO):
  A. RAW clock      : day ~ genes
  B. SEVERITY-ORTHOGONALIZED clock (headline): residualize genes AND day on
     score_sacrifice within each train fold, predict day-residual from gene-residual.
     -> isolates duration-beyond-severity, cannot cheat via severity.

Controls / adversarial checks:
  - severity-only baseline: predict day from score alone (how much is just severity?)
  - cross-target: same pipeline predicting SCORE instead of day (axes separable?)
  - permutation null: shuffle day across animals, refit clock B, p = P(perm >= obs)
  - batch leakage: is day confounded with run_date / slide within RR?

Outputs: runs/duration_clock/{results.json, report.txt, clock_genes.csv, figures/}.
First successful run caches per-animal metadata to runs/duration_clock/rr_meta.csv so
later runs do not need the external volume.

    PYTHONPATH="$PWD" python scripts/duration_clock.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kruskal
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.preprocessing import StandardScaler

# OpenMP guard (see immunoformer-segfault-blocker): import the package first.
from immunotransformer.train import resolve_device  # noqa: F401

RRPB = "runs/rr_within_relapse/pseudobulk.npz"
RRMAP2 = ("/Volumes/moldiassd/RRMAP2_xenium_adata/kmeans_separated/"
          "RRMAP2_xenium_all_samples.cellcharter.companion.ready.with_metadata.rerun.h5ad")
OUT = "runs/duration_clock"
META_CACHE = os.path.join(OUT, "rr_meta.csv")
N_PERM = 50
# Single l1_ratio (vs the [.1,.5,.9] grid) -> 3x fewer inner fits; the clock is robust
# to the exact mix and the LOAO+perm validation is what matters. n_alphas=50 halves the
# path length. n_jobs=None: for these tiny per-fold fits joblib spawn overhead dominates.
L1_RATIOS = 0.5
ENET_KW = dict(l1_ratio=L1_RATIOS, cv=5, n_alphas=50, max_iter=20000, n_jobs=None)
# Per-fold unsupervised top-variance prefilter: 5101 genes -> TOP_VAR. Computed on the
# train fold only (after residualization), so no leakage; cuts the elastic-net cost.
TOP_VAR = 1000


def _topvar(Xtr, k=TOP_VAR):
    """Indices of the top-k highest-variance columns of the training matrix."""
    if Xtr.shape[1] <= k:
        return np.arange(Xtr.shape[1])
    v = Xtr.var(axis=0)
    return np.sort(np.argsort(-v)[:k])


# --------------------------------------------------------------------------- IO
def load_meta(animals):
    """Per-animal day_of_sacrifice / score / batch. Cache to avoid re-reading h5ad."""
    if os.path.exists(META_CACHE):
        m = pd.read_csv(META_CACHE).set_index("sample_name")
        return m.reindex(pd.Index(animals.astype(str)))
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    try:
        import anndata as ad
    except ImportError as e:
        raise SystemExit(f"anndata needed to build metadata cache: {e}")
    try:
        a = ad.read_h5ad(RRMAP2, backed="r")
    except (OSError, PermissionError) as e:
        raise SystemExit(
            "Cannot read the RRMAP2 h5ad and no cached runs/duration_clock/rr_meta.csv "
            f"exists.\n  {e}\nLikely macOS Full Disk Access (TCC) is blocking "
            "/Volumes/moldiassd. Grant Full Disk Access to the host app and retry, or "
            "run this once in a terminal that already has access.")
    o = a.obs
    cols = ["day_of_sacrifice", "score_sacrifice", "sample_id", "run_date", "region"]
    cols = [c for c in cols if c in o.columns]
    m = (o[o["sample_name"].astype(str).isin(set(animals.astype(str)))]
         .drop_duplicates("sample_name").set_index("sample_name")[cols])
    os.makedirs(OUT, exist_ok=True)
    m.to_csv(META_CACHE)
    return m.reindex(pd.Index(animals.astype(str)))


def load():
    d = np.load(RRPB, allow_pickle=True)
    pb = d["pb"].astype(float)
    genes = d["genes"].astype(str)
    animals = d["animals"].astype(str)
    stage = d["stage"].astype(str)
    score = d["score"].astype(float)
    meta = load_meta(animals)
    day = meta["day_of_sacrifice"].to_numpy(float)
    keep = ~np.isnan(day) & ~np.isnan(score)
    return (pb[keep], genes, animals[keep], stage[keep], score[keep],
            day[keep], meta.loc[meta.index[keep]] if False else meta.iloc[np.where(keep)[0]])


# ----------------------------------------------------------------- LOAO clocks
def loao_clock(X, y, covar=None, seed=0):
    """Leave-one-animal-out predictions. If covar given, residualize X (2D) and y (1D)
    on it within each train fold (severity-orthogonalized clock). Target stays 1D so
    LinearRegression.predict returns a 1D array and no broadcasting occurs."""
    n = len(y)
    pred = np.full(n, np.nan)
    for i in range(n):
        tr = np.ones(n, bool); tr[i] = False
        Xtr, Xte = X[tr], X[i:i + 1]
        ytr = y[tr]
        if covar is not None:
            ctr, cte = covar[tr], covar[i:i + 1]
            fx = LinearRegression().fit(ctr, Xtr)        # X: 2D -> 2D residual
            fy = LinearRegression().fit(ctr, ytr)        # y: 1D -> 1D residual
            Xtr = Xtr - fx.predict(ctr)
            Xte = Xte - fx.predict(cte)
            ytr = ytr - fy.predict(ctr)
        sel = _topvar(Xtr)                               # train-only variance prefilter
        Xtr, Xte = Xtr[:, sel], Xte[:, sel]
        sc = StandardScaler().fit(Xtr)
        model = ElasticNetCV(random_state=seed, **ENET_KW)
        model.fit(sc.transform(Xtr), ytr)
        pred[i] = model.predict(sc.transform(Xte))[0]
    return pred


def loao_target(y, covar):
    """For clock B the target is residualized per-fold, so 'true' differs per fold.
    Recompute the full-data residual target for a stable evaluation reference."""
    f = LinearRegression().fit(covar, y)
    return y - f.predict(covar).ravel()


def final_coefs(X, y, genes, covar=None, seed=0):
    """Full-data fit to report the clock-gene signature (nonzero elastic-net coefs)."""
    if covar is not None:
        fx = LinearRegression().fit(covar, X)
        ft = LinearRegression().fit(covar, y)
        X = X - fx.predict(covar)
        y = y - ft.predict(covar).ravel()
    sel = _topvar(X)                                     # match LOAO feature space
    Xs, gsel = X[:, sel], genes[sel]
    sc = StandardScaler().fit(Xs)
    model = ElasticNetCV(random_state=seed, **ENET_KW).fit(sc.transform(Xs), y)
    c = model.coef_
    nz = np.where(c != 0)[0]
    order = nz[np.argsort(-np.abs(c[nz]))]
    return pd.DataFrame({"gene": gsel[order], "coef": c[order]}), float(model.alpha_)


# ------------------------------------------------------------------------ main
def main():
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    pb, genes, animals, stage, score, day, meta = load()
    n = len(day)
    print(f"[load] RR n={n} animals, {pb.shape[1]} genes, "
          f"day {day.min():.0f}-{day.max():.0f}, score {score.min():.2f}-{score.max():.2f}")
    sc_col = score.reshape(-1, 1)

    # --- clock A: raw day ~ genes ---
    predA = loao_clock(pb, day)
    rhoA = spearmanr(predA, day).statistic
    r2A = 1 - np.sum((predA - day) ** 2) / np.sum((day - day.mean()) ** 2)
    print(f"[clockA] raw day~genes  Spearman={rhoA:+.3f}", flush=True)

    # --- clock B: severity-orthogonalized (headline) ---
    predB = loao_clock(pb, day, covar=sc_col)
    dayB = loao_target(day, sc_col)               # full-data residual reference
    rhoB = spearmanr(predB, dayB).statistic
    r2B = 1 - np.sum((predB - dayB) ** 2) / np.sum((dayB - dayB.mean()) ** 2)
    print(f"[clockB] severity-orthogonalized  Spearman={rhoB:+.3f}", flush=True)

    # --- severity-only baseline: predict day from score alone ---
    predS = loao_clock(sc_col, day)
    rhoS = spearmanr(predS, day).statistic
    print(f"[baseline] day~score  Spearman={rhoS:+.3f}", flush=True)

    # --- cross-target control: same orthogonalized pipeline predicting SCORE ---
    # (residualize genes & score on DAY; can the clock features read severity?)
    predX = loao_clock(pb, score, covar=day.reshape(-1, 1))
    scoreX = loao_target(score, day.reshape(-1, 1))
    rhoX = spearmanr(predX, scoreX).statistic
    print(f"[cross] clock-feats->score  Spearman={rhoX:+.3f}", flush=True)

    # --- permutation null on clock B ---
    print(f"[perm] running {N_PERM} permutations of clock B ...", flush=True)
    rng = np.random.default_rng(0)
    perm_rho = np.empty(N_PERM)
    for k in range(N_PERM):
        yp = day[rng.permutation(n)]
        pk = loao_clock(pb, yp, covar=sc_col)
        perm_rho[k] = spearmanr(pk, loao_target(yp, sc_col)).statistic
        if (k + 1) % 10 == 0:
            print(f"[perm] {k + 1}/{N_PERM}  (running p~{(1 + np.sum(perm_rho[:k+1] >= rhoB)) / (2 + k):.3f})",
                  flush=True)
    p_perm = (1 + np.sum(perm_rho >= rhoB)) / (1 + N_PERM)

    # --- batch leakage: is day confounded with run_date / slide within RR? ---
    batch = {}
    for col in ["run_date", "sample_id", "region"]:
        if col in meta.columns:
            vals = meta[col].astype(str).to_numpy()
            groups = [day[vals == u] for u in np.unique(vals) if (vals == u).sum() >= 2]
            if len(groups) >= 2:
                st = kruskal(*groups)
                batch[col] = {"n_levels": int(len(np.unique(vals))),
                              "kruskal_p_day": round(float(st.pvalue), 4)}

    # --- clock-gene signature ---
    coefs, alphaB = final_coefs(pb, day, genes, covar=sc_col)
    coefs.to_csv(os.path.join(OUT, "clock_genes.csv"), index=False)

    # --- fit-array cache for downstream figures / HTML report (no recompute needed) ---
    np.savez(os.path.join(OUT, "fit_cache.npz"),
             predA=predA, predB=predB, dayB=dayB, day=day, score=score,
             predS=predS, animals=animals.astype(str),
             region=meta["region"].astype(str).to_numpy(),
             perm_rho=perm_rho, rhoA=rhoA, rhoB=rhoB, rhoS=rhoS, p_perm=p_perm)

    results = {
        "n_animals": int(n),
        "day_range": [float(day.min()), float(day.max())],
        "clock_A_raw": {"spearman": round(float(rhoA), 3), "r2_loao": round(float(r2A), 3)},
        "clock_B_severity_orthogonalized": {
            "spearman": round(float(rhoB), 3), "r2_loao": round(float(r2B), 3),
            "perm_p": round(float(p_perm), 4), "perm_rho_mean": round(float(perm_rho.mean()), 3),
            "n_clock_genes": int(len(coefs)), "alpha": round(alphaB, 5)},
        "severity_only_baseline_spearman": round(float(rhoS), 3),
        "cross_target_predict_score_spearman": round(float(rhoX), 3),
        "batch_leakage_day": batch,
        "top_clock_genes": coefs.head(25).round(4).to_dict("records"),
        "interpretation": (
            "clock_B is the de-confounded result (duration beyond severity). Compare to "
            "severity_only_baseline (day from score alone) and cross_target (can clock "
            "features read severity?). Low batch_leakage kruskal_p would flag day~batch "
            "confounding within RR."),
        "caveats": ("RR only, strain-clean. n is small (~33 animals); LOAO + permutation "
                    "null guard against overfit. Pseudobulk (animal-level); spatial Tier-2 "
                    "MIL is a separate build. CHRONIC excluded (day aliases run_date)."),
    }
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    # ---- report ----
    L = ["\n=== DURATION CLOCK — molecular clock of accumulated EAE damage (RR) ==="]
    L.append(f"n={n} animals, day {day.min():.0f}-{day.max():.0f} post-induction\n")
    L.append(f"  Clock A (raw, day~genes)            : Spearman={rhoA:+.3f}  R2(LOAO)={r2A:+.3f}")
    L.append(f"  Clock B (severity-orthogonalized)   : Spearman={rhoB:+.3f}  R2(LOAO)={r2B:+.3f}"
             f"  perm p={p_perm:.4f}  ({len(coefs)} clock genes)")
    L.append(f"  -- controls --")
    L.append(f"  severity-only baseline (day~score)  : Spearman={rhoS:+.3f}")
    L.append(f"  cross-target (clock feats -> score) : Spearman={rhoX:+.3f}")
    if batch:
        L.append(f"  batch leakage (day ~ batch within RR):")
        for c, v in batch.items():
            L.append(f"     {c}: {v['n_levels']} levels, Kruskal p(day)={v['kruskal_p_day']}")
    L.append(f"\n  top clock genes (severity-orthogonalized coefs):")
    L.append("    " + ", ".join(f"{r['gene']}({r['coef']:+.2f})" for r in results["top_clock_genes"][:15]))
    report = "\n".join(L)
    print(report)
    with open(os.path.join(OUT, "report.txt"), "w") as fh:
        fh.write(report + "\n")

    _plot(predB, dayB, coefs, OUT)
    print(f"\n[done] -> {OUT}/ (results.json, report.txt, clock_genes.csv, figures/)")


def _plot(predB, dayB, coefs, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))
    axA.scatter(dayB, predB, s=70, edgecolor="k", linewidth=.4, color="#4c72b0")
    lim = [min(dayB.min(), predB.min()), max(dayB.max(), predB.max())]
    axA.plot(lim, lim, "--", color="grey", lw=1)
    rho = spearmanr(predB, dayB).statistic
    axA.set_xlabel("true duration residual (day | severity)")
    axA.set_ylabel("predicted (LOAO)")
    axA.set_title(f"Severity-orthogonalized duration clock\nLOAO Spearman={rho:+.2f}", fontsize=11)
    axA.spines[["top", "right"]].set_visible(False)
    top = coefs.head(15).iloc[::-1]
    colors = ["#c44e52" if c > 0 else "#4c72b0" for c in top["coef"]]
    axB.barh(top["gene"], top["coef"], color=colors)
    axB.axvline(0, color="k", lw=.6)
    axB.set_xlabel("elastic-net coefficient (duration | severity)")
    axB.set_title("Top clock genes (red=accrues, blue=declines)", fontsize=11)
    axB.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figures", "duration_clock.png"), dpi=130, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "figures", "duration_clock.pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
