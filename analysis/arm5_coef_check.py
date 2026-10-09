"""Arm 5 diagnostic: did any niche-composition feature get a non-zero coefficient in any fold?

Arm 5 (pseudobulk + Global_niche fractions) matched arm 2 to two decimals. Two ways that can
happen: (a) the fractions are removed before fitting (top-variance prefilter on mixed
scales), or (b) they reach the elastic net and receive zero weight. baseline_ladder.loao_keep
applies the prefilter to genes only and always passes the 27 fractions; this re-runs exactly
that fold loop and records, per fold, which fractions entered the fit and their coefficients.

    PYTHONPATH="$PWD:$PWD/scripts" python analysis/arm5_coef_check.py
"""

from __future__ import annotations

import immunotransformer  # noqa: F401  (OpenMP guard — must precede numpy; see HANDOFF.md)

import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baseline_ladder as B  # noqa: E402
from duration_clock import ENET_KW, _topvar  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
OUT = os.path.join(B.OUT, "arm5_coefficient_check")


def main():
    pb, genes, fracs, info = B.load_all()
    niche, niche_names = fracs["arm3"]
    rows, preds = [], []
    for key, col, model, covs, _ in B.TARGETS:
        idx = B.cohort(info, col, model, key)
        sub = info.iloc[idx].reset_index(drop=True)
        y = sub[col].to_numpy(float)
        covar = sub[covs].to_numpy(float) if covs else None
        Xg, Xk = pb[idx], niche[idx]
        X = np.hstack([Xg, Xk])
        ng = Xg.shape[1]
        for i in range(len(y)):
            tr = np.ones(len(y), bool); tr[i] = False
            Xtr, ytr = X[tr], y[tr]
            if covar is not None:
                Xtr = Xtr - LinearRegression().fit(covar[tr], Xtr).predict(covar[tr])
                ytr = ytr - LinearRegression().fit(covar[tr], ytr).predict(covar[tr])
            sel = np.r_[_topvar(Xtr[:, :ng]), np.arange(ng, X.shape[1])]
            sc = StandardScaler().fit(Xtr[:, sel])
            m = ElasticNetCV(random_state=0, **ENET_KW).fit(sc.transform(Xtr[:, sel]), ytr)
            coef = m.coef_
            niche_coef = coef[-Xk.shape[1]:]
            gene_nz = int((coef[:-Xk.shape[1]] != 0).sum())
            rows.append({"target": key, "fold_held_out": sub.sample_name[i],
                         "n_features_fit": len(sel), "niche_features_fit": Xk.shape[1],
                         "n_gene_nonzero": gene_nz,
                         "n_niche_nonzero": int((niche_coef != 0).sum()),
                         "niche_nonzero": ";".join(f"{niche_names[j]}={niche_coef[j]:+.4f}"
                                                   for j in np.where(niche_coef != 0)[0]),
                         "max_abs_niche_coef": float(np.abs(niche_coef).max()),
                         "max_abs_gene_coef": float(np.abs(coef[:-Xk.shape[1]]).max()),
                         "alpha": float(m.alpha_)})
        print(f"[{key}] done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT + ".csv", index=False)

    pr = pd.read_csv(os.path.join(B.OUT, "loao_predictions_arms1-5.csv"))
    L = ["ARM 5 COEFFICIENT CHECK — do niche fractions enter and receive weight?", "=" * 66,
         "Prefilter: top-1000-variance on genes only; all 27 Global_niche fractions are passed",
         "to the scaler and elastic net in every fold (n_features_fit = 1000 + 27).", "",
         f"  {'target':22s} {'folds':>5s} {'folds w/ niche coef!=0':>23s} "
         f"{'median #niche!=0':>16s} {'median #genes!=0':>16s} {'max|arm5-arm2| pred':>20s}"]
    for key, g in df.groupby("target", sort=False):
        p = pr[pr.target == key].pivot(index="sample_name", columns="arm", values="pred")
        diff = float((p["arm5"] - p["arm2"]).abs().max())
        L.append(f"  {key:22s} {len(g):5d} {int((g.n_niche_nonzero > 0).sum()):23d} "
                 f"{g.n_niche_nonzero.median():16.0f} {g.n_gene_nonzero.median():16.0f} "
                 f"{diff:20.4f}")
    nz = df[df.n_niche_nonzero > 0]
    top = (nz.niche_nonzero.str.split(";").explode().str.split("=").str[0]
           .value_counts().head(8))
    L += ["", "Niche fractions most often selected (fold count): " +
          ", ".join(f"{k} {v}" for k, v in top.items()),
          "", "Per-fold detail: arm5_coefficient_check.csv"]
    report = "\n".join(L)
    print(report)
    with open(OUT + ".txt", "w") as fh:
        fh.write(report + "\n")


if __name__ == "__main__":
    main()
