"""Learning curve on animals: data-scale limit or architecture limit? (EXPLORATORY,
follow-up to docs/negative-result-scaling.md; not part of the WORKORDER)

Arm 2 (pseudobulk elastic net) and arm 6 (attention-MIL, regression head) refit with only a
fraction of each fold's training animals. Same LOAO folds, same held-out animal, same
in-fold residualisation, encoder fit on the subsampled training animals only. If arm 6
gains faster than arm 2 as animals are added, the limit at n ~ 30 is data scale rather than
architecture. Three random subsamples per fraction (one arm 6 seed each); fraction 1.0 is
read from the existing ladder results (arm 2 deterministic; arm 6 3-seed ensemble).

    PYTHONPATH="$PWD:$PWD/scripts" python analysis/learning_curve.py
"""

from __future__ import annotations

import immunotransformer  # noqa: F401  (OpenMP guard — must precede numpy; see HANDOFF.md)

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mil_loao as M  # noqa: E402
from baseline_ladder import OUT, TARGETS, cohort, load_all  # noqa: E402
from duration_clock import ENET_KW, _topvar, loao_target  # noqa: E402
from immunotransformer.stats import bootstrap_spearman  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
LC_TARGETS = ["day_of_sacrifice", "days_since_last_peak"]
FRACTIONS = [0.5, 0.75]
N_DRAWS = 3
STEM = os.path.join(OUT, "learning_curve")


def arm2_predict(X, y, covar, tr, te):
    Xtr, Xte, ytr = X[tr], X[te], y[tr]
    if covar is not None:
        fx = LinearRegression().fit(covar[tr], Xtr)
        fy = LinearRegression().fit(covar[tr], ytr)
        Xtr, Xte = Xtr - fx.predict(covar[tr]), Xte - fx.predict(covar[te])
        ytr = ytr - fy.predict(covar[tr])
    sel = _topvar(Xtr)
    sc = StandardScaler().fit(Xtr[:, sel])
    m = ElasticNetCV(random_state=0, **{**ENET_KW, "cv": min(5, int(tr.sum()) - 1)})
    return float(m.fit(sc.transform(Xtr[:, sel]), ytr).predict(sc.transform(Xte[:, sel]))[0])


def main():
    pb, _, _, info = load_all()
    d = np.load("runs/rr_within_relapse/pseudobulk_all67.npz", allow_pickle=True)
    smeta = pd.DataFrame(d["section_meta"].tolist(), columns=d["section_meta_cols"])
    smeta = smeta[~smeta.meta_sample_id.isin(M.EXCLUDE_SECTIONS)]
    Xc, section, _ = M.load_cells(set(smeta.meta_sample_id))
    Xn = M.lognorm(Xc)
    sec2animal = dict(zip(smeta.meta_sample_id, smeta.sample_name))
    cell_animal = np.array([sec2animal.get(s, "") for s in section])
    specs = [t for t in TARGETS if t[0] in LC_TARGETS]
    model_name = specs[0][2]
    cohort_animals = info[info.model == model_name].sample_name.to_numpy()
    cmask = np.isin(cell_animal, cohort_animals)

    rows = []
    if os.path.exists(STEM + ".csv"):
        rows = pd.read_csv(STEM + ".csv").to_dict("records")
    done = {(r["target"], r["fraction"], r["draw"], r["held_out"]) for r in rows}
    for frac in FRACTIONS:
        for draw in range(N_DRAWS):
            rng = np.random.default_rng(1000 * draw + int(frac * 100))
            for held in cohort_animals:
                todo = [t for t in specs if held in set(info.sample_name.iloc[
                    cohort(info, t[1], t[2], t[0])]) and (t[0], frac, draw, held) not in done]
                if not todo:
                    continue
                t0 = time.time()
                others = np.array([a for a in cohort_animals if a != held])
                keep = set(rng.choice(others, int(round(frac * len(others))), replace=False))
                train_rows = np.where(cmask & np.isin(cell_animal, list(keep)))[0]
                hvg, pca = M.fit_encoder(Xn, train_rows)
                E = np.zeros((Xn.shape[0], M.PCA_DIM), np.float32)
                E[cmask] = M.encode(Xn[cmask], hvg, pca)
                for key, col, _, covs, _ in todo:
                    idx = cohort(info, col, model_name, key)
                    sub = info.iloc[idx].reset_index(drop=True)
                    tr = sub.sample_name.isin(keep).to_numpy()
                    te = (sub.sample_name == held).to_numpy()
                    y = sub[col].to_numpy(float)
                    cov = sub[covs].to_numpy(float)
                    p2 = arm2_predict(pb[idx], y, cov, tr, te)
                    f = LinearRegression().fit(cov[tr], y[tr])
                    resid = y - f.predict(cov)
                    mu, sd = resid[tr].mean(), resid[tr].std()
                    bags_tr, y_tr, w_tr = [], [], []
                    for a in sub.sample_name[tr]:
                        secs = [s for s, an in sec2animal.items() if an == a]
                        for s in secs:
                            bags_tr.append(np.where(section == s)[0])
                            y_tr.append((resid[sub.sample_name == a][0] - mu) / sd)
                            w_tr.append(1.0 / len(secs))
                    bags_te = [np.where(section == s)[0] for s, an in sec2animal.items()
                               if an == held]
                    p6 = float(M.train_predict(E, bags_tr, np.array(y_tr), np.array(w_tr),
                                               bags_te, seed=draw).mean()) * sd + mu
                    rows.append({"target": key, "fraction": frac, "draw": draw,
                                 "held_out": held, "n_train": int(tr.sum()),
                                 "arm2": p2, "arm6": p6})
                pd.DataFrame(rows).to_csv(STEM + ".csv", index=False)
                print(f"  f={frac} draw={draw} held out {held}: {time.time() - t0:.0f}s",
                      flush=True)
    summarise(pd.DataFrame(rows), info)


def summarise(df, info):
    full = json.load(open(os.path.join(OUT, "results.json")))["targets"]
    L = ["LEARNING CURVE ON ANIMALS (exploratory) — data-scale or architecture limit?",
         "=" * 72,
         "Same LOAO folds; each fold's training animals subsampled to the fraction shown;",
         f"{N_DRAWS} random subsamples per fraction (rho per draw, then mean and range).",
         "Fraction 1.0 = the main ladder (arm 6 = 3-seed ensemble). rho vs the full-data",
         "residual target, as in the ladder.", ""]
    res = {}
    for key in LC_TARGETS:
        spec = [t for t in TARGETS if t[0] == key][0]
        sub = info.iloc[cohort(info, spec[1], spec[2], key)]
        ref = pd.Series(loao_target(sub[spec[1]].to_numpy(float), sub[spec[3]].to_numpy(float)),
                        index=sub.sample_name)
        L += [f"TARGET {key} (n = {len(sub)})",
              f"  {'fraction':>8s} {'~n train':>8s} {'arm 2 rho (range)':>22s} "
              f"{'arm 6 rho (range)':>22s}"]
        res[key] = []
        for frac, g in df[df.target == key].groupby("fraction"):
            r2 = [spearmanr(h.set_index("held_out").arm2.loc[ref.index], ref).statistic
                  for _, h in g.groupby("draw")]
            r6 = [spearmanr(h.set_index("held_out").arm6.loc[ref.index], ref).statistic
                  for _, h in g.groupby("draw")]
            res[key].append({"fraction": frac, "arm2": r2, "arm6": r6})
            L.append(f"  {frac:8.2f} {g.n_train.median():8.0f} {np.mean(r2):+8.3f} "
                     f"({min(r2):+.2f}, {max(r2):+.2f}) {np.mean(r6):+8.3f} "
                     f"({min(r6):+.2f}, {max(r6):+.2f})")
        f2 = full[key]["arms"]["arm2"]["rho"]
        f6 = full[key]["arms"].get("arm6", {}).get("rho", np.nan)
        L += [f"  {1.0:8.2f} {len(sub) - 1:8d} {f2:+8.3f} {'':14s} {f6:+8.3f}", ""]
        res[key].append({"fraction": 1.0, "arm2": [f2], "arm6": [f6]})
    ss = pd.read_csv(os.path.join(OUT, "arm6_seed_spread.csv")).set_index("target")
    L += ["COMPARABILITY: subsampled points use ONE arm 6 seed per draw; the 1.0 point above is",
          "the 3-seed ensemble. Single-seed mean at 1.0 (arm6_seed_spread.csv): " + ", ".join(
              f"{k} {ss.loc[k, ['seed0', 'seed1', 'seed2']].mean():+.3f}" for k in LC_TARGETS
              if k in ss.index) + ".", ""]
    L += ["READING: compare slopes. Arm 6 rising faster than arm 2 from 0.5 -> 1.0 suggests a",
          "data-scale limit; parallel or flatter curves suggest the architecture (as",
          "configured) adds little at this scale. 3 draws per point; treat as indicative."]
    report = "\n".join(L)
    print(report)
    with open(STEM + ".txt", "w") as fh:
        fh.write(report + "\n")
    with open(STEM + ".json", "w") as fh:
        json.dump(res, fh, indent=2)


if __name__ == "__main__":
    if os.environ.get("SUMMARY_ONLY"):
        summarise(pd.read_csv(STEM + ".csv"), load_all()[3])
    else:
        main()
