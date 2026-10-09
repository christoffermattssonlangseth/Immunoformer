"""Task 3 — tuned attention-MIL (arm 6t): the fairness fix.

Arm 2 (ElasticNetCV) tunes its regularisation inside every fold; arm 6 ran on fixed
hyperparameters, so every arm-6 loss was confounded with "not tuned". Here arm 6 gets an
inner hyperparameter loop, selected on the outer fold's TRAINING animals only.

Outer folds: identical to the ladder (LOAO over the target's animals, RR cohort).
Inner loop (per outer fold, per target):
  * inner validation split = INNER_VAL_FRAC of the outer-training animals (seeded),
  * encoder (HVG + PCA), residualisation on covariates and z-scaling all fit on the
    inner-TRAINING animals only,
  * grid: lr x hidden dim (attn dim = hidden/2) x dropout; each config trained up to
    MAX_EPOCHS with early stopping (patience PATIENCE) on inner-validation MSE,
  * selection: lowest inner-validation MSE; its best epoch is kept.
Final: encoder + residualisation on all outer-training animals, chosen config trained for
the chosen number of epochs, SEEDS seeds averaged, held-out animal predicted.
Sharded across processes/machines: SHARD=i NSHARDS=n. SUMMARY_ONLY=1 merges shards.

    SHARD=0 NSHARDS=1 PYTHONPATH="$PWD:$PWD/scripts" python analysis/mil_tuned.py
"""

from __future__ import annotations

import immunotransformer  # noqa: F401  (OpenMP guard — must precede numpy; see HANDOFF.md)

import glob
import itertools
import json
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mil_loao as M  # noqa: E402
from baseline_ladder import OUT as LADDER, TARGETS, cohort, load_all  # noqa: E402
from duration_clock import loao_target  # noqa: E402
from immunotransformer.model import GatedAttentionMIL  # noqa: E402
from immunotransformer.stats import bootstrap_spearman, compare_rhos  # noqa: E402

OUT = "runs/arm6_tuned"
TUNE_TARGETS = ["day_of_sacrifice", "days_since_last_peak"]
GRID = {"lr": [3e-4, 1e-3, 3e-3], "hidden": [64, 128, 256], "dropout": [0.1, 0.3]}
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", "60"))
PATIENCE = 10
INNER_VAL_FRAC = 0.2
SEEDS = [0, 1, 2]
SHARD = int(os.environ.get("SHARD", "0"))
NSHARDS = int(os.environ.get("NSHARDS", "1"))
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))


def make_model(cfg):
    return GatedAttentionMIL(in_dim=M.PCA_DIM, num_classes=2, proj_dim=cfg["hidden"],
                             attn_dim=cfg["hidden"] // 2, dropout=cfg["dropout"],
                             head="regression")


def _predict(model, E, bags):
    model.eval()
    with torch.no_grad():
        return np.array([model(torch.from_numpy(E[r[:M.MAX_CELLS_PER_BAG]]))[0].item()
                         for r in bags])


def fit(E, bags_tr, y_tr, w_tr, cfg, epochs, seed, val=None):
    """Train; with val=(bags, owners, y_by_owner) track inner-val MSE and early-stop."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=M.WD)
    best = (np.inf, 0)
    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        for step, i in enumerate(rng.permutation(len(bags_tr)), 1):
            rows = bags_tr[i]
            if len(rows) > M.MAX_CELLS_PER_BAG:
                rows = rng.choice(rows, M.MAX_CELLS_PER_BAG, replace=False)
            out, _ = model(torch.from_numpy(E[rows]))
            (w_tr[i] * (out.squeeze() - float(y_tr[i])) ** 2 / M.GRAD_ACCUM).backward()
            if step % M.GRAD_ACCUM == 0:
                opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
        if val is not None:
            vb, vo, vy = val
            p = pd.Series(_predict(model, E, vb)).groupby(np.asarray(vo)).mean()
            mse = float(np.mean([(p[a] - vy[a]) ** 2 for a in p.index]))
            if mse < best[0]:
                best = (mse, ep)
            elif ep - best[1] >= PATIENCE:
                break
    return model, best


def bags_for(animals, z, sec2animal, section):
    bags, owners, ys, ws = [], [], [], []
    for a in animals:
        secs = [s for s, an in sec2animal.items() if an == a]
        for s in secs:
            bags.append(np.where(section == s)[0]); owners.append(a)
            ys.append(z[a]); ws.append(1.0 / len(secs))
    return bags, owners, np.array(ys), np.array(ws)


def residual_z(sub, col, covs, train_mask):
    y = sub[col].to_numpy(float)
    cov = sub[covs].to_numpy(float)
    resid = y - LinearRegression().fit(cov[train_mask], y[train_mask]).predict(cov)
    mu, sd = resid[train_mask].mean(), resid[train_mask].std()
    return dict(zip(sub.sample_name, (resid - mu) / sd)), mu, sd


def main():
    os.makedirs(OUT, exist_ok=True)
    pb, _, _, info = load_all()
    d = np.load("runs/rr_within_relapse/pseudobulk_all67.npz", allow_pickle=True)
    smeta = pd.DataFrame(d["section_meta"].tolist(), columns=d["section_meta_cols"])
    smeta = smeta[~smeta.meta_sample_id.isin(M.EXCLUDE_SECTIONS)]
    X, section, _ = M.load_cells(set(smeta.meta_sample_id))
    Xn = M.lognorm(X)
    sec2animal = dict(zip(smeta.meta_sample_id, smeta.sample_name))
    cell_animal = np.array([sec2animal.get(s, "") for s in section])
    specs = [t for t in TARGETS if t[0] in TUNE_TARGETS]
    cohort_animals = info[info.model == specs[0][2]].sample_name.to_numpy()
    cmask = np.isin(cell_animal, cohort_animals)
    configs = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    path = os.path.join(OUT, f"shard_{SHARD}.csv")
    rows = pd.read_csv(path).to_dict("records") if os.path.exists(path) else []
    done = {(r["target"], r["held_out"]) for r in rows}

    def encode_on(animals):
        tr_rows = np.where(cmask & np.isin(cell_animal, list(animals)))[0]
        hvg, pca = M.fit_encoder(Xn, tr_rows)
        E = np.zeros((Xn.shape[0], M.PCA_DIM), np.float32)
        E[cmask] = M.encode(Xn[cmask], hvg, pca)
        return E

    for k, held in enumerate(cohort_animals):
        if k % NSHARDS != SHARD:
            continue
        todo = [t for t in specs if (t[0], held) not in done and held in set(
            info.sample_name.iloc[cohort(info, t[1], t[2], t[0])])]
        if not todo:
            continue
        t0 = time.time()
        E_outer = encode_on([a for a in cohort_animals if a != held])
        for key, col, model_name, covs, _ in todo:
            sub = info.iloc[cohort(info, col, model_name, key)].reset_index(drop=True)
            outer_tr = sub.sample_name[sub.sample_name != held].to_numpy()
            rng = np.random.default_rng(zlib.crc32(f"{key}|{held}".encode()))
            inner_val = set(rng.choice(outer_tr, max(3, int(round(INNER_VAL_FRAC * len(outer_tr)))),
                                       replace=False))
            inner_tr = [a for a in outer_tr if a not in inner_val]
            # ---- inner loop: everything fit on inner-train animals only
            E_in = encode_on(inner_tr)
            z_in, _, _ = residual_z(sub, col, covs, sub.sample_name.isin(inner_tr).to_numpy())
            btr, _, ytr, wtr = bags_for(inner_tr, z_in, sec2animal, section)
            bv, ov, _, _ = bags_for(list(inner_val), z_in, sec2animal, section)
            scores = []
            for cfg in configs:
                _, (mse, ep) = fit(E_in, btr, ytr, wtr, cfg, MAX_EPOCHS, seed=0,
                                   val=(bv, ov, z_in))
                scores.append({**cfg, "val_mse": mse, "best_epoch": ep})
            sc = pd.DataFrame(scores).sort_values("val_mse")
            best = sc.iloc[0]
            cfg = {"lr": float(best.lr), "hidden": int(best.hidden), "dropout": float(best.dropout)}
            # ---- final: all outer-training animals
            z, mu, sd = residual_z(sub, col, covs, (sub.sample_name != held).to_numpy())
            b, _, yv, wv = bags_for(outer_tr, z, sec2animal, section)
            bte = [np.where(section == s)[0] for s, an in sec2animal.items() if an == held]
            seed_preds = []
            for s in SEEDS:
                m, _ = fit(E_outer, b, yv, wv, cfg, int(best.best_epoch), seed=s)
                seed_preds.append(float(_predict(m, E_outer, bte).mean()) * sd + mu)
            rows.append({"target": key, "held_out": held, **cfg,
                         "epochs": int(best.best_epoch), "inner_val_mse": float(best.val_mse),
                         "inner_val_mse_default": float(sc[(sc.lr == 1e-3) & (sc.hidden == 128)
                                                           & (sc.dropout == 0.1)].val_mse.iloc[0]),
                         "pred": float(np.mean(seed_preds)),
                         "seed_preds": ";".join(f"{p:.4f}" for p in seed_preds)})
            pd.DataFrame(rows).to_csv(path, index=False)
        print(f"  [shard {SHARD}] held out {held}: {len(todo)} targets, {time.time() - t0:.0f}s",
              flush=True)


def summarise():
    df = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(OUT, "shard_*.csv"))])
    _, _, _, info = load_all()
    lad = json.load(open(os.path.join(LADDER, "results.json")))["targets"]
    a6 = pd.read_csv(os.path.join(LADDER, "arm6_predictions.csv"))
    L = ["TASK 3 — TUNED ATTENTION-MIL (arm 6t) vs untuned arm 6 and arm 2", "=" * 66,
         f"Inner loop on outer-training animals only: {INNER_VAL_FRAC:.0%} inner-val split; grid "
         f"{GRID}; early stopping (max {MAX_EPOCHS}, patience {PATIENCE}); final refit "
         f"{len(SEEDS)} seeds. Outer LOAO folds identical to the ladder.",
         "Zero-feature folds: not applicable to arm 6/6t (no feature selection); arm 2 counts "
         "from the ladder.", ""]
    res = {}
    for key in TUNE_TARGETS:
        spec = [t for t in TARGETS if t[0] == key][0]
        sub = info.iloc[cohort(info, spec[1], spec[2], key)]
        y = sub[spec[1]].to_numpy(float)
        ref = loao_target(y, sub[spec[3]].to_numpy(float))
        g = df[df.target == key].set_index("held_out")
        if not set(sub.sample_name) <= set(g.index):
            L.append(f"{key}: incomplete ({len(g)}/{len(sub)} folds)"); continue
        p6t = g.loc[sub.sample_name, "pred"].to_numpy(float)
        b = bootstrap_spearman(p6t, ref, n_boot=2000, seed=0)
        p6 = a6[a6.target == key].set_index("sample_name").loc[sub.sample_name, "pred"].to_numpy(float)
        b6 = bootstrap_spearman(p6, ref, n_boot=2000, seed=0)
        a2 = lad[key]["arms"]["arm2"]
        n = len(sub)
        r = {"arm6t": [float(b["rho"]), float(b["ci_lo"]), float(b["ci_hi"])],
             "arm6": [float(b6["rho"]), float(b6["ci_lo"]), float(b6["ci_hi"])],
             "arm2": [a2["rho"], *a2["ci"]],
             "p_6t_vs_2": compare_rhos(b["rho"], n, a2["rho"], n)["p_value"],
             "p_6t_vs_6": compare_rhos(b["rho"], n, b6["rho"], n)["p_value"],
             "chosen": g[["lr", "hidden", "dropout", "epochs"]].astype(str).agg("/".join, axis=1)
                        .value_counts().head(5).to_dict(),
             "inner_gain_vs_default": float((g.inner_val_mse_default - g.inner_val_mse).median())}
        res[key] = r
        L += [f"TARGET {key} (RR, n = {n})",
              f"  {'arm':28s} {'rho':>7s} {'95% CI':>17s} {'width':>6s}",
              f"  {'2  pseudobulk (tuned EN)':28s} {r['arm2'][0]:+7.3f} [{r['arm2'][1]:+.2f}, {r['arm2'][2]:+.2f}] {r['arm2'][2]-r['arm2'][1]:6.2f}   zero-feature folds {a2.get('n_null_folds', 0)}/{n}",
              f"  {'6  attention-MIL (untuned)':28s} {r['arm6'][0]:+7.3f} [{r['arm6'][1]:+.2f}, {r['arm6'][2]:+.2f}] {r['arm6'][2]-r['arm6'][1]:6.2f}",
              f"  {'6t attention-MIL (tuned)':28s} {r['arm6t'][0]:+7.3f} [{r['arm6t'][1]:+.2f}, {r['arm6t'][2]:+.2f}] {r['arm6t'][2]-r['arm6t'][1]:6.2f}",
              f"  Fisher z p: 6t vs 2 = {r['p_6t_vs_2']:.3f}; 6t vs 6 = {r['p_6t_vs_6']:.3f}",
              f"  chosen lr/hidden/dropout/epochs (top 5 by folds): {r['chosen']}",
              f"  median inner-val MSE gain of chosen config over the untuned default: "
              f"{r['inner_gain_vs_default']:+.3f}", ""]
    report = "\n".join(L)
    print(report)
    with open(os.path.join(OUT, "report.txt"), "w") as fh:
        fh.write(report + "\n")
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(res, fh, indent=2)


if __name__ == "__main__":
    summarise() if os.environ.get("SUMMARY_ONLY") else main()
