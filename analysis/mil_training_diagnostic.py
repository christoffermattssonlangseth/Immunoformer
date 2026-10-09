"""Is arm 6 undertrained on days_since_last_peak? (diagnostic for WORKORDER Task 4)

Arm 6 (attention-MIL, regression head, 30 fixed epochs) reaches rho 0.37 on
days_since_last_peak vs 0.70 for pseudobulk (arm 2). Before reading that gap, check whether
the head simply has not fit the target. Same LOAO folds, encoder and settings as
analysis/mil_loao.py; one seed; training extended to the last checkpoint, recording at
each CHECKPOINTS epoch:
  * train loss (weighted MSE on the z-scored residual target, epoch mean)        — leak-free
  * in-sample rho: animal-level predictions on the TRAINING animals vs their
    targets                                                                       — leak-free
  * the held-out animal's prediction                                       — test-informed
Underfit = training rho stays low / loss still falling at epoch 30. Fit-but-no-transfer =
training rho high while held-out rho stays flat. The held-out curve is reported as a
diagnostic only; it is NOT used to choose arm 6's epoch count (that would select on the
test animals). day_of_sacrifice is run alongside as the control target where arm 6 works.

    PYTHONPATH="$PWD:$PWD/scripts" python analysis/mil_training_diagnostic.py
"""

from __future__ import annotations

import immunotransformer  # noqa: F401  (OpenMP guard — must precede numpy; see HANDOFF.md)

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mil_loao as M  # noqa: E402
from baseline_ladder import OUT, TARGETS, cohort, load_all  # noqa: E402
from duration_clock import loao_target  # noqa: E402
from immunotransformer.model import GatedAttentionMIL  # noqa: E402

DIAG_TARGETS = os.environ.get("DIAG_TARGETS", "days_since_last_peak,day_of_sacrifice").split(",")
CHECKPOINTS = [10, 20, 30, 60, 90, 120]
SEED = 0
STEM = os.path.join(OUT, "arm6_training_diagnostic")


def _animal_preds(model, bags, owners):
    model.eval()
    with torch.no_grad():
        p = np.array([model(torch.from_numpy(E_[rows[:M.MAX_CELLS_PER_BAG]]))[0].item()
                      for rows in bags])
    model.train()
    return pd.Series(p).groupby(np.asarray(owners)).mean()


def train_with_checkpoints(bags_tr, owners_tr, y_tr, w_tr, z_animal, bags_te, z_held, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = GatedAttentionMIL(in_dim=M.PCA_DIM, num_classes=2, proj_dim=M.PROJ_DIM,
                              attn_dim=M.ATTN_DIM, dropout=M.DROPOUT, head="regression")
    opt = torch.optim.AdamW(model.parameters(), lr=M.LR, weight_decay=M.WD)
    out = []
    for epoch in range(1, max(CHECKPOINTS) + 1):
        model.train()
        opt.zero_grad()
        losses = []
        for step, i in enumerate(rng.permutation(len(bags_tr)), 1):
            rows = bags_tr[i]
            if len(rows) > M.MAX_CELLS_PER_BAG:
                rows = rng.choice(rows, M.MAX_CELLS_PER_BAG, replace=False)
            pred, _ = model(torch.from_numpy(E_[rows]))
            loss = w_tr[i] * (pred.squeeze() - float(y_tr[i])) ** 2
            losses.append(loss.item())
            (loss / M.GRAD_ACCUM).backward()
            if step % M.GRAD_ACCUM == 0:
                opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
        if epoch in CHECKPOINTS:
            tr = _animal_preds(model, bags_tr, owners_tr)
            te = _animal_preds(model, bags_te, ["held"] * len(bags_te))["held"]
            ztr = np.array([z_animal[a] for a in tr.index])
            out.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "train_loss_eval": float(np.mean((tr.values - ztr) ** 2)),
                        "heldout_loss": float((te - z_held) ** 2), "heldout_true_z": z_held,
                        "train_rho": float(spearmanr(tr.values,
                                                     [z_animal[a] for a in tr.index]).statistic),
                        "heldout_pred_z": float(te)})
    return out


def main():
    global E_
    pb, _, _, info = load_all()
    d = np.load("runs/rr_within_relapse/pseudobulk_all67.npz", allow_pickle=True)
    smeta = pd.DataFrame(d["section_meta"].tolist(), columns=d["section_meta_cols"])
    smeta = smeta[~smeta.meta_sample_id.isin(M.EXCLUDE_SECTIONS)]
    X, section, _ = M.load_cells(set(smeta.meta_sample_id))
    Xn = M.lognorm(X)
    sec2animal = dict(zip(smeta.meta_sample_id, smeta.sample_name))
    cell_animal = np.array([sec2animal.get(s, "") for s in section])
    specs = [t for t in TARGETS if t[0] in DIAG_TARGETS]
    model_name = specs[0][2]
    assert all(t[2] == model_name for t in specs), "diagnostic targets must share a cohort"
    cohort_animals = info[info.model == model_name].sample_name.to_numpy()
    cmask = np.isin(cell_animal, cohort_animals)

    rows = []
    for held in cohort_animals:
        todo = [t for t in specs if held in set(info.sample_name.iloc[cohort(info, t[1], t[2],
                                                                               t[0])])]
        if not todo:
            continue
        t0 = time.time()
        train_rows = np.where(cmask & (cell_animal != held))[0]
        hvg, pca = M.fit_encoder(Xn, train_rows)
        E_ = np.zeros((Xn.shape[0], M.PCA_DIM), np.float32)
        E_[cmask] = M.encode(Xn[cmask], hvg, pca)
        for key, col, _, covs, _ in todo:
            sub = info.iloc[cohort(info, col, model_name, key)].reset_index(drop=True)
            tr = (sub.sample_name != held).to_numpy()
            y = sub[col].to_numpy(float)
            resid = y - LinearRegression().fit(sub[covs].to_numpy(float)[tr], y[tr]).predict(
                sub[covs].to_numpy(float)) if covs else y.copy()
            mu, sd = resid[tr].mean(), resid[tr].std()
            z = dict(zip(sub.sample_name, (resid - mu) / sd))
            bags_tr, owners, y_tr, w_tr = [], [], [], []
            for a in sub.sample_name[tr]:
                secs = [s for s, an in sec2animal.items() if an == a]
                for s in secs:
                    bags_tr.append(np.where(section == s)[0]); owners.append(a)
                    y_tr.append(z[a]); w_tr.append(1.0 / len(secs))
            bags_te = [np.where(section == s)[0] for s, an in sec2animal.items() if an == held]
            for c in train_with_checkpoints(bags_tr, owners, y_tr, w_tr, z, bags_te, z[held],
                                            SEED):
                rows.append({"target": key, "held_out": held, **c,
                             "heldout_pred": c["heldout_pred_z"] * sd + mu})
        pd.DataFrame(rows).to_csv(STEM + ".csv", index=False)
        print(f"  held out {held}: {len(todo)} targets, {time.time() - t0:.0f}s", flush=True)
    summarise(pd.DataFrame(rows), info)


def summarise(df, info):
    L = ["ARM 6 TRAINING DIAGNOSTIC — undertrained, or fit-but-no-transfer?", "=" * 66,
         f"One seed, same LOAO folds/encoder as arm 6; checkpoints {CHECKPOINTS}. Reported",
         "arm 6 uses 30 epochs (fixed in advance). train_* columns are leak-free;",
         "held-out rho is test-informed and is NOT used to pick the epoch count.", ""]
    res = {}
    for key in df.target.unique():
        spec = [t for t in TARGETS if t[0] == key][0]
        sub = info.iloc[cohort(info, spec[1], spec[2], key)]
        y = sub[spec[1]].to_numpy(float)
        ref = pd.Series(loao_target(y, sub[spec[3]].to_numpy(float)) if spec[3] else y,
                        index=sub.sample_name)
        L += [f"TARGET {key} (n = {len(sub)}); losses are MSE in z-units of the residual target",
              f"  {'epoch':>5s} {'train loss':>10s} {'train MSE (eval)':>16s} "
              f"{'train rho (median, IQR)':>26s} {'held-out MSE':>12s} {'held-out rho':>12s}"]
        res[key] = []
        for ep, g in df[df.target == key].groupby("epoch"):
            g = g.set_index("held_out")
            ho = spearmanr(g.heldout_pred.loc[ref.index], ref.values).statistic
            q = g.train_rho.quantile([0.25, 0.5, 0.75])
            L.append(f"  {ep:5d} {g.train_loss.mean():10.3f} {g.train_loss_eval.mean():16.3f} "
                     f"{q[0.5]:+12.3f} [{q[0.25]:+.2f}, {q[0.75]:+.2f}] "
                     f"{g.heldout_loss.mean():12.3f} {ho:+12.3f}")
            res[key].append({"epoch": int(ep), "train_loss": float(g.train_loss.mean()),
                             "train_mse_eval": float(g.train_loss_eval.mean()),
                             "heldout_mse": float(g.heldout_loss.mean()),
                             "train_rho_median": float(q[0.5]), "heldout_rho": float(ho)})
        L.append("")
    L += ["Held-out MSE of a constant predictor (training mean) is ~1 in these units; below",
          "1 means the model beats predicting the mean.", "", "READING",
          "  train rho low and loss still falling at 30 -> undertrained: rerun arm 6 longer",
          "  (epoch count fixed in advance from the TRAIN curve, not the held-out one).",
          "  train rho high by 30 while held-out rho stays flat -> not undertraining; the gap",
          "  is generalisation from ~27 training animals."]
    report = "\n".join(L)
    print(report)
    with open(STEM + ".txt", "w") as fh:
        fh.write(report + "\n")
    with open(STEM + ".json", "w") as fh:
        json.dump(res, fh, indent=2)


E_ = None

if __name__ == "__main__":
    main()
