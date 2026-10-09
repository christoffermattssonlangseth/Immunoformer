"""ARM 6 of the baseline ladder — attention-MIL (Immunoformer, regression head) under the same
leave-one-animal-out folds as arms 1-5 (WORKORDER Task 4).

Bags = sections (`meta_sample_id`); an animal's prediction is the mean over its sections.
Each section is capped at CELL_CAP cells (fixed seed) once, at load, and cached; the model
additionally subsamples max_cells_per_bag per epoch as in immunotransformer/data.py.

Inside every fold (held-out animal excluded from everything that is fitted):
  * HVG selection (N_HVG genes) on a subsample of TRAINING cells,
  * PCA encoder (PCA_DIM) fit on a subsample of TRAINING cells, then applied to all cells,
  * target residualised on its covariates with a LinearRegression fit on training animals
    only, then z-scored with training mean/sd (prediction is mapped back to residual units),
  * GatedAttentionMIL(head="regression") trained for a FIXED number of epochs — no early
    stopping, because under LOAO the only held-out animal is the test animal.
The encoder depends only on which animal is held out, so one encoder per (cohort, held-out
animal) is shared by every target of that cohort. Training-bag loss is weighted
1/n_sections(animal) so animals count equally. Predictions are averaged over SEEDS.

Only y is residualised (cells cannot be); arms 2-5 residualise X as well. Because the
residual target is uncorrelated with the covariates in the training animals, the model
gains nothing by reading severity or day, but the asymmetry is noted in the report.

Section C2_G3_Mid_1 is excluded (Task 0 reassignment rule).

    PYTHONPATH="$PWD:$PWD/scripts" python analysis/mil_loao.py
"""

from __future__ import annotations

import immunotransformer  # noqa: F401  (OpenMP guard — must precede numpy; see HANDOFF.md)

import os
import sys
import time

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_ladder import ATLAS, EXCLUDE_SECTIONS, OUT, TARGETS, cohort, load_all  # noqa: E402
from immunotransformer.model import GatedAttentionMIL  # noqa: E402

CELL_CACHE = os.path.join(OUT, "mil_cells.npz")
PRED_OUT = os.environ.get("PRED_OUT", os.path.join(OUT, "arm6_predictions.csv"))
MAX_FOLDS = int(os.environ.get("MAX_FOLDS", "0"))  # >0: smoke test, first N folds per cohort
# RUN_ADJ_TARGETS="days_since_last_peak,onset_day_rr": train arm 6r instead — the same targets
# with a run_date indicator added to the covariates (residualisation fit on the training
# animals of each fold only, exactly like the severity/day covariates).
POOLING = os.environ.get("POOLING", "attention")   # "mean" = arm 6b, the no-attention control
RUN_ADJ_TARGETS = [t for t in os.environ.get("RUN_ADJ_TARGETS", "").split(",") if t]
CELL_CAP = 4096
N_HVG = 2000
PCA_DIM = 64
FIT_SUBSAMPLE = 100_000
EPOCHS = int(os.environ.get("EPOCHS", "30"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
MAX_CELLS_PER_BAG = 2048
LR, WD, GRAD_ACCUM = 1e-3, 1e-4, 4
PROJ_DIM, ATTN_DIM, DROPOUT = 128, 64, 0.1
CHUNK_ROWS = 100_000
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))


# ---------------------------------------------------------------- cells

def _obs_codes(obs, k):
    g = obs[k]
    cats = np.array([c.decode() if isinstance(c, bytes) else c for c in g["categories"][:]])
    return g["codes"][:], cats


def load_cells(sections):
    """Counts for up to CELL_CAP cells per wanted section -> CSR, section labels, genes."""
    if os.path.exists(CELL_CACHE):
        d = np.load(CELL_CACHE, allow_pickle=True)
        X = sp.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"]))
        return X, d["section"].astype(str), d["genes"].astype(str)
    rng = np.random.default_rng(0)
    with h5py.File(ATLAS, "r") as f:
        codes, cats = _obs_codes(f["obs"], "meta_sample_id")
        var = f["var"]
        genes = np.array([x.decode() if isinstance(x, bytes) else x
                          for x in var[var.attrs["_index"]][:]])
        want = np.isin(cats, list(sections))
        keep_rows = []
        for ci in np.where(want)[0]:
            rows = np.where(codes == ci)[0]
            if len(rows) > CELL_CAP:
                rows = np.sort(rng.choice(rows, CELL_CAP, replace=False))
            keep_rows.append(rows)
        keep_rows = np.sort(np.concatenate(keep_rows))
        C = f["layers/counts"]
        indptr = C["indptr"][:]
        parts, t0 = [], time.time()
        for a in range(0, len(codes), CHUNK_ROWS):
            e = min(a + CHUNK_ROWS, len(codes))
            sel = keep_rows[(keep_rows >= a) & (keep_rows < e)]
            if not len(sel):
                continue
            lo, hi = indptr[a], indptr[e]
            Xc = sp.csr_matrix((C["data"][lo:hi], C["indices"][lo:hi], indptr[a:e + 1] - lo),
                               shape=(e - a, len(genes)))
            parts.append(Xc[sel - a])
            print(f"  [cells] {e:,}/{len(codes):,}  kept {sum(p.shape[0] for p in parts):,}  "
                  f"{time.time() - t0:.0f}s", flush=True)
    X = sp.vstack(parts).tocsr().astype(np.float32)
    section = cats[codes[keep_rows]]
    np.savez(CELL_CACHE, data=X.data, indices=X.indices, indptr=X.indptr,
             shape=np.array(X.shape), section=section, genes=genes)
    return X, section, genes


def lognorm(X):
    """Per-cell log CP10k on the full panel (library size from all genes), CSR."""
    lib = np.asarray(X.sum(1)).ravel()
    lib[lib == 0] = 1
    Y = sp.diags((1e4 / lib).astype(np.float32)) @ X
    Y.data = np.log1p(Y.data)
    return Y.tocsr()


# ---------------------------------------------------------------- per-fold encoder

def fit_encoder(Xn, train_rows, seed=0):
    rng = np.random.default_rng(seed)
    sub = rng.choice(train_rows, min(FIT_SUBSAMPLE, len(train_rows)), replace=False)
    Xs = Xn[sub]
    mean = np.asarray(Xs.mean(0)).ravel()
    var = np.asarray(Xs.multiply(Xs).mean(0)).ravel() - mean ** 2
    disp = var / np.maximum(mean, 1e-6)                    # dispersion on training cells
    hvg = np.sort(np.argsort(-disp)[:N_HVG])
    pca = PCA(n_components=PCA_DIM, random_state=seed).fit(Xs[:, hvg].toarray())
    return hvg, pca


def encode(Xn, hvg, pca):
    out = np.empty((Xn.shape[0], PCA_DIM), np.float32)
    for a in range(0, Xn.shape[0], CHUNK_ROWS):
        out[a:a + CHUNK_ROWS] = pca.transform(Xn[a:a + CHUNK_ROWS][:, hvg].toarray())
    return out


# ---------------------------------------------------------------- model

def train_predict(E, bags_tr, y_tr, w_tr, bags_te, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = GatedAttentionMIL(in_dim=PCA_DIM, num_classes=2, proj_dim=PROJ_DIM,
                              attn_dim=ATTN_DIM, dropout=DROPOUT, head="regression",
                              pooling=POOLING)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        for step, i in enumerate(rng.permutation(len(bags_tr)), 1):
            rows = bags_tr[i]
            if len(rows) > MAX_CELLS_PER_BAG:
                rows = rng.choice(rows, MAX_CELLS_PER_BAG, replace=False)
            out, _ = model(torch.from_numpy(E[rows]))
            loss = w_tr[i] * (out.squeeze() - float(y_tr[i])) ** 2 / GRAD_ACCUM
            loss.backward()
            if step % GRAD_ACCUM == 0:
                opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
    model.eval()
    with torch.no_grad():
        return np.array([model(torch.from_numpy(E[rows[:MAX_CELLS_PER_BAG]]))[0].item()
                         for rows in bags_te])


# ---------------------------------------------------------------- main

def merge_seed_files(paths):
    """Average per-seed predictions across runs (e.g. seed 0 file + seeds 1-2 file)."""
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    rows = []
    for (tgt, a), g in df.groupby(["target", "sample_name"]):
        seeds = [float(v) for s in g.seed_preds for v in str(s).split(";")]
        rows.append({"target": tgt, "sample_name": a, "pred": float(np.mean(seeds)),
                     "seed_preds": ";".join(f"{v:.4f}" for v in seeds), "n_seeds": len(seeds)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "arm6_predictions.csv"), index=False)
    print(f"[merge] {len(paths)} files -> arm6_predictions.csv; seeds per animal: "
          f"{out.n_seeds.value_counts().to_dict()}")


def main():
    os.makedirs(OUT, exist_ok=True)
    if os.environ.get("MERGE"):   # MERGE="file1.csv,file2.csv"
        merge_seed_files(os.environ["MERGE"].split(","))
        return
    pb, genes_pb, fracs, info = load_all()
    d = np.load("runs/rr_within_relapse/pseudobulk_all67.npz", allow_pickle=True)
    smeta = pd.DataFrame(d["section_meta"].tolist(), columns=d["section_meta_cols"])
    smeta = smeta[~smeta.meta_sample_id.isin(EXCLUDE_SECTIONS)]
    X, section, _ = load_cells(set(smeta.meta_sample_id))
    Xn = lognorm(X)
    sec2animal = dict(zip(smeta.meta_sample_id, smeta.sample_name))
    cell_animal = np.array([sec2animal.get(s, "") for s in section])
    print(f"[cells] {X.shape[0]:,} cells x {X.shape[1]} genes, "
          f"{len(set(section))} sections", flush=True)

    done = pd.read_csv(PRED_OUT) if os.path.exists(PRED_OUT) else pd.DataFrame(
        columns=["target", "sample_name", "pred", "seed_preds"])
    rows_out = done.to_dict("records")
    targets = TARGETS
    if RUN_ADJ_TARGETS:
        for m_ in info.model.unique():
            assert info[info.model == m_].run_date.nunique() == 2, "one run indicator per cohort"
        info["run_indicator"] = (info.run_date == "20260506").astype(float)
        targets = [(k + "_runadj", c, m_, covs + ["run_indicator"], kd)
                   for k, c, m_, covs, kd in TARGETS if k in RUN_ADJ_TARGETS]
    for model_name in sorted({t[2] for t in targets}):
        specs = [t for t in targets if t[2] == model_name]
        cohort_animals = info[info.model == model_name].sample_name.to_numpy()
        cmask = np.isin(cell_animal, cohort_animals)
        print(f"[{model_name}] {len(cohort_animals)} animals, {cmask.sum():,} cells, "
              f"targets {[t[0] for t in specs]}", flush=True)
        for held in (cohort_animals[:MAX_FOLDS] if MAX_FOLDS else cohort_animals):
            todo = [t for t in specs if held in set(info.sample_name.iloc[cohort(
                info, t[1], t[2], t[0])]) and not ((done.target == t[0]) &
                                                    (done.sample_name == held)).any()]
            if not todo:
                continue
            t0 = time.time()
            train_rows = np.where(cmask & (cell_animal != held))[0]
            hvg, pca = fit_encoder(Xn, train_rows)
            E = np.zeros((Xn.shape[0], PCA_DIM), np.float32)
            E[cmask] = encode(Xn[cmask], hvg, pca)
            for key, col, _, covs, _ in todo:
                idx = cohort(info, col, model_name, key)
                sub = info.iloc[idx].reset_index(drop=True)
                tr = sub.sample_name != held
                y = sub[col].to_numpy(float)
                if covs:
                    cov = sub[covs].to_numpy(float)
                    f = LinearRegression().fit(cov[tr], y[tr])
                    resid = y - f.predict(cov)
                else:
                    resid = y.copy()
                mu, sd = resid[tr].mean(), resid[tr].std()
                z = dict(zip(sub.sample_name, (resid - mu) / sd))
                bags_tr, y_tr, w_tr = [], [], []
                for a in sub.sample_name[tr]:
                    secs = [s for s, an in sec2animal.items() if an == a]
                    for s in secs:
                        bags_tr.append(np.where(section == s)[0]); y_tr.append(z[a])
                        w_tr.append(1.0 / len(secs))
                bags_te = [np.where(section == s)[0] for s, an in sec2animal.items() if an == held]
                seed_preds = [float(train_predict(E, bags_tr, np.array(y_tr), np.array(w_tr),
                                                  bags_te, sd_).mean()) for sd_ in SEEDS]
                pred = float(np.mean(seed_preds)) * sd + mu
                rows_out.append({"target": key, "sample_name": held, "pred": pred,
                                 "seed_preds": ";".join(f"{p * sd + mu:.4f}" for p in seed_preds)})
            pd.DataFrame(rows_out).to_csv(PRED_OUT, index=False)   # checkpoint every fold
            print(f"  [{model_name}] held out {held}: {len(todo)} targets, "
                  f"{time.time() - t0:.0f}s", flush=True)
    print(f"[done] -> {PRED_OUT}")


if __name__ == "__main__":
    main()
