"""BATCH IDENTIFIABILITY — can run / slide / region / model be recovered from the transcriptome?

Existing batch checks (`duration_clock.py`) ask whether the LABEL is associated with batch
(Kruskal–Wallis). That is a different question from whether batch is RECOVERABLE from the
data. If run identity is predictable from expression, any model trained across runs can use
it as a label proxy — and `run_date` is aliased with `model` in two of the three Xenium runs
(WORKORDER Task 1).

Units. `run_date`, `run_id` and `model` are constant within an animal -> classified on
ANIMAL pseudobulk (n = 67). `sample_id` (slide) and `region` vary between an animal's
sections -> classified on SECTION pseudobulk (`meta_sample_id`, n = 158). Every split is
GroupKFold on animal (`sample_name`); PCA (top 50, on standardised log-CP10k) is fit inside
the training fold. Null = label permutation at the animal level (an animal's whole label
vector moves together, to an animal with the same number of sections).

Classifiers. HistGradientBoosting (as specified) with min_samples_leaf=5 — the default of 20
barely splits on ~50 training animals and would under-state recoverability. An L2 logistic
regression on the same PCs is reported alongside as a linear comparator.

Section C2_G3_Mid_1 (WORKORDER Task 0 reassignment rule). It is labelled C_M16_2 in the
atlas, but its sheet row carries C_M16_1's Animal ID and its tissue is male by Xist while
C_M16_2 is female. Reassignment is a decision, not a cleanup, and has not been confirmed
against the raw scoring sheets by a human -> DEFAULT: the section is EXCLUDED.
`SECTION_MODE=reassign` regroups it under C_M16_1 and writes results_reassigned.json as a
labelled sensitivity check; the default report compares the two.

    PYTHONPATH="$PWD" python scripts/batch_identifiability.py
    SECTION_MODE=reassign PYTHONPATH="$PWD" python scripts/batch_identifiability.py
    REPORT_ONLY=1 PYTHONPATH="$PWD" python scripts/batch_identifiability.py   # re-render
"""

from __future__ import annotations

import immunotransformer  # noqa: F401  (OpenMP guard — must precede numpy; see HANDOFF.md)

import json
import os
import time

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ATLAS = os.environ.get(
    "RRMAP2_H5AD",
    os.path.expanduser(
        "~/Downloads/RRMAP2_xenium_all_samples.cellcharter.companion.ready.with_metadata."
        "rerun.with_AnnoL1Curated_with_Region_Anno2to4Updated.h5ad"),
)
CACHE = "runs/rr_within_relapse/pseudobulk_all67.npz"
OUT = "runs/batch_identifiability"
QUESTIONED = {"C2_G3_Mid_1": "C_M16_1"}  # meta_sample_id -> candidate animal (Task 0)
SECTION_MODE = os.environ.get("SECTION_MODE", "exclude")  # exclude | reassign
N_BOOT = 2000
MAY_RUN = "20260506"

N_PCS = 50
N_FOLDS = 5
N_PERM = int(os.environ.get("N_PERM", "200"))
N_JOBS = int(os.environ.get("N_JOBS", "8"))
SEED = 0
CHUNK_ROWS = 100_000

META_COLS = ["sample_name", "meta_sample_id", "sample_id", "run_date", "run_id", "region",
             "model", "stage", "condition", "instrument_or_flowcell"]
ANIMAL_TARGETS = ["run_date", "run_id", "model"]
SECTION_TARGETS = ["sample_id", "region"]


# ---------------------------------------------------------------- pseudobulk (h5py stream)

def _obs_col(obs, k):
    g = obs[k]
    if isinstance(g, h5py.Group):
        cats = [c.decode() if isinstance(c, bytes) else c for c in g["categories"][:]]
        return pd.Categorical.from_codes(g["codes"][:], cats)
    return g[:]


def build_pseudobulk():
    """Section-level summed counts [158, n_genes] streamed from layers/counts in row chunks."""
    with h5py.File(ATLAS, "r") as f:
        obs = f["obs"]
        cells = pd.DataFrame({k: _obs_col(obs, k) for k in META_COLS}).astype(str)
        var = f["var"]
        genes = np.array([x.decode() if isinstance(x, bytes) else x
                          for x in var[var.attrs["_index"]][:]])
        sections = np.array(sorted(cells.meta_sample_id.unique()))
        code = pd.Categorical(cells.meta_sample_id, categories=sections).codes
        C = f["layers/counts"]
        indptr = C["indptr"][:]
        n_cells, n_genes = len(cells), len(genes)
        sums = np.zeros((len(sections), n_genes))
        t0 = time.time()
        for a in range(0, n_cells, CHUNK_ROWS):
            e = min(a + CHUNK_ROWS, n_cells)
            lo, hi = indptr[a], indptr[e]
            X = sp.csr_matrix((C["data"][lo:hi], C["indices"][lo:hi], indptr[a:e + 1] - lo),
                              shape=(e - a, n_genes))
            G = sp.csr_matrix((np.ones(e - a), (code[a:e], np.arange(e - a))),
                              shape=(len(sections), e - a))
            sums += (G @ X).toarray()
            print(f"  [pseudobulk] {e:,}/{n_cells:,} cells  {time.time() - t0:.0f}s", flush=True)
    meta = (cells.drop_duplicates("meta_sample_id").set_index("meta_sample_id")
            .loc[sections].reset_index())
    meta["n_cells"] = np.bincount(code, minlength=len(sections))
    return sums, genes, meta


def load_or_build():
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        meta = pd.DataFrame(d["section_meta"].tolist(), columns=d["section_meta_cols"])
        meta["n_cells"] = meta.n_cells.astype(int)
        return d["section_counts"], d["genes"].astype(str), meta
    sums, genes, meta = build_pseudobulk()
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, section_counts=sums, genes=genes,
             section_meta=meta.to_numpy(dtype=object), section_meta_cols=np.array(meta.columns))
    return sums, genes, meta


def log_cp10k(counts):
    return np.log1p(counts / counts.sum(1, keepdims=True) * 1e4)


# ---------------------------------------------------------------- classification

def _models(n_train):
    k = min(N_PCS, n_train - 1)
    hgb = make_pipeline(StandardScaler(), PCA(n_components=k, random_state=SEED),
                        HistGradientBoostingClassifier(min_samples_leaf=5, max_iter=100,
                                                       random_state=SEED))
    lr = make_pipeline(StandardScaler(), PCA(n_components=k, random_state=SEED),
                       StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    return {"hgb": hgb, "logreg": lr}


def cv_predict(X, y, groups, which=("hgb", "logreg")):
    """Out-of-fold labels (+ P(class 1) if binary), GroupKFold on animal; all transforms
    are fit inside each training fold."""
    classes = np.unique(y)
    n_splits = min(N_FOLDS, len(np.unique(groups)))
    pred = {m: np.empty(len(y), dtype=object) for m in which}
    proba = {m: np.full(len(y), np.nan) for m in which}
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            for m in which:
                pred[m][te] = y[tr][0]
                proba[m][te] = float(y[tr][0] == classes[-1])
            continue
        models = _models(len(tr))
        for m in which:
            fit = models[m].fit(X[tr], y[tr])
            pred[m][te] = fit.predict(X[te])
            if len(classes) == 2:
                proba[m][te] = fit.predict_proba(X[te])[:, list(fit.classes_).index(classes[-1])]
    return pred, proba


def _scores(y, pred, proba):
    out = {}
    for m in pred:
        s = {"ba": float(balanced_accuracy_score(y, pred[m].astype(str)))}
        if len(np.unique(y)) == 2:
            s["auc"] = float(roc_auc_score(y == np.unique(y)[-1], proba[m]))
        out[m] = s
    return out


def animal_block_permutation(y, groups, rng):
    """Move each animal's whole label vector to another animal with the same section count."""
    y_perm = y.copy()
    animals = pd.Series(groups).drop_duplicates().to_numpy()
    idx = {a: np.where(groups == a)[0] for a in animals}
    sizes = pd.Series({a: len(v) for a, v in idx.items()})
    for _, same in sizes.groupby(sizes):
        src = same.index.to_numpy()
        dst = rng.permutation(src)
        for a, b in zip(src, dst):
            y_perm[idx[a]] = y[idx[b]]
    return y_perm


def _perm_one(X, y, groups, seed):
    yp = animal_block_permutation(y, groups, np.random.default_rng(seed))
    return _scores(yp, *cv_predict(X, yp, groups))


def animal_bootstrap_auc(y, proba, groups, rng):
    """CI for the out-of-fold AUC, resampling ANIMALS (all of an animal's units move together).
    Conditional on the fitted folds: it captures sampling of animals, not refit variance."""
    animals = np.unique(groups)
    idx = {a: np.where(groups == a)[0] for a in animals}
    pos = y == np.unique(y)[-1]
    vals = []
    for _ in range(N_BOOT):
        take = np.concatenate([idx[a] for a in rng.choice(animals, len(animals))])
        if pos[take].all() or (~pos[take]).all():
            continue
        vals.append(roc_auc_score(pos[take], proba[take]))
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def _null_summary(obs, null):
    null = np.asarray(null)
    return {"observed": obs, "null_mean": float(null.mean()), "null_sd": float(null.std()),
            "null_q05": float(np.quantile(null, 0.05)),
            "null_q50": float(np.quantile(null, 0.50)),
            "null_q95": float(np.quantile(null, 0.95)), "null_max": float(null.max()),
            "p_perm": float((1 + (null >= obs).sum()) / (1 + len(null))),
            "null": [round(float(v), 4) for v in null]}


def evaluate(X, y, groups, label):
    t0 = time.time()
    pred, proba = cv_predict(X, y, groups)
    obs = _scores(y, pred, proba)
    nulls = Parallel(n_jobs=N_JOBS)(delayed(_perm_one)(X, y, groups, SEED + 1 + i)
                                    for i in range(N_PERM))
    animals_per_class = (pd.DataFrame({"y": y, "g": groups}).drop_duplicates()
                         .y.value_counts().to_dict())
    out = {"n": int(len(y)), "n_animals": int(len(np.unique(groups))),
           "n_classes": int(len(np.unique(y))), "chance": 1 / len(np.unique(y)),
           "class_counts": pd.Series(y).value_counts().to_dict(),
           "animals_per_class": animals_per_class}
    rng = np.random.default_rng(SEED)
    for m in obs:
        out[m] = {"balanced_accuracy": _null_summary(obs[m]["ba"], [n[m]["ba"] for n in nulls])}
        if "auc" in obs[m]:
            out[m]["auc"] = _null_summary(obs[m]["auc"], [n[m]["auc"] for n in nulls])
            out[m]["auc"]["ci95_animal_bootstrap"] = animal_bootstrap_auc(y, proba[m], groups,
                                                                          rng)
    out["confusion_hgb"] = pd.crosstab(pd.Series(y, name="true"),
                                       pd.Series(pred["hgb"].astype(str), name="pred")
                                       ).to_dict(orient="index")
    print(f"  [{label}] hgb {obs['hgb']['ba']:.3f}  logreg {obs['logreg']['ba']:.3f}  "
          f"chance {out['chance']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    return out


def run_stratum(sec_counts, sec_meta, name, targets=None):
    """Targets on one stratum: animal-level for animal-constant labels, section otherwise."""
    res = {}
    Xs = log_cp10k(sec_counts)
    gs = sec_meta.sample_name.to_numpy()
    animals = sec_meta.sample_name.drop_duplicates().to_numpy()
    a_counts = np.vstack([sec_counts[gs == a].sum(0) for a in animals])
    Xa = log_cp10k(a_counts)
    a_meta = sec_meta.drop_duplicates("sample_name").set_index("sample_name").loc[animals]
    for t in ANIMAL_TARGETS + SECTION_TARGETS:
        if targets and t not in targets:
            continue
        unit = "animal" if t in ANIMAL_TARGETS else "section"
        y = (a_meta[t] if unit == "animal" else sec_meta[t]).to_numpy().astype(str)
        if len(np.unique(y)) < 2:
            res[t] = {"skipped": f"single class {np.unique(y).tolist()}"}
            continue
        X, g = (Xa, animals) if unit == "animal" else (Xs, gs)
        res[t] = {"unit": unit, **evaluate(X, y, g, f"{name}:{t}")}
    return res


# ---------------------------------------------------------------- main

def apply_section_mode(meta, mode):
    q = meta.meta_sample_id.isin(QUESTIONED)
    if mode == "reassign":
        meta.loc[q, "sample_name"] = meta.loc[q, "meta_sample_id"].map(QUESTIONED)
        return meta, int(q.sum())
    return meta[~q].reset_index(drop=True), int(q.sum())


def _tabs(meta):
    animals = meta.drop_duplicates("sample_name")
    return {
        "run_date x model": pd.crosstab(animals.run_date, animals.model),
        "run_id x model": pd.crosstab(animals.run_id, animals.model),
        "run_date x stage": pd.crosstab(animals.stage, [animals.model, animals.run_date]),
        "sample_id x model (sections)": pd.crosstab(meta.sample_id, meta.model),
    }


def _may_tabs(meta):
    may = meta[meta.run_date == MAY_RUN]
    return {
        "May 2026: slide (sample_id) x model, sections": pd.crosstab(may.sample_id, may.model),
        "May 2026: region x model, sections": pd.crosstab(may.region, may.model),
        "May 2026: sub-run x model, animals": pd.crosstab(
            may.drop_duplicates("sample_name").run_id, may.drop_duplicates("sample_name").model),
    }


def may_region_matched(counts, meta):
    """Model within May 2026 using lumbar (L) sections only: cervical tissue exists only for
    RR in this run, and region is recoverable, so whole-animal pseudobulk mixes the two."""
    m = ((meta.run_date == MAY_RUN) & (meta.region == "L")).to_numpy()
    print(f"[May 2026, L sections only: {meta[m].sample_name.nunique()} animals]")
    return run_stratum(counts[m], meta[m].reset_index(drop=True), "may_L", targets=["model"])


def results_path(mode):
    return os.path.join(OUT, "results.json" if mode == "exclude" else "results_reassigned.json")


def main():
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    counts, genes, meta = load_or_build()
    keep = (meta.n_cells > 0).to_numpy()
    counts, meta = counts[keep], meta[keep].reset_index(drop=True)
    q = meta.meta_sample_id.isin(QUESTIONED).to_numpy()
    meta, n_q = apply_section_mode(meta, SECTION_MODE)
    if SECTION_MODE != "reassign":
        counts = counts[~q]

    if os.environ.get("REPORT_ONLY") or os.environ.get("EXTRA_ONLY"):
        with open(results_path(SECTION_MODE)) as fh:
            results = json.load(fh)
        if os.environ.get("EXTRA_ONLY"):
            results["may2026_L_only"] = may_region_matched(counts, meta)
            with open(results_path(SECTION_MODE), "w") as fh:
                json.dump(results, fh, indent=2, default=str)
        _plot(results)
        _report(results, _tabs(meta), _may_tabs(meta))
        return

    print(f"[data] {len(meta)} sections, {meta.sample_name.nunique()} animals, "
          f"{len(genes)} genes; section mode '{SECTION_MODE}' ({n_q} section: {QUESTIONED})")
    tabs, may_tabs = _tabs(meta), _may_tabs(meta)
    animals = meta.drop_duplicates("sample_name")
    results = {"section_mode": SECTION_MODE, "questioned_section": QUESTIONED,
               "n_sections": len(meta), "n_animals": len(animals),
               "settings": {"n_pcs": N_PCS, "n_folds": N_FOLDS, "n_perm": N_PERM,
                            "n_boot": N_BOOT, "hgb_min_samples_leaf": 5, "logreg_C": 0.1}}
    print("[all runs]")
    results["all"] = run_stratum(counts, meta, "all")

    m = (meta.run_date == MAY_RUN).to_numpy()
    may = meta[m].reset_index(drop=True)
    may_animals = may.drop_duplicates("sample_name")
    results["may2026_stratum"] = {
        "n_animals_by_model": may_animals.model.value_counts().to_dict(),
        "stages_by_model": may_animals.groupby("model").stage.value_counts()
                                      .unstack(fill_value=0).to_dict(orient="index"),
        "n_slides": int(may.sample_id.nunique()),
        "slides_mixing_models": int((may.groupby("sample_id").model.nunique() > 1).sum()),
        "slides_by_model": may.groupby("model").sample_id.nunique().to_dict(),
    }
    print(f"[May 2026 only: {len(may_animals)} animals]")
    results["may2026"] = run_stratum(counts[m], may, "may")

    results["may2026_L_only"] = may_region_matched(counts, meta)

    # one physical sub-run (run5 holds both models; run6 is RR-only)
    m5 = ((meta.run_date == MAY_RUN) & (meta.run_id == "run5")).to_numpy()
    print(f"[May 2026 run5 only: {meta[m5].sample_name.nunique()} animals]")
    results["may2026_run5"] = run_stratum(counts[m5], meta[m5].reset_index(drop=True),
                                          "may_run5", targets=["model"])

    # run identity WITHIN one model (run_date is aliased with model overall)
    for model, key in (("CHRONIC", "chronic_only"), ("RELAPSE REMITTING", "rr_only")):
        mm = (meta.model == model).to_numpy()
        print(f"[{model} only: {meta[mm].sample_name.nunique()} animals]")
        results[key] = run_stratum(counts[mm], meta[mm].reset_index(drop=True), key,
                                   targets=["run_date", "run_id"])
        # stage-matched: only stages present in every run of this model, so run != stage
        a = meta[mm].drop_duplicates("sample_name")
        ct = pd.crosstab(a.stage, a.run_date)
        shared = ct.index[(ct > 0).all(1)].tolist()
        ms = mm & meta.stage.isin(shared).to_numpy()
        print(f"[{model} stage-matched {shared}: {meta[ms].sample_name.nunique()} animals]")
        sm = run_stratum(counts[ms], meta[ms].reset_index(drop=True), key + "_sm",
                         targets=["run_date"])
        results[key + "_stage_matched"] = {**sm, "shared_stages": shared}

    with open(results_path(SECTION_MODE), "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    if SECTION_MODE == "exclude":
        for k, t in {**tabs, **may_tabs}.items():
            t.to_csv(os.path.join(OUT, k.split(" (")[0].split(",")[0].replace(":", "")
                                  .replace(" ", "_") + ".csv"))
        _plot(results)
        _report(results, tabs, may_tabs)
    else:
        print(f"[sensitivity] wrote {results_path(SECTION_MODE)}; the default report compares it")


# ---------------------------------------------------------------- figure

INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
OBS_COLOR, NULL_COLOR = "#2a78d6", "#b7b6b0"
STRATA = [("all", "All runs"), ("chronic_only_stage_matched", "Chronic\nstage-matched"),
          ("rr_only_stage_matched", "RR\nstage-matched"), ("may2026", "May 2026 run"),
          ("may2026_L_only", "May 2026\nlumbar only"), ("may2026_run5", "May 2026\nrun5 only")]


def _targets(res):
    return {t: v for t, v in res.items() if isinstance(v, dict) and "hgb" in v}


def _plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strata = [(k, v) for k, v in STRATA if k in results]
    widths = [max(1.4, len(_targets(results[k]))) for k, _ in strata]
    fig, axes = plt.subplots(1, len(strata), figsize=(15, 4.2), sharey=True,
                             gridspec_kw={"width_ratios": widths})
    fig.patch.set_facecolor(SURFACE)
    for ax, (key, title) in zip(axes, strata):
        ax.set_facecolor(SURFACE)
        res = _targets(results[key])
        for i, (t, v) in enumerate(res.items()):
            b = v["hgb"]["balanced_accuracy"]
            ax.plot([i - 0.3, i + 0.3], [v["chance"]] * 2, color=MUTED, lw=1, ls=":")
            ax.add_patch(plt.Rectangle((i - 0.25, b["null_q05"]), 0.5,
                                       b["null_q95"] - b["null_q05"], color=NULL_COLOR, lw=0))
            ax.plot(i, b["observed"], "o", ms=9, color=OBS_COLOR, mec=SURFACE, mew=2)
            ax.annotate(f"{b['observed']:.2f}", (i, b["observed"]), xytext=(10, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK)
        ax.set_xticks(range(len(res)),
                      [f"{t}\n{v['unit']}, k={v['n_classes']}\n{v['n_animals']} animals"
                       for t, v in res.items()], fontsize=7, color=INK)
        ax.set_xlim(-0.6, len(res) - 0.4)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, loc="left", fontsize=9, color=INK)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(axis="y", color="#e6e5e0", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("balanced accuracy (HGB, GroupKFold on animal)", fontsize=8, color=MUTED)
    fig.text(0.01, 0.01, "dot = observed · grey bar = permutation null 5th-95th pct · "
             f"dotted = chance 1/k · section mode: {results['section_mode']}", fontsize=7,
             color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(OUT, "figures", "batch_identifiability.png"), dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- report

def _metric_line(name, s):
    return (f"{name} {s['observed']:.3f}  null {s['null_mean']:.3f} ± {s['null_sd']:.3f} "
            f"[q05 {s['null_q05']:.3f}, q95 {s['null_q95']:.3f}, max {s['null_max']:.3f}]  "
            f"p = {s['p_perm']:.3f}")


def _block(t, v):
    if "skipped" in v:
        return [f"  {t}: skipped ({v['skipped']})"]
    L = [f"  {t}  [{v['unit']}-level, {v['n']} units from {v['n_animals']} animals, "
         f"k = {v['n_classes']}, chance = {v['chance']:.3f}]"]
    if v["n_classes"] <= 6:
        L.append("    animals per class: " + ", ".join(
            f"{k} {n}" for k, n in sorted(v["animals_per_class"].items())))
        if v["unit"] == "section":
            L.append("    sections per class: " + ", ".join(
                f"{k} {n}" for k, n in sorted(v["class_counts"].items())))
    else:
        c = pd.Series(v["class_counts"])
        L.append(f"    units per class: min {c.min()}, median {c.median():.0f}, max {c.max()}")
    for m, lab in (("hgb", "HGB   "), ("logreg", "logreg")):
        L.append(f"    {lab} " + _metric_line("balanced acc", v[m]["balanced_accuracy"]))
        if "auc" in v[m]:
            a = v[m]["auc"]
            lo, hi = a["ci95_animal_bootstrap"]
            L.append(f"           " + _metric_line("AUC", a) +
                     f"  95% CI (animal bootstrap) [{lo:.3f}, {hi:.3f}], width {hi - lo:.3f}")
    return L


def _p(v):
    return min(v["hgb"]["balanced_accuracy"]["p_perm"], v["logreg"]["balanced_accuracy"]["p_perm"])


def _ba(v, m="hgb"):
    return v[m]["balanced_accuracy"]["observed"]


def _table(t):
    return "  " + t.to_string().replace("\n", "\n  ")


def _sensitivity(r):
    path = results_path("reassign")
    if not os.path.exists(path):
        return ["  results_reassigned.json not found — run with SECTION_MODE=reassign."]
    with open(path) as fh:
        s = json.load(fh)

    def ba(res, m):
        b = res[m]["balanced_accuracy"]
        return b["observed"] if isinstance(b, dict) else b  # older results stored a float

    L = [f"  {'stratum':28s} {'target':10s} {'HGB excl':>9s} {'HGB reas':>9s} "
         f"{'logreg excl':>12s} {'logreg reas':>12s}"]
    for key, _ in STRATA:
        if key not in r:
            continue
        for t, v in _targets(r[key]).items():
            w = s.get(key, {}).get(t)
            if not w or "hgb" not in w:
                continue
            L.append(f"  {key:28s} {t:10s} {ba(v, 'hgb'):9.3f} {ba(w, 'hgb'):9.3f} "
                     f"{ba(v, 'logreg'):12.3f} {ba(w, 'logreg'):12.3f}")
    return L


def _report(r, tabs, may_tabs):
    s = r["settings"]
    L = ["BATCH IDENTIFIABILITY — is batch recoverable from the transcriptome?", "=" * 72, "",
         f"{r['n_sections']} sections / {r['n_animals']} animals; log-CP10k pseudobulk; top "
         f"{s['n_pcs']} PCs fit inside each fold; GroupKFold({s['n_folds']}) on animal.",
         f"Null = {s['n_perm']} animal-level label permutations (smallest attainable p = "
         f"{1 / (s['n_perm'] + 1):.3f}). Balanced accuracy, chance = 1/k. HGB primary, L2",
         "logreg = linear comparator. AUC (binary targets) with a 95% CI from "
         f"{s['n_boot']} animal-level bootstrap",
         "resamples of the out-of-fold predictions (conditional on the folds; excludes refit "
         "variance).", "",
         "SECTION C2_G3_Mid_1 — provenance and handling",
         "  Atlas and sheet label it C_M16_2. Evidence it belongs to C_M16_1: (1) its sheet row",
         "  carries C_M16_1's Animal ID (1058085) — independent of curves; (2) its tissue is",
         "  male by Xist (0.6% Xist+ cells vs >= 14% in every female section), C_M16_2 is female",
         "  and C_M16_1 male — independent of curves; (3) its score curve equals C_M16_1's —",
         "  CIRCULAR for curve-based analyses, carries no independent weight. Not yet confirmed",
         "  against the raw scoring sheets by a human.",
         f"  This report: section mode = '{r['section_mode']}' (default exclude). Sensitivity "
         "(reassigned to C_M16_1) at the end.", ""]

    for key, title in (("all", "ALL RUNS"),):
        L.append(title)
        for t, v in r[key].items():
            L += _block(t, v)
        L.append("")

    for key, lab in (("chronic_only", "CHRONIC"), ("rr_only", "RELAPSE REMITTING")):
        L.append(f"RUN WITHIN ONE MODEL — {lab} only (stage mix differs by run; see "
                 "run_date x stage)")
        for t, v in r[key].items():
            L += _block(t, v)
        sm = r[key + "_stage_matched"]
        L.append(f"  stage-matched (only stages present in every run: {sm['shared_stages']}):")
        L += _block("run_date", sm["run_date"])
        L.append("")

    may = r["may2026_stratum"]
    L += [f"MAY 2026 RUN ONLY ({MAY_RUN}) — the only run containing both EAE models",
          f"  animals per model: {may['n_animals_by_model']}",
          "  stages:"]
    for model, st in may["stages_by_model"].items():
        L.append(f"    {model}: " + ", ".join(f"{k} {v}" for k, v in st.items() if v))
    L += ["", "  CONFOUND CHECK FIRST — has the confound been removed, or moved one level down?",
          f"  {may['n_slides']} slides in this run; {may['slides_mixing_models']} hold both "
          f"models (slides per model: {may['slides_by_model']}).",
          "  -> Within May 2026, model is FULLY ALIASED WITH SLIDE: every slide carries one",
          "     model only. The within-run model contrast is therefore also a between-slide",
          "     contrast.",
          f"     Slide identity within this run: HGB {_ba(r['may2026']['sample_id']):.2f} / "
          f"logreg {_ba(r['may2026']['sample_id'], 'logreg'):.2f} vs chance "
          f"{r['may2026']['sample_id']['chance']:.2f} (p = {_p(r['may2026']['sample_id']):.3f}).",
          "     A per-slide fingerprint near chance argues against a slide-specific artefact,",
          "     but cannot exclude a preparation batch shared by all slides of one model."]
    for k, t in may_tabs.items():
        L += [f"  {k}:", _table(t), ""]
    for t, v in r["may2026"].items():
        L += _block(t, v)
    if "may2026_L_only" in r:
        L += ["", "MAY 2026, LUMBAR (L) SECTIONS ONLY — region-matched: cervical sections exist "
              "only for RR in this run"]
        for t, v in _targets(r["may2026_L_only"]).items():
            L += _block(t, v)
    L += ["", "MAY 2026, SUB-RUN run5 ONLY — the only physical sub-run holding both models"]
    for t, v in _targets(r["may2026_run5"]).items():
        L += _block(t, v)
    L.append("")

    a, m, m5 = r["all"], r["may2026"], r["may2026_run5"]
    cs, rs = r["chronic_only_stage_matched"]["run_date"], r["rr_only_stage_matched"]["run_date"]
    mauc = m["model"]["hgb"]["auc"]
    lo, hi = mauc["ci95_animal_bootstrap"]
    L += ["SUMMARY (numbers; no verdict without them)",
          f"  run_date, all runs: HGB {_ba(a['run_date']):.2f} / logreg "
          f"{_ba(a['run_date'], 'logreg'):.2f} vs chance {a['run_date']['chance']:.2f}, "
          f"p = {_p(a['run_date']):.3f}, {a['run_date']['n_animals']} animals "
          "(inflated: run is aliased with model).",
          f"  run_date within chronic, stage-matched: HGB {_ba(cs):.2f} / logreg "
          f"{_ba(cs, 'logreg'):.2f} vs chance 0.50, HGB p = "
          f"{cs['hgb']['balanced_accuracy']['p_perm']:.3f}, logreg p = "
          f"{cs['logreg']['balanced_accuracy']['p_perm']:.3f}, {cs['n_animals']} animals "
          f"({cs['animals_per_class']}).",
          f"  run_date within RR, stage-matched: HGB {_ba(rs):.2f} / logreg "
          f"{_ba(rs, 'logreg'):.2f} vs chance 0.50, HGB p = "
          f"{rs['hgb']['balanced_accuracy']['p_perm']:.3f}, logreg p = "
          f"{rs['logreg']['balanced_accuracy']['p_perm']:.3f}, {rs['n_animals']} animals "
          f"({rs['animals_per_class']}).",
          f"  model within May 2026: HGB balanced acc {_ba(m['model']):.2f} vs chance 0.50 "
          f"(p = {m['model']['hgb']['balanced_accuracy']['p_perm']:.3f}); HGB AUC "
          f"{mauc['observed']:.2f}, 95% CI [{lo:.2f}, {hi:.2f}]; "
          f"{m['model']['animals_per_class']}; model fully aliased with slide; cervical sections "
          "only in RR (AUC 1.00 is perfect separation, so the bootstrap CI is degenerate — "
          f"the permutation null q95 is {mauc['null_q95']:.2f}).",
          *([f"  model within May 2026, lumbar sections only (region-matched): HGB "
             f"{_ba(r['may2026_L_only']['model']):.2f} / logreg "
             f"{_ba(r['may2026_L_only']['model'], 'logreg'):.2f}, p = "
             f"{_p(r['may2026_L_only']['model']):.3f}, HGB AUC "
             f"{r['may2026_L_only']['model']['hgb']['auc']['observed']:.2f}, "
             f"{r['may2026_L_only']['model']['animals_per_class']}."]
            if "may2026_L_only" in r else []),
          f"  model within run5 only: HGB {_ba(m5['model']):.2f} / logreg "
          f"{_ba(m5['model'], 'logreg'):.2f}, p = {_p(m5['model']):.3f}, "
          f"{m5['model']['animals_per_class']} — 3 RR animals, underpowered.",
          f"  run5 vs run6 within May 2026: HGB {_ba(m['run_id']):.2f} / logreg "
          f"{_ba(m['run_id'], 'logreg'):.2f}, p = {_p(m['run_id']):.3f}.", "",
          "FRAMING (permanent, design property)",
          "  Each disease model uses a different mouse strain (RR = SJL/PLP, chronic = "
          "C57BL/6/MOG).",
          "  No stratum separates disease-model biology from strain. The within-run result",
          "  supports only: 'the RR/chronic difference is not purely a run artefact'. Any",
          "  description of runs/rr_vs_chronic as detecting disease-model biology is wrong and",
          "  must read 'strain-or-model'. The cross-run rr_vs_chronic AUC 0.998 is superseded;",
          "  cite the within-run May 2026 AUC above, with its CI and the slide aliasing.", "",
          "SENSITIVITY — C2_G3_Mid_1 excluded (default) vs reassigned to C_M16_1 "
          "(balanced accuracy)"]
    L += _sensitivity(r)
    L += ["", "CROSS-TABULATIONS (animals unless noted)"]
    for k, t in tabs.items():
        L += [f"  {k}:", _table(t), ""]
    L.append("figures/: batch_identifiability.png")
    report = "\n".join(L)
    print(report)
    with open(os.path.join(OUT, "report.txt"), "w") as fh:
        fh.write(report + "\n")


if __name__ == "__main__":
    main()
