"""Phase 0 feasibility gate for the relapse-phase model (docs/relapse-phase-model-design.md).

Question: does the RR relapse cycle carry molecular PHASE information BEYOND clinical
severity, and can the direction of travel (ascending vs descending) be decoded at
matched severity? If neither holds, the relapse cycle is "just severity" and we do
NOT build the deep model.

Runs on the cached animal-level pseudobulk (runs/rr_within_relapse/pseudobulk.npz),
so it needs no model and no 14GB reload. Animal is the unit of independence ->
leave-one-animal-out (LOAO) CV throughout, with permutation nulls because n is tiny.

Two gate tests (non-circular: features are severity-residualized, label-blind PCs):

  TEST A  PHASE-BEYOND-SEVERITY (well powered): peak(13) vs remission(9).
          Severity separates these trivially, so we residualize each gene on
          score_sacrifice, PCA the residuals (label-blind), and ask whether the
          residual molecular structure STILL separates peak from remission.
          AUC > permutation null  =>  phase != severity.

  TEST B  DIRECTION AT MATCHED SEVERITY (the killer, thin): onset(4, ascending)
          vs remission(9, descending). These already overlap in severity
          (0.25-1.25), so this is the real "getting worse vs recovering" contrast.
          Reported with a loud n=4 caveat -> POC-grade, not a significance claim.

Plus an ILLUSTRATION (mildly circular, labelled): the decomposition double
dissociation -- the acute-oscillator program should separate peak/remission (phase)
but NOT REM1/REM2 (accrual); the ratchet program should do the opposite.

    python scripts/rr_phase_feasibility.py
"""

from __future__ import annotations

import json
import os

import numpy as np
from numpy.random import default_rng
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

CACHE = "runs/rr_within_relapse/pseudobulk.npz"
OSC_JSON = "runs/rr_cycle_oscillation/results.json"
OUT_DIR = "runs/rr_phase_feasibility"
ASCENDING = {"ONSET1", "ONSET2"}
APEX = {"PEAK1", "PEAK2", "PEAK3"}
DESCENDING = {"REMISSION1", "REMISSION2"}


def residualize_on_severity(pb, score):
    """Remove the linear effect of clinical severity from every gene (label-blind).

    Animals with missing score are dropped upstream. Returns residual matrix.
    """
    s = score.copy()
    s = (s - s.mean()) / (s.std() + 1e-9)
    X = np.c_[np.ones_like(s), s]                  # [n, 2]
    beta, *_ = np.linalg.lstsq(X, pb, rcond=None)  # [2, g]
    return pb - X @ beta


def loao_auc(feat, y, C=0.5):
    """Pooled leave-one-animal-out AUC of L2-logistic on `feat` predicting binary y."""
    n = len(y)
    proba = np.zeros(n)
    for i in range(n):
        tr = np.arange(n) != i
        if len(set(y[tr])) < 2:
            proba[i] = y[tr].mean()               # degenerate fold -> prior
            continue
        clf = LogisticRegression(C=C, max_iter=2000)
        clf.fit(feat[tr], y[tr])
        proba[i] = clf.predict_proba(feat[i:i + 1])[0, 1]
    return roc_auc_score(y, proba), proba


def perm_null(feat, y, obs_auc, B=2000, seed=0, C=0.5):
    """Permutation p-value: shuffle labels across animals, recompute LOAO AUC."""
    rng = default_rng(seed)
    ge = 1
    for _ in range(B):
        yp = rng.permutation(y)
        a, _ = loao_auc(feat, yp, C=C)
        ge += a >= obs_auc
    return ge / (B + 1)


def contrast(pb_res, pb_raw, score, stage, pos_set, neg_set, name, n_pc=8, B=2000):
    """One LOAO contrast: severity-only baseline vs severity-residualized molecular PCs."""
    m = np.isin(stage, list(pos_set | neg_set))
    y = np.isin(stage[m], list(pos_set)).astype(int)
    sc = score[m]
    res = pb_res[m]

    # severity-only baseline (does clinical score alone separate them?)
    auc_sev, _ = loao_auc(sc.reshape(-1, 1), y)

    # severity-residualized, label-blind PCs of the molecular residual
    k = min(n_pc, res.shape[0] - 1)
    pcs = PCA(n_components=k, random_state=0).fit_transform(
        (res - res.mean(0)) / (res.std(0) + 1e-9))
    auc_mol, proba = loao_auc(pcs, y)
    p_mol = perm_null(pcs, y, auc_mol, B=B)

    return {
        "name": name,
        "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
        "pos": sorted(pos_set), "neg": sorted(neg_set),
        "severity_only_auc": round(float(auc_sev), 3),
        "residualized_molecular_auc": round(float(auc_mol), 3),
        "perm_p": round(float(p_mol), 4),
        "n_pcs": k,
    }


def program_score(pb, genes, gene_list):
    gi = {g: i for i, g in enumerate(genes)}
    cols = [gi[g] for g in gene_list if g in gi]
    sub = pb[:, cols]
    z = (sub - sub.mean(0)) / (sub.std(0) + 1e-9)
    return z.mean(1), len(cols)


def main():
    os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)
    d = np.load(CACHE, allow_pickle=True)
    pb, genes, stage, score = d["pb"], d["genes"].astype(str), d["stage"].astype(str), d["score"].astype(float)
    keep = ~np.isnan(score)
    pb, stage, score = pb[keep], stage[keep], score[keep]
    print(f"[load] {pb.shape[0]} RR animals (with score) x {pb.shape[1]} genes")

    pb_res = residualize_on_severity(pb, score)

    # ---- gate tests ----
    A = contrast(pb_res, pb, score, stage, APEX, DESCENDING,
                 "TEST A: phase-beyond-severity (peak vs remission)")
    B = contrast(pb_res, pb, score, stage, ASCENDING, DESCENDING,
                 "TEST B: direction at matched severity (onset vs remission)")

    # also: raw (non-residualized) onset-vs-remission, to show severity is NOT doing it
    mB = np.isin(stage, list(ASCENDING | DESCENDING))
    yB = np.isin(stage[mB], list(ASCENDING)).astype(int)
    auc_raw_mol, _ = loao_auc(
        PCA(n_components=min(8, mB.sum() - 1), random_state=0).fit_transform(
            (pb[mB] - pb[mB].mean(0)) / (pb[mB].std(0) + 1e-9)), yB)

    # ---- decomposition double-dissociation (illustration; gene lists from same data) ----
    osc = json.load(open(OSC_JSON)) if os.path.exists(OSC_JSON) else None
    decomp = None
    if osc:
        osc_genes = [r["gene"] for r in osc["top_acute_oscillating"]]
        rat_genes = [r["gene"] for r in osc["top_ratchet_cumulative"]]
        osc_s, n_osc = program_score(pb, genes, osc_genes)
        rat_s, n_rat = program_score(pb, genes, rat_genes)

        def grp(S):
            return np.isin(stage, list(S))
        peak_m, rem_m = grp(APEX), grp(DESCENDING)
        rem1_m, rem2_m = stage == "REMISSION1", stage == "REMISSION2"
        decomp = {
            "n_oscillator_genes": n_osc, "n_ratchet_genes": n_rat,
            "oscillator_peak_minus_rem": round(float(osc_s[peak_m].mean() - osc_s[rem_m].mean()), 3),
            "oscillator_rem2_minus_rem1": round(float(osc_s[rem2_m].mean() - osc_s[rem1_m].mean()), 3),
            "ratchet_peak_minus_rem": round(float(rat_s[peak_m].mean() - rat_s[rem_m].mean()), 3),
            "ratchet_rem2_minus_rem1": round(float(rat_s[rem2_m].mean() - rat_s[rem1_m].mean()), 3),
        }

    # ---- Hal alone, severity-residualized ----
    gi = {g: i for i, g in enumerate(genes)}
    hal_res = pb_res[:, gi["Hal"]]
    hal = {
        "resid_Hal_peak_minus_rem": round(float(hal_res[np.isin(stage, list(APEX))].mean()
                                                 - hal_res[np.isin(stage, list(DESCENDING))].mean()), 3),
        "resid_Hal_onset_minus_rem": round(float(hal_res[np.isin(stage, list(ASCENDING))].mean()
                                                  - hal_res[np.isin(stage, list(DESCENDING))].mean()), 3),
    }

    results = {"n_animals": int(pb.shape[0]), "test_A": A, "test_B": B,
               "onset_vs_rem_raw_molecular_auc": round(float(auc_raw_mol), 3),
               "decomposition_illustration": decomp, "hal_residual": hal}

    # ---- gate verdict ----
    gate_A = A["residualized_molecular_auc"] > 0.65 and A["perm_p"] < 0.05
    gate_B = B["residualized_molecular_auc"] > 0.70  # thin; descriptive, not gated on p
    verdict = ("PROCEED — phase carries molecular signal beyond severity"
               if gate_A else
               "STOP — no phase signal beyond severity; the cycle is severity")
    results["gate"] = {"test_A_passes": bool(gate_A),
                       "test_B_suggestive": bool(gate_B), "verdict": verdict}

    # ---- report ----
    L = [f"\n=== Phase 0 feasibility gate (n={pb.shape[0]} RR animals, LOAO CV) ===",
         "Features = severity-residualized, label-blind PCs. Permutation nulls (B=2000).",
         f"\n-- {A['name']} --",
         f"   peak(n={A['n_pos']}) vs remission(n={A['n_neg']})",
         f"   severity-only AUC = {A['severity_only_auc']}  (trivially high: peaks are severe)",
         f"   severity-RESIDUALIZED molecular AUC = {A['residualized_molecular_auc']}  "
         f"(perm p={A['perm_p']})",
         f"   => {'phase signal SURVIVES severity removal' if gate_A else 'no phase beyond severity'}",
         f"\n-- {B['name']} --",
         f"   onset(n={B['n_pos']}, ascending) vs remission(n={B['n_neg']}, descending), "
         f"matched severity",
         f"   severity-only AUC = {B['severity_only_auc']}  (low: groups overlap in score)",
         f"   residualized molecular AUC = {B['residualized_molecular_auc']}  (perm p={B['perm_p']})",
         f"   raw (non-resid) molecular AUC = {results['onset_vs_rem_raw_molecular_auc']}",
         f"   *** n=4 onset — POC-grade, NOT a significance claim ***"]
    if decomp:
        L += [f"\n-- Decomposition double-dissociation (ILLUSTRATION; gene lists from same data) --",
              f"   oscillator program ({decomp['n_oscillator_genes']} genes): "
              f"peak-rem={decomp['oscillator_peak_minus_rem']:+.2f} (phase, expect BIG), "
              f"REM2-REM1={decomp['oscillator_rem2_minus_rem1']:+.2f} (accrual, expect ~0)",
              f"   ratchet program ({decomp['n_ratchet_genes']} genes): "
              f"peak-rem={decomp['ratchet_peak_minus_rem']:+.2f} (phase, expect ~0), "
              f"REM2-REM1={decomp['ratchet_rem2_minus_rem1']:+.2f} (accrual, expect >0)"]
    L += [f"\n-- Hal (severity-residualized) --",
          f"   resid peak-rem = {hal['resid_Hal_peak_minus_rem']:+.2f}   "
          f"resid onset-rem = {hal['resid_Hal_onset_minus_rem']:+.2f}",
          f"\n=== GATE: {verdict} ==="]
    report = "\n".join(L)
    print(report)

    with open(os.path.join(OUT_DIR, "report.txt"), "w") as fh:
        fh.write(report + "\n")
    with open(os.path.join(OUT_DIR, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    _plot(pb, pb_res, stage, score, OUT_DIR)
    print(f"\n[done] -> {OUT_DIR}/ (report.txt, results.json, figures/)")


def _plot(pb, pb_res, stage, score, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    cyc = ["PLP CFA", "ONSET1", "ONSET2", "PEAK1", "REMISSION1", "PEAK2", "REMISSION2", "PEAK3"]
    cmap = {s: plt.cm.twilight(i / len(cyc)) for i, s in enumerate(cyc)}

    def scatter(ax, X, title):
        z = (X - X.mean(0)) / (X.std(0) + 1e-9)
        p = PCA(n_components=2, random_state=0).fit_transform(z)
        for s in cyc:
            m = stage == s
            if m.sum():
                ax.scatter(p[m, 0], p[m, 1], color=cmap[s], label=s, s=60, edgecolor="k", linewidth=0.4)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.spines[["top", "right"]].set_visible(False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    scatter(axes[0], pb, "Raw pseudobulk PCA\n(PC1 ~ severity)")
    scatter(axes[1], pb_res, "Severity-residualized PCA\n(does phase remain?)")
    axes[1].legend(fontsize=8, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figures", "phase_pca.pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
