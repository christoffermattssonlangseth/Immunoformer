"""Gene-level two-clock decomposition on the CONTINUOUS day-of-sacrifice axis.

The companion `rr_cycle_oscillation.py` splits genes into oscillating vs ratchet using
discrete relapse STAGES (PEAK1/REM1/PEAK2/REM2/PEAK3) — its accrual signal is
floor_drift = REM2 - REM1, a two-point contrast between remission stages. This script
re-asks the same question on the same axis the duration *clock* regresses on:
`day_of_sacrifice` (11-49 dpi), continuous, with clinical severity partialled out.

For every gene we compute two partial correlations over the 33 RR animals:

  r(gene, day | score)   -> ACCRUAL / duration axis  (rises with time, severity removed)
  r(gene, score | day)   -> SEVERITY / oscillation axis (tracks the relapse swings, time removed)

This is the gene-level mirror of the multivariate clock's severity-orthogonalization:
a gene high on the day|score axis is one the clock reads as accruing; a gene high on the
score|day axis but flat on day|score oscillates with severity and resets — it cannot be
what a severity-orthogonalized duration clock keys on.

  partial r_xy.z = (r_xy - r_xz r_yz) / sqrt((1-r_xz^2)(1-r_yz^2))
  t = r sqrt(df/(1-r^2)),  df = n - 3  (one covariate partialled)

Reuses the cached RR pseudobulk (no 14GB reload) + the duration-clock meta for day.

    PYTHONPATH="$PWD" python scripts/duration_gene_decomposition.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

CACHE = "runs/rr_within_relapse/pseudobulk.npz"
META = "runs/duration_clock/rr_meta.csv"
CLOCK_GENES = "runs/duration_clock/clock_genes.csv"
OUT_DIR = "runs/duration_gene_decomposition"


def bh_fdr(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    q[order] = (p[order] * n) / (np.arange(n) + 1)
    q[order] = np.minimum.accumulate(q[order][::-1])[::-1]
    return np.clip(q, 0, 1)


def partial_corr_vec(X, y, z):
    """Partial corr of each column of X with y, controlling for z. Returns (r, p)."""
    def z_(a):
        a = a - a.mean(0)
        s = a.std(0)
        return a / np.where(s == 0, 1, s)
    Xz, yz, zz = z_(X), z_(y), z_(z)
    n = X.shape[0]
    r_xy = (Xz * yz[:, None]).mean(0)
    r_xz = (Xz * zz[:, None]).mean(0)
    r_yz = float((yz * zz).mean())
    denom = np.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))
    denom = np.where(denom < 1e-9, np.nan, denom)
    r = (r_xy - r_xz * r_yz) / denom
    r = np.clip(np.nan_to_num(r), -0.999, 0.999)
    df = n - 3  # n - 2 - 1 covariate
    t = r * np.sqrt(df / (1 - r ** 2))
    p = 2 * stats.t.sf(np.abs(t), df)
    return r, p


def main():
    os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)
    if not os.path.exists(CACHE):
        raise SystemExit(f"cache {CACHE} not found — run scripts/rr_within_relapse.py first.")
    d = np.load(CACHE, allow_pickle=True)
    pb, genes, animals = d["pb"], d["genes"].astype(str), d["animals"].astype(str)
    meta = pd.read_csv(META).set_index("sample_name")

    keep = np.array([a in meta.index for a in animals])
    pb, animals = pb[keep], animals[keep]
    day = meta.loc[animals, "day_of_sacrifice"].to_numpy(float)
    score = meta.loc[animals, "score_sacrifice"].to_numpy(float)
    n = len(animals)
    print(f"[load] {n} RR animals x {pb.shape[1]} genes | day {day.min():.0f}-{day.max():.0f} dpi "
          f"| corr(day,score)={np.corrcoef(day, score)[0, 1]:+.2f}")

    # the two orthogonalized axes
    r_day, p_day = partial_corr_vec(pb, day, score)          # accrual | severity removed
    r_score, p_score = partial_corr_vec(pb, score, day)      # severity | time removed
    # raw (uncontrolled) day correlation, for reference
    r_day_raw = np.array([stats.spearmanr(pb[:, j], day)[0] if np.ptp(pb[:, j]) > 0 else 0.0
                          for j in range(pb.shape[1])])

    df = pd.DataFrame({
        "gene": genes,
        "r_day_given_score": r_day, "p_day": p_day,
        "r_score_given_day": r_score, "p_score": p_score,
        "r_day_raw_spearman": np.nan_to_num(r_day_raw),
    })
    df["q_day"] = bh_fdr(df["p_day"].to_numpy())
    df["q_score"] = bh_fdr(df["p_score"].to_numpy())

    # classify on the continuous axes (mirrors the stage acute/ratchet split)
    ACC, SEV = 0.30, 0.30  # partial-r thresholds
    df["class"] = "other"
    accrue = (df.r_day_given_score >= ACC)
    decline = (df.r_day_given_score <= -ACC)
    sev_only = (df.r_score_given_day.abs() >= SEV) & (df.r_day_given_score.abs() < ACC)
    df.loc[sev_only, "class"] = "severity_oscillating"
    df.loc[accrue, "class"] = "accrual_up"
    df.loc[decline, "class"] = "accrual_down"

    # cross-check vs the supervised duration clock signature
    clock = pd.read_csv(CLOCK_GENES) if os.path.exists(CLOCK_GENES) else pd.DataFrame(columns=["gene", "coef"])
    clock_set = set(clock.gene)
    df["in_clock"] = df.gene.isin(clock_set)

    top_accrue = df.sort_values("r_day_given_score", ascending=False).head(25)
    top_decline = df.sort_values("r_day_given_score").head(25)
    sv_df = df[df["class"] == "severity_oscillating"]
    top_sev = sv_df.reindex(
        sv_df.r_score_given_day.abs().sort_values(ascending=False).index).head(25)

    # agreement: does the gene-level day axis line up with the clock coefficients?
    if len(clock_set):
        merged = df.merge(clock, on="gene", how="inner")
        rho_clock = float(stats.spearmanr(merged.r_day_given_score, merged.coef)[0])
        sign_agree = float((np.sign(merged.r_day_given_score) == np.sign(merged.coef)).mean())
    else:
        rho_clock, sign_agree, merged = np.nan, np.nan, pd.DataFrame()

    results = {
        "n_animals": n,
        "day_range": [float(day.min()), float(day.max())],
        "corr_day_score": float(np.corrcoef(day, score)[0, 1]),
        "thresholds": {"accrual_partial_r": ACC, "severity_partial_r": SEV},
        "n_accrual_up": int(accrue.sum()),
        "n_accrual_down": int(decline.sum()),
        "n_severity_oscillating": int(sev_only.sum()),
        "n_day_sig_q05": int((df.q_day < 0.05).sum()),
        "clock_agreement": {
            "spearman_day_partial_vs_clock_coef": rho_clock,
            "sign_agreement_frac": sign_agree,
            "n_clock_genes_overlap": int(len(merged)),
        },
        "note": ("r_day_given_score is the severity-orthogonalized accrual axis (the gene-level "
                 "duration clock); r_score_given_day is the time-orthogonalized severity axis "
                 "(oscillation). day vs score are near-orthogonal across animals, so a gene can "
                 "load on one, the other, or both."),
        "top_accrual_up": top_accrue[["gene", "r_day_given_score", "r_score_given_day",
                                      "q_day", "in_clock"]].to_dict("records"),
        "top_accrual_down": top_decline[["gene", "r_day_given_score", "r_score_given_day",
                                         "q_day", "in_clock"]].to_dict("records"),
        "top_severity_oscillating": top_sev[["gene", "r_score_given_day", "r_day_given_score",
                                             "q_score", "in_clock"]].to_dict("records"),
    }
    with open(os.path.join(OUT_DIR, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    df.sort_values("r_day_given_score", ascending=False).to_csv(
        os.path.join(OUT_DIR, "gene_day_decomposition.csv"), index=False)

    _plot(df, OUT_DIR)
    _report(results, df)
    print(f"[done] -> {OUT_DIR}/ (results.json, gene_day_decomposition.csv, figures/)")


def _plot(df, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.scatter(df.r_score_given_day, df.r_day_given_score, s=7, c="#cfcfcf", zorder=1)
    up = df[df["class"] == "accrual_up"]
    dn = df[df["class"] == "accrual_down"]
    sv = df[df["class"] == "severity_oscillating"]
    ax.scatter(sv.r_score_given_day, sv.r_day_given_score, s=16, c="#e8590c", alpha=.7,
               zorder=2, label=f"severity / oscillating (n={len(sv)})")
    ax.scatter(up.r_score_given_day, up.r_day_given_score, s=18, c="#c0392b", alpha=.8,
               zorder=3, label=f"accrue with day ↑ (n={len(up)})")
    ax.scatter(dn.r_score_given_day, dn.r_day_given_score, s=18, c="#2c6fbb", alpha=.8,
               zorder=3, label=f"decline with day ↓ (n={len(dn)})")
    lab = pd.concat([
        up.sort_values("r_day_given_score", ascending=False).head(8),
        dn.sort_values("r_day_given_score").head(8),
        sv.reindex(sv.r_score_given_day.abs().sort_values(ascending=False).index).head(6),
    ])
    for r in lab.itertuples():
        ax.annotate(r.gene, (r.r_score_given_day, r.r_day_given_score), fontsize=7.2,
                    xytext=(3, 2), textcoords="offset points")
    ax.axhline(0, color="#aaa", lw=.6)
    ax.axvline(0, color="#aaa", lw=.6)
    for v in (0.30, -0.30):
        ax.axhline(v, color="#c0392b", lw=.5, ls="--")
        ax.axvline(v, color="#e8590c", lw=.5, ls="--")
    ax.set_xlabel("r(gene, score | day)   →   severity / oscillation axis")
    ax.set_ylabel("r(gene, day | score)   →   accrual / duration axis")
    ax.set_title("Gene-level two clocks on the continuous day axis\n"
                 "(partial correlations, severity-orthogonalized)", fontsize=12)
    ax.legend(fontsize=8.5, frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figures", "day_decomposition.png"),
                dpi=135, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "figures", "day_decomposition.pdf"), bbox_inches="tight")
    plt.close(fig)


def _report(results, df):
    L = [f"\n=== Gene-level two-clock decomposition on day-of-sacrifice (n={results['n_animals']}) ==="]
    L.append(f"day {results['day_range'][0]:.0f}-{results['day_range'][1]:.0f} dpi  "
             f"corr(day,score)={results['corr_day_score']:+.2f}")
    L.append(f"accrual↑={results['n_accrual_up']}  accrual↓={results['n_accrual_down']}  "
             f"severity/oscillating={results['n_severity_oscillating']}  "
             f"(day q<0.05: {results['n_day_sig_q05']})")
    ca = results["clock_agreement"]
    L.append(f"clock agreement: Spearman(day-partial, clock coef)={ca['spearman_day_partial_vs_clock_coef']:+.2f}"
             f"  sign-agree={ca['sign_agreement_frac']:.2f}  over {ca['n_clock_genes_overlap']} clock genes")
    L.append("\n-- Top ACCRUE with day (severity removed) --")
    for r in results["top_accrual_up"][:12]:
        L.append(f"   {r['gene']:<14} r_day|score={r['r_day_given_score']:+.2f}  "
                 f"r_score|day={r['r_score_given_day']:+.2f}  q={r['q_day']:.2g}"
                 f"{'  [clock]' if r['in_clock'] else ''}")
    L.append("\n-- Top DECLINE with day (severity removed) --")
    for r in results["top_accrual_down"][:12]:
        L.append(f"   {r['gene']:<14} r_day|score={r['r_day_given_score']:+.2f}  "
                 f"r_score|day={r['r_score_given_day']:+.2f}  q={r['q_day']:.2g}"
                 f"{'  [clock]' if r['in_clock'] else ''}")
    L.append("\n-- Top SEVERITY / OSCILLATING (tracks score, flat on day) --")
    for r in results["top_severity_oscillating"][:12]:
        L.append(f"   {r['gene']:<14} r_score|day={r['r_score_given_day']:+.2f}  "
                 f"r_day|score={r['r_day_given_score']:+.2f}  q={r['q_score']:.2g}"
                 f"{'  [clock]' if r['in_clock'] else ''}")
    report = "\n".join(L)
    print(report)
    with open(os.path.join(OUT_DIR, "report.txt"), "w") as fh:
        fh.write(report + "\n")


if __name__ == "__main__":
    main()
