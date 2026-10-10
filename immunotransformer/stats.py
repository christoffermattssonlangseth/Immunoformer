"""Statistical framework for comparing transfer performance across MS models.

All inference is at the *bag* (section) level — not the cell level — because
that is the unit of independence in this MIL design.  Mixing within a bag
would inflate sample size and underestimate variance.

Provided functions
------------------
bootstrap_spearman(preds, trues, n_boot, seed)
    Bootstrap CI for Spearman rho by resampling bags with replacement.

permutation_test(preds, trues, n_perm, seed)
    One-sided permutation test: H0: rho <= 0 (model no better than chance).

compare_rhos(r1, n1, r2, n2)
    Fisher z-test for H0: rho1 == rho2.  Returns z-statistic and two-sided p.

full_report(eval_results)
    Accepts a list of EvalResult objects (one per dataset) and returns a
    summary DataFrame with CIs and pairwise comparisons.

Design rationale
----------------
We resample bags (sections) rather than cells because:
  - Cells within a bag are not independent (same tissue section, same animal).
  - Animals in val were held out during training, so bags are approximately
    independent observations for the purposes of evaluating generalisation.
  - The number of bags is the effective sample size for statistical inference.

For the cross-etiology comparison, we use Spearman rho rather than MAE because
the absolute stage scale differs between source and target label spaces.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy.stats import norm, spearmanr


# ---------------------------------------------------------------------------
# Core bootstrap
# ---------------------------------------------------------------------------

def bootstrap_spearman(
    preds: np.ndarray,
    trues: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Bootstrap confidence interval for Spearman rho at the bag level.

    Parameters
    ----------
    preds : array-like, shape (n_bags,)
        Model predictions (continuous or integer stage scores).
    trues : array-like, shape (n_bags,)
        Ground-truth ordinal labels.
    n_boot : int
        Number of bootstrap resamples.
    ci : float
        Width of the central CI (e.g. 0.95 for 95%).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict with keys: rho, ci_lo, ci_hi, se, n_bags
    """
    preds = np.asarray(preds, dtype=float)
    trues = np.asarray(trues, dtype=float)
    n = len(preds)

    if n < 4:
        warnings.warn(
            f"Only {n} bags — bootstrap CI will be unreliable.", stacklevel=2
        )

    rho_obs = float(spearmanr(preds, trues).correlation)

    rng = np.random.default_rng(seed)
    boot_rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bp, bt = preds[idx], trues[idx]
        if len(set(bt)) < 2:
            boot_rhos[i] = np.nan
            continue
        boot_rhos[i] = spearmanr(bp, bt).correlation

    boot_rhos = boot_rhos[~np.isnan(boot_rhos)]
    alpha = 1.0 - ci
    ci_lo = float(np.percentile(boot_rhos, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_rhos, 100 * (1 - alpha / 2)))
    se = float(np.std(boot_rhos))

    return {
        "rho": rho_obs,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "se": se,
        "n_bags": n,
        "ci_level": ci,
    }


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def permutation_test(
    preds: np.ndarray,
    trues: np.ndarray,
    n_perm: int = 5000,
    seed: int = 0,
) -> dict:
    """One-sided permutation test: H0: rho <= 0 (random association).

    Shuffles the true labels and computes the fraction of permuted rhos that
    exceed the observed rho.  This gives a p-value for the hypothesis that the
    model is informative about disease stage.

    Parameters
    ----------
    preds, trues : array-like (n_bags,)
    n_perm : int
    seed : int

    Returns
    -------
    dict with keys: rho_obs, p_value, n_perm, n_bags
    """
    preds = np.asarray(preds, dtype=float)
    trues = np.asarray(trues, dtype=float)

    rho_obs = float(spearmanr(preds, trues).correlation)
    rng = np.random.default_rng(seed)
    null_rhos = np.empty(n_perm)

    for i in range(n_perm):
        shuffled = rng.permutation(trues)
        if len(set(shuffled)) < 2:
            null_rhos[i] = 0.0
        else:
            null_rhos[i] = spearmanr(preds, shuffled).correlation

    p_value = float((null_rhos >= rho_obs).mean())

    return {
        "rho_obs": rho_obs,
        "p_value": p_value,
        "n_perm": n_perm,
        "n_bags": len(preds),
    }


# ---------------------------------------------------------------------------
# Compare two Spearman correlations
# ---------------------------------------------------------------------------

def compare_rhos(
    rho1: float,
    n1: int,
    rho2: float,
    n2: int,
) -> dict:
    """Fisher z-test: H0 rho1 == rho2.  Two-sided.

    Used to compare, e.g., within-domain validation rho vs transfer rho.
    A significant result (p < 0.05) means the two settings produce
    meaningfully different ordinal correlations.

    Parameters
    ----------
    rho1, rho2 : float
        Observed Spearman correlations.
    n1, n2 : int
        Number of bags in each evaluation.

    Returns
    -------
    dict with keys: z_stat, p_value, rho1, rho2, n1, n2
    """
    # Clip to avoid arctanh singularity
    r1 = np.clip(rho1, -0.9999, 0.9999)
    r2 = np.clip(rho2, -0.9999, 0.9999)

    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)

    # Standard error of the difference under H0
    se = np.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    z_stat = (z1 - z2) / se
    p_value = float(2 * norm.sf(abs(z_stat)))  # two-sided

    return {
        "z_stat": float(z_stat),
        "p_value": p_value,
        "rho1": rho1,
        "n1": n1,
        "rho2": rho2,
        "n2": n2,
    }


def paired_bootstrap_delta(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    ref: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict:
    """Paired bootstrap of rho(pred_a, ref) - rho(pred_b, ref) over the SAME units.

    Use this instead of `compare_rhos` when both predictions are scored on the same
    animals: resampling units jointly keeps the correlation between the two rhos, which
    the independent-samples Fisher test ignores. Returns the observed difference, its
    percentile CI and the fraction of resamples with difference <= 0.
    """
    a, b, r = (np.asarray(v, dtype=float) for v in (pred_a, pred_b, ref))
    if not (len(a) == len(b) == len(r)):
        raise ValueError("pred_a, pred_b and ref must have the same length")
    rng = np.random.default_rng(seed)
    n = len(r)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        da, db = spearmanr(a[i], r[i]).statistic, spearmanr(b[i], r[i]).statistic
        if np.isfinite(da) and np.isfinite(db):
            diffs.append(da - db)
    diffs = np.asarray(diffs)
    alpha = (1 - ci) / 2
    return {
        "diff": float(spearmanr(a, r).statistic - spearmanr(b, r).statistic),
        "ci_lo": float(np.quantile(diffs, alpha)),
        "ci_hi": float(np.quantile(diffs, 1 - alpha)),
        "frac_le_0": float((diffs <= 0).mean()),
        "n": n,
        "n_boot_valid": int(len(diffs)),
    }


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def full_report(
    eval_results,
    n_boot: int = 2000,
    n_perm: int = 5000,
    reference_tag: Optional[str] = None,
    seed: int = 0,
) -> dict:
    """Compute per-dataset statistics and pairwise comparisons.

    Parameters
    ----------
    eval_results : list of EvalResult
        One entry per evaluation run (e.g., rrmap2_val, optic_nerve, mtdna_dsb).
    n_boot : int
        Bootstrap resamples for CI estimation.
    n_perm : int
        Permutation resamples for null p-values.
    reference_tag : str or None
        Tag of the within-domain result to use as the comparison baseline for
        Fisher z-tests.  Defaults to the first result in the list.
    seed : int

    Returns
    -------
    dict with:
        "per_dataset" : list of dicts (tag, rho, ci_lo, ci_hi, p_permutation)
        "pairwise"    : list of dicts (tag_a, tag_b, z_stat, p_fisher)
        "summary_table" : printable string
    """
    from itertools import combinations

    if not eval_results:
        raise ValueError("eval_results is empty.")

    ref_tag = reference_tag or eval_results[0].tag

    per_dataset = []
    for res in eval_results:
        preds = np.array([b.pred_score for b in res.bags])
        trues = np.array([b.true_label for b in res.bags])

        boot = bootstrap_spearman(preds, trues, n_boot=n_boot, seed=seed)
        perm = permutation_test(preds, trues, n_perm=n_perm, seed=seed)

        per_dataset.append(
            {
                "tag": res.tag,
                "n_bags": res.n_bags,
                "rho": boot["rho"],
                "ci_lo": boot["ci_lo"],
                "ci_hi": boot["ci_hi"],
                "se": boot["se"],
                "mae": res.mae,
                "accuracy": res.accuracy,
                "p_permutation": perm["p_value"],
            }
        )

    pairwise = []
    for a, b in combinations(per_dataset, 2):
        cmp = compare_rhos(a["rho"], a["n_bags"], b["rho"], b["n_bags"])
        pairwise.append(
            {
                "tag_a": a["tag"],
                "tag_b": b["tag"],
                "z_stat": cmp["z_stat"],
                "p_fisher": cmp["p_value"],
            }
        )

    # Build a readable summary table
    lines = [
        "\n=== Transfer benchmark: Spearman rho (bootstrapped 95% CI) ===\n",
        f"{'Dataset':<25} {'Bags':>5} {'rho':>6} {'95% CI':>18} {'p(perm)':>10} {'MAE':>7}",
        "-" * 75,
    ]
    for d in per_dataset:
        ci_str = f"[{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]"
        lines.append(
            f"{d['tag']:<25} {d['n_bags']:>5} {d['rho']:>+6.3f} "
            f"{ci_str:>18} {d['p_permutation']:>10.4f} {d['mae']:>7.3f}"
        )

    lines.append("\n--- Pairwise Fisher z-tests ---")
    for p in pairwise:
        sig = "**" if p["p_fisher"] < 0.01 else ("*" if p["p_fisher"] < 0.05 else "ns")
        lines.append(
            f"  {p['tag_a']} vs {p['tag_b']}: "
            f"z={p['z_stat']:+.3f}  p={p['p_fisher']:.4f}  {sig}"
        )
    lines.append("")

    summary = "\n".join(lines)
    print(summary)

    return {
        "per_dataset": per_dataset,
        "pairwise": pairwise,
        "summary_table": summary,
    }
