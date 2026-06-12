"""Cross-etiology transfer experiment.

Trains on RRMAP2 (relapse-remitting EAE, spinal cord) and evaluates:
  1. RRMAP2 held-out val (within-domain upper bound)
  2. Optic nerve EAE (same immune etiology, different anatomy)
  3. mtDNA-DSB oligodendrocyte model (inside-out / non-immune etiology)

python scripts/transfer_experiment.py --config configs/transfer_experiment.yaml

The script produces:
  runs/<run_name>/
    best.pt, encoder.pkl, hvg_genes.npy, run_manifest.json  (from training)
    rrmap2_val_attention.npz
    optic_nerve_attention.npz
    mtdna_dsb_attention.npz
    transfer_results.json
    transfer_report.txt
    figures/
      rho_comparison.pdf
      attention_top_cells.pdf   (if cell_type_col is set)

Usage notes
-----------
*   The val set for step 1 comes directly from train.py's animal-level split.
    We re-run evaluate() on the same val bags to get a consistent EvalResult
    object with attention arrays (train.py doesn't return bag-level detail).

*   The label_order for optic_nerve and mtdna_dsb must be specified in the
    config even if they are shorter / differently named than the RRMAP2 labels.
    Spearman rho is computed between the *ranking* of predicted scores and the
    *ranking* of target labels — absolute scale mismatch is irrelevant.

*   Set cell_type_col to the obs column that holds cell-type annotations if you
    want the attention figure.  Leave null to skip it.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "source": {
        "h5ad_path": "data/rrmap2.h5ad",
        "train_config": "configs/rrmap2_stage.yaml",
        "run_dir": "runs/transfer_experiment",
    },
    "targets": [
        {
            "tag": "optic_nerve",
            "h5ad_path": "data/optic_nerve.h5ad",
            "obs_animal_id": "sample_name",
            "obs_section_id": "meta_sample_id",
            "obs_label": "stage",
            # Adjust to your actual optic nerve label vocabulary:
            "label_order": [
                "naive", "EAE_pre", "EAE_peak", "EAE_late",
            ],
            "layer": "counts",
        },
        {
            "tag": "mtdna_dsb",
            "h5ad_path": "data/mtdna_dsb.h5ad",
            "obs_animal_id": "animal_id",
            "obs_section_id": "section_id",
            "obs_label": "timepoint",
            # Adjust to your actual mtDNA-DSB label vocabulary:
            "label_order": [
                "WT", "2wk_DSB", "4wk_DSB", "8wk_DSB",
            ],
            "layer": "counts",
        },
    ],
    "eval": {
        "max_cells_per_bag": 2048,
        "n_boot": 2000,
        "n_perm": 5000,
        "cell_type_col": None,   # set to obs column name for attention figures
        "device": "auto",
        "seed": 0,
    },
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(cfg: dict) -> None:
    src = cfg["source"]
    ev_cfg = cfg["eval"]
    run_dir = src["run_dir"]
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "figures"), exist_ok=True)

    # ------------------------------------------------------------------ Train
    print("\n" + "=" * 60)
    print("STEP 1 — Training on source dataset (RRMAP2)")
    print("=" * 60)

    from immunotransformer.config import Config
    from immunotransformer.train import train

    train_cfg = Config.from_yaml(src["train_config"])
    train_cfg.train.out_dir = run_dir
    best_metrics = train(train_cfg)

    # ------------------------------------------------------------------ Val (source)
    print("\n" + "=" * 60)
    print("STEP 2 — Within-domain evaluation (RRMAP2 val)")
    print("=" * 60)

    from immunotransformer.evaluate import TransferEvaluator

    evaluator = TransferEvaluator.from_run(run_dir, device=ev_cfg["device"])

    # Score only the held-out val animals, so this is a genuine within-domain
    # reference and not contaminated by the training bags. Falls back to the
    # whole set (with a warning) for older runs whose manifest predates this.
    val_animals = evaluator.manifest.get("val_animals")
    if val_animals:
        restrict = set(val_animals)
    else:
        restrict = None
        print("  [warn] manifest has no val_animals — within-domain eval will "
              "include TRAINING bags and be optimistic. Re-run train.py.")

    obs = train_cfg.obs
    rrmap2_result = evaluator.evaluate(
        h5ad_path=src["h5ad_path"],
        obs_animal_id=obs.animal_id,
        obs_section_id=obs.section_id,
        obs_label=obs.label,
        target_label_order=obs.label_order,
        layer=train_cfg.data.layer,
        tag="rrmap2_val",
        max_cells_per_bag=ev_cfg["max_cells_per_bag"],
        seed=ev_cfg["seed"],
        restrict_animals=restrict,
    )

    # ------------------------------------------------------------------ Targets
    all_results = [rrmap2_result]

    for target_cfg in cfg["targets"]:
        print(f"\n{'=' * 60}")
        print(f"STEP {2 + len(all_results)} — Transfer: {target_cfg['tag']}")
        print("=" * 60)

        result = evaluator.evaluate(
            h5ad_path=target_cfg["h5ad_path"],
            obs_animal_id=target_cfg["obs_animal_id"],
            obs_section_id=target_cfg["obs_section_id"],
            obs_label=target_cfg["obs_label"],
            target_label_order=target_cfg["label_order"],
            layer=target_cfg.get("layer", "counts"),
            tag=target_cfg["tag"],
            max_cells_per_bag=ev_cfg["max_cells_per_bag"],
            seed=ev_cfg["seed"],
        )
        all_results.append(result)

    # ------------------------------------------------------------------ Stats
    print("\n" + "=" * 60)
    print("STEP — Statistical comparison")
    print("=" * 60)

    from immunotransformer.stats import full_report

    report = full_report(
        all_results,
        n_boot=ev_cfg["n_boot"],
        n_perm=ev_cfg["n_perm"],
        reference_tag="rrmap2_val",
        seed=ev_cfg["seed"],
    )

    report_path = os.path.join(run_dir, "transfer_report.txt")
    with open(report_path, "w") as fh:
        fh.write(report["summary_table"])

    results_path = os.path.join(run_dir, "transfer_results.json")
    with open(results_path, "w") as fh:
        json.dump(
            {
                "best_training_metrics": best_metrics,
                "per_dataset": report["per_dataset"],
                "pairwise": report["pairwise"],
            },
            fh,
            indent=2,
        )
    print(f"\nResults saved -> {results_path}")
    print(f"Report saved  -> {report_path}")

    # ------------------------------------------------------------------ Figures
    _plot_rho_comparison(report["per_dataset"], run_dir)
    if ev_cfg.get("cell_type_col"):
        _plot_attention_celltypes(
            all_results,
            ev_cfg["cell_type_col"],
            run_dir,
        )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _plot_rho_comparison(per_dataset: list[dict], run_dir: str) -> None:
    """Bar chart: Spearman rho with 95% bootstrap CI per dataset."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[skip] matplotlib not available — skipping rho comparison plot")
        return

    tags = [d["tag"] for d in per_dataset]
    rhos = [d["rho"] for d in per_dataset]
    ci_lo = [d["ci_lo"] for d in per_dataset]
    ci_hi = [d["ci_hi"] for d in per_dataset]
    yerr_lo = [r - lo for r, lo in zip(rhos, ci_lo)]
    yerr_hi = [hi - r for r, hi in zip(rhos, ci_hi)]

    # Colour by etiology: rrmap2_val = blue, optic_nerve = teal, mtdna_dsb = coral
    palette = {
        "rrmap2_val": "#4C72B0",
        "optic_nerve": "#55A868",
        "mtdna_dsb": "#C44E52",
    }
    colors = [palette.get(t, "#8172B2") for t in tags]

    fig, ax = plt.subplots(figsize=(max(4, len(tags) * 1.6), 4))
    x = np.arange(len(tags))
    ax.bar(x, rhos, color=colors, alpha=0.85, zorder=2)
    ax.errorbar(
        x, rhos,
        yerr=[yerr_lo, yerr_hi],
        fmt="none", color="black", capsize=5, linewidth=1.5, zorder=3,
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=20, ha="right")
    ax.set_ylabel("Spearman ρ (bag-level)")
    ax.set_title("Disease stage prediction: within-domain vs. transfer")
    ax.set_ylim(-0.1, 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    fig_path = os.path.join(run_dir, "figures", "rho_comparison.pdf")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {fig_path}")


def _plot_attention_celltypes(
    results,
    cell_type_col: str,
    run_dir: str,
) -> None:
    """For each dataset, plot mean attention weight per cell type.

    Requires that the target AnnData is still in memory (which it isn't after
    evaluate() returns).  In practice, call this from a notebook after
    re-loading the .h5ad files and the saved attention .npz arrays.

    This function generates a template notebook cell that the user can adapt.
    """
    nb_path = os.path.join(run_dir, "figures", "attention_analysis_template.py")
    template = '''"""
Attention weight analysis template.
Run this after transfer_experiment.py to overlay attention weights with cell types.
Requires the target .h5ad files to be reloaded.
"""
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd

run_dir = "{run_dir}"
cell_type_col = "{cell_type_col}"

datasets = [
    dict(tag="rrmap2_val",   h5ad="data/rrmap2.h5ad"),
    dict(tag="optic_nerve",  h5ad="data/optic_nerve.h5ad"),
    dict(tag="mtdna_dsb",    h5ad="data/mtdna_dsb.h5ad"),
]

fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 5), sharey=False)

for ax, ds in zip(axes, datasets):
    adata = ad.read_h5ad(ds["h5ad"])
    attn_npz = np.load(f"{{run_dir}}/{{ds['tag']}}_attention.npz", allow_pickle=True)

    # Aggregate attention per cell type across all bags
    cell_scores = np.zeros(adata.n_obs)
    cell_counts = np.zeros(adata.n_obs, dtype=int)
    for key in attn_npz.files:
        # key format: "bag_<section_id>"
        # The npz doesn't store cell indices — join via obs
        pass  # TODO: extend evaluate() to save cell_idx alongside attn_weights

    # Placeholder: use the BagResult.cell_idx from the EvalResult object
    # (returned by evaluator.evaluate() in memory)
    ax.set_title(ds["tag"])
    ax.set_xlabel("Mean attention weight")
    ax.set_ylabel("Cell type")

plt.tight_layout()
plt.savefig(f"{{run_dir}}/figures/attention_top_cells.pdf", bbox_inches="tight")
print("Saved attention figure.")
'''.format(run_dir=run_dir, cell_type_col=cell_type_col)

    with open(nb_path, "w") as fh:
        fh.write(template)
    print(f"[template] attention analysis template -> {nb_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Cross-etiology transfer experiment for Immunoformer"
    )
    ap.add_argument(
        "--config",
        default=None,
        help="Path to transfer_experiment.yaml. Uses built-in defaults if omitted.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved config and exit without running training.",
    )
    args = ap.parse_args()

    if args.config:
        with open(args.config) as fh:
            user_cfg = yaml.safe_load(fh) or {}
        cfg = {**DEFAULT_CONFIG, **user_cfg}
        # Deep-merge nested dicts
        for key in ("source", "eval"):
            cfg[key] = {**DEFAULT_CONFIG.get(key, {}), **user_cfg.get(key, {})}
        if "targets" in user_cfg:
            cfg["targets"] = user_cfg["targets"]
    else:
        cfg = DEFAULT_CONFIG

    if args.dry_run:
        import pprint
        print("Resolved config:")
        pprint.pprint(cfg)
        return

    run(cfg)


if __name__ == "__main__":
    main()
