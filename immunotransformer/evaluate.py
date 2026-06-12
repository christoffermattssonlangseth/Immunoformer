"""Transfer evaluation: apply a trained Stage-1 model to a new dataset.

This module is the counterpart to train.py.  It loads the artifacts saved by
_save_run_artifacts() and applies identical preprocessing to a target dataset
that may have a different animal/section/label layout — and even a different
label vocabulary (cross-etiology transfer).

Typical usage
-------------
from immunotransformer.evaluate import TransferEvaluator

ev = TransferEvaluator.from_run("runs/rrmap2_stage")
results = ev.evaluate(
    h5ad_path="data/optic_nerve.h5ad",
    obs_animal_id="sample_name",
    obs_section_id="meta_sample_id",
    obs_label="stage",
    target_label_order=["EAE_d0", "EAE_d7", "EAE_d14", "EAE_d21"],
    layer="counts",
    tag="optic_nerve",
)
# results: EvalResult with .mae, .spearman, .bags (per-bag detail), .attention

Notes
-----
*   Gene subset alignment: the evaluator selects only the genes that were in the
    training HVG set (by name).  Genes missing from the target panel are
    zero-padded; genes present in target but absent from training are dropped.
    For same-panel Xenium data this is a no-op.

*   Label alignment: the evaluator maps target labels to integer ranks using
    `target_label_order`.  This does NOT need to match the source `label_order`
    — Spearman rho is computed between the model's predicted source-scale scores
    and the target's own rank integers.  This is the appropriate metric for
    cross-etiology transfer where absolute stage values are not comparable.

*   Attention: per-cell attention weights are returned for every bag, indexed
    back to the original cell indices in the target AnnData so you can overlay
    with cell-type annotations.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field
from typing import Optional

import anndata as ad
import numpy as np
import torch
from scipy.stats import spearmanr

from .data import BagDataset, build_bags, collate_single, normalize_expression
from .encoders import CellEncoder
from .model import GatedAttentionMIL
from .losses import coral_predict
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BagResult:
    section_id: str
    animal_id: str
    true_label: int          # integer rank in target_label_order
    pred_label: int          # CORAL prediction (integer rank in source space)
    pred_score: float        # continuous score = pred_label (same thing for now)
    attn_weights: np.ndarray  # [N_cells] soft attention over the bag's cells
    cell_idx: np.ndarray     # original row indices in target AnnData


@dataclass
class EvalResult:
    tag: str
    mae: float
    spearman: float
    accuracy: float
    n_bags: int
    bags: list[BagResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "tag": self.tag,
            "mae": self.mae,
            "spearman": self.spearman,
            "accuracy": self.accuracy,
            "n_bags": self.n_bags,
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class TransferEvaluator:
    """Loads a trained run and evaluates it on arbitrary target datasets."""

    def __init__(
        self,
        model: GatedAttentionMIL,
        encoder: CellEncoder,
        training_gene_names: np.ndarray,
        source_label_order: list[str],
        manifest: dict,
        device: torch.device,
        run_dir: str,
    ):
        self.model = model
        self.encoder = encoder
        self.training_gene_names = training_gene_names
        self.source_label_order = source_label_order
        self.manifest = manifest
        self.device = device
        self.run_dir = run_dir

    # ------------------------------------------------------------------
    @classmethod
    def from_run(
        cls,
        run_dir: str,
        device: str = "auto",
    ) -> "TransferEvaluator":
        """Load all artifacts saved by train._save_run_artifacts().

        Parameters
        ----------
        run_dir : str
            Directory that contains best.pt, encoder.pkl, hvg_genes.npy,
            run_manifest.json (produced by train.py).
        device : str
            "auto", "cpu", "cuda", or "mps".
        """
        _device = _resolve_device(device)

        # Manifest
        manifest_path = os.path.join(run_dir, "run_manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"run_manifest.json not found in {run_dir}. "
                "Re-run train.py with the patched version to generate it."
            )
        with open(manifest_path) as fh:
            manifest = json.load(fh)

        label_order: list[str] = manifest["label_order"]
        num_classes = len(label_order)

        # Encoder
        with open(os.path.join(run_dir, "encoder.pkl"), "rb") as fh:
            encoder: CellEncoder = pickle.load(fh)

        # Gene names
        hvg_genes = np.load(
            os.path.join(run_dir, "hvg_genes.npy"), allow_pickle=True
        )

        # Model — reconstruct architecture from manifest + encoder out_dim.
        # Older runs (pre-manifest-model-block) fall back to ModelConfig defaults.
        from .config import ModelConfig
        mcfg = ModelConfig()
        march = manifest.get("model", {})
        model = GatedAttentionMIL(
            in_dim=encoder.out_dim,
            num_classes=num_classes,
            proj_dim=march.get("proj_dim", mcfg.proj_dim),
            attn_dim=march.get("attn_dim", mcfg.attn_dim),
            dropout=march.get("dropout", mcfg.dropout),
        )
        state = torch.load(
            os.path.join(run_dir, "best.pt"),
            map_location=_device,
            weights_only=True,
        )
        model.load_state_dict(state)
        model.to(_device)
        model.eval()

        return cls(
            model=model,
            encoder=encoder,
            training_gene_names=hvg_genes,
            source_label_order=label_order,
            manifest=manifest,
            device=_device,
            run_dir=run_dir,
        )

    # ------------------------------------------------------------------
    def evaluate(
        self,
        h5ad_path: str,
        obs_animal_id: str,
        obs_section_id: str,
        obs_label: str,
        target_label_order: list[str],
        layer: Optional[str] = "counts",
        tag: str = "target",
        max_cells_per_bag: int = 2048,
        min_cells_per_bag: int = 64,
        seed: int = 0,
        save_attention: bool = True,
        restrict_animals: Optional[set] = None,
    ) -> EvalResult:
        """Run inference on a target dataset.

        Parameters
        ----------
        h5ad_path : str
            Path to the target .h5ad file.
        obs_animal_id : str
            Column in adata.obs that identifies the animal (for grouping).
        obs_section_id : str
            Column in adata.obs that identifies the section (bag unit).
        obs_label : str
            Column in adata.obs with the ordinal label.
        target_label_order : list[str]
            Ordered list of label values, from least to most severe.
            Does NOT need to match source_label_order.
        layer : str or None
            Raw counts layer name (same convention as training config).
        tag : str
            Human-readable name for this evaluation run (used in output files).
        max_cells_per_bag : int
            Cap on cells per bag (uses deterministic head slice in eval mode).
        min_cells_per_bag : int
            Bags smaller than this are skipped.
        save_attention : bool
            If True, save a <tag>_attention.npz to run_dir.
        restrict_animals : set or None
            If given, keep only bags whose animal_id is in this set. Used for the
            within-domain reference, so it evaluates the held-out val animals
            rather than the whole source set (which would include training bags).
        """
        print(f"\n[evaluate:{tag}] loading {h5ad_path}")
        adata = ad.read_h5ad(h5ad_path)
        print(f"  {adata.n_obs:,} cells x {adata.n_vars:,} genes")

        # 1. Align gene panel to the training HVG set
        adata = _align_genes(adata, self.training_gene_names)
        print(f"  gene panel aligned -> {adata.n_vars} genes")

        # 2. Normalise expression
        X = normalize_expression(adata, layer)

        # 3. Build a minimal Config-like namespace for build_bags / BagDataset
        from .config import Config, ObsSchema, DataConfig, ModelConfig, TrainConfig
        cfg = Config(
            obs=ObsSchema(
                animal_id=obs_animal_id,
                section_id=obs_section_id,
                label=obs_label,
                label_order=target_label_order,
            ),
            data=DataConfig(
                max_cells_per_bag=max_cells_per_bag,
                min_cells_per_bag=min_cells_per_bag,
                seed=seed,
            ),
        )

        bags = build_bags(adata, cfg)
        if restrict_animals is not None:
            bags = [b for b in bags if b.animal_id in restrict_animals]
            print(f"  restricted to {len(restrict_animals)} animals")
        if not bags:
            raise RuntimeError(
                f"No bags built for {tag}. "
                "Check obs column names and label_order."
            )
        print(f"  {len(bags)} bags built")

        # 4. Apply the TRAINING encoder (no re-fitting on target data)
        X_enc = self.encoder.transform(X)

        # 5. Inference
        dataset = BagDataset(bags, X_enc, cfg, train=False)
        loader = DataLoader(
            dataset, batch_size=1, shuffle=False, collate_fn=collate_single
        )

        bag_results: list[BagResult] = []
        preds_all, trues_all = [], []
        # collate_single yields the section_id as a plain string (batch size is
        # always 1 bag), so look bags up by that string directly.
        bag_by_section = {b.section_id: b for b in bags}

        with torch.no_grad():
            for cells, label, sec in loader:
                cells = cells.to(self.device)
                logits, attn = self.model(cells)
                p = coral_predict(logits).item()

                # Recover cell_idx and animal_id for this bag.
                bag = bag_by_section[sec]

                bag_results.append(
                    BagResult(
                        section_id=sec,
                        animal_id=bag.animal_id,
                        true_label=label.item(),
                        pred_label=int(p),
                        pred_score=float(p),
                        attn_weights=attn.detach().cpu().numpy(),
                        cell_idx=bag.cell_idx,
                    )
                )
                preds_all.append(p)
                trues_all.append(label.item())

        preds_arr = np.array(preds_all)
        trues_arr = np.array(trues_all)

        mae = float(np.abs(preds_arr - trues_arr).mean())
        acc = float((preds_arr.round().astype(int) == trues_arr).mean())
        rho = (
            float(spearmanr(preds_arr, trues_arr).correlation)
            if len(set(trues_all)) > 1
            else float("nan")
        )

        print(
            f"  [result] MAE={mae:.3f}  acc={acc:.3f}  "
            f"spearman_rho={rho:.3f}  n_bags={len(bag_results)}"
        )

        result = EvalResult(
            tag=tag,
            mae=mae,
            spearman=rho,
            accuracy=acc,
            n_bags=len(bag_results),
            bags=bag_results,
        )

        if save_attention:
            attn_path = os.path.join(self.run_dir, f"{tag}_attention.npz")
            np.savez(
                attn_path,
                **{
                    f"bag_{r.section_id}": r.attn_weights
                    for r in bag_results
                },
            )
            print(f"  attention saved -> {attn_path}")

        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_device(choice: str) -> torch.device:
    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _align_genes(adata: ad.AnnData, training_genes: np.ndarray) -> ad.AnnData:
    """Subset / zero-pad target adata to match training gene panel.

    All Xenium experiments in this project use the same 5101-gene mouse panel,
    so in practice this is a no-op.  The function is here for robustness in
    case future experiments use a different panel version.

    Parameters
    ----------
    adata : AnnData
        Target dataset.
    training_genes : np.ndarray of str
        Gene names in the order used during training (post-HVG selection).

    Returns
    -------
    AnnData with vars re-ordered / zero-padded to match training_genes.
    """
    target_genes = set(adata.var_names)
    common = [g for g in training_genes if g in target_genes]
    missing = [g for g in training_genes if g not in target_genes]

    if missing:
        import warnings
        warnings.warn(
            f"{len(missing)} training genes absent from target panel "
            f"(e.g. {missing[:3]}). They will be zero-padded.",
            stacklevel=2,
        )

    if not missing:
        # Fast path: just reorder
        return adata[:, list(training_genes)]

    # Slow path: build a new matrix with zero columns for missing genes
    from scipy.sparse import issparse, hstack, csr_matrix

    X_common = adata[:, common].X
    zero_block = csr_matrix((adata.n_obs, len(missing)), dtype=np.float32)

    # Build column-order matching training_genes
    gene_to_col: dict[str, int] = {g: i for i, g in enumerate(training_genes)}
    common_positions = [gene_to_col[g] for g in common]
    missing_positions = [gene_to_col[g] for g in missing]

    import scipy.sparse as sp
    out_X = sp.lil_matrix((adata.n_obs, len(training_genes)), dtype=np.float32)
    for new_col, old_col in zip(common_positions, range(len(common))):
        out_X[:, new_col] = (
            X_common[:, old_col].toarray() if issparse(X_common) else X_common[:, old_col:old_col+1]
        )
    out_X = out_X.tocsr()

    import pandas as pd
    new_var = pd.DataFrame(index=list(training_genes))
    return ad.AnnData(X=out_X, obs=adata.obs.copy(), var=new_var)
