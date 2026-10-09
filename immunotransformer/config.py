"""Configuration schema for the Stage 1 baseline.

Everything that depends on *your* AnnData layout lives here, so adapting to the
real Xenium object means editing one block, not the code. Load from YAML with
`Config.from_yaml(path)`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import yaml


@dataclass
class ObsSchema:
    """Which `adata.obs` columns hold the metadata we need.

    Defaults are placeholders — set these to your real column names.
    """
    animal_id: str = "animal_id"      # grouping key for leakage-free splits
    section_id: str = "section_id"    # one bag = one section; falls back to animal_id if missing
    label: str = "timepoint"          # the ordinal target (e.g. EAE stage)
    # Ordered list of label categories, low -> high. Required for ordinal encoding.
    label_order: list[str] = field(default_factory=lambda: [
        "control", "pre_onset", "onset", "peak", "late_1", "late_2",
    ])
    region: Optional[str] = "region"  # optional covariate/label (may be None)
    model: Optional[str] = "model"    # disease model (optional)


@dataclass
class DataConfig:
    h5ad_path: str = "data/eae_xenium.h5ad"
    layer: Optional[str] = None        # adata.layers[...] for raw counts; None = adata.X
    max_cells_per_bag: int = 2048      # random subsample per section per epoch (memory)
    min_cells_per_bag: int = 64        # drop tiny sections
    encoder: str = "pca"               # "identity" | "pca" | "fm" (frozen foundation model)
    pca_dim: int = 64
    n_hvg: int = 0                     # 0 = use all genes; else select N highly-variable genes
    hvg_subsample: int = 50_000        # cells sampled to compute HVG stats; avoids the ~2x
                                       # full-panel RAM spike from copying the whole object
    val_fraction: float = 0.25         # fraction of *animals* held out
    seed: int = 0


@dataclass
class ModelConfig:
    proj_dim: int = 128                # per-cell projection before attention
    attn_dim: int = 64                 # gated-attention hidden dim
    dropout: float = 0.1
    head: str = "coral"                # "coral" (ordinal) | "regression" (continuous target);
                                       # train.py drives coral only; analysis/mil_loao.py
                                       # drives regression under LOAO folds


@dataclass
class TrainConfig:
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_accum: int = 4                # bags per optimizer step (batch_size is always 1 bag)
    device: str = "auto"              # "auto" | "cpu" | "cuda" | "mps"
    out_dir: str = "runs/baseline"
    log_every: int = 20


@dataclass
class Config:
    obs: ObsSchema = field(default_factory=ObsSchema)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def num_classes(self) -> int:
        return len(self.obs.label_order)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
        return cls(
            obs=ObsSchema(**(raw.get("obs") or {})),
            data=DataConfig(**(raw.get("data") or {})),
            model=ModelConfig(**(raw.get("model") or {})),
            train=TrainConfig(**(raw.get("train") or {})),
        )

    def to_yaml(self, path: str) -> None:
        with open(path, "w") as fh:
            yaml.safe_dump(asdict(self), fh, sort_keys=False)
