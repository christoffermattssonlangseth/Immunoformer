"""ImmunoTransformer — Stage 1 baseline.

A frozen/pluggable per-cell encoder feeds a gated attention-MIL pool and an
ordinal (CORAL) head, predicting a section/animal-level label (e.g. EAE
timepoint) from a bag of Xenium cells. Animal-level train/val splitting is
enforced to avoid the per-animal leakage that would otherwise inflate scores.

See docs/le-quesne-framework-applications.md for the design rationale.
"""

from .config import Config

__all__ = ["Config"]
__version__ = "0.0.1"
