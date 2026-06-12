"""ImmunoTransformer — Stage 1 baseline.

A frozen/pluggable per-cell encoder feeds a gated attention-MIL pool and an
ordinal (CORAL) head, predicting a section/animal-level label (e.g. EAE
timepoint) from a bag of Xenium cells. Animal-level train/val splitting is
enforced to avoid the per-animal leakage that would otherwise inflate scores.

See docs/le-quesne-framework-applications.md for the design rationale.
"""

# --- OpenMP duplicate-runtime guard (macOS conda) -------------------------
# This env ships several OpenMP runtimes: the conda libomp + Intel libiomp5,
# plus copies bundled inside torch and sklearn. When numba/sklearn/torch spin
# up their thread pools together the duplicate runtimes collide and the process
# hard-segfaults (exit 139) right after data load — the multi-session "RRMAP2
# segfault" blocker. Tolerating the duplicate AND pinning threads avoids it.
# These must be set before numpy/torch load, so they live at the top of the
# package import. setdefault() means an explicit env var still wins (e.g. raise
# OMP_NUM_THREADS for a heavier run). See HANDOFF.md.
import os as _os

for _k, _v in {
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMBA_NUM_THREADS": "1",
}.items():
    _os.environ.setdefault(_k, _v)
# --------------------------------------------------------------------------

from .config import Config

__all__ = ["Config"]
__version__ = "0.0.1"
