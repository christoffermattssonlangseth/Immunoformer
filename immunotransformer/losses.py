"""Ordinal regression via CORAL (Cao et al. 2019).

EAE timepoint is ordered (control < pre_onset < onset < peak < late...), so we
predict it as K-1 *cumulative* binary tasks ("is the stage > k?") with shared
weights and rank-consistent thresholds — strictly better than treating the
stages as unordered classes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoralHead(nn.Module):
    """Shared projection to a scalar + (K-1) ordered bias thresholds."""

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.proj = nn.Linear(in_dim, 1, bias=False)
        self.thresholds = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # logits[b, k] = w·x_b - threshold_k  -> [B, K-1]
        return self.proj(x) - self.thresholds


def coral_targets(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """y -> binary levels: levels[b, k] = 1 if label_b > k else 0."""
    ks = torch.arange(num_classes - 1, device=labels.device)
    return (labels.unsqueeze(1) > ks.unsqueeze(0)).float()


def coral_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    targets = coral_targets(labels, logits.shape[1] + 1)
    return F.binary_cross_entropy_with_logits(logits, targets)


@torch.no_grad()
def coral_predict(logits: torch.Tensor) -> torch.Tensor:
    """Predicted level = number of thresholds passed."""
    return (torch.sigmoid(logits) > 0.5).sum(dim=1)
