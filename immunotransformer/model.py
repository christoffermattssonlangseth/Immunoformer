"""Gated attention-MIL (Ilse et al. 2018) + CORAL ordinal head.

The attention weights are the interpretable readout: after training, the cells a
section's prediction leans on are the analogue of Le Quesne's HPC proportions —
cross-check them against known EAE populations (perivascular T cells, DAM
microglia, reactive astrocytes).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .losses import CoralHead


class GatedAttentionMIL(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, proj_dim: int = 128,
                 attn_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Gated attention: V (tanh) * U (sigmoid) -> w
        self.attn_V = nn.Linear(proj_dim, attn_dim)
        self.attn_U = nn.Linear(proj_dim, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1)
        self.head = CoralHead(proj_dim, num_classes)

    def forward(self, cells: torch.Tensor):
        """cells: [N, in_dim] for ONE section. Returns (logits[1,K-1], attn[N])."""
        h = self.proj(cells)                              # [N, P]
        a = self.attn_w(torch.tanh(self.attn_V(h)) * torch.sigmoid(self.attn_U(h)))
        a = torch.softmax(a, dim=0)                       # [N, 1]
        z = (a * h).sum(dim=0, keepdim=True)              # [1, P]
        logits = self.head(z)                             # [1, K-1]
        return logits, a.squeeze(1)
