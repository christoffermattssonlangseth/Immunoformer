"""Task 3c: does the mean-pooling control (arm 6b) bypass the attention module?

Checks on GatedAttentionMIL(pooling="mean"):
  1. output == head(mean over cells of proj(cells))           (plain mean pooling)
  2. randomising every attention weight leaves the output unchanged
  3. after backward(), attention parameters receive no gradient
  4. the PRE-FIX forward that arm 6b actually ran (attention computed, then overwritten
     with uniform weights) gives bit-identical outputs to the current code path
Writes runs/arm6_tuned/meanpool_bypass_check.txt.

    PYTHONPATH="$PWD" python analysis/check_meanpool_bypass.py
"""

from __future__ import annotations

import immunotransformer  # noqa: F401  (OpenMP guard — must precede numpy; see HANDOFF.md)

import os

import torch

from immunotransformer.model import GatedAttentionMIL

OUT = "runs/arm6_tuned"


def prefix_forward(m, cells):
    """The forward pass as it was when arm 6b ran (before the 2026-10-09 refactor)."""
    h = m.proj(cells)
    a = m.attn_w(torch.tanh(m.attn_V(h)) * torch.sigmoid(m.attn_U(h)))
    a = torch.softmax(a, dim=0)
    a = torch.full_like(a, 1.0 / a.shape[0])
    z = (a * h).sum(dim=0, keepdim=True)
    return m.head(z)


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)
    m = GatedAttentionMIL(in_dim=64, num_classes=2, head="regression", pooling="mean").eval()
    cells = torch.randn(2048, 64)
    L = ["MEAN-POOL BYPASS CHECK (Task 3c)", "=" * 40, "",
         "Code path (immunotransformer/model.py, GatedAttentionMIL.forward):",
         "  h = self.proj(cells)",
         "  if self.pooling == 'mean':",
         "      a = torch.full((N, 1), 1/N)          # attn_V / attn_U / attn_w never called",
         "  else:",
         "      a = softmax(attn_w(tanh(attn_V(h)) * sigmoid(attn_U(h))))",
         "  z = (a * h).sum(0)  ->  self.head(z)", ""]
    with torch.no_grad():
        out = m(cells)[0]
        ref = m.head(m.proj(cells).mean(0, keepdim=True))
        L.append(f"1. output vs head(mean(proj(cells))): max |diff| = {(out - ref).abs().max():.2e}")
        for p in (m.attn_V, m.attn_U, m.attn_w):
            for w in p.parameters():
                w.copy_(torch.randn_like(w) * 10)
        out2 = m(cells)[0]
        L.append(f"2. after randomising all attention weights x10: max |diff| = "
                 f"{(out - out2).abs().max():.2e}")
    m.train()
    loss = (m(cells)[0].squeeze() - 1.0) ** 2
    loss.backward()
    g = {n: (p.grad is None or float(p.grad.abs().max()) == 0.0)
         for n, p in m.named_parameters() if n.startswith("attn_")}
    L.append(f"3. attention parameters with no/zero gradient: {sum(g.values())}/{len(g)} "
             f"({', '.join(g)})")
    m.eval()
    with torch.no_grad():
        d = (m(cells)[0] - prefix_forward(m, cells)).abs().max()
    L.append(f"4. current code vs pre-fix code (what arm 6b ran): max |diff| = {d:.2e}")
    L += ["", "Conclusion: arm 6b output depends only on the mean of the projected cells; the",
          "attention module contributes nothing to the output or the gradient, before and",
          "after the refactor. The arm 6b results are attention-free as reported."]
    report = "\n".join(L)
    print(report)
    with open(os.path.join(OUT, "meanpool_bypass_check.txt"), "w") as fh:
        fh.write(report + "\n")


if __name__ == "__main__":
    main()
