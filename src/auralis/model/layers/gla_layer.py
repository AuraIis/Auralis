"""Gated Linear Attention (GLA) layer — pure-PyTorch reference.

Linear-attention variant with a learnable per-head decay ``alpha``.
Computes an outer-product hidden state ``S_t = alpha * S_{t-1} + k v^T``
and reads with ``q S_t``, then gates the output.

Reference: Yang et al., "Gated Linear Attention Transformers with
Hardware-Efficient Training" (2024). v1's Triton kernel gave ~20x speedup;
production path will swap this implementation for that kernel on GPU.
The interface stays the same so ``HelixBlock`` never changes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GLALayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 16,
        d_head: int = 128,
        d_state: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        # d_state is nominally the "memory key dimension". GLA ties it to d_head
        # in most published variants; accept an override but default to d_head.
        self.d_state = d_state or d_head

        # Projections (bias-free, matches Llama/Mistral convention)
        self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.g_proj = nn.Linear(d_model, n_heads * d_head, bias=False)  # output gate

        # alpha: per-head decay in (0, 1). Bias init ~log(0.9/(1-0.9)) so
        # initial decay is ~0.9 — similar to RetNet / Mamba dt init in spirit.
        self.alpha_proj = nn.Linear(d_model, n_heads, bias=True)
        with torch.no_grad():
            self.alpha_proj.bias.fill_(2.2)                    # sigmoid(2.2) ≈ 0.9

        self.out_proj = nn.Linear(n_heads * d_head, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,                                       # [B, L, d_model]
        state: torch.Tensor | None = None,                     # [B, H, d_head, d_head]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = x.shape
        H, D = self.n_heads, self.d_head

        q = self.q_proj(x).view(B, L, H, D)
        k = self.k_proj(x).view(B, L, H, D)
        v = self.v_proj(x).view(B, L, H, D)
        g = torch.sigmoid(self.g_proj(x).view(B, L, H, D))
        alpha = torch.sigmoid(self.alpha_proj(x))              # [B, L, H] in (0,1)

        out, new_state = self._gla_scan(q, k, v, alpha, state)
        out = out * g                                          # per-channel gate
        return self.out_proj(out.reshape(B, L, H * D)), new_state

    # ------------------------------------------------------------------
    # Sequential linear-attention scan (O(L) in time, pure torch).
    # Production GPU path: chunkwise kernel from flash-linear-attention.
    # ------------------------------------------------------------------
    def _gla_scan(
        self,
        q: torch.Tensor,                                       # [B, L, H, D]
        k: torch.Tensor,                                       # [B, L, H, D]
        v: torch.Tensor,                                       # [B, L, H, D]
        alpha: torch.Tensor,                                   # [B, L, H]
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, H, D = q.shape
        # Scale queries (as in softmax attention) to keep inner products sane.
        q = q * (D ** -0.5)

        if state is None:
            S = torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)
        else:
            S = state.to(q.dtype)

        # Per-head, per-step: S = alpha * S + k_t v_t^T ; out = q_t S
        alpha = alpha.permute(0, 2, 1)                         # [B, H, L]
        outs = []
        for t in range(L):
            k_t = k[:, t]                                      # [B, H, D]
            v_t = v[:, t]                                      # [B, H, D]
            q_t = q[:, t]                                      # [B, H, D]
            a_t = alpha[..., t].unsqueeze(-1).unsqueeze(-1)    # [B, H, 1, 1]

            # Outer product k_t v_t^T → [B, H, D, D]
            update = torch.einsum("bhd,bhe->bhde", k_t, v_t)
            S = a_t * S + update

            # Read: q_t dot S
            y_t = torch.einsum("bhd,bhde->bhe", q_t, S)        # [B, H, D]
            outs.append(y_t)

        out = torch.stack(outs, dim=1)                         # [B, L, H, D]
        return out, S


__all__ = ["GLALayer"]
