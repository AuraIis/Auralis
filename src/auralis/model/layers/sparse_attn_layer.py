"""Sparse attention layer — sliding window + global tokens.

Pure-PyTorch reference. Late layers (22-27 in the 28-layer stack) use this
to support long-context retrieval (needle-in-haystack). Earlier layers stay
on Mamba / GLA which are linear in sequence length.

Attention pattern (per query position ``t``):

- attend to the last ``window_size`` positions ``[t - window_size + 1, t]``
- attend to the first ``global_tokens`` positions ``[0, global_tokens)`` — these
  act as a shared "summary bus" that every position can see
- causal: never attend to positions ``> t``

Production GPU path: swap for ``flash_attn`` with a sliding-window mask.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from auralis.model.utils.rotary import RotaryEmbedding, apply_rotary_pos_emb


class SparseAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 16,
        d_head: int = 128,
        window_size: int = 1024,
        global_tokens: int = 32,
        use_rope: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.window_size = window_size
        self.global_tokens = global_tokens
        self.use_rope = use_rope

        self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.out_proj = nn.Linear(n_heads * d_head, d_model, bias=False)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,                                       # [B, L, d_model]
        rope: tuple[torch.Tensor, torch.Tensor] | None = None, # (cos, sin) each [L, d_head]
    ) -> tuple[torch.Tensor, None]:
        B, L, _ = x.shape
        H, D = self.n_heads, self.d_head

        q = self.q_proj(x).view(B, L, H, D)
        k = self.k_proj(x).view(B, L, H, D)
        v = self.v_proj(x).view(B, L, H, D)

        if self.use_rope and rope is not None:
            q, k = apply_rotary_pos_emb(q, k, rope[0], rope[1])

        # [B, L, H, D] → [B, H, L, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * (D ** -0.5)      # [B, H, L, L]

        mask = self._build_mask(L, device=x.device)                      # [L, L]
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v)                                      # [B, H, L, D]

        out = out.transpose(1, 2).contiguous().view(B, L, H * D)
        return self.out_proj(out), None

    # ------------------------------------------------------------------
    # Mask: True = blocked (will be -inf before softmax)
    # ------------------------------------------------------------------
    def _build_mask(self, L: int, device: torch.device) -> torch.Tensor:
        i = torch.arange(L, device=device).unsqueeze(1)                  # [L, 1] query pos
        j = torch.arange(L, device=device).unsqueeze(0)                  # [1, L] key pos

        # Causal: block future
        causal = j > i

        # Within-window: keep j in [i - window + 1, i]
        outside_window = (i - j) >= self.window_size

        # Global tokens always attendable (override outside_window for first N keys)
        global_ok = j < self.global_tokens

        blocked = causal | (outside_window & ~global_ok)
        return blocked


__all__ = ["SparseAttentionLayer"]
