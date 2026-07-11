"""Regression tests for the sparse-attention layer's mask + backends.

Two subtleties this pins:
- The native masked-softmax path is the ONE that trains whenever global_tokens
  != 0 (``_use_flash`` returns False for any global_tokens > 0), i.e. for the
  shipped helix_v2_1b.yaml (global_tokens=32). So the authoritative check for the
  real config is native-vs-an-explicit-reference, NOT native-vs-flash.
- flash_attn is used only when global_tokens == 0 (the *_flash.yaml config the
  base trained with), with window_size=(window-1, 0). The -1 is a hand-matched
  off-by-one against the native ``(i-j) >= window_size`` mask; the GPU arm pins it.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from auralis.model.layers.sparse_attn_layer import (
    SparseAttentionLayer,
    _FLASH_AVAILABLE,
)


def _reference_masked_attention(q, k, v, window_size, global_tokens):
    """Explicit float64 windowed+global causal attention. `allowed` is stated
    directly (not by paraphrasing _build_mask), so a mask bug diverges."""
    B, L, H, D = q.shape
    qh = q.transpose(1, 2).double()
    kh = k.transpose(1, 2).double()
    vh = v.transpose(1, 2).double()
    scores = torch.matmul(qh, kh.transpose(-2, -1)) * (D ** -0.5)
    i = torch.arange(L).unsqueeze(1)
    j = torch.arange(L).unsqueeze(0)
    allowed = (j <= i) & (((i - j) < window_size) | (j < global_tokens))
    scores = scores.masked_fill(~allowed, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, vh)
    return out.transpose(1, 2)


def _make(q_seed, B, L, H, D):
    g = torch.Generator().manual_seed(q_seed)
    return (torch.randn(B, L, H, D, generator=g, dtype=torch.float64),
            torch.randn(B, L, H, D, generator=g, dtype=torch.float64),
            torch.randn(B, L, H, D, generator=g, dtype=torch.float64))


def test_native_mask_matches_explicit_reference_cpu():
    """Native masked-softmax == explicit windowed+global causal reference.
    Uses the shipped global_tokens > 0 regime (window < L) so BOTH the window
    edge and the global-token save path are exercised."""
    B, L, H, D = 2, 40, 2, 16
    window_size, global_tokens = 8, 4
    layer = SparseAttentionLayer(
        d_model=H * D, n_heads=H, d_head=D,
        window_size=window_size, global_tokens=global_tokens, use_rope=False,
    ).double()
    q, k, v = _make(0, B, L, H, D)

    # Guard the guard: the config must actually exercise window masking AND
    # global saves, else the test is vacuous.
    ii = torch.arange(L).unsqueeze(1)
    jj = torch.arange(L).unsqueeze(0)
    window_masked = ((jj <= ii) & ((ii - jj) >= window_size) & (jj >= global_tokens)).any()
    global_saved = ((jj <= ii) & ((ii - jj) >= window_size) & (jj < global_tokens)).any()
    assert window_masked and global_saved

    out_native = layer._native(q, k, v)
    out_ref = _reference_masked_attention(q, k, v, window_size, global_tokens)
    # _native computes softmax in fp32 regardless of input dtype (numerical
    # stability), so a float64 reference floors at ~1e-7; a real mask bug is
    # ~100x+ (an off-by-one flips whole rows), so this tolerance still bites.
    torch.testing.assert_close(out_native, out_ref, atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(
    not (_FLASH_AVAILABLE and torch.cuda.is_available()),
    reason="flash-attn + CUDA required for the fused-kernel window parity check",
)
def test_native_matches_flash_at_global_tokens_zero_gpu():
    """At global_tokens=0 (the *_flash.yaml regime), native must match
    flash_attn_func — pins the window_size=(w-1, 0) off-by-one against the kernel."""
    from flash_attn import flash_attn_func

    B, L, H, D = 2, 128, 4, 64
    window_size = 16
    dev = "cuda"
    layer = SparseAttentionLayer(
        d_model=H * D, n_heads=H, d_head=D,
        window_size=window_size, global_tokens=0, use_rope=False,
    ).to(dev)
    q = torch.randn(B, L, H, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(B, L, H, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(B, L, H, D, device=dev, dtype=torch.bfloat16)
    out_flash = flash_attn_func(q, k, v, causal=True, window_size=(window_size - 1, 0))
    out_native = layer._native(q, k, v)
    torch.testing.assert_close(out_native.float(), out_flash.float(), atol=2e-2, rtol=2e-2)
