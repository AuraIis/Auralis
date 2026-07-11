"""Regression tests for the GLA native scan.

Root cause of the guarded bug: the native sequential scan applied the decay
gate to the *value* axis (``alpha[:, t].unsqueeze(-2)`` -> ``[B,H,1,D]``)
instead of the *key* axis. Because the shipped configs use ``d_k == d_v``, the
tensor shapes stayed valid, so no test crashed — yet the recurrence diverged
~92% (mean relative) from the fused ``fla.ops.gla.chunk_gla`` kernel that
training actually used. The only existing GLA tests checked forward shape and
that backward runs, so the axis error was invisible.

These tests pin the correct semantics two ways:

1. CPU, always-on: the native scan must equal an explicit per-key-gated GLA
   recurrence written with zero broadcasting ambiguity.
2. GPU, opt-in: the native scan must match ``chunk_gla`` (the training kernel)
   to within chunked-fp tolerance.
"""
from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

from auralis.model.layers.gla_layer import GLALayer, _FLA_AVAILABLE


def _project(layer: GLALayer, x: torch.Tensor):
    B, L, _ = x.shape
    H, D = layer.n_heads, layer.d_head
    q = layer.q_proj(x).view(B, L, H, D)
    k = layer.k_proj(x).view(B, L, H, D)
    v = layer.v_proj(x).view(B, L, H, D)
    log_alpha = -F.softplus(-layer.alpha_proj(x).view(B, L, H, D))
    return q, k, v, log_alpha


def _explicit_reference(q, k, v, log_alpha):
    """Unambiguous GLA recurrence: the gate is indexed by the KEY channel.

    S[b,h,i,j] <- alpha[b,h,t,i] * S[b,h,i,j] + k[b,h,t,i] * v[b,h,t,j]
    out[b,h,t,j] = sum_i q_scaled[b,h,t,i] * S[b,h,i,j]

    Written so the decayed axis (i = key) is impossible to misread.
    """
    B, L, H, D = q.shape
    qs = q * (D ** -0.5)
    alpha = torch.exp(log_alpha)
    S = torch.zeros(B, H, D, D, dtype=q.dtype)  # [.., key, value]
    outs = []
    for t in range(L):
        a_key = alpha[:, t].unsqueeze(-1)  # [B,H,D,1] -> per key channel
        upd = torch.einsum("bhi,bhj->bhij", k[:, t], v[:, t])
        S = a_key * S + upd
        outs.append(torch.einsum("bhi,bhij->bhj", qs[:, t], S))
    return torch.stack(outs, dim=1)


def test_native_scan_decays_key_axis_cpu():
    """The native scan must match the explicit per-key-gated reference.

    Fails on the pre-fix code (which decayed the value axis) under non-uniform
    per-channel gates; passes once the decay is applied to the key axis.
    """
    torch.manual_seed(0)
    B, L, H, D = 2, 24, 3, 16
    layer = GLALayer(d_model=H * D, n_heads=H, d_head=D).double()
    x = torch.randn(B, L, H * D, dtype=torch.float64)
    q, k, v, log_alpha = _project(layer, x)

    # Guard the guard: the gates must actually vary across channels, otherwise
    # the two axes would coincide and the test could not detect the bug.
    assert torch.exp(log_alpha).std(dim=-1).mean() > 1e-3

    out_native, _ = layer._native_scan(q, k, v, log_alpha, None)
    out_ref = _explicit_reference(q, k, v, log_alpha)
    torch.testing.assert_close(out_native, out_ref, atol=1e-10, rtol=1e-8)


@pytest.mark.skipif(
    not (_FLA_AVAILABLE and torch.cuda.is_available()),
    reason="fla + CUDA required for the fused-kernel parity check",
)
def test_native_scan_matches_fla_kernel_gpu():
    """Authoritative check: native scan == the fused chunk_gla training kernel."""
    from fla.ops.gla import chunk_gla

    torch.manual_seed(0)
    B, L, H, D = 2, 64, 4, 32
    dev = "cuda"
    layer = GLALayer(d_model=H * D, n_heads=H, d_head=D).to(dev).float()
    x = torch.randn(B, L, H * D, device=dev)
    q, k, v, log_alpha = _project(layer, x)

    out_fla, _ = chunk_gla(
        q, k, v, log_alpha, scale=D ** -0.5,
        initial_state=None, output_final_state=False,
    )
    out_native, _ = layer._native_scan(q, k, v, log_alpha, None)
    # chunked fp accumulation vs a sequential fp32 scan: a few e-3 is expected.
    torch.testing.assert_close(out_native.float(), out_fla.float(), atol=5e-3, rtol=1e-2)


@pytest.mark.skipif(
    not (_FLA_AVAILABLE and torch.cuda.is_available()),
    reason="fla + CUDA required for the fused-kernel parity check",
)
def test_native_scan_accumulates_state_in_fp32_under_bf16_autocast():
    """Under bf16 autocast at the trained length, the native scan must still match
    the fp32-internal fla kernel. A native scan that accumulates its recurrent
    state in bf16 (the pre-fix behaviour) drifts ~10% mean-relative over L=2048
    and fails this; accumulating S in fp32 keeps it within kernel tolerance.
    """
    from fla.ops.gla import chunk_gla

    torch.manual_seed(0)
    B, L, H, D = 2, 2048, 4, 64   # the ACTUAL trained context length
    dev = "cuda"
    layer = GLALayer(d_model=H * D, n_heads=H, d_head=D).to(dev)
    x = torch.randn(B, L, H * D, device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        q, k, v, log_alpha = _project(layer, x)
        out_native, _ = layer._native_scan(q, k, v, log_alpha, None)
    out_fla, _ = chunk_gla(
        q, k, v, log_alpha, scale=D ** -0.5,
        initial_state=None, output_final_state=False,
    )
    # bf16 I/O rounding at both ends allows ~2%; a bf16-STATE regression is ~10%.
    torch.testing.assert_close(out_native.float(), out_fla.float(), atol=2e-2, rtol=2e-2)
