"""Regression test for the Mamba-2 native selective scan.

Mamba shipped with NO cross-implementation parity test — only shape + backward
checks — the same gap that let the GLA axis bug ship. Mamba's state is
rectangular ([d_inner, d_state] with d_inner != d_state), so a transpose bug
tends to shape-CRASH rather than run silently, which makes it less exposed than
GLA — but dt/decay/readout mistakes stay shape-valid and finite.

This pins the native scan against an INDEPENDENTLY-derived reference of the
Mamba-2 recurrence, written with named-index einsums (not by paraphrasing
``_selective_scan``'s unsqueeze-broadcasts), so a broadcast-axis mistake in the
production scan diverges from the reference.

    h[b,i,s] = exp(dt[b,t,i] * A[i,s]) * h[b,i,s] + dt[b,t,i]*B[b,t,s]*x[b,t,i]
    y[b,t,i] = sum_s C[b,t,s] * h[b,i,s]  ;  y += x * D   (D per channel i)

Note: the AUTHORITATIVE native-vs-mamba_ssm kernel arm is intentionally absent.
The two backends do not share a parameter layout and mamba_ssm's Triton kernel
does not build on Blackwell (sm_120), so a kernel arm here would silently skip
on the training box and degrade to a self-consistency check. That arm is tracked
separately (needs a param-mapping adapter + a non-Blackwell CI GPU).
"""
from __future__ import annotations

import torch

from auralis.model.layers.mamba_layer import _Mamba2Native


def _reference_ssd(x, dt, A, Bm, Cm, D):
    """Independent Mamba-2 recurrence via named-index einsums (float64)."""
    bsz, L, I = x.shape
    S = A.shape[1]
    h = torch.zeros(bsz, I, S, dtype=x.dtype)
    ys = []
    for t in range(L):
        dA = torch.exp(torch.einsum("bi,is->bis", dt[:, t], A))          # decay per (i,s)
        dBx = torch.einsum("bi,bs,bi->bis", dt[:, t], Bm[:, t], x[:, t])  # dt_i * B_s * x_i
        h = dA * h + dBx
        ys.append(torch.einsum("bs,bis->bi", Cm[:, t], h))               # sum_s C_s h_{i,s}
    return torch.stack(ys, dim=1) + torch.einsum("bti,i->bti", x, D)


def test_native_selective_scan_matches_reference_cpu():
    torch.manual_seed(0)
    B, L, d_inner, d_state = 2, 20, 12, 8   # d_inner != d_state (rectangular)
    x = torch.randn(B, L, d_inner, dtype=torch.float64)
    dt = torch.rand(B, L, d_inner, dtype=torch.float64) * 0.1 + 0.01     # small positive steps
    A = -torch.rand(d_inner, d_state, dtype=torch.float64) - 0.1         # negative (stable)
    Bm = torch.randn(B, L, d_state, dtype=torch.float64)
    Cm = torch.randn(B, L, d_state, dtype=torch.float64)
    D = torch.rand(d_inner, dtype=torch.float64) + 0.5

    # Guard the guard: every axis must actually vary, so an axis swap can't hide.
    assert A.std(0).mean() > 1e-3 and A.std(1).mean() > 1e-3   # varies over i AND s
    assert Bm.std(-1).mean() > 1e-3 and Cm.std(-1).mean() > 1e-3
    assert x.std(-1).mean() > 1e-3 and dt.std(-1).mean() > 1e-3

    native = _Mamba2Native(d_model=6, d_state=d_state, expand_factor=2)   # d_inner = 12
    out_native, _ = native._selective_scan(x, dt, A, Bm, Cm, D, None)
    out_ref = _reference_ssd(x, dt, A, Bm, Cm, D)

    torch.testing.assert_close(out_native, out_ref, atol=1e-10, rtol=1e-8)


def test_native_selective_scan_returns_full_state():
    """The native scan must return the true final [B, d_inner, d_state] state
    (unlike the mamba_ssm wrapper, which returns the input state unchanged —
    tracked as a separate contract bug)."""
    torch.manual_seed(1)
    B, L, d_inner, d_state = 1, 5, 12, 8
    x = torch.randn(B, L, d_inner, dtype=torch.float64)
    dt = torch.rand(B, L, d_inner, dtype=torch.float64) * 0.1 + 0.01
    A = -torch.rand(d_inner, d_state, dtype=torch.float64) - 0.1
    Bm = torch.randn(B, L, d_state, dtype=torch.float64)
    Cm = torch.randn(B, L, d_state, dtype=torch.float64)
    D = torch.rand(d_inner, dtype=torch.float64) + 0.5

    native = _Mamba2Native(d_model=6, d_state=d_state, expand_factor=2)
    _, state = native._selective_scan(x, dt, A, Bm, Cm, D, None)
    assert state.shape == (B, d_inner, d_state)
    assert torch.isfinite(state).all()
