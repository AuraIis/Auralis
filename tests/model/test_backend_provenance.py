"""Backend-provenance tests: make the ~6 ambient-env math paths visible and
enforced, so a silent fused->native fallback becomes a loud, recorded fact.

The gate keys off find_spec (reliable 'installed?') x import-succeeded x env
request — an ImportError alone is NOT a reliable 'absent' signal (a broken
install can raise it), so a REQUESTED kernel that is unavailable for ANY reason
must abort. The abort tests force cuda.is_available()=True so the enforcement
path is exercised on a CPU CI box too (guard-the-guard).
"""
from __future__ import annotations

import pytest
import torch

from auralis.model import backend_info
from auralis.model.layers import gla_layer, mamba_layer, sparse_attn_layer
from auralis.model.layers.gla_layer import GLALayer
from auralis.model.layers.plain_attn_layer import PlainAttentionLayer
from auralis.model.layers.sparse_attn_layer import SparseAttentionLayer


def test_last_backend_records_actual_executed_path_cpu():
    """On CPU no fused kernel runs; the tap must record 'native' (not the
    hypothetical on_cuda=True answer backend_info used to hardcode)."""
    gla = GLALayer(d_model=64, n_heads=2, d_head=32)
    gla(torch.randn(1, 8, 64))
    assert gla._last_backend == "native"

    sp = SparseAttentionLayer(d_model=64, n_heads=2, d_head=32, use_rope=False)
    sp(torch.randn(1, 8, 64))
    assert sp._last_backend == "native"

    pa = PlainAttentionLayer(d_model=64, n_heads=2, d_head=32, use_rope=False)
    pa(torch.randn(1, 8, 64))
    assert pa._last_backend == "torch_sdpa"   # NOT 'native' — the old lie


def test_assert_no_broken_kernels_passes_when_available():
    """With all kernels importable (or CPU early-return), no request can abort."""
    backend_info.assert_no_broken_kernels()   # must not raise in a healthy env


def test_assert_raises_when_requested_kernel_is_broken(monkeypatch):
    """Installed-but-broken (find_spec present, import failed) + requested -> abort.
    This is the case a plain `except ImportError` would have swallowed."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gla_layer, "_FLA_AVAILABLE", False)
    monkeypatch.setattr(gla_layer, "_FLA_PRESENT", True)
    monkeypatch.setattr(gla_layer, "_FLA_IMPORT_ERROR", ImportError("libcuda.so.1: cannot open"))
    monkeypatch.setenv("AURALIS_USE_GLA_KERNEL", "1")
    with pytest.raises(RuntimeError, match="FAILED to import"):
        backend_info.assert_no_broken_kernels()


def test_assert_raises_when_requested_kernel_absent(monkeypatch):
    """Genuinely absent (find_spec None) but requested -> abort with a clear
    'unset the flag' message, not a silent native run."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sparse_attn_layer, "_FLASH_AVAILABLE", False)
    monkeypatch.setattr(sparse_attn_layer, "_FLASH_PRESENT", False)
    monkeypatch.setenv("AURALIS_USE_FLASH_ATTN", "1")
    with pytest.raises(RuntimeError, match="NOT INSTALLED"):
        backend_info.assert_no_broken_kernels()


def test_assert_silent_when_unavailable_kernel_not_requested(monkeypatch):
    """Unavailable but NOT requested (intentional native run) -> no raise."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gla_layer, "_FLA_AVAILABLE", False)
    monkeypatch.setattr(gla_layer, "_FLA_PRESENT", True)
    monkeypatch.setattr(gla_layer, "_FLA_IMPORT_ERROR", ImportError("boom"))
    monkeypatch.delenv("AURALIS_USE_GLA_KERNEL", raising=False)
    monkeypatch.delenv("AURALIS_USE_CUDA_KERNELS", raising=False)
    backend_info.assert_no_broken_kernels()   # not requested -> no raise


def test_assert_early_returns_on_cpu(monkeypatch):
    """On a box without CUDA the kernels never run, so a stray flag must not
    block an intended native run."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(gla_layer, "_FLA_AVAILABLE", False)
    monkeypatch.setattr(gla_layer, "_FLA_PRESENT", False)
    monkeypatch.setenv("AURALIS_USE_GLA_KERNEL", "1")
    backend_info.assert_no_broken_kernels()   # CPU -> early return, no raise
