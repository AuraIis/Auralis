"""Introspect which back-end each layer actually uses.

Answers the question "did my env flags actually land in the model?" at
run-start time, without having to wait for OOM / slow speeds to hint.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn


def _mamba_backend(layer: nn.Module) -> str:
    # Our Mamba2Layer wraps either _Mamba2Native or _Mamba2CUDA.
    inner = getattr(layer, "_impl", layer)
    t = type(inner).__name__
    if t.endswith("CUDA"):
        return "mamba_ssm"
    return "native"


def _gla_backend(layer: nn.Module) -> str:
    # Prefer the ACTUALLY-executed backend recorded by the last forward; only
    # fall back to a hypothetical (on_cuda=True) resolution before any forward.
    actual = getattr(layer, "_last_backend", None)
    if actual is not None:
        return actual
    try:
        from auralis.model.layers import gla_layer
        fla = gla_layer._FLA_AVAILABLE and gla_layer._use_fla(on_cuda=True)
        return "fla?" if fla else "native?"   # '?' = pre-forward hypothetical
    except Exception:
        return "native?"


def _sparse_backend(layer: nn.Module) -> str:
    actual = getattr(layer, "_last_backend", None)
    if actual is not None:
        return "flash_attn" if actual == "flash" else actual
    try:
        from auralis.model.layers import sparse_attn_layer
        gt = getattr(layer, "global_tokens", 0) or 0
        flash = sparse_attn_layer._FLASH_AVAILABLE and sparse_attn_layer._use_flash(on_cuda=True, global_tokens=gt)
        return "flash_attn?" if flash else "native?"
    except Exception:
        return "native?"


def describe_model_backends(model: nn.Module) -> dict[str, Any]:
    """Walk HelixBlocks and return {layer_idx: (type, backend)} + counts."""
    per_layer: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    blocks = getattr(model, "blocks", [])
    for idx, block in enumerate(blocks):
        attn = getattr(block, "attn", None)
        lc = getattr(block, "layer_config", None)
        layer_type = getattr(lc, "type", "unknown") if lc else "unknown"
        if layer_type == "mamba":
            backend = _mamba_backend(attn)
        elif layer_type == "gla":
            backend = _gla_backend(attn)
        elif layer_type == "sparse_attention":
            backend = _sparse_backend(attn)
        elif layer_type == "plain_attention":
            # Plain attention uses F.scaled_dot_product_attention which auto-
            # selects flash / mem-efficient / math internally — report it as
            # sdpa (NOT 'native': the old label was affirmatively wrong).
            backend = getattr(attn, "_last_backend", None) or "torch_sdpa"
        else:
            backend = "unknown"
        per_layer.append({"idx": str(idx), "type": layer_type, "backend": backend})
        key = f"{layer_type}:{backend}"
        counts[key] = counts.get(key, 0) + 1
    return {"per_layer": per_layer, "summary": counts}


def format_backend_summary(desc: dict[str, Any]) -> str:
    lines = ["  layer-backend summary:"]
    for k, v in sorted(desc["summary"].items()):
        lines.append(f"    {k:32s} {v:>3d}")
    return "\n".join(lines)


def assert_no_broken_kernels() -> None:
    """Abort if a REQUESTED fused kernel is unavailable for ANY reason — the
    silent 'you thought you trained fused, you trained native' failure.

    Two unavailable cases, both fatal WHEN THE KERNEL WAS REQUESTED:
      * present-but-broken: the package is installed (find_spec) but its import
        raised (ABI / CUDA / version / missing transitive dep). An ImportError
        alone is NOT a reliable 'absent' signal, so we key off find_spec.
      * genuinely absent: not installed. If you requested it via the env flag,
        that is still a mismatch — unset the flag for an intentional native run.

    Only enforced when CUDA is available (the kernels never run on CPU, so a
    stray flag on a CPU box must not block an intended native run). Call this as
    EARLY as possible — before warm-start / resume / any forward — so a broken
    kernel fails at startup, not after producing a baseline on the wrong backend.
    """
    import os

    import torch

    from auralis.model.layers import gla_layer, mamba_layer, sparse_attn_layer

    if not torch.cuda.is_available():
        return

    all_on = os.environ.get("AURALIS_USE_CUDA_KERNELS", "0") == "1"
    problems: list[str] = []
    checks = [
        (gla_layer._FLA_AVAILABLE, gla_layer._FLA_PRESENT, gla_layer._FLA_IMPORT_ERROR,
         "AURALIS_USE_GLA_KERNEL", "fla (GLA)"),
        (mamba_layer._MAMBA_SSM_AVAILABLE, mamba_layer._MAMBA_SSM_PRESENT, mamba_layer._MAMBA_SSM_IMPORT_ERROR,
         "AURALIS_USE_MAMBA_KERNEL", "mamba_ssm (Mamba)"),
        (sparse_attn_layer._FLASH_AVAILABLE, sparse_attn_layer._FLASH_PRESENT, sparse_attn_layer._FLASH_IMPORT_ERROR,
         "AURALIS_USE_FLASH_ATTN", "flash-attn (Sparse)"),
    ]
    for available, present, err, flag, name in checks:
        requested = all_on or os.environ.get(flag, "") == "1"
        if requested and not available:
            if present:
                problems.append(f"{name}: installed but FAILED to import ({type(err).__name__}: {err})")
            else:
                problems.append(f"{name}: requested but NOT INSTALLED (unset {flag} for an intended native run)")
    if problems:
        raise RuntimeError(
            "Requested fused kernels are unavailable; the run would silently fall "
            "back to the unverified native path:\n  " + "\n  ".join(problems)
        )


__all__ = ["describe_model_backends", "format_backend_summary", "assert_no_broken_kernels"]
