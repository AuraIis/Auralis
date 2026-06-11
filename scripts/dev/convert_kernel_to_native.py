"""Convert a KERNEL-format Helix v2 checkpoint to NATIVE (pure-PyTorch) format.

Background
----------
Checkpoints trained with the CUDA back-ends save the Mamba layers in
``mamba_ssm.Mamba2`` (SSD) layout — keys live under ``blocks.N.attn._impl.inner.*``
(fused ``in_proj``, per-head ``A_log / D / dt_bias``, a gated ``norm``, etc.).
The repo's older native Mamba layer (``_Mamba2Native``) is a *different*
architecture (Mamba-1 style: separate ``dt_proj`` / ``x_proj``, full
``[d_inner, d_state]`` ``A_log``) and **cannot represent** those weights — so a
straight native load fails with hundreds of missing/extra keys and, if forced,
produces garbage.

The fix is NOT a weight remap (the two Mamba variants are not interconvertible).
It is a pure-PyTorch SSD back-end (``_Mamba2NativeSSD`` in
``layers/mamba_layer.py``, selected by ``AURALIS_MAMBA_NATIVE_SSD=1``) whose
parameter tree is byte-identical to ``mamba_ssm.Mamba2``. With that back-end the
kernel checkpoint loads with ZERO Mamba remapping — GLA / sparse-attn / norm /
embedding keys are already shared between the two back-ends.

So "conversion" here is intentionally thin and verifiable:

1. load the kernel checkpoint, strip the ``_orig_mod.`` torch.compile prefix;
2. build the native-SSD model from the SAME model config and assert the state
   dict loads with strict=True (0 missing / 0 unexpected / 0 shape mismatch);
3. write a new ``.pt`` carrying only ``{"model": <clean state dict>}`` plus a
   ``conversion`` provenance block.

The strict-load assertion is the contract: a converted checkpoint that does not
load strict-clean into the native-SSD model is rejected, not silently saved.

Usage
-----
    python scripts/dev/convert_kernel_to_native.py \
        --in  /path/to/best.pt \
        --out /path/to/best.native.pt \
        --model-config configs/model/helix_v2_1b_flash.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

# Force the pure-PyTorch SSD back-end and make sure NO kernel back-end is
# selected (conversion is CPU-only; mamba_ssm / fla / flash-attn need not exist).
os.environ["AURALIS_MAMBA_NATIVE_SSD"] = "1"
os.environ.pop("AURALIS_USE_MAMBA_KERNEL", None)
os.environ.pop("AURALIS_USE_CUDA_KERNELS", None)
os.environ.pop("AURALIS_USE_GLA_KERNEL", None)
os.environ.pop("AURALIS_USE_FLASH_ATTN", None)

from auralis.model import build_model  # noqa: E402


def _strip_compile_prefix(state: dict) -> dict:
    return {k.replace("_orig_mod.", ""): v for k, v in state.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="in_path", type=Path, required=True, help="kernel-format .pt")
    p.add_argument("--out", dest="out_path", type=Path, required=True, help="native-format .pt to write")
    p.add_argument("--model-config", type=Path, required=True)
    args = p.parse_args()

    print(f"[convert] loading kernel checkpoint: {args.in_path}")
    payload = torch.load(args.in_path, map_location="cpu", weights_only=False)
    if "model" not in payload:
        raise SystemExit("checkpoint has no 'model' key")
    state = _strip_compile_prefix(payload["model"])
    state = {k: (v.float() if torch.is_floating_point(v) else v) for k, v in state.items()}
    print(f"[convert] kernel state dict: {len(state)} tensors")

    print(f"[convert] building native-SSD model from {args.model_config}")
    model = build_model(args.model_config)
    backend = model.blocks[0].attn.backend
    print(f"[convert] mamba backend = {backend!r} (must be 'native_ssd')")
    if backend != "native_ssd":
        raise SystemExit("native-SSD back-end not active — AURALIS_MAMBA_NATIVE_SSD wiring broken")

    model_keys = set(model.state_dict())
    ckpt_keys = set(state)
    missing = sorted(model_keys - ckpt_keys)     # model needs, ckpt lacks
    unexpected = sorted(ckpt_keys - model_keys)   # ckpt has, model lacks
    msd = model.state_dict()
    shape_mm = [
        (k, tuple(msd[k].shape), tuple(state[k].shape))
        for k in (model_keys & ckpt_keys) if msd[k].shape != state[k].shape
    ]

    print(f"[convert] key diff: missing={len(missing)} unexpected={len(unexpected)} "
          f"shape_mismatch={len(shape_mm)}")
    if missing:
        print("  first missing:", missing[:5])
    if unexpected:
        print("  first unexpected:", unexpected[:5])
    if shape_mm:
        print("  first shape mismatch:", shape_mm[:5])

    # The contract: strict load must succeed cleanly.
    incompatible = model.load_state_dict(state, strict=True)
    # load_state_dict(strict=True) raises on mismatch; reaching here = clean.
    assert not incompatible.missing_keys and not incompatible.unexpected_keys, incompatible
    print("[convert] strict load OK: 0 missing / 0 unexpected / 0 shape mismatch")

    out = {
        "model": model.state_dict(),
        "conversion": {
            "source": str(args.in_path),
            "source_step": payload.get("state", {}).get("step"),
            "source_best_val_loss": payload.get("state", {}).get("best_val_loss"),
            "mamba_backend": "native_ssd",
            "model_config": str(args.model_config),
            "note": "kernel(mamba_ssm.Mamba2 SSD) -> native_ssd pure-PyTorch; param tree identical, no remap",
        },
    }
    # carry the original state block (step etc.) for downstream tooling
    if "state" in payload:
        out["state"] = payload["state"]

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out_path)
    sz = args.out_path.stat().st_size
    print(f"[convert] wrote {args.out_path} ({sz/1e9:.2f} GB)")
    print("[convert] DONE")


if __name__ == "__main__":
    main()
