#!/usr/bin/env python3
"""Micro-benchmark Helix training throughput variants.

This intentionally uses synthetic token batches so it measures model/trainer
overhead, not disk or dataloader speed. It is for comparing variants under the
same shape, not for reporting absolute production throughput.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))


def configure_env(args: argparse.Namespace) -> None:
    os.environ["AURALIS_USE_MAMBA_KERNEL"] = "1" if args.kernels else "0"
    os.environ["AURALIS_USE_GLA_KERNEL"] = "1" if args.kernels else "0"
    os.environ["AURALIS_USE_FLASH_ATTN"] = "1" if args.kernels else "0"
    torch.set_float32_matmul_precision(args.matmul_precision)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def load_model(args: argparse.Namespace, device: torch.device):
    from auralis.model import build_model
    from auralis.training.utils import apply_gradient_checkpointing

    model = build_model(args.model_config).to(device)
    apply_gradient_checkpointing(model, args.gradient_checkpointing)
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state = {k.replace("_orig_mod.", ""): v for k, v in payload["model"].items()}
        missing, extra = model.load_state_dict(state, strict=False)
        if missing or extra:
            raise RuntimeError(
                f"checkpoint mismatch: missing={len(missing)} extra={len(extra)} "
                f"first_missing={missing[:3]} first_extra={extra[:3]}"
            )
    model.train()
    if args.compile:
        model = torch.compile(model, mode=args.compile_mode)
    return model


def make_batch(args: argparse.Namespace, device: torch.device) -> dict[str, torch.Tensor]:
    x = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.seq_length),
        dtype=torch.long,
        device=device,
    )
    return {"input_ids": x, "labels": x.clone()}


def one_step(
    *,
    model,
    optimizer,
    batch: dict[str, torch.Tensor],
    grad_accum: int,
    dtype: torch.dtype,
    force_micro_sync: bool,
    clip_grad: bool,
    mark_cudagraph_steps: bool,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss_total = 0.0
    for _ in range(grad_accum):
        if mark_cudagraph_steps and hasattr(torch, "compiler"):
            mark = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
            if callable(mark):
                mark()
        with torch.autocast(device_type="cuda", dtype=dtype):
            out = model(input_ids=batch["input_ids"], labels=batch["labels"])
            loss = out["loss"] / grad_accum
        loss.backward()
        loss_total += float(loss.detach().item())
        if force_micro_sync:
            torch.cuda.synchronize()
    if clip_grad:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss_total


def bench_variant(
    args: argparse.Namespace, force_micro_sync: bool, clip_grad: bool
) -> dict[str, float | str]:
    device = torch.device("cuda")
    model = load_model(args, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-8,
        fused=args.fused_optimizer,
    )
    batch = make_batch(args, device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    for _ in range(args.warmup):
        one_step(
            model=model,
            optimizer=optimizer,
            batch=batch,
            grad_accum=args.grad_accum,
            dtype=dtype,
            force_micro_sync=force_micro_sync,
            clip_grad=clip_grad,
            mark_cudagraph_steps=args.mark_cudagraph_steps,
        )
    torch.cuda.synchronize()

    times: list[float] = []
    losses: list[float] = []
    tokens_per_step = args.batch_size * args.seq_length * args.grad_accum
    for _ in range(args.iters):
        t0 = time.perf_counter()
        loss = one_step(
            model=model,
            optimizer=optimizer,
            batch=batch,
            grad_accum=args.grad_accum,
            dtype=dtype,
            force_micro_sync=force_micro_sync,
            clip_grad=clip_grad,
            mark_cudagraph_steps=args.mark_cudagraph_steps,
        )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        times.append(dt)
        losses.append(loss)

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    del model, optimizer, batch
    torch.cuda.empty_cache()
    return {
        "variant": f"sync={force_micro_sync},clip={clip_grad}",
        "seconds_per_step_avg": statistics.mean(times),
        "seconds_per_step_min": min(times),
        "tokens_per_second_avg": tokens_per_step / statistics.mean(times),
        "tokens_per_second_best": tokens_per_step / min(times),
        "loss_last": losses[-1],
        "peak_alloc_gb": peak_gb,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=200000)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--matmul-precision", default="high")
    parser.add_argument("--kernels", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--mark-cudagraph-steps", action="store_true")
    parser.add_argument("--fused-optimizer", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    configure_env(args)
    print(f"gpu={torch.cuda.get_device_name(0)}")
    print(
        f"shape=batch{args.batch_size} seq{args.seq_length} accum{args.grad_accum} "
        f"tokens/step={args.batch_size * args.seq_length * args.grad_accum}"
    )
    for force_sync, clip in [(True, True), (False, True), (False, False)]:
        result = bench_variant(args, force_micro_sync=force_sync, clip_grad=clip)
        print(result, flush=True)


if __name__ == "__main__":
    main()
