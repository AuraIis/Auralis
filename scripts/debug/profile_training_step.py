#!/usr/bin/env python3
"""Profile one or more Helix training steps.

The profiler is intentionally separate from the production trainer. It can load
any compatible checkpoint, run synthetic batches with the same shapes, and
write both a Chrome trace and a compact operator summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))


def configure_runtime(args: argparse.Namespace) -> None:
    os.environ["AURALIS_USE_MAMBA_KERNEL"] = "1" if args.kernels else "0"
    os.environ["AURALIS_USE_GLA_KERNEL"] = "1" if args.kernels else "0"
    os.environ["AURALIS_USE_FLASH_ATTN"] = "1" if args.kernels else "0"
    torch.set_float32_matmul_precision(args.matmul_precision)
    if torch.cuda.is_available():
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
    input_ids = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.seq_length),
        dtype=torch.long,
        device=device,
    )
    return {"input_ids": input_ids, "labels": input_ids.clone()}


def training_step(
    *,
    model,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    grad_accum: int,
    amp_dtype: torch.dtype,
    clip_grad: bool,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss_total = 0.0
    for micro_idx in range(grad_accum):
        with record_function(f"microbatch_{micro_idx:03d}"):
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                with record_function("model_forward"):
                    out = model(input_ids=batch["input_ids"], labels=batch["labels"])
                with record_function("loss_scale"):
                    loss = out["loss"] / grad_accum
            with record_function("backward"):
                loss.backward()
            loss_total += float(loss.detach().item())
    if clip_grad:
        with record_function("grad_clip"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    with record_function("optimizer_step"):
        optimizer.step()
    return loss_total


def write_operator_summary(prof, out_txt: Path, row_limit: int) -> None:
    table_cuda = prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=row_limit,
    )
    table_cpu = prof.key_averages().table(
        sort_by="self_cpu_time_total",
        row_limit=row_limit,
    )
    out_txt.write_text(
        "# CUDA time\n\n"
        + table_cuda
        + "\n\n# CPU time\n\n"
        + table_cpu
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=200000)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--profile-steps", type=int, default=2)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--matmul-precision", default="high")
    parser.add_argument("--kernels", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--clip-grad", action="store_true")
    parser.add_argument("--row-limit", type=int, default=80)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this profiler")
    configure_runtime(args)
    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-8,
        fused=True,
    )
    batch = make_batch(args, device)
    amp_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    for _ in range(args.warmup):
        training_step(
            model=model,
            optimizer=optimizer,
            batch=batch,
            grad_accum=args.grad_accum,
            amp_dtype=amp_dtype,
            clip_grad=args.clip_grad,
        )
    torch.cuda.synchronize()

    trace_path = args.output_dir / "trace.json"
    summary_path = args.output_dir / "operator_summary.txt"
    meta_path = args.output_dir / "profile_meta.json"

    t0 = time.perf_counter()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        with_flops=True,
    ) as prof:
        for step_idx in range(args.profile_steps):
            with record_function(f"profile_step_{step_idx:03d}"):
                training_step(
                    model=model,
                    optimizer=optimizer,
                    batch=batch,
                    grad_accum=args.grad_accum,
                    amp_dtype=amp_dtype,
                    clip_grad=args.clip_grad,
                )
            prof.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    prof.export_chrome_trace(str(trace_path))
    write_operator_summary(prof, summary_path, args.row_limit)
    tokens = args.batch_size * args.seq_length * args.grad_accum * args.profile_steps
    meta = {
        "gpu": torch.cuda.get_device_name(0),
        "model_config": str(args.model_config),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "batch_size": args.batch_size,
        "seq_length": args.seq_length,
        "grad_accum": args.grad_accum,
        "profile_steps": args.profile_steps,
        "tokens_profiled": tokens,
        "elapsed_seconds": elapsed,
        "tokens_per_second": tokens / max(elapsed, 1e-9),
        "peak_alloc_gb": torch.cuda.max_memory_allocated() / 1e9,
        "kernels": args.kernels,
        "compile": args.compile,
        "compile_mode": args.compile_mode if args.compile else None,
        "gradient_checkpointing": args.gradient_checkpointing,
        "clip_grad": args.clip_grad,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    print(f"trace: {trace_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
