"""Training-throughput benchmark for Helix v2 on the real GPU.

Builds the exact corpus20b model + data + optimizer stack and runs real train
steps (forward + backward + optimizer), reporting median tokens/sec, step
time, and peak VRAM. Used to measure each perf lever (micro-batch size,
gradient checkpointing, torch.compile, sync removal) before changing the
production trainer.

Examples (inside the container, repo at /workspace/v2data)::

    # baseline (matches corpus20b_codeheavy.yaml: micro=1 accum=32 ckpt on)
    python3 scripts/dev/bench_train.py --micro-batch 1 --grad-accum 32 --ckpt on

    # candidate: bigger micro-batch, no checkpointing
    python3 scripts/dev/bench_train.py --micro-batch 8 --grad-accum 4 --ckpt off

    # correctness: fixed-seed loss series (compare across configs/changes)
    python3 scripts/dev/bench_train.py --micro-batch 1 --grad-accum 4 --steps 8 \
        --warmup 0 --correctness
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path,
                   default=REPO / "configs" / "training" / "corpus20b_codeheavy.yaml")
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=32)
    p.add_argument("--ckpt", choices=["on", "off"], default="on")
    p.add_argument("--compile", dest="compile_mode", default="off",
                   help="off | default | reduce-overhead | max-autotune")
    p.add_argument("--steps", type=int, default=30, help="measured steps")
    p.add_argument("--warmup", type=int, default=5, help="untimed warmup steps")
    p.add_argument("--sync-mode", choices=["legacy", "off"], default="off",
                   help="legacy = 2x cuda.synchronize + loss.item() per micro-batch "
                        "(old trainer hot path); off = sync once per step")
    p.add_argument("--correctness", action="store_true",
                   help="print per-step loss series (fixed seed) and exit")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    import os
    import torch

    from auralis.model import build_model
    from auralis.training.dataset import MixedDataLoader
    from auralis.training.optimizer import build_optimizer
    from auralis.training.utils import apply_gradient_checkpointing, load_yaml, set_seed

    cfg = load_yaml(args.config)
    tcfg = cfg["training"]
    dcfg = cfg["data"]
    seq = int(dcfg["seq_length"])

    # Same runtime knobs as train_phase1.py
    torch.set_float32_matmul_precision(str(tcfg.get("matmul_precision", "high")))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    kcfg = tcfg.get("kernels") or {}
    enabled = bool(kcfg.get("enabled", False))
    os.environ["AURALIS_USE_MAMBA_KERNEL"] = "1" if enabled and kcfg.get("mamba", True) else "0"
    os.environ["AURALIS_USE_GLA_KERNEL"] = "1" if enabled and kcfg.get("gla", True) else "0"
    os.environ["AURALIS_USE_FLASH_ATTN"] = "1" if enabled and kcfg.get("flash_attention", True) else "0"

    set_seed(args.seed)
    device = torch.device("cuda")
    model = build_model(REPO / cfg["model"]["config_path"]).to(device)
    n_params = sum(pm.numel() for pm in model.parameters())
    apply_gradient_checkpointing(model, args.ckpt == "on")
    if args.compile_mode != "off":
        mode = None if args.compile_mode == "default" else args.compile_mode
        model = torch.compile(model, mode=mode) if mode else torch.compile(model)
    model.train()

    loader = MixedDataLoader(
        data_dir=dcfg["data_dir"], mix_ratios=dcfg["mix_ratios"],
        batch_size=args.micro_batch, seq_length=seq,
        seed=int(dcfg.get("dataloader_seed", 42)), split="train",
        val_split_bytes=int(dcfg.get("val_split_bytes", 0)),
    )
    data_iter = iter(loader)
    optimizer = build_optimizer(model, tcfg["optimizer"])
    clip = float(tcfg.get("gradient_clip_norm", 1.0))
    tokens_per_step = args.micro_batch * args.grad_accum * seq

    print(f"model={n_params/1e9:.2f}B micro={args.micro_batch} accum={args.grad_accum} "
          f"tok/step={tokens_per_step} ckpt={args.ckpt} compile={args.compile_mode} "
          f"sync={args.sync_mode}", flush=True)

    def autocast():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    losses: list[float] = []
    step_times: list[float] = []
    data_times: list[float] = []
    total = args.warmup + args.steps
    for step in range(total):
        if step == args.warmup:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            bench_t0 = time.time()
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=device)
        dt = 0.0
        for _ in range(args.grad_accum):
            td0 = time.time()
            batch = next(data_iter)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            if args.sync_mode == "legacy":
                torch.cuda.synchronize()
            dt += time.time() - td0
            with autocast():
                out = model(input_ids=batch["input_ids"], labels=batch["labels"])
                loss = out["loss"] / args.grad_accum
            loss.backward()
            if args.sync_mode == "legacy":
                loss_sum += loss.item()  # item() = stall, like the old trainer
                torch.cuda.synchronize()
            else:
                loss_sum += loss.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=False)
        optimizer.step()
        torch.cuda.synchronize()
        step_t = time.time() - t0
        step_loss = float(loss_sum.item()) if args.sync_mode != "legacy" else float(loss_sum)
        if step >= args.warmup:
            step_times.append(step_t)
            data_times.append(dt)
            losses.append(step_loss)
        if args.correctness:
            print(f"  step {step}: loss {step_loss:.6f}", flush=True)
        elif step % 5 == 0:
            print(f"  step {step}/{total}: {step_t:.2f}s loss {step_loss:.3f}", flush=True)

    if args.correctness:
        return
    wall = time.time() - bench_t0
    med = statistics.median(step_times)
    res = {
        "micro_batch": args.micro_batch, "grad_accum": args.grad_accum,
        "ckpt": args.ckpt, "compile": args.compile_mode, "sync_mode": args.sync_mode,
        "tokens_per_step": tokens_per_step,
        "median_step_s": round(med, 3),
        "mean_step_s": round(statistics.mean(step_times), 3),
        "median_tokens_per_s": round(tokens_per_step / med, 1),
        "mean_tokens_per_s": round(tokens_per_step * len(step_times) / wall, 1),
        "data_frac": round(sum(data_times) / wall, 4),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "mean_loss": round(statistics.mean(losses), 4),
    }
    print(json.dumps(res, indent=2), flush=True)
    if args.json_out:
        args.json_out.write_text(json.dumps(res) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
