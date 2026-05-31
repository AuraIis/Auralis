#!/usr/bin/env python3
"""Tiny contrastive tuner for learning probes.

Normal SFT did not reliably move the red neuro-map links. This script directly
optimizes target continuations against dangerous negative continuations:

    loss = target_nll + weight * softplus(target_nll - negative_nll + margin)

It is a diagnostic/prototype, not a final trainer. Promotion still requires all
hard and fresh gates.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))


def _maybe_enable_mamba_kernel() -> bool:
    if os.environ.get("AURALIS_USE_MAMBA_KERNEL") == "1":
        return True
    if not torch.cuda.is_available():
        return False
    try:
        import mamba_ssm  # noqa: F401
    except ImportError:
        return False
    os.environ["AURALIS_USE_MAMBA_KERNEL"] = "1"
    return True


_KERNEL_ACTIVE = _maybe_enable_mamba_kernel()

from auralis.model import build_model  # noqa: E402
from auralis.tokenizer.chat_template import build_inference_prompt  # noqa: E402
from auralis.training.optimizer import build_optimizer, build_scheduler  # noqa: E402


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def load_checkpoint_weights(model, checkpoint: Path, device: torch.device) -> None:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = {k.replace("_orig_mod.", ""): v for k, v in payload["model"].items()}
    missing, extra = model.load_state_dict(state, strict=False)
    if missing or extra:
        raise SystemExit(f"state_dict mismatch: missing={len(missing)} extra={len(extra)}")


def save_checkpoint(model, optimizer, scheduler, step: int, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"contrastive_probe_step_{step}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "kind": "contrastive_probe_tune",
        },
        path,
    )
    return path


def load_probe_pairs(path: Path, system: str) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    pairs: list[dict[str, Any]] = []
    for probe in data.get("probes", []):
        prompt = build_inference_prompt([{"role": "user", "content": probe["prompt"]}], default_system=system)
        for target in probe.get("target_answers", []):
            for negative in probe.get("negative_answers", []):
                pairs.append(
                    {
                        "id": probe["id"],
                        "category": probe.get("category", "unknown"),
                        "prompt_text": probe["prompt"],
                        "prompt": prompt,
                        "target": target,
                        "negative": negative,
                    }
                )
    if not pairs:
        raise SystemExit(f"no target/negative pairs in {path}")
    return pairs


def sequence_nll(
    model,
    sp: spm.SentencePieceProcessor,
    prompt: str,
    continuation: str,
    device: torch.device,
) -> torch.Tensor:
    prompt_ids = sp.EncodeAsIds(prompt)
    cont_ids = sp.EncodeAsIds(continuation)
    ids = prompt_ids + cont_ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    labels = torch.tensor(cont_ids, dtype=torch.long, device=device)
    start = len(prompt_ids) - 1
    with autocast_context(device):
        logits = model(input_ids=input_ids)["logits"][0, start : start + len(cont_ids)].float()
    return F.cross_entropy(logits, labels, reduction="mean")


def _load_render_function(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import renderer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.render


def write_outputs(trace: dict[str, Any], trace_json: Path | None, trace_html: Path | None, neuro_html: Path | None) -> None:
    if trace_json:
        trace_json.parent.mkdir(parents=True, exist_ok=True)
        trace_json.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    if trace_html:
        render_dashboard = _load_render_function(REPO / "scripts/eval/learning_trace_dashboard.py")
        trace_html.parent.mkdir(parents=True, exist_ok=True)
        trace_html.write_text(render_dashboard(trace), encoding="utf-8")
    if neuro_html:
        render_neuro = _load_render_function(REPO / "scripts/eval/learning_neuro_map.py")
        neuro_html.parent.mkdir(parents=True, exist_ok=True)
        neuro_html.write_text(render_neuro(trace, auto_refresh=5), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, default=REPO / "tokenizer/helix_v2_tokenizer.model")
    ap.add_argument("--learning-probes", type=Path, default=REPO / "eval/learning_trace_de_core.yaml")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--lr", type=float, default=5e-8)
    ap.add_argument("--warmup-steps", type=int, default=8)
    ap.add_argument("--contrastive-weight", type=float, default=1.0)
    ap.add_argument("--desired-margin", type=float, default=0.75)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--trace-json", type=Path, default=None)
    ap.add_argument("--trace-html", type=Path, default=None)
    ap.add_argument("--neuro-html", type=Path, default=None)
    ap.add_argument(
        "--generation-system",
        default=(
            "Du bist Auralis, ein hilfreicher deutscher KI-Assistent. "
            "Antworte korrekt, knapp und ehrlich. Wenn etwas unsicher oder erfunden ist, sage das deutlich."
        ),
    )
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    pairs = load_probe_pairs(args.learning_probes, args.generation_system)
    print(f"loaded {len(pairs)} contrastive pairs from {args.learning_probes}")

    model = build_model(args.model_config).to(device)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    load_checkpoint_weights(model, args.checkpoint, device)
    print(f"loaded checkpoint: {args.checkpoint}")
    print(f"mamba backend: {'mamba_ssm' if _KERNEL_ACTIVE else 'native'}")

    optimizer = build_optimizer(
        model,
        {"name": "adamw", "lr": args.lr, "betas": [0.9, 0.95], "weight_decay": 0.0, "eps": 1e-8},
    )
    scheduler = build_scheduler(
        optimizer,
        {"type": "cosine", "warmup_steps": args.warmup_steps, "min_lr_ratio": 0.1},
        total_steps=args.steps,
    )

    # Reuse the existing visual probe evaluator.
    smoke_spec = importlib.util.spec_from_file_location("smoke_sft_de", REPO / "scripts/sft/smoke_sft_de.py")
    if smoke_spec is None or smoke_spec.loader is None:
        raise RuntimeError("cannot import smoke_sft_de")
    smoke = importlib.util.module_from_spec(smoke_spec)
    sys.modules[smoke_spec.name] = smoke
    smoke_spec.loader.exec_module(smoke)
    learning_probes = smoke.load_learning_probes(args.learning_probes)

    trace: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "model_config": str(args.model_config),
        "probe_file": str(args.learning_probes),
        "steps": args.steps,
        "history": [],
    }

    def eval_trace(step: int, loss_value: float | None, elapsed: float) -> None:
        rows = smoke.evaluate_learning_probes(model, sp, learning_probes, device, args.generation_system)
        trace["history"].append(
            {
                "step": step,
                "train_loss": loss_value,
                "val_loss": None,
                "val_by_category": {},
                "lr": scheduler.get_last_lr()[0] if step else None,
                "elapsed_seconds": elapsed,
                "probes": rows,
            }
        )
        write_outputs(trace, args.trace_json, args.trace_html, args.neuro_html)
        print(f"step {step:4d} trace")
        for row in rows:
            margin = row.get("margin")
            margin_s = "n/a" if margin is None else f"{margin:+.3f}"
            flags = f" forbidden={row['forbidden_hits']}" if row.get("forbidden_hits") else ""
            print(f"  {row['id']}: target_nll={row['target_nll']:.3f} margin={margin_s}{flags}")

    t0 = time.time()
    eval_trace(0, None, 0.0)
    rng = random.Random(args.seed)
    model.train()
    for step in range(1, args.steps + 1):
        batch = rng.choices(pairs, k=args.batch_size)
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for item in batch:
            target_nll = sequence_nll(model, sp, item["prompt"], item["target"], device)
            negative_nll = sequence_nll(model, sp, item["prompt"], item["negative"], device)
            contrastive = F.softplus(target_nll - negative_nll + args.desired_margin)
            loss = target_nll + args.contrastive_weight * contrastive
            (loss / args.batch_size).backward()
            losses.append(float(loss.item()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        loss_value = sum(losses) / len(losses)
        if not math.isfinite(loss_value):
            raise RuntimeError(f"non-finite loss at step {step}: {loss_value}")
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            print(f"step {step:4d} | contrastive_loss={loss_value:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")
            eval_trace(step, loss_value, time.time() - t0)

    path = save_checkpoint(model, optimizer, scheduler, args.steps, args.output_dir)
    trace["saved_checkpoint"] = str(path)
    write_outputs(trace, args.trace_json, args.trace_html, args.neuro_html)
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
