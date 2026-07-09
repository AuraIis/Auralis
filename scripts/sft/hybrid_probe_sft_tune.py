#!/usr/bin/env python3
"""Hybrid SFT + contrastive probe tuner.

This is a diagnostic repair trainer for the German response-fix work. Pure SFT
kept chasing individual gates, while pure contrastive probe tuning improved
probe margins but sometimes flipped answer polarity elsewhere. This script
combines both objectives:

    loss = sft_weight * assistant_sft_loss
         + probe_weight * (target_nll
                           + contrastive_weight * softplus(target_nll - negative_nll + margin))

The learning probes are training diagnostics here, not a leaderboard. Promotion
still requires source-disjoint semantic gates.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))


def load_smoke_module():
    spec = importlib.util.spec_from_file_location(
        "smoke_sft_de", REPO / "scripts/sft/smoke_sft_de.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import smoke_sft_de")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


smoke = load_smoke_module()

from auralis.model import build_model  # noqa: E402
from auralis.tokenizer.chat_template import build_inference_prompt  # noqa: E402
from auralis.training.optimizer import build_optimizer, build_scheduler  # noqa: E402


def save_checkpoint(model, optimizer, scheduler, step: int, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"hybrid_probe_sft_step_{step}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "kind": "hybrid_probe_sft_tune",
        },
        path,
    )
    return path


def load_probe_pairs(path: Path, system: str) -> list[dict[str, Any]]:
    probes = smoke.load_learning_probes(path)
    pairs: list[dict[str, Any]] = []
    for probe in probes:
        prompt = build_inference_prompt(
            [{"role": "user", "content": probe.prompt}], default_system=system
        )
        for target in probe.target_answers:
            for negative in probe.negative_answers:
                pairs.append(
                    {
                        "id": probe.id,
                        "category": probe.category,
                        "prompt_text": probe.prompt,
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
    with smoke.autocast_context(device):
        logits = model(input_ids=input_ids)["logits"][0, start : start + len(cont_ids)].float()
    return F.cross_entropy(logits, labels, reduction="mean")


def _load_render_function(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import renderer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.render


def write_outputs(
    trace: dict[str, Any], trace_json: Path | None, trace_html: Path | None, neuro_html: Path | None
) -> None:
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, default=REPO / "tokenizer/helix_v2_tokenizer.model")
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument(
        "--learning-probes", type=Path, default=REPO / "eval/learning_trace_de_core.yaml"
    )
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--lr", type=float, default=4e-8)
    ap.add_argument("--warmup-steps", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=1, help="SFT micro-batch size.")
    ap.add_argument(
        "--grad-accum", type=int, default=4, help="SFT gradient accumulation batches per step."
    )
    ap.add_argument("--probe-batch-size", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--train-limit", type=int, default=0)
    ap.add_argument("--val-limit", type=int, default=0)
    ap.add_argument("--sft-weight", type=float, default=1.0)
    ap.add_argument("--probe-weight", type=float, default=0.35)
    ap.add_argument("--contrastive-weight", type=float, default=0.8)
    ap.add_argument("--desired-margin", type=float, default=0.55)
    ap.add_argument("--eos-loss-weight", type=float, default=8.0)
    ap.add_argument("--category-weights", default="")
    ap.add_argument("--family-balanced-sampler", action="store_true")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--learning-trace-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--trace-json", type=Path, default=None)
    ap.add_argument("--trace-html", type=Path, default=None)
    ap.add_argument("--neuro-html", type=Path, default=None)
    ap.add_argument("--diag-json", type=Path, default=None)
    ap.add_argument(
        "--generation-system",
        default=(
            "Du bist Auralis, ein hilfreicher deutscher KI-Assistent. "
            "Antworte korrekt, knapp und ehrlich. Wenn etwas unsicher oder erfunden ist, sage das deutlich."
        ),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    smoke.set_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    train_examples = smoke.load_examples(args.train, sp, args.max_length, args.train_limit or None)
    val_examples = smoke.load_examples(args.val, sp, args.max_length, args.val_limit or None)
    category_weights = smoke.parse_category_weights(args.category_weights)
    sft_iter = smoke.batches(
        train_examples,
        args.batch_size,
        rng,
        category_weights=category_weights,
        family_balanced=args.family_balanced_sampler,
    )
    probe_pairs = load_probe_pairs(args.learning_probes, args.generation_system)
    learning_probes = smoke.load_learning_probes(args.learning_probes)
    print(f"loaded {len(probe_pairs)} contrastive pairs from {args.learning_probes}", flush=True)

    model = build_model(args.model_config).to(device)
    smoke.apply_gradient_checkpointing(model, enabled=(device.type == "cuda"))
    loaded_step = smoke.load_checkpoint_weights(model, args.checkpoint, device)
    print(f"loaded checkpoint: {args.checkpoint} (source step={loaded_step})", flush=True)

    optimizer = build_optimizer(
        model,
        {"name": "adamw", "lr": args.lr, "betas": [0.9, 0.95], "weight_decay": 0.0, "eps": 1e-8},
    )
    scheduler = build_scheduler(
        optimizer,
        {"type": "cosine", "warmup_steps": args.warmup_steps, "min_lr_ratio": 0.1},
        total_steps=args.steps,
    )
    pad_id = sp.pad_id() if sp.pad_id() >= 0 else 0
    eos_ids = sp.EncodeAsIds("<|end|>")
    eos_id = eos_ids[-1] if eos_ids else -1
    trace: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "model_config": str(args.model_config),
        "train": str(args.train),
        "val": str(args.val),
        "probe_file": str(args.learning_probes),
        "steps": args.steps,
        "objective": {
            "sft_weight": args.sft_weight,
            "probe_weight": args.probe_weight,
            "contrastive_weight": args.contrastive_weight,
            "desired_margin": args.desired_margin,
            "eos_loss_weight": args.eos_loss_weight,
        },
        "history": [],
    }
    diag: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "train": str(args.train),
        "val": str(args.val),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "probe_pairs": len(probe_pairs),
        "steps": args.steps,
        "losses": [],
    }

    def eval_trace(step: int, loss_row: dict[str, float] | None, elapsed: float) -> None:
        val = smoke.eval_loss(
            model, val_examples, pad_id, device, max_batches=8, batch_size=args.batch_size
        )
        val_by_category = smoke.eval_loss_by_category(
            model,
            val_examples,
            pad_id,
            device,
            max_batches=4,
            batch_size=args.batch_size,
        )
        probe_rows = smoke.evaluate_learning_probes(
            model, sp, learning_probes, device, args.generation_system
        )
        trace["history"].append(
            {
                "step": step,
                "train_loss": None if loss_row is None else loss_row["total"],
                "val_loss": val,
                "val_by_category": val_by_category,
                "lr": scheduler.get_last_lr()[0] if step else None,
                "elapsed_seconds": elapsed,
                "loss_components": loss_row,
                "probes": probe_rows,
            }
        )
        write_outputs(trace, args.trace_json, args.trace_html, args.neuro_html)
        print(f"step {step:4d} trace | val_loss={val:.4f}", flush=True)
        for row in probe_rows:
            margin = row.get("margin")
            margin_s = "n/a" if margin is None else f"{margin:+.3f}"
            flags = f" forbidden={row['forbidden_hits']}" if row.get("forbidden_hits") else ""
            print(
                f"  {row['id']}: target_nll={row['target_nll']:.3f} margin={margin_s}{flags}",
                flush=True,
            )

    t0 = time.time()
    eval_trace(0, None, 0.0)
    model.train()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        sft_values: list[float] = []
        for _ in range(args.grad_accum):
            batch = smoke.collate(next(sft_iter), pad_id, device)
            with smoke.autocast_context(device):
                out = model(input_ids=batch["input_ids"])
                sft_loss = smoke.weighted_shift_loss(
                    out["logits"],
                    batch["labels"],
                    eos_id=eos_id,
                    eos_loss_weight=args.eos_loss_weight,
                )
            ((args.sft_weight * sft_loss) / max(1, args.grad_accum)).backward()
            sft_values.append(float(sft_loss.item()))

        probe_batch = rng.choices(probe_pairs, k=args.probe_batch_size)
        probe_values: list[float] = []
        target_values: list[float] = []
        contrast_values: list[float] = []
        for item in probe_batch:
            target_nll = sequence_nll(model, sp, item["prompt"], item["target"], device)
            negative_nll = sequence_nll(model, sp, item["prompt"], item["negative"], device)
            contrastive = F.softplus(target_nll - negative_nll + args.desired_margin)
            probe_loss = target_nll + args.contrastive_weight * contrastive
            ((args.probe_weight * probe_loss) / max(1, args.probe_batch_size)).backward()
            target_values.append(float(target_nll.item()))
            contrast_values.append(float(contrastive.item()))
            probe_values.append(float(probe_loss.item()))

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        loss_row = {
            "total": args.sft_weight * (sum(sft_values) / len(sft_values))
            + args.probe_weight * (sum(probe_values) / len(probe_values)),
            "sft": sum(sft_values) / len(sft_values),
            "probe": sum(probe_values) / len(probe_values),
            "probe_target_nll": sum(target_values) / len(target_values),
            "probe_contrastive": sum(contrast_values) / len(contrast_values),
        }
        if not all(math.isfinite(v) for v in loss_row.values()):
            raise RuntimeError(f"non-finite loss row at step {step}: {loss_row}")
        diag["losses"].append({"step": step, **loss_row, "lr": scheduler.get_last_lr()[0]})
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            print(
                f"step {step:4d} | total={loss_row['total']:.4f} "
                f"sft={loss_row['sft']:.4f} probe={loss_row['probe']:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}",
                flush=True,
            )
        if step == 1 or step % args.learning_trace_every == 0 or step == args.steps:
            eval_trace(step, loss_row, time.time() - t0)

    path = save_checkpoint(model, optimizer, scheduler, args.steps, args.output_dir)
    trace["saved_checkpoint"] = str(path)
    diag["saved_checkpoint"] = str(path)
    write_outputs(trace, args.trace_json, args.trace_html, args.neuro_html)
    if args.diag_json:
        args.diag_json.parent.mkdir(parents=True, exist_ok=True)
        args.diag_json.write_text(json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {path}", flush=True)


if __name__ == "__main__":
    main()
