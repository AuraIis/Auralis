#!/usr/bin/env python3
"""Tiny supervised-finetuning smoke for rescued German Helix data.

This is intentionally small and boring:

- loads a Helix checkpoint
- reads rescued ``*.helix.jsonl`` SFT data
- masks loss to assistant content + assistant ``<|end|>``
- runs a few optimizer steps
- prints before/after greedy generations for fixed probes

The goal is not to produce a final model. It is a gate: if this does not learn
cleanly and stop cleanly, do not start a larger SFT run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_gradient_checkpointing(model, enabled: bool) -> None:
    if enabled and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    elif not enabled and hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


HELIX_TURN_RE = re.compile(r"<\|(system|user|assistant)\|>\n(.*?)\n<\|end\|>\n", re.DOTALL)


@dataclass
class SFTExample:
    input_ids: list[int]
    labels: list[int]
    source: str = ""
    category: str = ""
    block: str = ""
    family: str = ""


@dataclass
class LearningProbe:
    id: str
    category: str
    prompt: str
    target_answers: list[str]
    negative_answers: list[str]
    expect_contains: list[str]
    forbid_contains: list[str]
    max_new_tokens: int = 64


def _encode(sp: spm.SentencePieceProcessor, text: str) -> list[int]:
    return list(sp.EncodeAsIds(text))


def encode_helix_sft(sp: spm.SentencePieceProcessor, text: str, max_length: int) -> SFTExample | None:
    """Encode one canonical Helix prompt with assistant-only labels."""
    pos = 0
    assistant_ranges: list[tuple[int, int]] = []
    result_ranges: list[tuple[int, int]] = []
    saw_assistant = False

    for match in HELIX_TURN_RE.finditer(text):
        if match.start() != pos:
            return None
        role = match.group(1)
        header = f"<|{role}|>\n"
        if role == "assistant":
            saw_assistant = True
            # Mask only the assistant's content plus its closing <|end|>. The
            # assistant-open tag remains prompt context, which matches
            # inference where generation starts right after that tag.
            astart, aend = match.start() + len(header), match.end()
            assistant_ranges.append((astart, aend))
            # Tool-Use Phase 2: the <result>...</result> block is INJECTED by the
            # harness at inference (the model stops at </tool> and never writes it).
            # Exclude it (plus surrounding newlines) from the loss, else the model
            # learns to fabricate results -> exactly the fake-<result> failure the
            # tool-gate forbids. (No-op for call_only traces with no <result>.)
            for m in re.finditer(r"\n?<result>.*?</result>\n?", text[astart:aend], re.DOTALL):
                result_ranges.append((astart + m.start(), astart + m.end()))
        pos = match.end()

    if pos != len(text) or not saw_assistant:
        return None

    proto = sp.EncodeAsImmutableProto(text)
    ids = [piece.id for piece in proto.pieces]
    labels: list[int] = []
    for piece in proto.pieces:
        if piece.end <= piece.begin:
            labels.append(-100)
            continue
        in_assistant = any(piece.begin >= s and piece.end <= e for s, e in assistant_ranges)
        in_result = any(piece.begin >= s and piece.end <= e for s, e in result_ranges)
        labels.append(piece.id if (in_assistant and not in_result) else -100)

    if len(ids) < 8:
        return None
    if len(ids) > max_length:
        return None
    if all(x == -100 for x in labels):
        return None

    # Guard against accidental tokenization drift from segment encoding.
    if sp.DecodeIds(ids) != text:
        return None
    return SFTExample(input_ids=ids, labels=labels)


def load_examples(path: Path, sp: spm.SentencePieceProcessor, max_length: int, limit: int | None) -> list[SFTExample]:
    out: list[SFTExample] = []
    dropped = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            ex = encode_helix_sft(sp, rec["text"], max_length=max_length)
            if ex is None:
                dropped += 1
                continue
            ex.source = str(rec.get("source", ""))
            ex.category = str(rec.get("category", ""))
            ex.block = str(rec.get("block", ""))
            ex.family = str(rec.get("family", ""))
            out.append(ex)
            if limit and len(out) >= limit:
                break
    if not out:
        raise SystemExit(f"no usable examples in {path} (dropped={dropped})")
    print(f"loaded {len(out):,} examples from {path} (dropped={dropped:,})", flush=True)
    return out


def collate(batch: list[SFTExample], pad_id: int, device: torch.device) -> dict[str, torch.Tensor]:
    max_len = max(len(ex.input_ids) for ex in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, ex in enumerate(batch):
        n = len(ex.input_ids)
        input_ids[i, :n] = torch.tensor(ex.input_ids, dtype=torch.long)
        labels[i, :n] = torch.tensor(ex.labels, dtype=torch.long)
    return {"input_ids": input_ids.to(device), "labels": labels.to(device)}


def parse_category_weights(raw: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    if not raw.strip():
        return weights
    for item in raw.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise SystemExit(f"bad --category-weights item: {item!r}; expected category=weight")
        key, value = item.split("=", 1)
        try:
            weight = float(value)
        except ValueError as exc:
            raise SystemExit(f"bad weight in --category-weights item: {item!r}") from exc
        if weight <= 0:
            raise SystemExit(f"category weight must be > 0: {item!r}")
        weights[key.strip()] = weight
    return weights


def batches(
    examples: list[SFTExample],
    batch_size: int,
    rng: random.Random,
    category_weights: dict[str, float] | None = None,
    family_balanced: bool = False,
    bucket: bool = False,
) -> Iterable[list[SFTExample]]:
    if family_balanced:
        by_family: dict[str, list[SFTExample]] = {}
        for ex in examples:
            family = ex.family or ex.block or ex.category or "unknown"
            by_family.setdefault(family, []).append(ex)
        families = sorted(by_family)
        if not families:
            raise SystemExit("family-balanced sampler requested but no family/block metadata found")
        while True:
            rng.shuffle(families)
            for family in families:
                group = by_family[family]
                if category_weights:
                    weights = [category_weights.get(ex.category, 1.0) for ex in group]
                    yield rng.choices(group, weights=weights, k=batch_size)
                else:
                    yield rng.choices(group, k=batch_size)

    if category_weights:
        weights = [category_weights.get(ex.category, 1.0) for ex in examples]
        while True:
            yield rng.choices(examples, weights=weights, k=batch_size)

    if bucket:
        # length-bucketed: sort by length, fixed-size batches of similar length
        # (minimal padding, NO cross-example contamination), shuffle batch order
        sorted_idx = sorted(range(len(examples)), key=lambda j: len(examples[j].input_ids))
        groups = [sorted_idx[i : i + batch_size] for i in range(0, len(sorted_idx), batch_size)]
        while True:
            rng.shuffle(groups)
            for g in groups:
                yield [examples[j] for j in g]

    order = list(range(len(examples)))
    while True:
        rng.shuffle(order)
        for i in range(0, len(order), batch_size):
            yield [examples[j] for j in order[i : i + batch_size]]


@torch.no_grad()
def eval_loss(model, examples: list[SFTExample], pad_id: int, device: torch.device, max_batches: int, batch_size: int) -> float:
    model.eval()
    losses: list[float] = []
    for i in range(0, min(len(examples), max_batches * batch_size), batch_size):
        batch = collate(examples[i : i + batch_size], pad_id, device)
        with autocast_context(device):
            out = model(input_ids=batch["input_ids"])
            loss = weighted_shift_loss(out["logits"], batch["labels"], eos_id=-1, eos_loss_weight=1.0)
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / max(1, len(losses))


@torch.no_grad()
def eval_loss_by_category(
    model,
    examples: list[SFTExample],
    pad_id: int,
    device: torch.device,
    max_batches: int,
    batch_size: int,
) -> dict[str, float]:
    by_category: dict[str, list[SFTExample]] = {}
    for ex in examples:
        by_category.setdefault(ex.category or "unknown", []).append(ex)
    out: dict[str, float] = {}
    for category, category_examples in sorted(by_category.items()):
        out[category] = eval_loss(
            model,
            category_examples,
            pad_id,
            device,
            max_batches=max_batches,
            batch_size=batch_size,
        )
    return out


def _norm_text(text: str) -> str:
    text = text.lower()
    replacements = {
        "ö": "oe",
        "ü": "ue",
        "ä": "ae",
        "ß": "ss",
        "osterreich": "oesterreich",
        "munchen": "muenchen",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return re.sub(r"\s+", " ", text).strip()


def _contains_any(answer: str, needles: list[str]) -> list[str]:
    norm = _norm_text(answer)
    return [needle for needle in needles if _norm_text(needle) in norm]


def load_learning_probes(path: Path) -> list[LearningProbe]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    defaults = data.get("defaults", {}) or {}
    probes: list[LearningProbe] = []
    for raw in data.get("probes", []):
        target_answers = list(raw.get("target_answers") or raw.get("targets") or [])
        if not target_answers:
            raise SystemExit(f"learning probe {raw.get('id')} has no target_answers")
        probes.append(
            LearningProbe(
                id=str(raw["id"]),
                category=str(raw.get("category", "unknown")),
                prompt=str(raw["prompt"]),
                target_answers=target_answers,
                negative_answers=list(raw.get("negative_answers") or raw.get("negatives") or []),
                expect_contains=list(raw.get("expect_contains") or []),
                forbid_contains=list(raw.get("forbid_contains") or []),
                max_new_tokens=int(raw.get("max_new_tokens") or defaults.get("max_new_tokens") or 64),
            )
        )
    if not probes:
        raise SystemExit(f"no learning probes found in {path}")
    return probes


@torch.no_grad()
def continuation_nll(
    model,
    sp: spm.SentencePieceProcessor,
    prompt: str,
    continuation: str,
    device: torch.device,
) -> dict[str, float | int]:
    prompt_ids = sp.EncodeAsIds(prompt)
    continuation_ids = sp.EncodeAsIds(continuation)
    if not continuation_ids:
        return {"avg_nll": float("inf"), "total_nll": float("inf"), "tokens": 0, "ppl": float("inf")}
    ids = prompt_ids + continuation_ids
    input_ids = torch.tensor([ids], device=device, dtype=torch.long)
    labels = torch.tensor(continuation_ids, device=device, dtype=torch.long)
    start = len(prompt_ids) - 1
    with autocast_context(device):
        logits = model(input_ids=input_ids)["logits"][0, start : start + len(continuation_ids)]
    losses = F.cross_entropy(logits.float(), labels, reduction="none")
    avg = float(losses.mean().item())
    total = float(losses.sum().item())
    return {"avg_nll": avg, "total_nll": total, "tokens": len(continuation_ids), "ppl": float(torch.exp(losses.mean()).item())}


@torch.no_grad()
def next_token_topk(
    model,
    sp: spm.SentencePieceProcessor,
    prompt: str,
    device: torch.device,
    k: int = 8,
) -> list[dict[str, float | str | int]]:
    input_ids = torch.tensor([sp.EncodeAsIds(prompt)], device=device, dtype=torch.long)
    with autocast_context(device):
        logits = model(input_ids=input_ids)["logits"][0, -1].float()
    probs = torch.softmax(logits, dim=-1)
    values, indices = torch.topk(probs, k=min(k, probs.numel()))
    out = []
    for prob, idx in zip(values.tolist(), indices.tolist()):
        out.append({"token_id": int(idx), "piece": sp.IdToPiece(int(idx)), "text": sp.DecodeIds([int(idx)]), "prob": float(prob)})
    return out


@torch.no_grad()
def evaluate_learning_probes(
    model,
    sp: spm.SentencePieceProcessor,
    probes: list[LearningProbe],
    device: torch.device,
    generation_system: str,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for probe in probes:
        prompt = build_inference_prompt([{"role": "user", "content": probe.prompt}], default_system=generation_system)
        answer = generate(model, sp, prompt, device, max_new_tokens=probe.max_new_tokens).strip()
        target_scores = [
            {"text": target, **continuation_nll(model, sp, prompt, target, device)}
            for target in probe.target_answers
        ]
        negative_scores = [
            {"text": negative, **continuation_nll(model, sp, prompt, negative, device)}
            for negative in probe.negative_answers
        ]
        best_target = min(target_scores, key=lambda item: float(item["avg_nll"]))
        best_negative = min(negative_scores, key=lambda item: float(item["avg_nll"])) if negative_scores else None
        target_nll = float(best_target["avg_nll"])
        negative_nll = float(best_negative["avg_nll"]) if best_negative else None
        margin = (negative_nll - target_nll) if negative_nll is not None else None
        expected_hits = _contains_any(answer, probe.expect_contains)
        forbidden_hits = _contains_any(answer, probe.forbid_contains)
        rows.append(
            {
                "id": probe.id,
                "category": probe.category,
                "prompt": probe.prompt,
                "answer": answer,
                "best_target": best_target,
                "best_negative": best_negative,
                "target_nll": target_nll,
                "negative_nll": negative_nll,
                "margin": margin,
                "expected_hits": expected_hits,
                "forbidden_hits": forbidden_hits,
                "top_next_tokens": next_token_topk(model, sp, prompt, device),
            }
        )
    model.train()
    return rows


@torch.no_grad()
def generate(model, sp: spm.SentencePieceProcessor, prompt: str, device: torch.device, max_new_tokens: int) -> str:
    model.eval()
    input_ids = torch.tensor([sp.EncodeAsIds(prompt)], device=device, dtype=torch.long)
    new_ids: list[int] = []
    end_ids = sp.EncodeAsIds("<|end|>")
    end_id = end_ids[-1] if end_ids else sp.PieceToId("<|end|>")
    for _ in range(max_new_tokens):
        with autocast_context(device):
            out = model(input_ids=input_ids)
        logits = out["logits"][0, -1]
        next_id = int(torch.argmax(logits).item())
        new_ids.append(next_id)
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == end_id:
            break
    model.train()
    return sp.DecodeIds(new_ids)


def weighted_shift_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    eos_id: int,
    eos_loss_weight: float,
) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    flat_labels = shift_labels.reshape(-1)
    losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        flat_labels,
        ignore_index=-100,
        reduction="none",
    )
    mask = flat_labels.ne(-100)
    weights = mask.to(losses.dtype)
    if eos_id >= 0 and eos_loss_weight != 1.0:
        weights = torch.where(flat_labels.eq(eos_id), weights * eos_loss_weight, weights)
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


def load_checkpoint_weights(model, checkpoint: Path, device: torch.device) -> int | None:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = {k.replace("_orig_mod.", ""): v for k, v in payload["model"].items()}
    missing, extra = model.load_state_dict(state, strict=False)
    if missing or extra:
        print(f"state_dict mismatch: missing={len(missing)} extra={len(extra)}", flush=True)
        print(f"  first missing: {missing[:3]}", flush=True)
        print(f"  first extra  : {extra[:3]}", flush=True)
        if _KERNEL_ACTIVE:
            print("  mamba backend: mamba_ssm", flush=True)
        raise SystemExit(2)
    return payload.get("state", {}).get("step")


def save_sft_checkpoint(model, optimizer, scheduler, step: int, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"sft_smoke_step_{step}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "kind": "sft_smoke",
        },
        path,
    )
    return path


def _load_render_function(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import renderer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.render


def write_learning_trace_outputs(
    trace: dict[str, Any],
    *,
    trace_json: Path | None,
    trace_html: Path | None,
    neuro_html: Path | None,
    auto_refresh: int,
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
        neuro_html.write_text(render_neuro(trace, auto_refresh=auto_refresh), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-config", type=Path, default=REPO / "configs/model/helix_v2_1b.yaml")
    ap.add_argument("--checkpoint", type=Path, default=REPO / "checkpoints/phase1_pretrain/best.pt")
    ap.add_argument("--tokenizer", type=Path, default=REPO / "tokenizer/helix_v2_tokenizer.model")
    ap.add_argument("--train", type=Path, default=REPO / "data/training/sft_rescued/balanced/de_strict/train.helix.jsonl")
    ap.add_argument("--val", type=Path, default=REPO / "data/training/sft_rescued/balanced/de_strict/val.helix.jsonl")
    ap.add_argument("--output-dir", type=Path, default=REPO / "checkpoints/sft_smoke_de")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=1536)
    ap.add_argument("--train-limit", type=int, default=512)
    ap.add_argument("--val-limit", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2.0e-5)
    ap.add_argument("--warmup-steps", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--save-final", action="store_true")
    ap.add_argument("--grad-ckpt", action="store_true", help="enable gradient checkpointing (slower, less VRAM)")
    ap.add_argument("--save-every", type=int, default=0, help="save checkpoint every N steps (0=off)")
    ap.add_argument("--bucket", action="store_true", help="length-bucketed batching (minimal padding, no contamination)")
    ap.add_argument(
        "--category-weights",
        default="",
        help="Optional comma-separated weighted sampler map, e.g. hallucination_guard=8,qa_de=2.",
    )
    ap.add_argument(
        "--family-balanced-sampler",
        action="store_true",
        help="Sample across family/block groups before category weighting.",
    )
    ap.add_argument("--eos-loss-weight", type=float, default=1.0, help="Extra loss weight for assistant <|end|> tokens.")
    ap.add_argument("--diag-json", type=Path, default=None, help="Optional JSON report with losses and generations.")
    ap.add_argument("--probe", action="append", default=None, help="Extra fixed generation probe. Repeatable.")
    ap.add_argument(
        "--learning-probes",
        type=Path,
        default=None,
        help="Optional YAML probes for visual learning traces with target/negative likelihoods.",
    )
    ap.add_argument(
        "--learning-trace-json",
        type=Path,
        default=None,
        help="Write per-step learning probe traces to this JSON file.",
    )
    ap.add_argument(
        "--learning-trace-html",
        type=Path,
        default=None,
        help="Continuously render the learning trace dashboard HTML to this file.",
    )
    ap.add_argument(
        "--learning-neuro-html",
        type=Path,
        default=None,
        help="Continuously render a live-style knowledge graph HTML to this file.",
    )
    ap.add_argument(
        "--learning-html-auto-refresh",
        type=int,
        default=10,
        help="Meta-refresh seconds for live HTML outputs. Use 0 to disable.",
    )
    ap.add_argument(
        "--learning-trace-every",
        type=int,
        default=0,
        help="Evaluate learning probes every N optimizer steps. Defaults to --eval-every when probes are set.",
    )
    ap.add_argument(
        "--generation-system",
        default=(
            "Du bist Auralis, ein hilfreicher deutscher KI-Assistent. "
            "Antworte korrekt, knapp und ehrlich. Wenn etwas unsicher oder erfunden ist, sage das deutlich."
        ),
    )
    ap.add_argument("--data-check-only", action="store_true",
                    help="Only load/tokenize examples and verify assistant masks; do not build the model.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    rng = random.Random(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"device: {device}", flush=True)

    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    pad_id = sp.PieceToId("<pad>")
    eos_id = sp.EncodeAsIds("<|end|>")[-1]
    train_examples = load_examples(args.train, sp, args.max_length, args.train_limit)
    val_examples = load_examples(args.val, sp, args.max_length, args.val_limit)
    train_masked = [sum(1 for x in ex.labels if x != -100) for ex in train_examples]
    val_masked = [sum(1 for x in ex.labels if x != -100) for ex in val_examples]
    category_weights = parse_category_weights(args.category_weights)
    train_by_category: dict[str, int] = {}
    val_by_category: dict[str, int] = {}
    train_by_family: dict[str, int] = {}
    for ex in train_examples:
        train_by_category[ex.category or "unknown"] = train_by_category.get(ex.category or "unknown", 0) + 1
        family = ex.family or ex.block or "unknown"
        train_by_family[family] = train_by_family.get(family, 0) + 1
    for ex in val_examples:
        val_by_category[ex.category or "unknown"] = val_by_category.get(ex.category or "unknown", 0) + 1
    print(
        "mask check: "
        f"train assistant tokens min/avg/max="
        f"{min(train_masked)}/{sum(train_masked)/len(train_masked):.1f}/{max(train_masked)}; "
        f"val={min(val_masked)}/{sum(val_masked)/len(val_masked):.1f}/{max(val_masked)}",
        flush=True,
    )
    print(f"train categories: {train_by_category}", flush=True)
    print(f"val categories: {val_by_category}", flush=True)
    print(f"train families: {len(train_by_family)} groups", flush=True)
    if category_weights:
        print(f"weighted sampler: {category_weights}", flush=True)
    if args.family_balanced_sampler:
        print("family-balanced sampler: enabled", flush=True)
    if args.data_check_only:
        print("data-check-only: PASS", flush=True)
        return
    learning_probes = load_learning_probes(args.learning_probes) if args.learning_probes else []
    learning_trace_every = args.learning_trace_every or args.eval_every
    learning_trace = {
        "checkpoint": str(args.checkpoint),
        "train": str(args.train),
        "val": str(args.val),
        "model_config": str(args.model_config),
        "probe_file": str(args.learning_probes) if args.learning_probes else None,
        "steps": args.steps,
        "history": [],
    }
    if learning_probes:
        print(f"learning probes: {len(learning_probes)} from {args.learning_probes}", flush=True)

    print(f"building model: {args.model_config}", flush=True)
    model = build_model(args.model_config).to(device)
    apply_gradient_checkpointing(model, enabled=args.grad_ckpt)
    loaded_step = load_checkpoint_weights(model, args.checkpoint, device)
    print(f"loaded checkpoint: {args.checkpoint} (pretrain step={loaded_step})", flush=True)
    print(f"mamba backend: {'mamba_ssm' if _KERNEL_ACTIVE else 'native'}", flush=True)

    optimizer = build_optimizer(
        model,
        {"name": "adamw", "lr": args.lr, "betas": [0.9, 0.95], "weight_decay": 0.0, "eps": 1e-8},
    )
    scheduler = build_scheduler(
        optimizer,
        {"type": "cosine", "warmup_steps": args.warmup_steps, "min_lr_ratio": 0.1},
        total_steps=args.steps,
    )

    probes = [
        "Was ist die Hauptstadt von Deutschland?",
        "Erklaere kurz, was Wasser ist.",
        "Schrieb Goethe Mein Kampf? Antworte kurz und korrekt.",
        "Ist Bonn heute die Hauptstadt von Deutschland?",
    ]
    if args.probe:
        probes.extend(args.probe)
    diag = {
        "checkpoint": str(args.checkpoint),
        "train": str(args.train),
        "val": str(args.val),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "category_weights": category_weights,
        "family_balanced_sampler": args.family_balanced_sampler,
        "eos_loss_weight": args.eos_loss_weight,
        "generation_system": args.generation_system,
        "train_by_category": train_by_category,
        "val_by_category": val_by_category,
        "history": [],
        "generations_before": [],
        "generations_after": [],
    }
    print("\n--- generations before SFT ---", flush=True)
    for probe in probes:
        prompt = build_inference_prompt([{"role": "user", "content": probe}], default_system=args.generation_system)
        answer = generate(model, sp, prompt, device, max_new_tokens=64)
        diag["generations_before"].append({"prompt": probe, "answer": answer})
        print(f"\nPROMPT: {probe}\n{answer!r}", flush=True)

    initial_val = eval_loss(model, val_examples, pad_id, device, max_batches=8, batch_size=args.batch_size)
    initial_by_category = eval_loss_by_category(
        model, val_examples, pad_id, device, max_batches=8, batch_size=args.batch_size
    )
    print(f"\ninitial val_loss={initial_val:.4f}", flush=True)
    print(f"initial val_by_category={initial_by_category}", flush=True)
    diag["initial_val_loss"] = initial_val
    diag["initial_val_by_category"] = initial_by_category
    if learning_probes:
        initial_learning = evaluate_learning_probes(model, sp, learning_probes, device, args.generation_system)
        learning_trace["history"].append(
            {
                "step": 0,
                "train_loss": None,
                "val_loss": initial_val,
                "val_by_category": initial_by_category,
                "lr": None,
                "elapsed_seconds": 0.0,
                "probes": initial_learning,
            }
        )
        write_learning_trace_outputs(
            learning_trace,
            trace_json=args.learning_trace_json,
            trace_html=args.learning_trace_html,
            neuro_html=args.learning_neuro_html,
            auto_refresh=args.learning_html_auto_refresh,
        )
        print("initial learning probes:", flush=True)
        for row in initial_learning:
            margin = row["margin"]
            margin_s = "n/a" if margin is None else f"{margin:+.3f}"
            flags = []
            if row["forbidden_hits"]:
                flags.append(f"forbidden={row['forbidden_hits']}")
            print(
                f"  {row['id']}: target_nll={row['target_nll']:.3f} margin={margin_s} "
                f"answer={row['answer'][:120]!r} {' '.join(flags)}",
                flush=True,
            )

    model.train()
    batch_iter = batches(
        train_examples,
        args.batch_size,
        rng,
        category_weights=category_weights,
        family_balanced=args.family_balanced_sampler,
        bucket=args.bucket,
    )
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, args.steps + 1):
        step_loss = 0.0
        for _ in range(args.grad_accum):
            batch = collate(next(batch_iter), pad_id, device)
            with autocast_context(device):
                out = model(input_ids=batch["input_ids"])
                loss = weighted_shift_loss(
                    out["logits"],
                    batch["labels"],
                    eos_id=eos_id,
                    eos_loss_weight=args.eos_loss_weight,
                ) / args.grad_accum
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite SFT loss at step {step}: {loss.item()}")
            loss.backward()
            step_loss += float(loss.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if args.save_every and step % args.save_every == 0:
            sp_path = save_sft_checkpoint(model, optimizer, scheduler, step, args.output_dir)
            print(f"  [checkpoint] step {step} -> {sp_path}", flush=True)

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            val = eval_loss(model, val_examples, pad_id, device, max_batches=8, batch_size=args.batch_size)
            val_by_category = eval_loss_by_category(
                model, val_examples, pad_id, device, max_batches=8, batch_size=args.batch_size
            )
            lr = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0
            print(f"step {step:4d} | train_loss={step_loss:.4f} | val_loss={val:.4f} | lr={lr:.2e} | {elapsed:.1f}s", flush=True)
            print(f"           val_by_category={val_by_category}", flush=True)
            diag["history"].append(
                {
                    "step": step,
                    "train_loss": step_loss,
                    "val_loss": val,
                    "val_by_category": val_by_category,
                    "lr": lr,
                    "elapsed_seconds": elapsed,
                }
            )
        if learning_probes and (
            step == 1 or step % learning_trace_every == 0 or step == args.steps
        ):
            # Reuse the latest scalar eval when the learning trace aligns with
            # normal eval. Otherwise compute a small validation snapshot here.
            if "val" not in locals() or not (step == 1 or step % args.eval_every == 0 or step == args.steps):
                val = eval_loss(model, val_examples, pad_id, device, max_batches=8, batch_size=args.batch_size)
                val_by_category = eval_loss_by_category(
                    model, val_examples, pad_id, device, max_batches=8, batch_size=args.batch_size
                )
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
            probe_rows = evaluate_learning_probes(model, sp, learning_probes, device, args.generation_system)
            learning_trace["history"].append(
                {
                    "step": step,
                    "train_loss": step_loss,
                    "val_loss": val,
                    "val_by_category": val_by_category,
                    "lr": lr,
                    "elapsed_seconds": elapsed,
                    "probes": probe_rows,
                }
            )
            write_learning_trace_outputs(
                learning_trace,
                trace_json=args.learning_trace_json,
                trace_html=args.learning_trace_html,
                neuro_html=args.learning_neuro_html,
                auto_refresh=args.learning_html_auto_refresh,
            )
            print("           learning_trace=", flush=True)
            for row in probe_rows:
                margin = row["margin"]
                margin_s = "n/a" if margin is None else f"{margin:+.3f}"
                print(
                    f"             {row['id']}: target_nll={row['target_nll']:.3f} "
                    f"margin={margin_s} answer={row['answer'][:100]!r}",
                    flush=True,
                )

    print("\n--- generations after SFT smoke ---", flush=True)
    for probe in probes:
        prompt = build_inference_prompt([{"role": "user", "content": probe}], default_system=args.generation_system)
        answer = generate(model, sp, prompt, device, max_new_tokens=96)
        diag["generations_after"].append({"prompt": probe, "answer": answer})
        print(f"\nPROMPT: {probe}\n{answer!r}", flush=True)

    if args.save_final:
        path = save_sft_checkpoint(model, optimizer, scheduler, args.steps, args.output_dir)
        diag["saved_checkpoint"] = str(path)
        print(f"\nsaved: {path}", flush=True)
    if args.diag_json:
        args.diag_json.parent.mkdir(parents=True, exist_ok=True)
        args.diag_json.write_text(json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"diag_json: {args.diag_json}", flush=True)
    if args.learning_trace_json:
        write_learning_trace_outputs(
            learning_trace,
            trace_json=args.learning_trace_json,
            trace_html=args.learning_trace_html,
            neuro_html=args.learning_neuro_html,
            auto_refresh=args.learning_html_auto_refresh,
        )
        print(f"learning_trace_json: {args.learning_trace_json}", flush=True)
    if args.learning_trace_html:
        print(f"learning_trace_html: {args.learning_trace_html}", flush=True)
    if args.learning_neuro_html:
        print(f"learning_neuro_html: {args.learning_neuro_html}", flush=True)
    print("\nSFT smoke finished", flush=True)


if __name__ == "__main__":
    main()
