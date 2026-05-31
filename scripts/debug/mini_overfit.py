"""Mini-overfit diagnostic for Helix.

The goal is to answer a narrow structural question: can the model/tokenizer/
loss/backprop stack learn a tiny perfect set? This is not useful training.

Expected behavior for a healthy stack:
- train/eval loss falls sharply on the tiny set
- greedy generations move toward the memorized answers
- no NaNs, state_dict/load, or masking failures
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import sentencepiece as spm
import torch

REPO = Path(__file__).resolve().parents[2]
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
from auralis.tokenizer.chat_template import (  # noqa: E402
    build_inference_prompt,
    build_training_prompt,
)


HELIX_TURN_RE = re.compile(r"<\|(system|user|assistant)\|>\n(.*?)\n<\|end\|>\n", re.DOTALL)


@dataclass
class Example:
    prompt: str
    answer: str
    input_ids: list[int]
    labels: list[int]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_sft(
    sp: spm.SentencePieceProcessor,
    prompt: str,
    answer: str,
    *,
    max_length: int,
) -> Example:
    text = build_training_prompt(
        [
            {"role": "system", "content": "Du bist Auralis. Antworte exakt und kurz."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
    )
    pos = 0
    assistant_ranges: list[tuple[int, int]] = []
    for match in HELIX_TURN_RE.finditer(text):
        if match.start() != pos:
            raise ValueError(f"non-canonical chat template near char {pos}")
        role = match.group(1)
        if role == "assistant":
            assistant_ranges.append((match.start() + len("<|assistant|>\n"), match.end()))
        pos = match.end()
    if pos != len(text) or not assistant_ranges:
        raise ValueError("missing assistant span")

    proto = sp.encode_as_immutable_proto(text)
    input_ids = [piece.id for piece in proto.pieces]
    labels: list[int] = []
    for piece in proto.pieces:
        in_assistant = any(
            piece.end > piece.begin and piece.begin >= start and piece.end <= end
            for start, end in assistant_ranges
        )
        labels.append(piece.id if in_assistant else -100)

    if len(input_ids) > max_length:
        raise ValueError(f"example too long: {len(input_ids)} > {max_length}")
    if all(x == -100 for x in labels):
        raise ValueError("assistant labels are fully masked")
    if sp.decode(input_ids) != text:
        raise ValueError("tokenizer roundtrip failed")
    return Example(prompt=prompt, answer=answer, input_ids=input_ids, labels=labels)


def make_examples(sp: spm.SentencePieceProcessor, max_length: int) -> list[Example]:
    pairs = [
        ("Was ist die Hauptstadt von Deutschland?", "Berlin."),
        ("Rechne 2 + 2.", "4."),
        ("Was ist Wasser bei Raumtemperatur?", "Eine Flüssigkeit."),
        ("Schreibe eine Python-Funktion add(a, b).", "def add(a, b):\n    return a + b"),
        ("Wer schrieb Faust?", "Johann Wolfgang von Goethe."),
        ("Antworte mit genau einem Wort: Welche Farbe hat Gras?", "grün"),
        ("Ist eval() für fremde Eingaben sicher?", "Nein."),
        ("Liegt Paris in Frankreich?", "Ja."),
    ]
    return [encode_sft(sp, prompt, answer, max_length=max_length) for prompt, answer in pairs]


def collate(examples: list[Example], pad_id: int, device: torch.device) -> dict[str, torch.Tensor]:
    max_len = max(len(ex.input_ids) for ex in examples)
    input_ids = torch.full((len(examples), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), max_len), -100, dtype=torch.long)
    for row, ex in enumerate(examples):
        n = len(ex.input_ids)
        input_ids[row, :n] = torch.tensor(ex.input_ids, dtype=torch.long)
        labels[row, :n] = torch.tensor(ex.labels, dtype=torch.long)
    return {"input_ids": input_ids.to(device), "labels": labels.to(device)}


@torch.no_grad()
def eval_loss(model, batch: dict[str, torch.Tensor]) -> float:
    model.eval()
    out = model(input_ids=batch["input_ids"], labels=batch["labels"])
    return float(out["loss"])


@torch.no_grad()
def generate(model, sp: spm.SentencePieceProcessor, prompt: str, device: torch.device, max_new_tokens: int) -> str:
    model.eval()
    text = build_inference_prompt(
        [
            {"role": "system", "content": "Du bist Auralis. Antworte exakt und kurz."},
            {"role": "user", "content": prompt},
        ]
    )
    ids = sp.encode(text, out_type=int)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    end_id = sp.piece_to_id("<|end|>")
    new_ids: list[int] = []
    for _ in range(max_new_tokens):
        out = model(input_ids=x)
        next_id = int(out["logits"][0, -1].argmax())
        if next_id == end_id:
            break
        new_ids.append(next_id)
        x = torch.cat([x, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
    return sp.decode(new_ids).strip()


def load_checkpoint(model, checkpoint: Path, device: torch.device) -> None:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = {key.replace("_orig_mod.", ""): value for key, value in payload["model"].items()}
    missing, extra = model.load_state_dict(state, strict=False)
    if missing or extra:
        raise SystemExit(f"checkpoint mismatch: missing={len(missing)} extra={len(extra)}")
    step = payload.get("state", {}).get("step", payload.get("step", "?"))
    print(f"loaded checkpoint: {checkpoint} step={step}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, default=REPO / "configs/model/helix_v2_debug_tiny.yaml")
    parser.add_argument("--tokenizer", type=Path, default=REPO / "tokenizer/helix_v2_tokenizer.model")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260512)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} mamba_kernel={_KERNEL_ACTIVE}", flush=True)

    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    examples = make_examples(sp, max_length=args.max_length)
    batch = collate(examples, sp.pad_id(), device)
    print(
        f"examples={len(examples)} batch_shape={tuple(batch['input_ids'].shape)} "
        f"assistant_tokens={(batch['labels'] != -100).sum().item()}",
        flush=True,
    )

    model = build_model(args.model_config).to(device)
    print(f"params={model.count_parameters()/1e6:.2f}M vocab={model.config.vocab_size:,}", flush=True)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, device)
    if model.config.vocab_size != sp.get_piece_size():
        raise SystemExit(f"vocab mismatch: model={model.config.vocab_size} tokenizer={sp.get_piece_size()}")

    print("\n--- before ---", flush=True)
    for ex in examples[:4]:
        print(f"Q: {ex.prompt}\nA: {generate(model, sp, ex.prompt, device, args.max_new_tokens)!r}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    initial = eval_loss(model, batch)
    print(f"\ninitial_loss={initial:.4f}", flush=True)

    start = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        out = model(input_ids=batch["input_ids"], labels=batch["labels"])
        loss = out["loss"]
        if not torch.isfinite(loss):
            raise SystemExit(f"non-finite loss at step {step}: {loss}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step % max(1, args.steps // 6) == 0 or step == args.steps:
            current = eval_loss(model, batch)
            elapsed = time.time() - start
            print(f"step={step:4d} train_loss={float(loss):.4f} eval_loss={current:.4f} elapsed={elapsed:.1f}s", flush=True)

    final = eval_loss(model, batch)
    print("\n--- after ---", flush=True)
    exact = 0
    for ex in examples:
        pred = generate(model, sp, ex.prompt, device, args.max_new_tokens)
        hit = pred.startswith(ex.answer)
        exact += int(hit)
        print(f"{'OK' if hit else 'NO'} Q: {ex.prompt}\n   target={ex.answer!r}\n   pred  ={pred!r}", flush=True)

    ratio = initial / max(final, 1e-9)
    print("\n=== SUMMARY ===", flush=True)
    print(f"initial_loss={initial:.4f}", flush=True)
    print(f"final_loss={final:.4f}", flush=True)
    print(f"loss_drop_ratio={ratio:.2f}x", flush=True)
    print(f"exact_prefix_hits={exact}/{len(examples)}", flush=True)
    passed = final < 0.25 or (ratio >= 20.0 and exact >= len(examples) // 2)
    print(f"mini_overfit={'PASS' if passed else 'FAIL'}", flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
