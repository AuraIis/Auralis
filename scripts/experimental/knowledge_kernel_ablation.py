"""Tiny A/B ablation for the Duden/DNA knowledge-kernel idea.

This trains two fresh debug-tiny Helix models for the same number of steps:

- plain: facts as normal prose
- kernel: the same facts in ``<memory>`` definition blocks

Then it evaluates answer-only loss on QA probes. This is a smoke test, not a
benchmark; the goal is to catch obvious "this format is easier/harder to learn"
signals before touching real pretraining.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import sentencepiece as spm
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from auralis.model import build_model  # noqa: E402
from scripts.experimental.knowledge_kernel import (  # noqa: E402
    build_outputs,
    sample_entries,
)


@dataclass
class AblationResult:
    name: str
    train_loss_initial: float
    train_loss_final: float
    qa_loss_initial: float
    qa_loss_final: float
    train_drop_ratio: float
    qa_drop_ratio: float
    generations: dict[str, str]
    seconds: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_or_build_kernel_dir(path: Path, tokenizer: Path) -> None:
    if (path / "plain_corpus.txt").exists() and (path / "current_kernel.txt").exists():
        return
    build_outputs(sample_entries(), path, tokenizer)


def make_lm_batch(
    sp: spm.SentencePieceProcessor,
    text: str,
    *,
    seq_len: int,
    repeat: int,
    max_rows: int | None = None,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    ids = sp.encode((text.strip() + "\n\n") * repeat, out_type=int)
    rows: list[list[int]] = []
    stride = seq_len
    for start in range(0, max(0, len(ids) - seq_len), stride):
        chunk = ids[start : start + seq_len]
        if len(chunk) == seq_len:
            rows.append(chunk)
    if not rows:
        raise ValueError("training text is too short for requested seq_len")
    if max_rows is not None:
        rows = rows[:max_rows]
    x = torch.tensor(rows, dtype=torch.long, device=device)
    return {"input_ids": x, "labels": x.clone()}


def encode_answer_loss_batch(
    sp: spm.SentencePieceProcessor,
    rows: list[dict[str, str]],
    *,
    style: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    encoded: list[tuple[list[int], list[int]]] = []
    for row in rows:
        if style == "kernel":
            prefix = f"<recall>\nBegriff: {row['term']}\nFrage: {row['instruction']}\nAntwort: "
            suffix = f"{row['output']}\n</recall>\n"
        else:
            prefix = f"Begriff: {row['term']}\nFrage: {row['instruction']}\nAntwort: "
            suffix = f"{row['output']}\n"
        prefix_ids = sp.encode(prefix, out_type=int)
        suffix_ids = sp.encode(suffix, out_type=int)
        ids = prefix_ids + suffix_ids
        labels = [-100] * len(prefix_ids) + suffix_ids
        encoded.append((ids, labels))

    max_len = max(len(ids) for ids, _ in encoded)
    pad_id = sp.pad_id()
    x = torch.full((len(encoded), max_len), pad_id, dtype=torch.long, device=device)
    y = torch.full((len(encoded), max_len), -100, dtype=torch.long, device=device)
    for row, (ids, labels) in enumerate(encoded):
        x[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        y[row, : len(labels)] = torch.tensor(labels, dtype=torch.long, device=device)
    return {"input_ids": x, "labels": y}


@torch.no_grad()
def loss(model, batch: dict[str, torch.Tensor]) -> float:
    model.eval()
    return float(model(input_ids=batch["input_ids"], labels=batch["labels"])["loss"])


@torch.no_grad()
def generate(
    model,
    sp: spm.SentencePieceProcessor,
    prompt: str,
    *,
    device: torch.device,
    max_new_tokens: int,
) -> str:
    model.eval()
    ids = sp.encode(prompt, out_type=int)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    end_ids = {sp.piece_to_id("</recall>"), sp.piece_to_id("<|end|>"), sp.eos_id()}
    new_ids: list[int] = []
    for _ in range(max_new_tokens):
        out = model(input_ids=x)
        next_id = int(out["logits"][0, -1].argmax())
        if next_id in end_ids:
            break
        new_ids.append(next_id)
        x = torch.cat([x, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
    return sp.decode(new_ids).strip()


def train_one(
    *,
    name: str,
    train_text: str,
    qa_rows: list[dict[str, str]],
    sp: spm.SentencePieceProcessor,
    args: argparse.Namespace,
    device: torch.device,
) -> AblationResult:
    set_seed(args.seed)
    model = build_model(args.model_config).to(device)
    if model.config.vocab_size != sp.get_piece_size():
        raise ValueError(
            f"vocab mismatch: model={model.config.vocab_size} tokenizer={sp.get_piece_size()}"
        )
    train_batch = make_lm_batch(
        sp,
        train_text,
        seq_len=args.seq_len,
        repeat=args.repeat,
        max_rows=args.max_train_rows,
        device=device,
    )
    qa_style = "kernel" if name == "kernel" else "plain"
    qa_batch = encode_answer_loss_batch(sp, qa_rows, style=qa_style, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    train_initial = loss(model, train_batch)
    qa_initial = loss(model, qa_batch)
    start = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        out = model(input_ids=train_batch["input_ids"], labels=train_batch["labels"])
        train_loss = out["loss"]
        if not torch.isfinite(train_loss):
            raise ValueError(f"{name}: non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step == args.steps or step % max(1, args.steps // 4) == 0:
            print(f"{name} step={step} train_loss={float(train_loss.detach()):.4f}", flush=True)

    elapsed = time.time() - start
    train_final = loss(model, train_batch)
    qa_final = loss(model, qa_batch)

    generations: dict[str, str] = {}
    for row in qa_rows[: args.generations]:
        if qa_style == "kernel":
            prompt = f"<recall>\nBegriff: {row['term']}\nFrage: {row['instruction']}\nAntwort: "
        else:
            prompt = f"Begriff: {row['term']}\nFrage: {row['instruction']}\nAntwort: "
        generations[row["instruction"]] = generate(
            model,
            sp,
            prompt,
            device=device,
            max_new_tokens=args.max_new_tokens,
        )

    return AblationResult(
        name=name,
        train_loss_initial=train_initial,
        train_loss_final=train_final,
        qa_loss_initial=qa_initial,
        qa_loss_final=qa_final,
        train_drop_ratio=train_initial / max(train_final, 1e-9),
        qa_drop_ratio=qa_initial / max(qa_final, 1e-9),
        generations=generations,
        seconds=elapsed,
    )


def write_report(path: Path, results: list[AblationResult]) -> None:
    by_name = {r.name: r for r in results}
    lines = [
        "# Knowledge Kernel A/B Ablation",
        "",
        "| Variant | Train Loss | QA Loss | Train Drop | QA Drop | Seconds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.name} | {result.train_loss_final:.4f} | "
            f"{result.qa_loss_final:.4f} | {result.train_drop_ratio:.2f}x | "
            f"{result.qa_drop_ratio:.2f}x | {result.seconds:.1f} |"
        )
    if "plain" in by_name and "kernel" in by_name:
        plain = by_name["plain"]
        kernel = by_name["kernel"]
        delta = plain.qa_loss_final - kernel.qa_loss_final
        winner = "kernel" if delta > 0 else "plain"
        lines.extend(
            [
                "",
                "## Signal",
                "",
                f"- Lower QA loss wins. Winner in this smoke: `{winner}`.",
                f"- QA loss delta plain-minus-kernel: `{delta:.4f}`.",
                "",
            ]
        )
    lines.extend(["## Generations", ""])
    for result in results:
        lines.append(f"### {result.name}")
        lines.append("")
        for question, answer in result.generations.items():
            lines.append(f"- Q: {question}")
            lines.append(f"  A: `{answer}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-dir", type=Path, default=Path("data/eval/knowledge_kernel_smoke"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/eval/knowledge_kernel_ablation")
    )
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/model/helix_v2_debug_tiny.yaml")
    )
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("tokenizer/helix_v2_tokenizer.model")
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--repeat", type=int, default=24)
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optionally cap both variants to the same number of seq_len rows.",
    )
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    load_or_build_kernel_dir(args.kernel_dir, args.tokenizer)
    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    plain = (args.kernel_dir / "plain_corpus.txt").read_text(encoding="utf-8")
    kernel = (args.kernel_dir / "current_kernel.txt").read_text(encoding="utf-8")
    qa_rows = [
        json.loads(line)
        for line in (args.kernel_dir / "qa_eval.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [
        train_one(
            name="plain",
            train_text=plain,
            qa_rows=qa_rows,
            sp=sp,
            args=args,
            device=device,
        ),
        train_one(
            name="kernel",
            train_text=kernel,
            qa_rows=qa_rows,
            sp=sp,
            args=args,
            device=device,
        ),
    ]
    (args.output_dir / "results.json").write_text(
        json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(args.output_dir / "report.md", results)
    print(f"wrote {args.output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
