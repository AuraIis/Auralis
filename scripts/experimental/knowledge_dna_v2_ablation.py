"""A/B/C ablation for Knowledge-DNA v2.

The script trains fresh Helix models on three corpus variants built by
``knowledge_dna_v2.py``:

* plain
* dna
* hybrid

It reports loss, simple generation metrics, tag echo rate, and counterfact
failures. The default model remains debug-tiny; pass the 100M config for the
larger pre-run experiment.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from auralis.model import build_model  # noqa: E402
from scripts.experimental.knowledge_dna_v2 import (  # noqa: E402
    VARIANT_FILES,
    build_outputs,
    sample_entries,
)


@dataclass
class GenerationMetric:
    question: str
    kind: str
    expected: str
    generated: str
    matched: bool
    forbidden_hit: bool
    tag_echo: bool


@dataclass
class VariantResult:
    name: str
    train_loss_initial: float
    train_loss_final: float
    probe_loss_initial: float
    probe_loss_final: float
    train_drop_ratio: float
    probe_drop_ratio: float
    exact_match_light: float
    counterfact_failure_rate: float
    tag_echo_rate: float
    generations: list[GenerationMetric]
    seconds: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_or_build_dna_dir(path: Path, tokenizer: Path, seed: int) -> None:
    if (
        all((path / filename).exists() for filename in VARIANT_FILES.values())
        and (path / "probes.jsonl").exists()
    ):
        return
    build_outputs(sample_entries(), path, tokenizer, seed)


def load_probes(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def make_lm_batch(
    sp: spm.SentencePieceProcessor,
    text: str,
    *,
    seq_len: int,
    repeat: int,
    max_rows: int | None,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    ids = sp.encode((text.strip() + "\n\n") * repeat, out_type=int)
    rows: list[list[int]] = []
    for start in range(0, max(0, len(ids) - seq_len), seq_len):
        chunk = ids[start : start + seq_len]
        if len(chunk) == seq_len:
            rows.append(chunk)
    if not rows:
        raise ValueError("training text is too short for requested seq_len")
    if max_rows is not None:
        rows = rows[:max_rows]
    x = torch.tensor(rows, dtype=torch.long, device=device)
    return {"input_ids": x, "labels": x.clone()}


def variant_prompt(row: dict[str, Any], variant: str) -> str:
    if variant in {"dna", "hybrid"}:
        return "\n".join(
            [
                "<recall>",
                f"Begriff: {row['term']}",
                f"Frage: {row['question']}",
                "Antwort: ",
            ]
        )
    return f"Begriff: {row['term']}\nFrage: {row['question']}\nAntwort: "


def variant_suffix(row: dict[str, Any], variant: str) -> str:
    if variant in {"dna", "hybrid"}:
        return f"{row['answer']}\n</recall>\n"
    return f"{row['answer']}\n"


def encode_probe_loss_batch(
    sp: spm.SentencePieceProcessor,
    rows: list[dict[str, Any]],
    *,
    variant: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    encoded: list[tuple[list[int], list[int]]] = []
    for row in rows:
        prefix_ids = sp.encode(variant_prompt(row, variant), out_type=int)
        suffix_ids = sp.encode(variant_suffix(row, variant), out_type=int)
        ids = prefix_ids + suffix_ids
        labels = [-100] * len(prefix_ids) + suffix_ids
        encoded.append((ids, labels))

    max_len = max(len(ids) for ids, _ in encoded)
    pad_id = sp.pad_id()
    x = torch.full((len(encoded), max_len), pad_id, dtype=torch.long, device=device)
    y = torch.full((len(encoded), max_len), -100, dtype=torch.long, device=device)
    for row_idx, (ids, labels) in enumerate(encoded):
        x[row_idx, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        y[row_idx, : len(labels)] = torch.tensor(labels, dtype=torch.long, device=device)
    return {"input_ids": x, "labels": y}


def normalize_for_match(text: str) -> str:
    return " ".join(text.lower().replace("\n", " ").split())


def score_generation(row: dict[str, Any], generated: str) -> GenerationMetric:
    normalized = normalize_for_match(generated)
    aliases = [str(v) for v in row.get("aliases") or []]
    expected_terms = aliases or [str(row["answer"])]
    matched = any(normalize_for_match(term) in normalized for term in expected_terms)
    forbidden_hit = any(
        normalize_for_match(term) in normalized for term in row.get("forbidden") or []
    )
    tag_echo = any(tag in generated for tag in ("<memory>", "</memory>", "<recall>", "</recall>"))
    return GenerationMetric(
        question=str(row["question"]),
        kind=str(row["kind"]),
        expected=str(row["answer"]),
        generated=generated,
        matched=matched,
        forbidden_hit=forbidden_hit,
        tag_echo=tag_echo,
    )


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
    stop_ids = {sp.eos_id(), sp.piece_to_id("</recall>"), sp.piece_to_id("<|end|>")}
    new_ids: list[int] = []
    for _ in range(max_new_tokens):
        out = model(input_ids=x)
        next_id = int(out["logits"][0, -1].argmax())
        if next_id in stop_ids:
            break
        new_ids.append(next_id)
        x = torch.cat([x, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
    return sp.decode(new_ids).strip()


def train_variant(
    *,
    name: str,
    train_text: str,
    probes: list[dict[str, Any]],
    sp: spm.SentencePieceProcessor,
    args: argparse.Namespace,
    device: torch.device,
) -> VariantResult:
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
    probe_batch = encode_probe_loss_batch(sp, probes, variant=name, device=device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )

    train_initial = loss(model, train_batch)
    probe_initial = loss(model, probe_batch)
    start = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        out = model(input_ids=train_batch["input_ids"], labels=train_batch["labels"])
        train_loss = out["loss"]
        if not torch.isfinite(train_loss):
            raise ValueError(f"{name}: non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        if step == 1 or step == args.steps or step % max(1, args.steps // 4) == 0:
            print(f"{name} step={step} train_loss={float(train_loss.detach()):.4f}", flush=True)

    seconds = time.time() - start
    train_final = loss(model, train_batch)
    probe_final = loss(model, probe_batch)

    metrics: list[GenerationMetric] = []
    for row in probes[: args.generations]:
        generated = generate(
            model,
            sp,
            variant_prompt(row, name),
            device=device,
            max_new_tokens=args.max_new_tokens,
        )
        metrics.append(score_generation(row, generated))

    exact = sum(1 for item in metrics if item.matched) / max(len(metrics), 1)
    counterfacts = [item for item in metrics if item.kind == "counterfact"]
    counterfact_failure = sum(
        1 for item in counterfacts if item.forbidden_hit or not item.matched
    ) / max(len(counterfacts), 1)
    tag_echo = sum(1 for item in metrics if item.tag_echo) / max(len(metrics), 1)
    return VariantResult(
        name=name,
        train_loss_initial=train_initial,
        train_loss_final=train_final,
        probe_loss_initial=probe_initial,
        probe_loss_final=probe_final,
        train_drop_ratio=train_initial / max(train_final, 1e-9),
        probe_drop_ratio=probe_initial / max(probe_final, 1e-9),
        exact_match_light=exact,
        counterfact_failure_rate=counterfact_failure,
        tag_echo_rate=tag_echo,
        generations=metrics,
        seconds=seconds,
    )


def write_report(path: Path, results: list[VariantResult], args: argparse.Namespace) -> None:
    by_name = {result.name: result for result in results}
    plain = by_name.get("plain")
    hybrid = by_name.get("hybrid")
    go = False
    reason = "missing plain or hybrid result"
    if plain and hybrid:
        probe_ok = hybrid.probe_loss_final <= plain.probe_loss_final
        exact_ok = hybrid.exact_match_light >= max(plain.exact_match_light, 0.10)
        counter_ok = (
            hybrid.counterfact_failure_rate <= plain.counterfact_failure_rate
            and hybrid.counterfact_failure_rate < 1.0
        )
        tag_ok = hybrid.tag_echo_rate <= 0.10
        go = probe_ok and exact_ok and counter_ok and tag_ok
        reason = (
            f"probe_ok={probe_ok}, exact_ok={exact_ok}, counter_ok={counter_ok}, tag_ok={tag_ok}"
        )

    lines = [
        "# Knowledge-DNA v2 Ablation Report",
        "",
        f"- Model config: `{args.model_config}`",
        f"- Steps: {args.steps}",
        f"- Seq len: {args.seq_len}",
        f"- Repeat: {args.repeat}",
        f"- Max train rows: {args.max_train_rows}",
        "",
        "## Results",
        "",
        "| Variant | Train Loss | Probe Loss | Train Drop | Probe Drop | Exact-Light | Counterfact Fail | Tag Echo | Seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.name} | {result.train_loss_final:.4f} | {result.probe_loss_final:.4f} | "
            f"{result.train_drop_ratio:.2f}x | {result.probe_drop_ratio:.2f}x | "
            f"{result.exact_match_light:.2%} | {result.counterfact_failure_rate:.2%} | "
            f"{result.tag_echo_rate:.2%} | {result.seconds:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Go / No-Go",
            "",
            f"- Decision: `{'GO' if go else 'NO-GO'}`",
            f"- Reason: {reason}",
            "- GO requires better/equal probe loss, some real answer matches, fewer absolute counterfact failures, and low tag echo.",
            "- GO means hybrid is allowed as a small 1-3% candidate source for the next 500M run.",
            "- NO-GO means Knowledge-DNA remains experimental and must not enter the main mix.",
            "",
            "## Generations",
            "",
        ]
    )
    for result in results:
        lines.append(f"### {result.name}")
        lines.append("")
        for item in result.generations:
            lines.append(f"- `{item.kind}` Q: {item.question}")
            lines.append(f"  - expected: {item.expected}")
            lines.append(f"  - generated: `{item.generated}`")
            lines.append(
                f"  - matched={item.matched}, forbidden_hit={item.forbidden_hit}, tag_echo={item.tag_echo}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dna-dir", type=Path, default=Path("data/eval/knowledge_dna_v2_smoke"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/eval/knowledge_dna_v2_ablation_smoke")
    )
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/model/helix_v2_debug_tiny.yaml")
    )
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("tokenizer/helix_v2_tokenizer.model")
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--repeat", type=int, default=24)
    parser.add_argument("--max-train-rows", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    return args


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"device={device}", flush=True)

    load_or_build_dna_dir(args.dna_dir, args.tokenizer, args.seed)
    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    probes = load_probes(args.dna_dir / "probes.jsonl")
    texts = {
        name: (args.dna_dir / filename).read_text(encoding="utf-8")
        for name, filename in VARIANT_FILES.items()
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [
        train_variant(
            name=name,
            train_text=texts[name],
            probes=probes,
            sp=sp,
            args=args,
            device=device,
        )
        for name in ("plain", "dna", "hybrid")
    ]
    (args.output_dir / "results.json").write_text(
        json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(args.output_dir / "report.md", results, args)
    print(f"wrote {args.output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
