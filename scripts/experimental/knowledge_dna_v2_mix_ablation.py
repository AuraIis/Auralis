"""Compare clean text training with a small Knowledge-DNA boost.

This is closer to the intended production use than the pure DNA ablation:

* baseline: samples only clean pretraining text
* dna_boost: samples the same clean text plus a small share of hybrid DNA rows

The batch sampler draws fresh rows every step, so a small DNA share can still be
seen over time without making the whole corpus artificial.
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
from scripts.experimental.knowledge_dna_v2_ablation import (  # noqa: E402
    GenerationMetric,
    generate,
    score_generation,
)


@dataclass
class MixResult:
    name: str
    train_loss_initial: float
    train_loss_final: float
    probe_loss_initial: float
    probe_loss_final: float
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


def read_lines_until(path: Path, max_lines: int, max_bytes: int) -> list[str]:
    rows: list[str] = []
    seen = 0
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(line)
            seen += len(line.encode("utf-8")) + 1
            if len(rows) >= max_lines or seen >= max_bytes:
                break
    if not rows:
        raise ValueError(f"no rows read from {path}")
    return rows


def block_rows(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    rows = [block.strip() for block in text.split("\n\n") if block.strip()]
    if not rows:
        raise ValueError(f"no DNA blocks read from {path}")
    return rows


def rows_to_sequences(
    sp: spm.SentencePieceProcessor,
    rows: list[str],
    *,
    seq_len: int,
    max_sequences: int,
) -> list[list[int]]:
    sequences: list[list[int]] = []
    for row in rows:
        ids = sp.encode(row + "\n", out_type=int)
        for start in range(0, max(0, len(ids) - 1), seq_len):
            chunk = ids[start : start + seq_len]
            if len(chunk) < seq_len:
                chunk = chunk + [sp.eos_id()] * (seq_len - len(chunk))
            sequences.append(chunk)
            if len(sequences) >= max_sequences:
                return sequences
    if not sequences:
        raise ValueError("could not build token sequences")
    return sequences


def make_batch(
    rows: list[list[int]], *, batch_size: int, rng: random.Random, device: torch.device
) -> dict[str, torch.Tensor]:
    chosen = [rng.choice(rows) for _ in range(batch_size)]
    x = torch.tensor(chosen, dtype=torch.long, device=device)
    return {"input_ids": x, "labels": x.clone()}


def make_mixed_batch(
    normal_rows: list[list[int]],
    dna_rows: list[list[int]],
    *,
    batch_size: int,
    dna_rows_per_batch: int,
    rng: random.Random,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    normal_n = batch_size - dna_rows_per_batch
    chosen = [rng.choice(normal_rows) for _ in range(normal_n)]
    chosen.extend(rng.choice(dna_rows) for _ in range(dna_rows_per_batch))
    rng.shuffle(chosen)
    x = torch.tensor(chosen, dtype=torch.long, device=device)
    return {"input_ids": x, "labels": x.clone()}


def prompt(row: dict[str, Any]) -> str:
    return f"Frage: {row['question']}\nAntwort: "


def suffix(row: dict[str, Any]) -> str:
    return f"{row['answer']}\n"


def encode_probe_batch(
    sp: spm.SentencePieceProcessor,
    probes: list[dict[str, Any]],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    encoded: list[tuple[list[int], list[int]]] = []
    for row in probes:
        prefix_ids = sp.encode(prompt(row), out_type=int)
        suffix_ids = sp.encode(suffix(row), out_type=int)
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


@torch.no_grad()
def loss(model, batch: dict[str, torch.Tensor]) -> float:
    model.eval()
    return float(model(input_ids=batch["input_ids"], labels=batch["labels"])["loss"])


@torch.no_grad()
def probe_loss(
    model,
    sp: spm.SentencePieceProcessor,
    probes: list[dict[str, Any]],
    *,
    probe_batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    weighted = 0.0
    total = 0
    for start in range(0, len(probes), probe_batch_size):
        chunk = probes[start : start + probe_batch_size]
        batch = encode_probe_batch(sp, chunk, device=device)
        value = float(model(input_ids=batch["input_ids"], labels=batch["labels"])["loss"])
        weighted += value * len(chunk)
        total += len(chunk)
    return weighted / max(total, 1)


def train_mix(
    *,
    name: str,
    normal_rows: list[list[int]],
    dna_rows: list[list[int]],
    probes: list[dict[str, Any]],
    sp: spm.SentencePieceProcessor,
    args: argparse.Namespace,
    device: torch.device,
) -> MixResult:
    set_seed(args.seed)
    rng = random.Random(args.seed + (17 if name == "dna_boost" else 0))
    model = build_model(args.model_config).to(device)
    if model.config.vocab_size != sp.get_piece_size():
        raise ValueError(
            f"vocab mismatch: model={model.config.vocab_size} tokenizer={sp.get_piece_size()}"
        )

    dna_rows_per_batch = (
        0 if name == "baseline" else max(1, round(args.batch_size * args.dna_share))
    )
    first_batch = (
        make_batch(normal_rows, batch_size=args.batch_size, rng=rng, device=device)
        if name == "baseline"
        else make_mixed_batch(
            normal_rows,
            dna_rows,
            batch_size=args.batch_size,
            dna_rows_per_batch=dna_rows_per_batch,
            rng=rng,
            device=device,
        )
    )
    train_initial = loss(model, first_batch)
    probe_initial = probe_loss(
        model,
        sp,
        probes,
        probe_batch_size=args.probe_batch_size,
        device=device,
    )
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )
    start = time.time()
    train_final = train_initial
    for step in range(1, args.steps + 1):
        model.train()
        batch = (
            make_batch(normal_rows, batch_size=args.batch_size, rng=rng, device=device)
            if name == "baseline"
            else make_mixed_batch(
                normal_rows,
                dna_rows,
                batch_size=args.batch_size,
                dna_rows_per_batch=dna_rows_per_batch,
                rng=rng,
                device=device,
            )
        )
        out = model(input_ids=batch["input_ids"], labels=batch["labels"])
        train_loss = out["loss"]
        if not torch.isfinite(train_loss):
            raise ValueError(f"{name}: non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        train_final = float(train_loss.detach())
        if step == 1 or step == args.steps or step % max(1, args.steps // 4) == 0:
            print(f"{name} step={step} train_loss={train_final:.4f}", flush=True)

    seconds = time.time() - start
    probe_final = probe_loss(
        model,
        sp,
        probes,
        probe_batch_size=args.probe_batch_size,
        device=device,
    )
    metrics: list[GenerationMetric] = []
    for row in probes[: args.generations]:
        generated = generate(
            model, sp, prompt(row), device=device, max_new_tokens=args.max_new_tokens
        )
        metrics.append(score_generation(row, generated))
    exact = sum(1 for item in metrics if item.matched) / max(len(metrics), 1)
    counterfacts = [item for item in metrics if item.kind == "counterfact"]
    counterfail = sum(1 for item in counterfacts if item.forbidden_hit or not item.matched) / max(
        len(counterfacts), 1
    )
    tagecho = sum(1 for item in metrics if item.tag_echo) / max(len(metrics), 1)
    return MixResult(
        name=name,
        train_loss_initial=train_initial,
        train_loss_final=train_final,
        probe_loss_initial=probe_initial,
        probe_loss_final=probe_final,
        probe_drop_ratio=probe_initial / max(probe_final, 1e-9),
        exact_match_light=exact,
        counterfact_failure_rate=counterfail,
        tag_echo_rate=tagecho,
        generations=metrics,
        seconds=seconds,
    )


def write_mix_files(
    out: Path, normal_lines: list[str], dna_blocks: list[str], args: argparse.Namespace
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    baseline = list(normal_lines)
    boosted = list(normal_lines)
    target_dna = max(1, round(len(normal_lines) * args.dna_share / max(1.0 - args.dna_share, 1e-9)))
    while len(boosted) < len(normal_lines) + target_dna:
        boosted.append(rng.choice(dna_blocks))
    rng.shuffle(baseline)
    rng.shuffle(boosted)
    (out / "baseline_mix.txt").write_text("\n".join(baseline) + "\n", encoding="utf-8")
    (out / "dna_boost_mix.txt").write_text("\n".join(boosted) + "\n", encoding="utf-8")
    (out / "mix_manifest.json").write_text(
        json.dumps(
            {
                "normal_lines": len(normal_lines),
                "dna_blocks_available": len(dna_blocks),
                "dna_share_target": args.dna_share,
                "dna_rows_per_batch": max(1, round(args.batch_size * args.dna_share)),
                "batch_size": args.batch_size,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def write_report(path: Path, results: list[MixResult], args: argparse.Namespace) -> None:
    by_name = {r.name: r for r in results}
    base = by_name["baseline"]
    boost = by_name["dna_boost"]
    probe_ok = boost.probe_loss_final <= base.probe_loss_final
    exact_ok = boost.exact_match_light >= base.exact_match_light
    counter_ok = boost.counterfact_failure_rate <= base.counterfact_failure_rate
    tag_ok = boost.tag_echo_rate <= 0.10
    go = probe_ok and exact_ok and counter_ok and tag_ok
    lines = [
        "# Knowledge-DNA v2 Mixed Ablation",
        "",
        f"- Model config: `{args.model_config}`",
        f"- Steps: {args.steps}",
        f"- Batch size: {args.batch_size}",
        f"- Seq len: {args.seq_len}",
        f"- DNA share target: {args.dna_share:.2%}",
        "",
        "| Variant | Train Loss | Probe Loss | Probe Drop | Exact-Light | Counterfact Fail | Tag Echo | Seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r.name} | {r.train_loss_final:.4f} | {r.probe_loss_final:.4f} | "
            f"{r.probe_drop_ratio:.2f}x | {r.exact_match_light:.2%} | "
            f"{r.counterfact_failure_rate:.2%} | {r.tag_echo_rate:.2%} | {r.seconds:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Go / No-Go",
            "",
            f"- Decision: `{'GO' if go else 'NO-GO'}`",
            f"- Reason: probe_ok={probe_ok}, exact_ok={exact_ok}, counter_ok={counter_ok}, tag_ok={tag_ok}",
            "",
            "## Generations",
            "",
        ]
    )
    for r in results:
        lines.append(f"### {r.name}")
        for item in r.generations:
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
    parser.add_argument("--normal-file", type=Path, required=True)
    parser.add_argument("--dna-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/eval/knowledge_dna_v2_mixed_100m")
    )
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/model/helix_v2_100m.yaml")
    )
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("tokenizer/helix_v2_tokenizer.model")
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--normal-lines", type=int, default=50_000)
    parser.add_argument("--normal-bytes", type=int, default=80_000_000)
    parser.add_argument("--normal-sequences", type=int, default=4096)
    parser.add_argument("--dna-sequences", type=int, default=4096)
    parser.add_argument("--dna-share", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--probe-batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"device={device}", flush=True)
    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    normal_lines = read_lines_until(args.normal_file, args.normal_lines, args.normal_bytes)
    dna_blocks = block_rows(args.dna_dir / "hybrid_corpus.txt")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_mix_files(args.output_dir, normal_lines, dna_blocks, args)
    normal_rows = rows_to_sequences(
        sp, normal_lines, seq_len=args.seq_len, max_sequences=args.normal_sequences
    )
    dna_rows = rows_to_sequences(
        sp, dna_blocks, seq_len=args.seq_len, max_sequences=args.dna_sequences
    )
    probes = [
        json.loads(line)
        for line in (args.dna_dir / "probes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    results = [
        train_mix(
            name=name,
            normal_rows=normal_rows,
            dna_rows=dna_rows,
            probes=probes,
            sp=sp,
            args=args,
            device=device,
        )
        for name in ("baseline", "dna_boost")
    ]
    (args.output_dir / "results.json").write_text(
        json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(args.output_dir / "report.md", results, args)
    print(f"wrote {args.output_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
