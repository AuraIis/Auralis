"""Tokenizer quality report.

Evaluates the trained Helix v2 tokenizer against the targets in
``configs/tokenizer/helix_v2.yaml`` using three held-out probe corpora:

- English: sampled from Wikipedia EN
- German:  sampled from the v1-reused cleaned/german.txt
- Code:    sampled from cleaned/code.txt

Metrics:

- tokens / 100 words      (lower is better; compare to SOTA ~130 EN)
- tokens / KB             (compression density)
- unknown-byte fallback rate (should be near 0 thanks to byte_fallback)
- chat-template round-trip: ``build_inference_prompt`` encoded then decoded
  must round-trip byte-exactly — extra insurance against v1's prompt bug.

Writes a Markdown report to ``tokenizer/quality_report.md`` and exits
non-zero if any quality target is missed.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import sentencepiece as spm
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.data._common import load_paths
from auralis.tokenizer.chat_template import build_inference_prompt


def _sample_lines(path: Path, n: int, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    reservoir: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            if len(reservoir) < n:
                reservoir.append(line)
            else:
                j = rng.randint(0, i)
                if j < n:
                    reservoir[j] = line
    return reservoir


def _metrics(sp: spm.SentencePieceProcessor, lines: list[str]) -> dict[str, float]:
    total_tokens = 0
    total_words = 0
    total_bytes = 0
    total_unk = 0
    for line in lines:
        ids = sp.EncodeAsIds(line)
        total_tokens += len(ids)
        total_words += len(line.split())
        total_bytes += len(line.encode("utf-8"))
        total_unk += sum(1 for i in ids if i == sp.unk_id())
    return {
        "tokens_per_100_words": (total_tokens / max(total_words, 1)) * 100.0,
        "tokens_per_kb": (total_tokens / max(total_bytes, 1)) * 1024.0,
        "unknown_rate": total_unk / max(total_tokens, 1),
        "n_samples": len(lines),
        "total_tokens": total_tokens,
        "total_words": total_words,
    }


def _check_chat_roundtrip(sp: spm.SentencePieceProcessor) -> dict[str, bool | str]:
    prompt = build_inference_prompt(
        [
            {"role": "system", "content": "Du bist Helix."},
            {"role": "user",   "content": "Hallo, wie geht's dir heute? 1+1=?"},
        ],
    )
    ids = sp.EncodeAsIds(prompt)
    decoded = sp.DecodeIds(ids)
    ok = decoded == prompt
    return {"round_trips": ok, "prompt": prompt, "decoded": decoded}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="helix_v2_tokenizer.model")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "tokenizer" / "helix_v2.yaml")
    parser.add_argument("--data-config", type=Path, default=None)
    parser.add_argument("--samples-per-language", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "tokenizer" / "quality_report.md")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    targets = cfg["quality_targets"]
    data_cfg = load_paths(args.data_config) if args.data_config else load_paths()
    data_root = Path(data_cfg["_data_root"])

    sp = spm.SentencePieceProcessor(model_file=str(args.model))
    print(f"Loaded tokenizer: {args.model} (vocab={sp.GetPieceSize():,})\n")

    sources = {
        "english": data_root / data_cfg["cleaned"]["english"],
        "german":  data_root / data_cfg["cleaned"]["german"],
        "code":    data_root / data_cfg["cleaned"]["code"],
    }
    target_map = {
        "english": targets["english_tokens_per_100_words_max"],
        "german":  targets["german_tokens_per_100_words_max"],
        "code":    targets["code_tokens_per_100_words_max"],
    }
    unk_target = float(targets["unknown_token_rate_max"])

    report_lines: list[str] = ["# Helix v2 Tokenizer Quality Report\n"]
    all_pass = True
    per_lang: dict[str, dict[str, float]] = {}

    for lang, src in sources.items():
        if not src.exists():
            report_lines.append(f"\n## {lang}\nMissing source: `{src}` — skipped.\n")
            continue
        lines = _sample_lines(src, args.samples_per_language)
        m = _metrics(sp, lines)
        per_lang[lang] = m
        t = target_map[lang]
        pass_tokens = m["tokens_per_100_words"] <= t
        pass_unk = m["unknown_rate"] <= unk_target
        all_pass = all_pass and pass_tokens and pass_unk
        report_lines += [
            f"\n## {lang}",
            f"- Samples: {m['n_samples']:,} lines ({m['total_words']:,} words)",
            f"- **Tokens / 100 words:** {m['tokens_per_100_words']:.1f}"
            f" (target ≤ {t}) — {'✓' if pass_tokens else '✗'}",
            f"- Tokens / KB: {m['tokens_per_kb']:.1f}",
            f"- Unknown-token rate: {m['unknown_rate']:.6f}"
            f" (target ≤ {unk_target}) — {'✓' if pass_unk else '✗'}",
        ]

    # Chat round-trip
    rt = _check_chat_roundtrip(sp)
    all_pass = all_pass and rt["round_trips"]
    report_lines += [
        "\n## Chat-template round-trip",
        f"- Round-trips byte-exact: {'✓' if rt['round_trips'] else '✗'}",
    ]
    if not rt["round_trips"]:
        report_lines += [
            f"- Original: `{rt['prompt']!r}`",
            f"- Decoded : `{rt['decoded']!r}`",
        ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"\nReport: {args.output}")
    print(f"Status: {'PASS' if all_pass else 'FAIL'}")
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
