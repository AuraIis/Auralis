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

    # data_paths.yaml now stores per-language lists of paths/globs. For the
    # quality report we just pick the first existing file per language as a
    # representative probe corpus (random sampling inside the file takes care
    # of diversity).
    def _first_existing(entries: list[str] | str) -> Path | None:
        if isinstance(entries, str):
            entries = [entries]
        for e in entries:
            p = data_root / e
            if p.is_file():
                return p
            # glob expand
            matches = sorted(data_root.glob(e))
            if matches:
                return matches[0]
        return None

    sources = {
        "english": _first_existing(data_cfg["cleaned"]["english"]),
        "german":  _first_existing(data_cfg["cleaned"]["german"]),
        "code":    _first_existing(data_cfg["cleaned"]["code"]),
    }
    # Per-language gate + which metric it applies to.
    # EN/DE use tokens/100-words (natural-language compression metric).
    # Code uses tokens/KB (byte-compression metric) because "words" is
    # poorly defined for code (3 "words" but many symbol tokens per line).
    gate_map = {
        "english": ("tokens_per_100_words", float(targets["english_tokens_per_100_words_max"])),
        "german":  ("tokens_per_100_words", float(targets["german_tokens_per_100_words_max"])),
        "code":    ("tokens_per_kb",        float(targets["code_tokens_per_kb_max"])),
    }
    unk_target = float(targets["unknown_token_rate_max"])

    report_lines: list[str] = ["# Helix v2 Tokenizer Quality Report\n"]
    all_pass = True
    per_lang: dict[str, dict[str, float]] = {}

    for lang, src in sources.items():
        if src is None or not src.exists():
            report_lines.append(f"\n## {lang}\nMissing source — skipped.\n")
            continue
        lines = _sample_lines(src, args.samples_per_language)
        m = _metrics(sp, lines)
        per_lang[lang] = m
        metric_key, target_value = gate_map[lang]
        gated_value = m[metric_key]
        pass_gate = gated_value <= target_value
        pass_unk = m["unknown_rate"] <= unk_target
        all_pass = all_pass and pass_gate and pass_unk
        gate_label = "Tokens / 100 words" if metric_key == "tokens_per_100_words" else "Tokens / KB"
        other_label = "Tokens / KB" if metric_key == "tokens_per_100_words" else "Tokens / 100 words"
        other_key = "tokens_per_kb" if metric_key == "tokens_per_100_words" else "tokens_per_100_words"
        report_lines += [
            f"\n## {lang}",
            f"- Samples: {m['n_samples']:,} lines ({m['total_words']:,} words)",
            f"- **{gate_label}:** {gated_value:.1f}"
            f" (target ≤ {target_value}) — {'✓' if pass_gate else '✗'}",
            f"- {other_label}: {m[other_key]:.1f}",
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
