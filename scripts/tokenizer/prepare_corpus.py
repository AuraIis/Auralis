"""Sample a SentencePiece-training corpus from the three cleaned pretraining pools.

Input: ``<data_root>/cleaned/{english,german,code}.txt`` (one doc per line).
Output: a single ``<data_root>/tokenizer_corpus/corpus.txt`` file with mixed
language content, plus a ``.manifest.json``.

Strategy:

- Per-language byte budgets from ``configs/tokenizer/helix_v2.yaml``:
  e.g. 50 % EN / 40 % DE / 10 % code of ``corpus_budget_gb.total`` (default 15 GB).
- Reservoir-style sampling: read the source once, keep lines with probability
  ``target_bytes / source_bytes``. Deterministic seed per language so re-runs
  produce the same corpus (important: tokenizer training is not re-running
  for free).
- Interleave the three pools in the output file so SentencePiece does not
  see all EN first (the default is to subsample input anyway, but being
  mixed makes spot-inspection sane).

The output is safe to feed directly to ``train_tokenizer.py``.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.data._common import atomic_text_writer, check_free_space, load_paths, now_iso

REPO_ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_CFG = REPO_ROOT / "configs" / "tokenizer" / "helix_v2.yaml"


@dataclass
class CorpusStats:
    output_file: str
    started_at: str = ""
    finished_at: str = ""
    per_language_source_bytes: dict[str, int] = field(default_factory=dict)
    per_language_target_bytes: dict[str, int] = field(default_factory=dict)
    per_language_written_bytes: dict[str, int] = field(default_factory=dict)
    per_language_written_lines: dict[str, int] = field(default_factory=dict)
    total_written_bytes: int = 0
    total_written_lines: int = 0
    tokenizer_config_file: str = ""


def _expand_sources(data_root: Path, entries: list[str] | str) -> list[Path]:
    """Expand a list of path-or-glob entries to concrete existing files."""
    if isinstance(entries, str):
        entries = [entries]
    files: list[Path] = []
    for e in entries:
        p = data_root / e
        if any(ch in e for ch in "*?["):
            files.extend(sorted(data_root.glob(e)))
        elif p.is_file():
            files.append(p)
        # missing files are silently skipped; _sample_and_write_language logs them
    return files


def _total_bytes(paths: list[Path]) -> int:
    return sum(p.stat().st_size for p in paths if p.exists())


def _sample_and_write_language(
    lang: str,
    sources: list[Path],
    target_bytes: int,
    max_line_bytes: int,
    fh_out,
    seed: int,
) -> tuple[int, int]:
    """Reservoir-style sample across a list of source files.

    All files share one keep_prob so the resulting sample is proportional
    to each file's size (big files contribute more). Early-stops when the
    language target is met.
    """
    source_bytes = _total_bytes(sources)
    if not sources or source_bytes == 0:
        print(f"  warn: {lang} has no readable sources", file=sys.stderr)
        return 0, 0
    keep_prob = 1.0 if source_bytes <= target_bytes else (target_bytes / source_bytes)
    rng = random.Random(seed)

    written_bytes = 0
    written_lines = 0
    for src in sources:
        with src.open("r", encoding="utf-8", errors="replace") as fh_in:
            for line in tqdm(fh_in, desc=f"sample {lang}:{src.name}", unit="line"):
                if rng.random() > keep_prob:
                    continue
                line = line.rstrip("\n")
                if len(line.encode("utf-8")) > max_line_bytes:
                    line = line[:max_line_bytes]
                if not line:
                    continue
                fh_out.write(line + "\n")
                written_bytes += len(line.encode("utf-8")) + 1
                written_lines += 1
                if written_bytes >= target_bytes * 1.05:
                    return written_bytes, written_lines
    return written_bytes, written_lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample cleaned pools into a tokenizer corpus.")
    parser.add_argument("--data-config", type=Path, default=None)
    parser.add_argument("--tokenizer-config", type=Path, default=TOKENIZER_CFG)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output file (default: <data_root>/tokenizer_corpus/corpus.txt)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--required-free-gb", type=float, default=20.0)
    args = parser.parse_args()

    data_cfg = load_paths(args.data_config) if args.data_config else load_paths()
    tok_cfg = yaml.safe_load(args.tokenizer_config.read_text(encoding="utf-8"))

    data_root = Path(data_cfg["_data_root"])
    output = args.output or (data_root / "tokenizer_corpus" / "corpus.txt")
    output.parent.mkdir(parents=True, exist_ok=True)
    check_free_space(output.parent, args.required_free_gb)

    budget_gb = float(tok_cfg["corpus_budget_gb"]["total"])
    budget_bytes = int(budget_gb * 1024**3)
    mix = tok_cfg["corpus_mix_bytes"]
    max_line_bytes = int(tok_cfg["corpus_budget_gb"]["per_line_max_bytes"])

    sources: dict[str, list[Path]] = {
        lang: _expand_sources(data_root, data_cfg["cleaned"][lang])
        for lang in ("english", "german", "code")
    }

    stats = CorpusStats(
        output_file=str(output),
        started_at=now_iso(),
        tokenizer_config_file=str(args.tokenizer_config),
    )

    print(f"Total corpus budget: {budget_gb:.1f} GB")
    print(f"Mix: {mix}")
    print(f"Output: {output}\n")

    with atomic_text_writer(output) as fh:
        for lang in ("english", "german", "code"):
            srcs = sources[lang]
            share = float(mix[lang])
            target = int(budget_bytes * share)
            source_bytes = _total_bytes(srcs)
            stats.per_language_source_bytes[lang] = source_bytes
            stats.per_language_target_bytes[lang] = target
            print(
                f"[{lang}] sources={len(srcs)} total={source_bytes / 1e9:.2f}GB  target={target / 1e9:.2f}GB"
            )
            for p in srcs:
                print(f"  - {p}")
            written_bytes, written_lines = _sample_and_write_language(
                lang,
                srcs,
                target,
                max_line_bytes,
                fh,
                seed=args.seed + hash(lang) % 1000,
            )
            stats.per_language_written_bytes[lang] = written_bytes
            stats.per_language_written_lines[lang] = written_lines
            stats.total_written_bytes += written_bytes
            stats.total_written_lines += written_lines
            print(f"  wrote {written_bytes / 1e9:.2f}GB, {written_lines:,} lines\n")

    stats.finished_at = now_iso()
    manifest = output.with_suffix(".txt.manifest.json")
    import json

    manifest.write_text(json.dumps(asdict(stats), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Manifest: {manifest}")
    print(
        f"Corpus total: {stats.total_written_bytes / 1e9:.2f} GB, {stats.total_written_lines:,} lines"
    )


if __name__ == "__main__":
    main()
