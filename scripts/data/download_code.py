"""Download Code pretraining data for Phase 1.

Sources (see ``Doc/SPEC_DATASETS.md`` §2.3):

- StarCoderData subset — 1B tokens across 9 languages with per-language shares
- Proof-Pile-2        — 250M tokens (math / formal reasoning)

StarCoderData is massive (>250B tokens); we stream and stop per language when
the per-language quota is filled.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterator

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.data._common import (
    DownloadStats,
    atomic_text_writer,
    check_free_space,
    clean_text,
    load_paths,
    now_iso,
    resolve,
)

# Code is token-dense (whitespace / symbols).
CODE_BYTES_PER_TOKEN = 3.5

LANG_SHARES: dict[str, float] = {
    "python":     0.30,
    "javascript": 0.20,
    "typescript": 0.10,
    "rust":       0.10,
    "cpp":        0.10,
    "go":         0.08,
    "java":       0.07,
    "shell":      0.03,
    "sql":        0.02,
}


def _open_streaming(name: str, **kwargs: Any):
    from datasets import load_dataset
    return load_dataset(name, split="train", streaming=True, **kwargs)


def _stream_starcoder_language(lang: str, filters: dict[str, Any]) -> Iterator[tuple[str, dict[str, int]]]:
    ds = _open_streaming("bigcode/starcoderdata", data_dir=lang)
    min_len = filters["min_length"]
    max_len = filters["max_length"]
    min_stars = filters["min_stars"]
    for ex in ds:
        stars = ex.get("max_stars_count", 0) or 0
        if stars < min_stars:
            yield None, {"low_stars": 1}
            continue
        content = ex.get("content", "") or ""
        if len(content) < min_len:
            yield None, {"too_short": 1}
            continue
        if len(content) > max_len:
            yield None, {"too_long": 1}
            continue
        # Keep newlines for code — they are semantically meaningful.
        yield content.replace("\r\n", "\n"), {}


def _download_starcoder(
    out_dir: Path,
    target_tokens: int,
    filters: dict[str, Any],
) -> DownloadStats:
    output_path = out_dir / "starcoderdata.txt"
    stats = DownloadStats(
        source="starcoderdata",
        output_file=str(output_path),
        target_tokens=target_tokens,
        estimated_bytes_per_token=CODE_BYTES_PER_TOKEN,
        started_at=now_iso(),
        filters_applied={**filters, "lang_shares": LANG_SHARES},
    )
    total_target_bytes = stats.target_bytes()

    with atomic_text_writer(output_path) as fh:
        for lang, share in LANG_SHARES.items():
            lang_target_bytes = int(total_target_bytes * share)
            lang_bytes = 0
            for line, reasons in tqdm(
                _stream_starcoder_language(lang, filters),
                desc=f"starcoder:{lang}",
                unit="file",
            ):
                if line is None:
                    for reason, count in reasons.items():
                        key = f"{lang}:{reason}"
                        stats.filtered_reasons[key] = stats.filtered_reasons.get(key, 0) + count
                        stats.filtered_total += count
                    continue
                # Wrap each file with a language marker so the tokenizer and
                # downstream model see a clear boundary.
                wrapped = f"<|code|>[{lang}]\n{line}\n<|endcode|>"
                fh.write(wrapped + "\n")
                sz = len(wrapped.encode("utf-8")) + 1
                stats.final_bytes += sz
                stats.final_docs += 1
                lang_bytes += sz
                if lang_bytes >= lang_target_bytes:
                    break

    stats.finished_at = now_iso()
    stats.write_manifest(output_path.with_suffix(".txt.manifest.json"))
    return stats


def _stream_proof_pile(_filters: dict[str, Any]) -> Iterator[tuple[str, dict[str, int]]]:
    ds = _open_streaming("EleutherAI/proof-pile-2", name="default")
    wanted = {"open-web-math", "arxiv"}
    for ex in ds:
        meta = ex.get("meta", {}) or {}
        subset = str(meta.get("subset", meta.get("config_name", ""))).lower()
        if subset and not any(w in subset for w in wanted):
            yield None, {"subset_not_wanted": 1}
            continue
        text = ex.get("text", "") or ex.get("content", "") or ""
        if len(text) < 200:
            yield None, {"too_short": 1}
            continue
        yield clean_text(text), {}


def _download_proof_pile(out_dir: Path, target_tokens: int, filters: dict[str, Any]) -> DownloadStats:
    output_path = out_dir / "proof_pile_2.txt"
    stats = DownloadStats(
        source="proof_pile_2",
        output_file=str(output_path),
        target_tokens=target_tokens,
        estimated_bytes_per_token=CODE_BYTES_PER_TOKEN,
        started_at=now_iso(),
        filters_applied=filters,
    )
    target_bytes = stats.target_bytes()
    with atomic_text_writer(output_path) as fh:
        for line, reasons in tqdm(_stream_proof_pile(filters), desc="proof_pile_2", unit="doc"):
            if line is None:
                for reason, count in reasons.items():
                    stats.filtered_reasons[reason] = stats.filtered_reasons.get(reason, 0) + count
                    stats.filtered_total += count
                continue
            fh.write(line + "\n")
            stats.final_bytes += len(line.encode("utf-8")) + 1
            stats.final_docs += 1
            if stats.final_bytes >= target_bytes:
                break
    stats.finished_at = now_iso()
    stats.write_manifest(output_path.with_suffix(".txt.manifest.json"))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Phase 1 Code pretraining data.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--sources", nargs="+", choices=["starcoder", "proof_pile"],
                        default=["starcoder", "proof_pile"])
    parser.add_argument("--required-free-gb", type=float, default=8.0)
    args = parser.parse_args()

    cfg = load_paths(args.config) if args.config else load_paths()
    out_dir = resolve(cfg, "raw", "code")
    out_dir.mkdir(parents=True, exist_ok=True)
    check_free_space(out_dir, args.required_free_gb)

    targets = cfg["phase1_targets"]["code"]
    filters = cfg["filters"]["code"]

    summaries: list[DownloadStats] = []
    if "starcoder" in args.sources:
        summaries.append(_download_starcoder(out_dir, int(targets["starcoder"]), filters))
    if "proof_pile" in args.sources:
        summaries.append(_download_proof_pile(out_dir, int(targets["proof_pile"]), filters))

    print("\n=== Summary ===")
    for s in summaries:
        print(f"  {s.source:15s} {s.final_docs:>10,} docs  {s.final_bytes/1e9:>6.2f} GB")


if __name__ == "__main__":
    main()
