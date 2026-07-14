"""Tokenize the cleaned Phase-1 corpora into memmap-ready binary files.

For each language configured under ``cleaned`` in ``configs/data_paths.yaml``,
this script concatenates its source files, encodes each document through the
trained Helix v2 SentencePiece model, inserts ``</s>`` between documents, and
writes the result as a flat ``uint32`` stream on the NAS.

Output layout (under ``<data_root>/tokenized/phase1/``):

- ``english.bin`` — flat uint32 token stream (no doc boundaries, <eos> separates)
- ``english.idx`` — int64 pairs ``[offset_tokens, n_tokens]`` per document
- ``english.manifest.json`` — source files, counts, elapsed, SP hash

Same for ``german`` and ``code``. A single pass on the NAS, atomic renames.

Resume-safe: an already-existing ``*.bin`` (not ``.tmp``) is skipped, so you can
re-run the script after restarting the machine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import sentencepiece as spm
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.data._common import check_free_space, load_paths, now_iso

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOKENIZER = REPO_ROOT / "tokenizer" / "helix_v2_tokenizer.model"


@dataclass
class TokenizeStats:
    language: str
    output_bin: str
    output_idx: str
    sources: list[str] = field(default_factory=list)
    source_fingerprints: list[dict] = field(default_factory=list)
    documents: int = 0
    tokens: int = 0
    empty_lines: int = 0
    bytes_in: int = 0
    tokenizer_sha256: str = ""
    tokens_per_byte: float = 0.0   # measured from the written bin (for bpb)
    started_at: str = ""
    finished_at: str = ""
    elapsed_seconds: float = 0.0


def _expand_sources(data_root: Path, entries: list[str] | str) -> list[Path]:
    if isinstance(entries, str):
        entries = [entries]
    out: list[Path] = []
    for e in entries:
        p = data_root / e
        if any(ch in e for ch in "*?["):
            out.extend(sorted(data_root.glob(e)))
        elif p.is_file():
            out.append(p)
    return out


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_fingerprints(paths: list[Path]) -> list[dict]:
    return [
        {
            "path": str(p),
            "size_bytes": p.stat().st_size,
            "mtime_ns": p.stat().st_mtime_ns,
        }
        for p in paths
    ]


def _existing_output_status(
    out_bin: Path,
    out_idx: Path,
    manifest_path: Path,
    tokenizer_hash: str,
    sources: list[Path],
) -> tuple[bool, str]:
    """Validate the complete output triplet before treating it as resumable."""
    missing = [
        str(p) for p in (out_bin, out_idx, manifest_path) if not p.is_file()
    ]
    if missing:
        return False, f"missing completion artifact(s): {', '.join(missing)}"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"unreadable manifest: {exc}"

    if manifest.get("tokenizer_sha256") != tokenizer_hash:
        return False, "tokenizer hash differs from manifest"
    if manifest.get("sources") != [str(p) for p in sources]:
        return False, "resolved source list differs from manifest"

    current_fingerprints = _source_fingerprints(sources)
    recorded_fingerprints = manifest.get("source_fingerprints")
    if recorded_fingerprints is not None:
        if recorded_fingerprints != current_fingerprints:
            return False, "source size or mtime differs from manifest"
    elif manifest.get("bytes_in") != sum(p.stat().st_size for p in sources):
        # Backward-compatible validation for older manifests.
        return False, "source byte total differs from legacy manifest"

    bin_size = out_bin.stat().st_size
    idx_size = out_idx.stat().st_size
    if bin_size % np.dtype(np.uint32).itemsize:
        return False, "bin size is not uint32-aligned"
    if idx_size % (2 * np.dtype(np.int64).itemsize):
        return False, "idx size is not int64-pair-aligned"
    tokens = bin_size // np.dtype(np.uint32).itemsize
    documents = idx_size // (2 * np.dtype(np.int64).itemsize)
    if manifest.get("tokens") != tokens:
        return False, "bin token count differs from manifest"
    if manifest.get("documents") != documents:
        return False, "idx document count differs from manifest"
    if documents:
        with out_idx.open("rb") as fh:
            fh.seek(-2 * np.dtype(np.int64).itemsize, os.SEEK_END)
            last_offset, last_length = np.fromfile(fh, dtype=np.int64, count=2)
        if int(last_offset + last_length) != tokens:
            return False, "last idx span does not end at bin token count"
    elif tokens:
        return False, "non-empty bin has an empty idx"
    return True, "validated bin/idx/manifest triplet"


def _write_manifest_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _iter_documents(paths: list[Path]) -> Iterable[tuple[str, str]]:
    """Yield (source_path, text) one document at a time."""
    for p in paths:
        with p.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line:
                    yield str(p), line


def _tokenize_language(
    lang: str,
    sources: list[Path],
    out_bin: Path,
    out_idx: Path,
    sp: spm.SentencePieceProcessor,
    batch_size: int,
    tokenizer_hash: str,
) -> TokenizeStats:
    stats = TokenizeStats(
        language=lang,
        output_bin=str(out_bin),
        output_idx=str(out_idx),
        sources=[str(p) for p in sources],
        source_fingerprints=_source_fingerprints(sources),
        started_at=now_iso(),
        tokenizer_sha256=tokenizer_hash,
        bytes_in=sum(p.stat().st_size for p in sources if p.exists()),
    )
    if not sources:
        raise FileNotFoundError(
            f"[{lang}] no source files resolved from config. "
            "Refuse to write an empty tokenized corpus."
        )

    eos = sp.eos_id()
    if eos is None or int(eos) < 0:
        raise ValueError(
            f"[{lang}] tokenizer reports no valid EOS id (got {eos!r}). Appending it and "
            "casting to uint32 would wrap to a huge out-of-vocab token (e.g. -1 -> 4294967295) "
            "and silently corrupt the entire corpus. Export the SentencePiece model with a "
            "defined </s> (eos_id >= 0) before tokenizing."
        )

    out_bin.parent.mkdir(parents=True, exist_ok=True)
    tmp_bin = out_bin.with_suffix(out_bin.suffix + ".tmp")
    tmp_idx = out_idx.with_suffix(out_idx.suffix + ".tmp")

    t0 = time.time()
    buffer: list[str] = []
    offset_tokens = 0

    def flush(f_bin, f_idx):
        nonlocal offset_tokens
        if not buffer:
            return
        # SentencePiece batched encode for throughput
        ids_batch = sp.encode(buffer, out_type=int)
        for ids in ids_batch:
            if not ids:
                stats.empty_lines += 1
                continue
            ids.append(eos)
            arr = np.asarray(ids, dtype=np.uint32)
            arr.tofile(f_bin)
            idx = np.asarray([offset_tokens, len(ids)], dtype=np.int64)
            idx.tofile(f_idx)
            offset_tokens += len(ids)
            stats.tokens += len(ids)
            stats.documents += 1
        buffer.clear()

    total_bytes = stats.bytes_in
    pbar = tqdm(total=total_bytes, unit="B", unit_scale=True, desc=f"tok {lang}")
    try:
        with tmp_bin.open("wb") as f_bin, tmp_idx.open("wb") as f_idx:
            last_pos = 0
            for src_path, text in _iter_documents(sources):
                buffer.append(text)
                pbar.update(len(text.encode("utf-8")))
                if len(buffer) >= batch_size:
                    flush(f_bin, f_idx)
            flush(f_bin, f_idx)
        os.replace(tmp_bin, out_bin)
        os.replace(tmp_idx, out_idx)
    finally:
        pbar.close()
        for t in (tmp_bin, tmp_idx):
            if t.exists():
                t.unlink()

    # Measure tokens/byte from the middle of the finished bin (the honest value
    # for bits-per-byte — decode tokens back to text, count UTF-8 bytes). The
    # config previously carried a hand-guessed 0.2338 for German which inflated
    # bpb ~30%; record the real number here so configs can use it.
    try:
        mm = np.memmap(out_bin, dtype=np.uint32, mode="r")
        n = int(mm.shape[0])
        take = min(300_000, n)
        lo = max(0, (n - take) // 2)
        sample_ids = [int(x) for x in mm[lo : lo + take]]
        sample_bytes = len(sp.decode(sample_ids).encode("utf-8"))
        stats.tokens_per_byte = round(take / max(1, sample_bytes), 6)
        del mm
    except Exception as exc:  # measurement is diagnostic; never fail the run
        print(f"  warn: tokens/byte measurement failed for {lang}: {exc}")

    stats.elapsed_seconds = round(time.time() - t0, 1)
    stats.finished_at = now_iso()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-config", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--languages", nargs="+", default=["english", "german", "code"],
                        choices=["english", "german", "code"])
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="Lines fed to SP.encode() per call.")
    parser.add_argument("--required-free-gb", type=float, default=100.0)
    parser.add_argument("--force", action="store_true",
                        help="Re-tokenize even if output already exists.")
    parser.add_argument("--output-subdir", default="phase1",
                        help="Subdirectory under <data_root>/tokenized/ to write the "
                             ".bin/.idx/.manifest files into. Default 'phase1'. Use e.g. "
                             "'curated_40b' to keep a phase-1 rollback anchor intact.")
    args = parser.parse_args()

    cfg = load_paths(args.data_config) if args.data_config else load_paths()
    data_root = Path(cfg["_data_root"])
    if not args.tokenizer.exists():
        sys.exit(f"tokenizer missing: {args.tokenizer}")

    tokenizer_hash = _sha256_file(args.tokenizer)
    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    print(f"tokenizer : {args.tokenizer} (sha={tokenizer_hash[:12]}…)")
    print(f"vocab     : {sp.GetPieceSize():,}")

    out_dir = data_root / "tokenized" / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    check_free_space(out_dir, args.required_free_gb)
    print(f"output dir: {out_dir}")

    all_stats: list[TokenizeStats] = []
    for lang in args.languages:
        out_bin = out_dir / f"{lang}.bin"
        out_idx = out_dir / f"{lang}.idx"
        manifest_path = out_bin.with_suffix(".bin.manifest.json")
        sources = _expand_sources(data_root, cfg["cleaned"][lang])
        for s in sources:
            print(f"  - {s}")
        if not args.force and any(
            p.exists() for p in (out_bin, out_idx, manifest_path)
        ):
            valid, detail = _existing_output_status(
                out_bin, out_idx, manifest_path, tokenizer_hash, sources,
            )
            if valid:
                print(f"\n[{lang}] already tokenized and verified → {out_bin}")
                continue
            print(
                f"\n[{lang}] incomplete/stale output ({detail}); rebuilding "
                "the triplet atomically"
            )
        else:
            print(f"\n[{lang}] tokenizing → {out_bin}")
        stats = _tokenize_language(
            lang=lang,
            sources=sources,
            out_bin=out_bin,
            out_idx=out_idx,
            sp=sp,
            batch_size=args.batch_size,
            tokenizer_hash=tokenizer_hash,
        )
        # Manifest is the completion marker and is replaced last. A crash after
        # bin/idx replacement cannot be mistaken for a completed resumable run.
        _write_manifest_atomic(manifest_path, asdict(stats))
        all_stats.append(stats)
        print(
            f"  {stats.documents:,} docs | {stats.tokens/1e9:.2f} B tokens | "
            f"{stats.elapsed_seconds/60:.1f} min"
        )

    print("\n=== Summary ===")
    for s in all_stats:
        print(f"  {s.language:8s} {s.documents:>12,} docs  {s.tokens/1e9:>6.2f} B tokens")


if __name__ == "__main__":
    main()
