#!/usr/bin/env python3
"""Deduplicate fresh German JSONL documents against line-based reference corpora.

The operation deliberately performs cross-dataset deduplication only: reference
documents are indexed, then fresh documents that exact- or near-match that index
are removed. The retained fresh documents are not inserted back into the index.

Example::

    python scripts/data/dedup_de_fresh.py \
        --fresh data/fresh/de_fresh.jsonl \
        --ref cleaned/fineweb2_de.txt cleaned/wikipedia_de.txt \
        --ref-manifest manifests/dedup_reference_sha256.json \
        --out data/fresh/de_fresh.dedup.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from datasketch import MinHash, MinHashLSH

DEFAULT_NUM_PERM = 64
DEFAULT_THRESHOLD = 0.85
DEFAULT_SHINGLE_SIZE = 5
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def normalize(text: str) -> str:
    """Return the normalization used by both exact and near deduplication."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def shingle_bytes(text: str, shingle_size: int) -> list[bytes]:
    words = normalize(text).split()
    if len(words) < shingle_size:
        normalized = normalize(text)
        return [normalized.encode("utf-8")] if normalized else []
    return [
        " ".join(words[index : index + shingle_size]).encode("utf-8")
        for index in range(len(words) - shingle_size + 1)
    ]


def minhash(text: str, *, num_perm: int, shingle_size: int) -> MinHash:
    value = MinHash(num_perm=num_perm)
    shingles = shingle_bytes(text, shingle_size)
    if shingles:
        value.update_batch(shingles)
    return value


def sha1_normalized(text: str) -> str:
    return hashlib.sha1(normalize(text).encode("utf-8"), usedforsecurity=False).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reference_manifest(path: Path, references: Sequence[Path]) -> dict[str, Any]:
    """Load and validate an ordered SHA-256 reference manifest.

    Paths may differ between host and container. Therefore order and basename are
    authoritative, while the original manifest path is retained for provenance.
    """
    raw_bytes = path.read_bytes()
    payload = json.loads(raw_bytes)
    if payload.get("schema_version") != 1:
        raise ValueError("reference manifest schema_version must be 1")
    if payload.get("hash_algorithm") != "sha256":
        raise ValueError("reference manifest hash_algorithm must be 'sha256'")

    entries = payload.get("references")
    if not isinstance(entries, list) or len(entries) != len(references):
        raise ValueError("reference manifest must contain one ordered entry per --ref file")

    normalized_entries: list[dict[str, Any]] = []
    for reference, entry in zip(references, entries, strict=True):
        if not isinstance(entry, dict):
            raise ValueError("each reference manifest entry must be an object")
        manifest_path = entry.get("path")
        expected_hash = str(entry.get("sha256", "")).lower()
        expected_size = entry.get("size_bytes")
        if not isinstance(manifest_path, str) or Path(manifest_path).name != reference.name:
            raise ValueError(
                f"reference order/name mismatch: manifest={manifest_path!r}, cli={reference}"
            )
        if not SHA256_RE.fullmatch(expected_hash):
            raise ValueError(f"invalid SHA-256 for reference {manifest_path!r}")
        actual_size = reference.stat().st_size
        if expected_size is not None and expected_size != actual_size:
            raise ValueError(
                f"reference size mismatch for {reference}: expected {expected_size}, "
                f"found {actual_size}"
            )
        normalized_entries.append(
            {
                "path": str(reference),
                "manifest_path": manifest_path,
                "size_bytes": actual_size,
                "sha256": expected_hash,
            }
        )

    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "hash_algorithm": "sha256",
        "references": normalized_entries,
    }


def iter_reference_documents(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            yield line.strip()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--fresh", required=True, type=Path, help="JSONL with a text field")
    parser.add_argument(
        "--ref", required=True, nargs="+", type=Path, help="line-per-document reference files"
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--ref-manifest",
        type=Path,
        help="ordered schema-v1 JSON manifest containing reference SHA-256 hashes",
    )
    parser.add_argument(
        "--verify-ref-hashes",
        action="store_true",
        help="recompute every reference SHA-256 before indexing (adds one full read pass)",
    )
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--num-perm", type=int, default=DEFAULT_NUM_PERM)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--shingle-size", type=int, default=DEFAULT_SHINGLE_SIZE)
    args = parser.parse_args(argv)

    if args.min_chars < 0:
        parser.error("--min-chars must be >= 0")
    if args.num_perm <= 0:
        parser.error("--num-perm must be > 0")
    if not 0 < args.threshold <= 1:
        parser.error("--threshold must be in (0, 1]")
    if args.shingle_size <= 0:
        parser.error("--shingle-size must be > 0")
    if args.verify_ref_hashes and args.ref_manifest is None:
        parser.error("--verify-ref-hashes requires --ref-manifest")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    references: list[Path] = args.ref
    manifest = (
        load_reference_manifest(args.ref_manifest, references)
        if args.ref_manifest is not None
        else None
    )

    if args.verify_ref_hashes and manifest is not None:
        for entry, reference in zip(manifest["references"], references, strict=True):
            actual_hash = sha256_file(reference)
            if actual_hash != entry["sha256"]:
                raise ValueError(
                    f"reference SHA-256 mismatch for {reference}: "
                    f"expected {entry['sha256']}, found {actual_hash}"
                )
            entry["verified"] = True
    elif manifest is not None:
        for entry in manifest["references"]:
            entry["verified"] = False

    lsh = MinHashLSH(threshold=args.threshold, num_perm=args.num_perm)
    exact: set[str] = set()
    started = time.monotonic()
    reference_documents = 0
    for reference in references:
        for text in iter_reference_documents(reference):
            if len(text) < args.min_chars:
                continue
            exact.add(sha1_normalized(text))
            lsh.insert(
                f"r{reference_documents}",
                minhash(text, num_perm=args.num_perm, shingle_size=args.shingle_size),
            )
            reference_documents += 1
            if reference_documents % 200_000 == 0:
                elapsed = max(1e-9, time.monotonic() - started)
                print(
                    f"  ref indexed {reference_documents:,} "
                    f"({reference_documents / elapsed:.0f}/s)",
                    flush=True,
                )
    print(
        f"[dedup] reference: {reference_documents:,} docs indexed in "
        f"{time.monotonic() - started:.0f}s",
        flush=True,
    )

    kept = dropped_exact = dropped_near = fresh_seen = invalid_json = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fresh_started = time.monotonic()
    with args.fresh.open(encoding="utf-8") as source, args.out.open("w", encoding="utf-8") as out:
        for line in source:
            line = line.rstrip("\n")
            if not line:
                continue
            fresh_seen += 1
            try:
                text = json.loads(line)["text"]
                if not isinstance(text, str):
                    raise TypeError("text must be a string")
            except (json.JSONDecodeError, KeyError, TypeError):
                invalid_json += 1
                continue

            if sha1_normalized(text) in exact:
                dropped_exact += 1
                continue
            if lsh.query(
                minhash(text, num_perm=args.num_perm, shingle_size=args.shingle_size)
            ):
                dropped_near += 1
                continue
            out.write(line + "\n")
            kept += 1

            if fresh_seen % 200_000 == 0:
                elapsed = max(1e-9, time.monotonic() - fresh_started)
                print(
                    f"  fresh {fresh_seen:,} | kept {kept:,} | exact-dup {dropped_exact:,} "
                    f"| near-dup {dropped_near:,} | {fresh_seen / elapsed:.0f}/s",
                    flush=True,
                )

    report = {
        "schema_version": 2,
        "fresh_seen": fresh_seen,
        "kept": kept,
        "dropped_exact": dropped_exact,
        "dropped_near": dropped_near,
        "invalid_json": invalid_json,
        "ref_docs": reference_documents,
        "drop_pct": round(100 * (dropped_exact + dropped_near) / max(1, fresh_seen), 3),
        "config": {
            "min_chars": args.min_chars,
            "num_perm": args.num_perm,
            "threshold": args.threshold,
            "shingle_size": args.shingle_size,
            "cross_dataset_only": True,
        },
        "inputs": {
            "fresh": {
                "path": str(args.fresh),
                "size_bytes": args.fresh.stat().st_size,
            },
            "reference_manifest": manifest,
        },
    }
    report_path = args.out.with_suffix(".dedup_report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[dedup] DONE {json.dumps(report)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
