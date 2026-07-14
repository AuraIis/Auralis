"""Integrity tests for pretraining tokenizer resume artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

pytest.importorskip("sentencepiece")

from scripts.data.tokenize_for_pretraining import (  # noqa: E402
    _existing_output_status,
    _source_fingerprints,
)


def _make_triplet(tmp_path: Path) -> tuple[Path, Path, Path, Path, str]:
    source = tmp_path / "source.txt"
    source.write_text("ein dokument\n", encoding="utf-8")
    out_bin = tmp_path / "german.bin"
    out_idx = tmp_path / "german.idx"
    manifest = tmp_path / "german.bin.manifest.json"
    tokenizer_hash = "a" * 64

    np.asarray([10, 11, 2], dtype=np.uint32).tofile(out_bin)
    np.asarray([0, 3], dtype=np.int64).tofile(out_idx)
    manifest.write_text(json.dumps({
        "sources": [str(source)],
        "source_fingerprints": _source_fingerprints([source]),
        "bytes_in": source.stat().st_size,
        "tokenizer_sha256": tokenizer_hash,
        "tokens": 3,
        "documents": 1,
    }), encoding="utf-8")
    return source, out_bin, out_idx, manifest, tokenizer_hash


def test_complete_triplet_is_resumable(tmp_path: Path) -> None:
    source, out_bin, out_idx, manifest, tokenizer_hash = _make_triplet(tmp_path)
    valid, detail = _existing_output_status(
        out_bin, out_idx, manifest, tokenizer_hash, [source],
    )
    assert valid
    assert "validated" in detail


def test_missing_idx_is_not_resumable(tmp_path: Path) -> None:
    source, out_bin, out_idx, manifest, tokenizer_hash = _make_triplet(tmp_path)
    out_idx.unlink()
    valid, detail = _existing_output_status(
        out_bin, out_idx, manifest, tokenizer_hash, [source],
    )
    assert not valid
    assert "missing completion artifact" in detail


def test_changed_source_is_not_resumable(tmp_path: Path) -> None:
    source, out_bin, out_idx, manifest, tokenizer_hash = _make_triplet(tmp_path)
    source.write_text("ein veraendertes dokument\n", encoding="utf-8")
    valid, detail = _existing_output_status(
        out_bin, out_idx, manifest, tokenizer_hash, [source],
    )
    assert not valid
    assert "source size or mtime" in detail


def test_idx_must_end_at_bin_token_count(tmp_path: Path) -> None:
    source, out_bin, out_idx, manifest, tokenizer_hash = _make_triplet(tmp_path)
    np.asarray([0, 2], dtype=np.int64).tofile(out_idx)
    valid, detail = _existing_output_status(
        out_bin, out_idx, manifest, tokenizer_hash, [source],
    )
    assert not valid
    assert "last idx span" in detail
