"""Tests for ``scripts/data/filter_quality.py``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.data.filter_quality import (
    _gopher_repetition_reason,
    _normalise,
    _passes,
    _repetition_score,
    _words_for_repetition,
    main,
)


def test_normalise_collapses_whitespace_for_text() -> None:
    assert _normalise("foo\n\nbar\t baz\r\n", preserve_newlines=False) == "foo bar baz"


def test_normalise_preserves_code_newlines_mode() -> None:
    assert _normalise("print('x')\r\n", preserve_newlines=True) == "print('x')"


def test_boilerplate_is_rejected() -> None:
    line = "Please accept all cookies before reading the rest of this article."
    assert _passes(
        line,
        min_length=10,
        max_length=1000,
        preserve_newlines=False,
        allow_mojibake=False,
        language="english",
    ) == "boilerplate"


def test_gopher_repetition_rejects_phrase_spam() -> None:
    line = "spam " * 30
    reason = _passes(
        line,
        min_length=10,
        max_length=1000,
        preserve_newlines=False,
        allow_mojibake=False,
        language="german",
    )
    assert reason is not None
    assert reason.startswith(("top_", "dup_"))


def test_legacy_repetition_mode_remains_reproducible() -> None:
    assert _passes(
        "spam " * 30,
        min_length=10,
        max_length=1000,
        preserve_newlines=False,
        allow_mojibake=False,
        repetition_mode="legacy",
    ) == "repetitive_legacy"


def test_gopher_does_not_reject_long_normal_prose_by_document_length() -> None:
    # Most function words repeat, so the legacy type/token score exceeds 0.60.
    # Content-bearing terms vary, so no long phrase dominates the document.
    sentences = [
        (
            f"Der Fachbegriff{i} wurde im Abschnitt{i} mit einer "
            f"Analyse{i} und dem Ergebnis{i} präzise{i} erläutert."
        )
        for i in range(1000)
    ]
    line = " ".join(sentences)
    assert _repetition_score(line) > 0.60
    assert _gopher_repetition_reason(line, "german") is None
    assert _passes(
        line,
        min_length=10,
        max_length=1_000_000,
        preserve_newlines=False,
        allow_mojibake=False,
        language="german",
    ) is None


def test_german_word_split_keeps_umlauts_and_eszett() -> None:
    assert _words_for_repetition("Größe, süß, Äpfel und Öl.") == [
        "grösse", "süss", "äpfel", "und", "öl",
    ]


def test_clean_line_passes() -> None:
    line = "This is a clean sentence about machine learning and data curation."
    assert _passes(
        line,
        min_length=10,
        max_length=1000,
        preserve_newlines=False,
        allow_mojibake=False,
        language="english",
    ) is None


def test_fineweb2_hq_jsonl_profile_validates_schema_and_accounts_chars(
    tmp_path: Path, monkeypatch,
) -> None:
    good = (
        "Das ist ein längerer, sauberer deutscher Beispieldatensatz. "
        "Er enthält genügend Inhalt für die konservative Eingangsprüfung. "
    ) * 3
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.txt"
    records = [
        json.dumps({"text": good, "quality_score": 0.42}, ensure_ascii=False),
        json.dumps({"text": good}, ensure_ascii=False),
        "{broken-json",
    ]
    input_path.write_text("\n".join(records) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "filter_quality.py",
        "--input", str(input_path),
        "--output", str(output_path),
        "--language", "german",
        "--input-format", "jsonl",
        "--source-profile", "fineweb2-hq",
    ])

    main()

    assert output_path.read_text(encoding="utf-8").strip() == _normalise(
        good, preserve_newlines=False,
    )
    manifest = json.loads(
        output_path.with_suffix(".txt.manifest.json").read_text(encoding="utf-8"),
    )
    assert manifest["lines_in"] == 3
    assert manifest["lines_written"] == 1
    assert manifest["dropped"] == {"invalid_source_schema": 1, "invalid_json": 1}
    assert manifest["chars_in"] == 2 * len(good)
    assert manifest["chars_written"] == len(_normalise(good, preserve_newlines=False))
    assert manifest["dropped_chars"]["invalid_source_schema"] == len(good)
    assert manifest["flags"]["repetition_mode"] == "off"
