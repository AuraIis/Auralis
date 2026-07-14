"""Tests for prompt-and-answer contamination matching."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.data.contamination_check import (  # noqa: E402
    _answer_candidates,
    _matches,
    _normalise,
)


def test_question_without_answer_is_not_a_contamination_hit() -> None:
    question = _normalise("Was ist die Hauptstadt von Deutschland?")
    answers = _answer_candidates({"expected_keywords": ["Berlin"]})
    assert not _matches(
        _normalise("Was ist die Hauptstadt von Deutschland? Diskutieren Sie."),
        question,
        answers,
    )


def test_question_and_expected_answer_are_a_hit() -> None:
    question = _normalise("Was ist die Hauptstadt von Deutschland?")
    answers = _answer_candidates({"expected_keywords": ["Berlin"]})
    line = _normalise("Was ist die Hauptstadt von Deutschland? Antwort: Berlin.")
    assert _matches(line, question, answers)
