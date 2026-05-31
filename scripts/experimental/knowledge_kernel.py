"""Build Duden/DNA-style knowledge blocks for pretraining experiments.

The goal is not to copy Duden. It is to test whether compact, structured
lexicon/fact blocks help a tiny model learn concepts more cleanly than the same
facts as plain prose.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

try:
    import sentencepiece as spm
except ImportError:  # pragma: no cover - handled by CLI error path.
    spm = None


FUTURE_TAGS = [
    "<|definition|>",
    "<|fact|>",
    "<|example|>",
    "<|source|>",
    "<|question|>",
    "<|answer|>",
    "<|end|>",
]
CURRENT_TAGS = ["<memory>", "</memory>", "<recall>", "</recall>", "<|end|>"]


@dataclass
class KnowledgeEntry:
    term: str
    definition: str
    facts: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)
    source: str = "curated"
    questions: list[dict[str, str]] = field(default_factory=list)

    def validate(self) -> None:
        if not self.term.strip():
            raise ValueError("entry has empty term")
        if len(self.definition.strip()) < 12:
            raise ValueError(f"{self.term}: definition is too short")
        for qa in self.questions:
            if not qa.get("question") or not qa.get("answer"):
                raise ValueError(f"{self.term}: question entries need question+answer")


def load_entries(path: Path) -> list[KnowledgeEntry]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    rows = raw["entries"] if isinstance(raw, dict) and "entries" in raw else raw
    if not isinstance(rows, list):
        raise ValueError("knowledge input must be a list or contain an entries list")
    entries = [KnowledgeEntry(**row) for row in rows]
    for entry in entries:
        entry.validate()
    return entries


def sample_entries() -> list[KnowledgeEntry]:
    return [
        KnowledgeEntry(
            term="Photosynthese",
            definition=(
                "Die Photosynthese ist ein biochemischer Prozess, bei dem Pflanzen, "
                "Algen und einige Bakterien Lichtenergie in chemische Energie umwandeln."
            ),
            facts=[
                "Chlorophyll nimmt Lichtenergie auf.",
                "Aus Kohlenstoffdioxid und Wasser entstehen Glucose und Sauerstoff.",
                "Die Lichtreaktion liefert Energie für den Calvin-Zyklus.",
            ],
            examples=[
                "Eine Sonnenblume nutzt Photosynthese, um aus Licht und CO2 energiereiche Stoffe aufzubauen.",
            ],
            related=["Chlorophyll", "Calvin-Zyklus", "Glucose", "Sauerstoff"],
            source="curated_biology_seed",
            questions=[
                {
                    "question": "Was entsteht bei der Photosynthese neben Glucose?",
                    "answer": "Sauerstoff.",
                },
                {
                    "question": "Welche Rolle spielt Chlorophyll?",
                    "answer": "Chlorophyll nimmt Lichtenergie auf.",
                },
            ],
        ),
        KnowledgeEntry(
            term="Berlin",
            definition="Berlin ist die Hauptstadt der Bundesrepublik Deutschland.",
            facts=[
                "Berlin ist ein deutsches Bundesland und zugleich eine Stadt.",
                "Die Stadt liegt im Nordosten Deutschlands.",
                "Der Regierungssitz Deutschlands befindet sich in Berlin.",
            ],
            examples=["Der Deutsche Bundestag tagt im Reichstagsgebäude in Berlin."],
            related=["Deutschland", "Bundestag", "Bundesland"],
            source="curated_geo_seed",
            questions=[
                {"question": "Was ist die Hauptstadt Deutschlands?", "answer": "Berlin."},
                {"question": "Liegt Berlin in Deutschland?", "answer": "Ja."},
            ],
        ),
        KnowledgeEntry(
            term="Johann Wolfgang von Goethe",
            definition=(
                "Johann Wolfgang von Goethe war ein deutscher Dichter, Schriftsteller "
                "und Naturforscher."
            ),
            facts=[
                "Goethe schrieb Faust.",
                "Goethe gehört zur Weimarer Klassik.",
                "Goethe ist nicht der Autor von Mein Kampf.",
            ],
            examples=["Die Tragödie Faust ist eines der bekanntesten Werke Goethes."],
            related=["Faust", "Weimarer Klassik", "Schiller"],
            source="curated_literature_seed",
            questions=[
                {"question": "Wer schrieb Faust?", "answer": "Johann Wolfgang von Goethe."},
                {"question": "Schrieb Goethe Mein Kampf?", "answer": "Nein."},
            ],
        ),
        KnowledgeEntry(
            term="Addition",
            definition="Die Addition ist eine Grundrechenart, bei der Zahlen zusammengezählt werden.",
            facts=[
                "Das Ergebnis einer Addition heißt Summe.",
                "2 + 2 ergibt 4.",
                "Addition ist kommutativ: a + b = b + a.",
            ],
            examples=["Wenn man zwei Äpfel und zwei Äpfel zusammenlegt, hat man vier Äpfel."],
            related=["Summe", "Rechnen", "Mathematik"],
            source="curated_math_seed",
            questions=[
                {"question": "Rechne 2 + 2.", "answer": "4."},
                {"question": "Wie heißt das Ergebnis einer Addition?", "answer": "Summe."},
            ],
        ),
        KnowledgeEntry(
            term="Python-Funktion",
            definition=(
                "Eine Python-Funktion ist ein benannter Codeblock, der mit def definiert "
                "wird und wiederverwendbare Logik enthalten kann."
            ),
            facts=[
                "Parameter stehen in Klammern hinter dem Funktionsnamen.",
                "return gibt einen Wert aus der Funktion zurück.",
                "Einrückung bestimmt in Python den Funktionskörper.",
            ],
            examples=["def add(a, b):\n    return a + b"],
            related=["def", "return", "Parameter", "Einrückung"],
            source="curated_code_seed",
            questions=[
                {
                    "question": "Wie definiert man in Python eine Funktion?",
                    "answer": "Mit def, zum Beispiel: def add(a, b):",
                },
                {"question": "Was macht return?", "answer": "return gibt einen Wert zurück."},
            ],
        ),
    ]


def plain_block(entry: KnowledgeEntry) -> str:
    parts = [f"{entry.term}: {entry.definition}"]
    if entry.facts:
        parts.append(" ".join(entry.facts))
    if entry.examples:
        parts.append("Beispiel: " + " ".join(entry.examples))
    if entry.related:
        parts.append("Verwandte Begriffe: " + ", ".join(entry.related) + ".")
    return "\n".join(parts)


def future_block(entry: KnowledgeEntry) -> str:
    lines = [
        "<|definition|>",
        f"Begriff: {entry.term}",
        f"Definition: {entry.definition}",
    ]
    if entry.facts:
        lines.extend(["<|fact|>", *[f"- {fact}" for fact in entry.facts]])
    if entry.examples:
        lines.extend(["<|example|>", *entry.examples])
    if entry.related:
        lines.append("Verwandte Begriffe: " + ", ".join(entry.related))
    lines.extend(["<|source|>", entry.source, "<|end|>"])
    return "\n".join(lines)


def current_block(entry: KnowledgeEntry) -> str:
    lines = [
        "<memory>",
        "Typ: definition",
        f"Begriff: {entry.term}",
        f"Definition: {entry.definition}",
    ]
    if entry.facts:
        lines.extend(["Fakten:", *[f"- {fact}" for fact in entry.facts]])
    if entry.examples:
        lines.extend(["Beispiele:", *entry.examples])
    if entry.related:
        lines.append("Verwandte Begriffe: " + ", ".join(entry.related))
    lines.extend([f"Quelle: {entry.source}", "</memory>"])
    return "\n".join(lines)


def qa_rows(entries: list[KnowledgeEntry]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for entry in entries:
        for qa in entry.questions:
            rows.append(
                {
                    "term": entry.term,
                    "instruction": qa["question"],
                    "output": qa["answer"],
                    "source": entry.source,
                }
            )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def token_audit(tokenizer_path: Path, texts: dict[str, str]) -> dict[str, Any]:
    if spm is None:
        return {"available": False, "reason": "sentencepiece not installed"}
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    out: dict[str, Any] = {
        "available": True,
        "tokenizer": str(tokenizer_path),
        "vocab_size": sp.get_piece_size(),
        "special_tokens": {},
        "texts": {},
    }
    for token in sorted(set(FUTURE_TAGS + CURRENT_TAGS)):
        piece_id = sp.piece_to_id(token)
        ids = sp.encode(token, out_type=int)
        out["special_tokens"][token] = {
            "piece_id": piece_id,
            "encoded_ids": ids,
            "registered": piece_id >= 0 and piece_id in ids and sp.decode(ids) == token,
        }
    for name, text in texts.items():
        ids = sp.encode(text, out_type=int)
        out["texts"][name] = {
            "chars": len(text),
            "tokens": len(ids),
            "chars_per_token": round(len(text) / max(len(ids), 1), 3),
            "unk": sum(1 for i in ids if i == sp.unk_id()),
        }
    return out


def write_report(path: Path, entries: list[KnowledgeEntry], audit: dict[str, Any]) -> None:
    lines = [
        "# Knowledge Kernel Smoke Report",
        "",
        f"- Entries: {len(entries)}",
        f"- QA rows: {len(qa_rows(entries))}",
        "",
        "## Tokenizer Tag Audit",
        "",
    ]
    if audit.get("available"):
        for token, info in audit["special_tokens"].items():
            status = "registered" if info["registered"] else "split"
            lines.append(f"- `{token}`: {status}, ids={info['encoded_ids']}")
        lines.extend(["", "## Text Efficiency", ""])
        for name, info in audit["texts"].items():
            lines.append(
                f"- `{name}`: {info['tokens']} tokens, "
                f"{info['chars_per_token']} chars/token, unk={info['unk']}"
            )
    else:
        lines.append(f"- tokenizer audit skipped: {audit.get('reason')}")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `current_kernel.txt` uses existing tokenizer specials and can be tested now.",
            "- `future_kernel.txt` shows the cleaner format for a future tokenizer with dedicated tags.",
            "- `plain_corpus.txt` is the baseline for ablation.",
            "- `qa_eval.jsonl` is the small probe set to test whether facts are learned.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_outputs(entries: list[KnowledgeEntry], out: Path, tokenizer: Path | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    plain = "\n\n".join(plain_block(entry) for entry in entries) + "\n"
    current = "\n\n".join(current_block(entry) for entry in entries) + "\n"
    future = "\n\n".join(future_block(entry) for entry in entries) + "\n"

    (out / "entries.json").write_text(
        json.dumps([asdict(entry) for entry in entries], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "plain_corpus.txt").write_text(plain, encoding="utf-8")
    (out / "current_kernel.txt").write_text(current, encoding="utf-8")
    (out / "future_kernel.txt").write_text(future, encoding="utf-8")
    write_jsonl(out / "qa_eval.jsonl", qa_rows(entries))

    texts = {
        "plain_corpus": plain,
        "current_kernel": current,
        "future_kernel": future,
    }
    audit = token_audit(tokenizer, texts) if tokenizer else {"available": False, "reason": "no tokenizer"}
    (out / "tokenizer_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(out / "report.md", entries, audit)
    print(f"wrote {len(entries)} entries to {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sample = sub.add_parser("sample", help="write a curated sample knowledge kernel")
    sample.add_argument("--output-dir", type=Path, required=True)
    sample.add_argument("--tokenizer", type=Path, default=Path("tokenizer/helix_v2_tokenizer.model"))
    sample.set_defaults(func=lambda a: build_outputs(sample_entries(), a.output_dir, a.tokenizer))

    build = sub.add_parser("build", help="build from a YAML/JSON entries file")
    build.add_argument("--input", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--tokenizer", type=Path, default=Path("tokenizer/helix_v2_tokenizer.model"))
    build.set_defaults(func=lambda a: build_outputs(load_entries(a.input), a.output_dir, a.tokenizer))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        args.func(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

