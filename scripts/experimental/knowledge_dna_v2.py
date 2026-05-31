"""Build Knowledge-DNA v2 corpora and probes.

This is a practical version of the "Duden/DNA" idea for Auralis. It avoids
changing the tokenizer and uses only special tokens that already exist in the
Helix v2 tokenizer: ``<memory>``, ``</memory>``, ``<recall>``, ``</recall>``,
and ``<|end|>``.

The output is intentionally split into three comparable variants:

* plain: facts as normal prose
* dna: structured memory/recall blocks
* hybrid: prose + memory + recall/counterfact/transfer tasks
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

try:
    import sentencepiece as spm
except ImportError:  # pragma: no cover - CLI reports this in tokenizer audit.
    spm = None


CURRENT_SPECIALS = ["<memory>", "</memory>", "<recall>", "</recall>", "<|end|>"]
VARIANT_FILES = {
    "plain": "plain_corpus.txt",
    "dna": "dna_corpus.txt",
    "hybrid": "hybrid_corpus.txt",
}


@dataclass
class Probe:
    question: str
    answer: str
    kind: str = "fact"
    aliases: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)

    def validate(self, term: str) -> None:
        if not self.question.strip():
            raise ValueError(f"{term}: probe has empty question")
        if not self.answer.strip():
            raise ValueError(f"{term}: probe has empty answer")
        if self.kind not in {"fact", "definition", "counterfact", "transfer", "format"}:
            raise ValueError(f"{term}: unsupported probe kind {self.kind!r}")


@dataclass
class KnowledgeDNAEntry:
    term: str
    definition: str
    facts: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)
    counterfacts: list[str] = field(default_factory=list)
    source: str = "curated"
    probes: list[Probe] = field(default_factory=list)

    def validate(self) -> None:
        if not self.term.strip():
            raise ValueError("entry has empty term")
        if len(self.definition.strip()) < 12:
            raise ValueError(f"{self.term}: definition is too short")
        if not self.facts:
            raise ValueError(f"{self.term}: at least one fact is required")
        if not self.probes:
            raise ValueError(f"{self.term}: at least one probe is required")
        for probe in self.probes:
            probe.validate(self.term)


def _probe_from_raw(raw: dict[str, Any]) -> Probe:
    return Probe(
        question=str(raw["question"]),
        answer=str(raw["answer"]),
        kind=str(raw.get("kind", "fact")),
        aliases=[str(v) for v in raw.get("aliases", [])],
        forbidden=[str(v) for v in raw.get("forbidden", [])],
    )


def _entry_from_raw(raw: dict[str, Any]) -> KnowledgeDNAEntry:
    entry = KnowledgeDNAEntry(
        term=str(raw["term"]),
        definition=str(raw["definition"]),
        facts=[str(v) for v in raw.get("facts", [])],
        examples=[str(v) for v in raw.get("examples", [])],
        related=[str(v) for v in raw.get("related", [])],
        counterfacts=[str(v) for v in raw.get("counterfacts", [])],
        source=str(raw.get("source", "curated")),
        probes=[_probe_from_raw(v) for v in raw.get("probes", [])],
    )
    entry.validate()
    return entry


def load_entries(path: Path) -> list[KnowledgeDNAEntry]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    rows = raw["entries"] if isinstance(raw, dict) and "entries" in raw else raw
    if not isinstance(rows, list):
        raise ValueError("knowledge DNA input must be a list or contain an entries list")
    return [_entry_from_raw(row) for row in rows]


def sample_entries() -> list[KnowledgeDNAEntry]:
    entries = [
        KnowledgeDNAEntry(
            term="Photosynthese",
            definition=(
                "Die Photosynthese ist ein biochemischer Prozess, bei dem Pflanzen, "
                "Algen und einige Bakterien Lichtenergie in chemische Energie umwandeln."
            ),
            facts=[
                "Chlorophyll nimmt Lichtenergie auf.",
                "Aus Kohlenstoffdioxid und Wasser entstehen Glucose und Sauerstoff.",
                "Der Calvin-Zyklus nutzt Energie aus der Lichtreaktion.",
            ],
            examples=[
                "Eine Sonnenblume baut mit Licht, Wasser und Kohlenstoffdioxid energiereiche Stoffe auf."
            ],
            related=["Chlorophyll", "Calvin-Zyklus", "Glucose", "Sauerstoff"],
            counterfacts=["Photosynthese ist keine Zellatmung."],
            source="curated_biology_seed",
            probes=[
                Probe(
                    "Was ist Photosynthese?",
                    "Photosynthese wandelt Lichtenergie in chemische Energie um.",
                    "definition",
                    aliases=["Lichtenergie", "chemische Energie"],
                ),
                Probe(
                    "Was entsteht bei der Photosynthese neben Glucose?",
                    "Sauerstoff.",
                    "fact",
                    aliases=["Sauerstoff"],
                ),
                Probe(
                    "Ist Photosynthese dasselbe wie Zellatmung?",
                    "Nein. Photosynthese baut energiereiche Stoffe auf; Zellatmung baut sie ab.",
                    "counterfact",
                    aliases=["Nein", "nicht dasselbe"],
                    forbidden=["Ja."],
                ),
            ],
        ),
        KnowledgeDNAEntry(
            term="Berlin",
            definition="Berlin ist die Hauptstadt der Bundesrepublik Deutschland.",
            facts=[
                "Berlin ist ein deutsches Bundesland und zugleich eine Stadt.",
                "Berlin liegt im Nordosten Deutschlands.",
                "Der Deutsche Bundestag tagt in Berlin.",
            ],
            examples=["Der Reichstag ist ein bekanntes Parlamentsgebaeude in Berlin."],
            related=["Deutschland", "Bundestag", "Bundesland"],
            counterfacts=["Berlin liegt nicht bei Frankfurt am Main."],
            source="curated_geo_seed",
            probes=[
                Probe(
                    "Was ist die Hauptstadt Deutschlands?",
                    "Berlin.",
                    "fact",
                    aliases=["Berlin"],
                    forbidden=["Frankfurt"],
                ),
                Probe(
                    "Liegt Berlin bei Frankfurt?",
                    "Nein. Berlin liegt im Nordosten Deutschlands, Frankfurt am Main liegt in Hessen.",
                    "counterfact",
                    aliases=["Nein", "Nordosten"],
                    forbidden=["Ja."],
                ),
                Probe(
                    "Warum ist Berlin fuer die deutsche Politik wichtig?",
                    "Berlin ist Hauptstadt und Sitz wichtiger Bundesorgane wie des Bundestags.",
                    "transfer",
                    aliases=["Hauptstadt", "Bundestag"],
                ),
            ],
        ),
        KnowledgeDNAEntry(
            term="Johann Wolfgang von Goethe",
            definition="Johann Wolfgang von Goethe war ein deutscher Dichter, Schriftsteller und Naturforscher.",
            facts=[
                "Goethe schrieb Faust.",
                "Goethe gehoert zur Weimarer Klassik.",
                "Goethe ist nicht der Autor von Mein Kampf.",
            ],
            examples=["Faust ist eines der bekanntesten Werke Goethes."],
            related=["Faust", "Weimarer Klassik", "Schiller"],
            counterfacts=["Mein Kampf wurde nicht von Goethe geschrieben."],
            source="curated_literature_seed",
            probes=[
                Probe(
                    "Wer schrieb Faust?",
                    "Johann Wolfgang von Goethe.",
                    "fact",
                    aliases=["Goethe"],
                    forbidden=["Hitler"],
                ),
                Probe(
                    "Schrieb Goethe Mein Kampf?",
                    "Nein. Goethe schrieb Faust; Mein Kampf wurde von Adolf Hitler geschrieben.",
                    "counterfact",
                    aliases=["Nein", "Faust"],
                    forbidden=["Ja."],
                ),
                Probe(
                    "Nenne eine literarische Epoche, mit der Goethe verbunden ist.",
                    "Weimarer Klassik.",
                    "fact",
                    aliases=["Weimarer Klassik"],
                ),
            ],
        ),
        KnowledgeDNAEntry(
            term="Addition",
            definition="Die Addition ist eine Grundrechenart, bei der Zahlen zusammengezaehlt werden.",
            facts=[
                "Das Ergebnis einer Addition heisst Summe.",
                "2 + 2 ergibt 4.",
                "Addition ist kommutativ: a + b = b + a.",
            ],
            examples=["Zwei Aepfel plus zwei Aepfel ergeben vier Aepfel."],
            related=["Summe", "Rechnen", "Mathematik"],
            counterfacts=["2 + 2 ergibt nicht 5."],
            source="curated_math_seed",
            probes=[
                Probe("Rechne 2 + 2.", "4.", "fact", aliases=["4"], forbidden=["5"]),
                Probe(
                    "Wie heisst das Ergebnis einer Addition?",
                    "Summe.",
                    "definition",
                    aliases=["Summe"],
                ),
                Probe(
                    "Ist 2 + 2 gleich 5?",
                    "Nein. 2 + 2 ergibt 4.",
                    "counterfact",
                    aliases=["Nein", "4"],
                    forbidden=["Ja."],
                ),
            ],
        ),
        KnowledgeDNAEntry(
            term="Python-Funktion",
            definition=(
                "Eine Python-Funktion ist ein benannter Codeblock, der mit def definiert "
                "wird und wiederverwendbare Logik enthalten kann."
            ),
            facts=[
                "Parameter stehen in Klammern hinter dem Funktionsnamen.",
                "return gibt einen Wert aus der Funktion zurueck.",
                "Einrueckung bestimmt in Python den Funktionskoerper.",
            ],
            examples=["def add(a, b):\n    return a + b"],
            related=["def", "return", "Parameter", "Einrueckung"],
            counterfacts=["Eine Python-Funktion wird nicht mit function wie in JavaScript definiert."],
            source="curated_code_seed",
            probes=[
                Probe(
                    "Wie definiert man in Python eine Funktion?",
                    "Mit def, zum Beispiel: def add(a, b):",
                    "fact",
                    aliases=["def"],
                    forbidden=["function"],
                ),
                Probe(
                    "Was macht return in einer Python-Funktion?",
                    "return gibt einen Wert aus der Funktion zurueck.",
                    "definition",
                    aliases=["Wert", "zurueck"],
                ),
                Probe(
                    "Ist Einrueckung in Python egal?",
                    "Nein. Einrueckung bestimmt in Python den Codeblock.",
                    "counterfact",
                    aliases=["Nein", "Codeblock"],
                    forbidden=["Ja."],
                ),
            ],
        ),
    ]
    for entry in entries:
        entry.validate()
    return entries


def plain_block(entry: KnowledgeDNAEntry) -> str:
    parts = [f"{entry.term}: {entry.definition}"]
    parts.extend(entry.facts)
    if entry.counterfacts:
        parts.extend(entry.counterfacts)
    if entry.examples:
        parts.append("Beispiel: " + " ".join(entry.examples))
    if entry.related:
        parts.append("Verwandte Begriffe: " + ", ".join(entry.related) + ".")
    return "\n".join(parts)


def dna_memory_block(entry: KnowledgeDNAEntry) -> str:
    lines = [
        "<memory>",
        "Typ: knowledge_dna",
        f"Begriff: {entry.term}",
        f"Definition: {entry.definition}",
        "Fakten:",
        *[f"- {fact}" for fact in entry.facts],
    ]
    if entry.counterfacts:
        lines.extend(["Abgrenzung:", *[f"- {fact}" for fact in entry.counterfacts]])
    if entry.examples:
        lines.extend(["Beispiele:", *entry.examples])
    if entry.related:
        lines.append("Verwandte Begriffe: " + ", ".join(entry.related))
    lines.extend([f"Quelle: {entry.source}", "</memory>"])
    return "\n".join(lines)


def recall_block(entry: KnowledgeDNAEntry, probe: Probe) -> str:
    return "\n".join(
        [
            "<recall>",
            f"Begriff: {entry.term}",
            f"Frage: {probe.question}",
            f"Antwort: {probe.answer}",
            "</recall>",
        ]
    )


def dna_block(entry: KnowledgeDNAEntry) -> str:
    blocks = [dna_memory_block(entry)]
    blocks.extend(recall_block(entry, probe) for probe in entry.probes)
    return "\n\n".join(blocks)


def hybrid_block(entry: KnowledgeDNAEntry) -> str:
    fact_probes = [probe for probe in entry.probes if probe.kind in {"fact", "definition"}]
    hard_probes = [probe for probe in entry.probes if probe.kind in {"counterfact", "transfer"}]
    blocks = [plain_block(entry), dna_memory_block(entry)]
    blocks.extend(recall_block(entry, probe) for probe in hard_probes)
    blocks.extend(recall_block(entry, probe) for probe in fact_probes[:1])
    return "\n\n".join(blocks)


def probe_rows(entries: list[KnowledgeDNAEntry]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        for probe in entry.probes:
            rows.append(
                {
                    "term": entry.term,
                    "question": probe.question,
                    "answer": probe.answer,
                    "kind": probe.kind,
                    "aliases": probe.aliases,
                    "forbidden": probe.forbidden,
                    "source": entry.source,
                }
            )
    return rows


def variant_texts(entries: list[KnowledgeDNAEntry]) -> dict[str, str]:
    return {
        "plain": "\n\n".join(plain_block(entry) for entry in entries) + "\n",
        "dna": "\n\n".join(dna_block(entry) for entry in entries) + "\n",
        "hybrid": "\n\n".join(hybrid_block(entry) for entry in entries) + "\n",
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def tokenizer_audit(tokenizer_path: Path | None, texts: dict[str, str]) -> dict[str, Any]:
    if tokenizer_path is None:
        return {"available": False, "reason": "no tokenizer"}
    if spm is None:
        return {"available": False, "reason": "sentencepiece not installed"}
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    audit: dict[str, Any] = {
        "available": True,
        "tokenizer": str(tokenizer_path),
        "vocab_size": sp.get_piece_size(),
        "special_tokens": {},
        "texts": {},
    }
    for token in CURRENT_SPECIALS:
        ids = sp.encode(token, out_type=int)
        piece_id = sp.piece_to_id(token)
        audit["special_tokens"][token] = {
            "piece_id": piece_id,
            "encoded_ids": ids,
            "registered": piece_id >= 0 and piece_id in ids and sp.decode(ids) == token,
        }
    for name, text in texts.items():
        ids = sp.encode(text, out_type=int)
        audit["texts"][name] = {
            "chars": len(text),
            "tokens": len(ids),
            "chars_per_token": round(len(text) / max(len(ids), 1), 3),
            "unk": sum(1 for token_id in ids if token_id == sp.unk_id()),
        }
    return audit


def write_report(out: Path, entries: list[KnowledgeDNAEntry], texts: dict[str, str], audit: dict[str, Any]) -> None:
    rows = probe_rows(entries)
    by_kind: dict[str, int] = {}
    for row in rows:
        by_kind[row["kind"]] = by_kind.get(row["kind"], 0) + 1

    lines = [
        "# Knowledge-DNA v2 Build Report",
        "",
        f"- Entries: {len(entries)}",
        f"- Probes: {len(rows)}",
        "- Probe kinds: "
        + ", ".join(f"{kind}={count}" for kind, count in sorted(by_kind.items())),
        "",
        "## Variant Sizes",
        "",
        "| Variant | Chars | Tokens | Chars/Token |",
        "|---|---:|---:|---:|",
    ]
    token_stats = audit.get("texts", {}) if audit.get("available") else {}
    for name, text in texts.items():
        stat = token_stats.get(name, {})
        lines.append(
            f"| {name} | {len(text)} | {stat.get('tokens', 'n/a')} | "
            f"{stat.get('chars_per_token', 'n/a')} |"
        )

    lines.extend(["", "## Tokenizer Specials", ""])
    if audit.get("available"):
        for token, info in audit["special_tokens"].items():
            status = "registered" if info["registered"] else "split"
            lines.append(f"- `{token}`: {status}, ids={info['encoded_ids']}")
    else:
        lines.append(f"- skipped: {audit.get('reason')}")

    lines.extend(
        [
            "",
            "## Intended Use",
            "",
            "- Use `plain` as baseline.",
            "- Use `dna` to test pure structured memory learning.",
            "- Use `hybrid` as the practical candidate for a 1-3% pretraining booster.",
        ]
    )
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def build_outputs(entries: list[KnowledgeDNAEntry], out: Path, tokenizer: Path | None, seed: int) -> None:
    rng = random.Random(seed)
    entries = list(entries)
    rng.shuffle(entries)

    out.mkdir(parents=True, exist_ok=True)
    texts = variant_texts(entries)
    for name, filename in VARIANT_FILES.items():
        (out / filename).write_text(texts[name], encoding="utf-8")

    rows = probe_rows(entries)
    write_jsonl(out / "probes.jsonl", rows)
    (out / "entries.json").write_text(
        json.dumps([asdict(entry) for entry in entries], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "version": "knowledge-dna-v2",
        "seed": seed,
        "entries": len(entries),
        "probes": len(rows),
        "variants": {
            name: {"file": filename, "chars": len(texts[name])}
            for name, filename in VARIANT_FILES.items()
        },
        "special_tokens": CURRENT_SPECIALS,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    audit = tokenizer_audit(tokenizer, texts)
    (out / "tokenizer_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(out, entries, texts, audit)
    print(f"wrote Knowledge-DNA v2 outputs to {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sample = sub.add_parser("sample", help="write curated sample DNA corpus")
    sample.add_argument("--output-dir", type=Path, required=True)
    sample.add_argument("--tokenizer", type=Path, default=Path("tokenizer/helix_v2_tokenizer.model"))
    sample.add_argument("--seed", type=int, default=20260513)
    sample.set_defaults(func=lambda a: build_outputs(sample_entries(), a.output_dir, a.tokenizer, a.seed))

    build = sub.add_parser("build", help="build DNA corpus from YAML/JSON")
    build.add_argument("--input", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--tokenizer", type=Path, default=Path("tokenizer/helix_v2_tokenizer.model"))
    build.add_argument("--seed", type=int, default=20260513)
    build.set_defaults(func=lambda a: build_outputs(load_entries(a.input), a.output_dir, a.tokenizer, a.seed))
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
