"""Synthetic Q&A generation from politik / domain corpora.

Implements the SwallowCode-style pattern: take raw structured text, ask
an LLM to convert it into high-quality instruction-following examples.
For us: turn 14k Plenarprotokoll-Reden + 10k Gerichtsentscheidungen +
4.6k MdB-Lebensläufe into 30k+ Q&A pairs ready for Phase-5 MoRA training.

Usage:
    OPENROUTER_API_KEY=sk-or-... \\
    python scripts/data/pipeline/synth_qa_pairs.py \\
        --input  /staging/politik_de/raw/bundestag_protokolle/bundestag_protokolle.jsonl \\
        --output /staging/politik_de/sft/protokolle_qa.jsonl \\
        --schema plenary \\
        --max-docs 100 \\
        --pairs-per-doc 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# Schema-specific prompts. Each schema corresponds to one of our politik
# JSONL inputs. The prompt instructs DeepSeek to produce high-quality
# question-answer pairs grounded ONLY in the source material — with the
# anti-hallucination guard from L-017.
PROMPTS: dict[str, str] = {

    "plenary": """\
Du bekommst eine Bundestags-Plenarrede. Erzeuge {n_pairs} Frage-Antwort-Paare,
die diese Rede als FAKTENGRUNDLAGE nutzen.

ABSOLUT KRITISCH:
- Antworten dürfen NUR Aussagen enthalten, die wörtlich oder paraphrasiert in
  der Rede stehen. Niemals Fakten erfinden.
- Wenn die Rede etwas nicht hergibt, formuliere keine Frage dazu.
- Verwende keine Spekulations-Marker wie "vermutlich", "wahrscheinlich",
  "soll", "angeblich".
- Jede Antwort muss die Quelle (Plenarprotokoll {wp}/{sitzung}, Sprecher
  {redner} {fraktion}) am Ende referenzieren.

Quellen-Metadaten:
  Wahlperiode:   {wp}
  Sitzung:       {sitzung}
  Redner:        {redner}
  Fraktion:      {fraktion}

Rede-Text:
{text}

Antworte als JSON-Array mit {n_pairs} Objekten, jeweils
{{"question": "...", "answer": "..."}}, sonst nichts.""",

    "caselaw": """\
Du bekommst eine deutsche Gerichtsentscheidung. Erzeuge {n_pairs} Frage-Antwort-Paare
basierend AUSSCHLIESSLICH auf dem Urteilstext.

REGELN:
- Antworten enthalten nur, was im Urteilstext steht.
- Bei rechtlichen Aussagen: nenne den genauen Paragraphen/Norm wenn vorhanden.
- Quelle in jeder Antwort: {courtType} {documentType} {decisionDate}, ECLI {ecli}.

Urteils-Metadaten:
  Gericht:       {courtType}
  Dokumenttyp:   {documentType}
  Datum:         {decisionDate}
  Aktenzeichen:  {fileNumbers}
  ECLI:          {ecli}

Urteilstext (gekürzt):
{body}

Antworte als JSON-Array mit {n_pairs} Q&A-Objekten:
{{"question": "...", "answer": "..."}}""",

    "politician": """\
Du bekommst die Stammdaten eines/r deutschen Bundestagsabgeordneten.
Erzeuge {n_pairs} Frage-Antwort-Paare nur basierend auf diesen Daten.

REGELN:
- Niemals Spekulation. Wenn ein Feld fehlt, frag nicht danach.
- Quelle in jeder Antwort: "Bundestag MdB-Stammdaten".

Politiker-Daten:
{record}

Erzeuge {n_pairs} Q&A-Objekte als JSON-Array.""",
}


def load_records(path: Path, schema: str, max_docs: int):
    """Yields enriched records ready for prompt-rendering."""
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_docs:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if schema == "plenary":
                yield {
                    "wp": rec.get("wp", "?"),
                    "sitzung": rec.get("sitzung", "?"),
                    "redner": rec.get("redner", "?"),
                    "fraktion": rec.get("fraktion", "?"),
                    "text": (rec.get("text") or "")[:6000],
                    "raw_id": rec.get("source_url", f"plenary_{i}"),
                }
            elif schema == "caselaw":
                yield {
                    "courtType": rec.get("courtType", "?"),
                    "documentType": rec.get("documentType", "?"),
                    "decisionDate": rec.get("decisionDate", "?"),
                    "fileNumbers": ", ".join(rec.get("fileNumbers") or []),
                    "ecli": rec.get("ecli", "?"),
                    "body": (rec.get("body") or "")[:8000],
                    "raw_id": rec.get("documentNumber", f"case_{i}"),
                }
            elif schema == "politician":
                # Reduced JSON dump — keep only fields useful for QA.
                useful = {k: v for k, v in rec.items()
                          if k in ("nachname", "vorname", "partei_kurz",
                                    "geburtsdatum", "geburtsort", "beruf",
                                    "vita_kurz", "mandate")}
                yield {
                    "record": json.dumps(useful, ensure_ascii=False, indent=2),
                    "raw_id": rec.get("id", f"pol_{i}"),
                }
            else:
                raise SystemExit(f"unknown schema: {schema}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--schema", choices=sorted(PROMPTS.keys()), required=True)
    p.add_argument("--max-docs", type=int, default=100)
    p.add_argument("--pairs-per-doc", type=int, default=3)
    p.add_argument("--model", default="deepseek/deepseek-chat-v3.1")
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("FATAL: OPENROUTER_API_KEY env var required")

    from distilabel.pipeline import Pipeline
    from distilabel.steps import LoadDataFromDicts
    from distilabel.steps.tasks import TextGeneration
    from distilabel.models import OpenAILLM

    template = PROMPTS[args.schema]
    print(f"Reading {args.input} ...", flush=True)
    docs = []
    for rec in load_records(args.input, args.schema, args.max_docs):
        rec_for_prompt = {**rec, "n_pairs": args.pairs_per_doc}
        docs.append({
            "instruction": template.format(**rec_for_prompt),
            "raw_id": rec["raw_id"],
        })
    print(f"  {len(docs)} source docs prepared, asking for "
          f"{args.pairs_per_doc} Q&A pairs each = "
          f"~{len(docs) * args.pairs_per_doc} target examples")

    llm = OpenAILLM(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        model=args.model,
        generation_kwargs={"temperature": 0.5, "max_new_tokens": 1500},
    )

    with Pipeline(name=f"synth-qa-{args.schema}") as pipeline:
        loader = LoadDataFromDicts(data=docs)
        gen_step = TextGeneration(
            llm=llm,
            input_batch_size=args.batch_size,
            num_generations=1,
        )
        loader >> gen_step

    distiset = pipeline.run(use_cache=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_total = 0
    n_pairs = 0
    n_parse_fail = 0
    with args.output.open("w", encoding="utf-8") as out_f:
        for split_name, ds in distiset.items():
            for row in ds:
                n_total += 1
                generation = (row.get("generation") or "").strip()
                # Try to parse JSON array from the response
                pairs = None
                try:
                    # Strip code fences if present
                    if generation.startswith("```"):
                        first_nl = generation.find("\n")
                        last_fence = generation.rfind("```")
                        generation = generation[first_nl + 1: last_fence].strip()
                    pairs = json.loads(generation)
                    if not isinstance(pairs, list):
                        pairs = None
                except json.JSONDecodeError:
                    pairs = None
                if pairs is None:
                    n_parse_fail += 1
                    continue
                for pair in pairs:
                    if not isinstance(pair, dict):
                        continue
                    q, a = pair.get("question"), pair.get("answer")
                    if not q or not a:
                        continue
                    out_f.write(json.dumps({
                        "source_id": row.get("raw_id"),
                        "schema": args.schema,
                        "messages": [
                            {"role": "user", "content": q},
                            {"role": "assistant", "content": a},
                        ],
                    }, ensure_ascii=False) + "\n")
                    n_pairs += 1

    print()
    print(f"=== Synth Q&A results ===")
    print(f"  source docs: {n_total}")
    print(f"  parse fails: {n_parse_fail}")
    print(f"  Q&A pairs:   {n_pairs}")
    print(f"  output:      {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
