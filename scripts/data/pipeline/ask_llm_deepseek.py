"""Ask-LLM quality scoring pipeline using DeepSeek V4 Flash via OpenRouter.

Replaces the Flan-T5 POC (which was too weak — see ask_llm_poc.py results)
with a real production-grade scorer. distilabel orchestrates batches and
retries; DeepSeek V4 Flash is cheap (~$0.14 per 1M input tokens) and
handles German fluently — both regressions of the Flan-T5 attempt.

Usage:
    OPENROUTER_API_KEY=sk-or-... \\
    python scripts/data/pipeline/ask_llm_deepseek.py \\
        --input  /staging/raw/fineweb_10bt/fineweb_10bt.txt \\
        --output /staging/cleaned/ask_llm/fineweb_10bt_scored.jsonl \\
        --max-docs 1000 \\
        --threshold 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable


SCORE_PROMPT = """\
You are a strict data-quality evaluator for an LLM-pretraining corpus.
Rate the document on a 1-5 scale:
1 = useless boilerplate, link spam, gibberish, or content with no informational value.
2 = mostly noise with occasional useful sentences.
3 = mediocre web text — readable but with grammatical issues or low information density.
4 = clean, informative, well-formed prose suitable for training.
5 = high-quality reference material (encyclopaedia, textbook, research summary).

Document head (first 1500 chars):
{doc_head}

Reply with this exact format on a single line:
Score: <digit from 1 to 5>"""


def read_blank_separated_docs(path: Path) -> Iterable[tuple[int, str]]:
    """Stream multi-line-with-blank-line-separator docs from a text file."""
    buf: list[str] = []
    n = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip() == "":
                if buf:
                    text = " ".join(b.strip() for b in buf if b.strip())
                    if text:
                        yield n, text
                        n += 1
                    buf.clear()
            else:
                buf.append(line)
        if buf:
            text = " ".join(b.strip() for b in buf if b.strip())
            if text:
                yield n, text


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-docs", type=int, default=100)
    p.add_argument("--threshold", type=int, default=3,
                   help="keep docs with score >= threshold (default 3)")
    p.add_argument("--head-chars", type=int, default=1500)
    p.add_argument("--model", default="qwen/qwen3.6-35b-a3b",
                   help="OpenRouter model id. Default: qwen/qwen3.6-35b-a3b — "
                        "matches the model the user runs locally on bitbastion "
                        "for self-consistency between corpus-filtering and "
                        "downstream judging. Reasoning is auto-disabled via "
                        "extra_body. deepseek/deepseek-chat-v3.1 is NOT "
                        "recommended — exhibits a 18%% degenerate-token bug "
                        "('棣棣棣') at short completion lengths.")
    p.add_argument("--batch-size", type=int, default=8,
                   help="distilabel parallelism per LLM call")
    args = p.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("FATAL: OPENROUTER_API_KEY env var is required")

    # distilabel pipeline: load docs as a generator, score via OpenAI-compatible
    # OpenRouter LLM, write JSONL.
    from distilabel.pipeline import Pipeline
    from distilabel.steps import LoadDataFromDicts
    from distilabel.steps.tasks import TextGeneration
    from distilabel.models import OpenAILLM

    print(f"Reading {args.input} ...", flush=True)
    docs = []
    for doc_id, text in read_blank_separated_docs(args.input):
        if doc_id >= args.max_docs:
            break
        docs.append({
            "doc_id": doc_id,
            "instruction": SCORE_PROMPT.format(doc_head=text[:args.head_chars]),
            "head": text[:200],
            "length_chars": len(text),
        })
    print(f"  {len(docs)} docs prepared", flush=True)

    # NOTE on the prompt + decoding:
    # The 5000-doc v1 run on deepseek/deepseek-chat-v3.1 (T=0, max=4,
    # "respond with EXACTLY one digit") hit a degenerate-token loop on
    # ~18% of inputs ("棣棣棣棣"); the bug is in deepseek-chat-v3.1
    # itself, not the prompt. Switched default to llama-3.3-70b/qwen3.x.
    #
    # For reasoning models (qwen3.6-*, qwen3-thinking, etc.) we MUST
    # disable thinking — otherwise max_new_tokens=16 is consumed by the
    # reasoning trace and content comes back empty. OpenRouter's
    # extra_body={"reasoning": {"enabled": false}} works for all models
    # that support it; non-reasoning models silently ignore it.
    llm = OpenAILLM(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        model=args.model,
        generation_kwargs={
            "temperature": 0.05,
            "max_new_tokens": 16,
            "extra_body": {"reasoning": {"enabled": False}},
        },
    )

    with Pipeline(name="ask-llm-deepseek") as pipeline:
        loader = LoadDataFromDicts(data=docs)
        score_step = TextGeneration(
            llm=llm,
            input_batch_size=args.batch_size,
            num_generations=1,
        )
        loader >> score_step

    distiset = pipeline.run(
        parameters={
            score_step.name: {
                "llm": {"generation_kwargs": {
                    "temperature": 0.05,
                    "max_new_tokens": 16,
                    "extra_body": {"reasoning": {"enabled": False}},
                }},
            },
        },
        use_cache=False,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_kept = 0
    n_total = 0
    histogram = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, "?": 0}

    # distiset structure: { leaf_step_name -> { split_name -> HF Dataset } }
    # We have one leaf (score_step) and the default 'train' split.
    def _iter_rows(distiset):
        for leaf_name, leaf in distiset.items():
            for split_name, ds in leaf.items():
                for row in ds:
                    yield row

    with args.output.open("w", encoding="utf-8") as out_f:
        for row in _iter_rows(distiset):
            n_total += 1
            generation = (row.get("generation") or "").strip()
            # Parse first digit
            score = None
            for ch in generation:
                if ch in "12345":
                    score = int(ch)
                    break
            histogram[score if score in histogram else "?"] += 1
            rec = {
                "doc_id": row.get("doc_id"),
                "score": score,
                "raw_response": generation,
                "head": row.get("head"),
                "length_chars": row.get("length_chars"),
                "kept": score is not None and score >= args.threshold,
            }
            if rec["kept"]:
                n_kept += 1
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print()
    print(f"=== Ask-LLM (DeepSeek) results ===")
    print(f"  scored:  {n_total} docs")
    print(f"  kept:    {n_kept} (threshold ≥ {args.threshold}) — {100*n_kept/max(n_total,1):.1f}%")
    print(f"  output:  {args.output}")
    print(f"  histogram:")
    for k in [1, 2, 3, 4, 5, "?"]:
        bar = "█" * histogram[k]
        print(f"    {k}: {histogram[k]:4d}  {bar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
