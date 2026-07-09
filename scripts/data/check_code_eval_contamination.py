#!/usr/bin/env python3
"""Decontamination check: code JSONLs vs HumanEval / MBPP (n-gram overlap).

~80% of the corpus20b code is synthetic (OpenCoder) — if a teacher saw
HumanEval/MBPP, pass@k on the annealed model is meaningless. This scans the
code JSONLs ({"text": ...}, one doc per line) for word-level n-gram overlap
against the eval prompts + canonical solutions and reports the contamination
rate; --emit-clean writes filtered copies.

Eval files (fetch once, ~1 MB total):
  HumanEval: https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz
  MBPP:      https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl

Method: normalize (lowercase, strip comments, collapse whitespace), split to
words, hash all 13-grams of each eval problem; a doc is contaminated if any
window shares an n-gram. Measured on 300-doc samples (n=13): code_multi 2.3%,
opc_algorithmic 28.0%, opc_snippets 4.0%, opc_qa 6.7% — OpenCoder algorithmic
is heavily MBPP-derived. Run on the container:

    python3 scripts/data/check_code_eval_contamination.py \
        --inputs /workspace/v2data/data/fresh/code_multi.jsonl,/workspace/v2data/data/fresh/opc_algorithmic.jsonl,/workspace/v2data/data/fresh/opc_snippets.jsonl,/workspace/v2data/data/fresh/opc_qa.jsonl \
        --humaneval /tmp/HumanEval.jsonl --mbpp /tmp/mbpp.jsonl --emit-clean
"""

from __future__ import annotations

import argparse
import gzip
import json
import pathlib
import re
import sys

NGRAM = 13  # GPT-3/PaLM decontamination standard; --ngram 10 is stricter (more FPs on idioms)
_WORD = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+|[^\sA-Za-z0-9_]")
_COMMENT = re.compile(r"#[^\n]*")


def words(text: str) -> list[str]:
    return _WORD.findall(_COMMENT.sub(" ", text.lower()))


def ngrams(ws: list[str], n: int = NGRAM):
    return {hash(tuple(ws[i : i + n])) for i in range(len(ws) - n + 1)}


def load_eval(path: pathlib.Path) -> list[str]:
    op = gzip.open if path.suffix == ".gz" else open
    probs = []
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            # HumanEval: prompt+canonical_solution; MBPP: text+code
            probs.append(
                (d.get("prompt") or d.get("text") or "")
                + "\n"
                + (d.get("canonical_solution") or d.get("code") or "")
            )
    return probs


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--inputs", required=True, help="comma-separated code JSONLs")
    ap.add_argument("--humaneval", type=pathlib.Path, default=None)
    ap.add_argument("--mbpp", type=pathlib.Path, default=None)
    ap.add_argument("--ngram", type=int, default=NGRAM)
    ap.add_argument(
        "--emit-clean", action="store_true", help="write <input>.decontam.jsonl without hits"
    )
    ap.add_argument(
        "--dump-hits", type=int, default=3, help="print first N contaminated docs per file"
    )
    a = ap.parse_args()
    if not (a.humaneval or a.mbpp):
        sys.exit("need --humaneval and/or --mbpp")

    bank: set[int] = set()
    nprob = 0
    for p in (a.humaneval, a.mbpp):
        if p:
            probs = load_eval(p)
            nprob += len(probs)
            for t in probs:
                bank |= ngrams(words(t), a.ngram)
    print(f"eval bank: {nprob} problems, {len(bank):,} distinct {a.ngram}-grams", flush=True)

    tot_docs = tot_hits = 0
    for inp in a.inputs.split(","):
        inp = pathlib.Path(inp)
        out = (
            open(inp.with_suffix(".decontam.jsonl"), "w", encoding="utf-8")
            if a.emit_clean
            else None
        )
        ndoc = nhit = shown = 0
        for line in open(inp, encoding="utf-8"):
            if not line.strip():
                continue
            ndoc += 1
            try:
                txt = json.loads(line)["text"]
            except Exception:
                continue
            ws = words(txt)
            hit = any(
                hash(tuple(ws[i : i + a.ngram])) in bank for i in range(len(ws) - a.ngram + 1)
            )
            if hit:
                nhit += 1
                if shown < a.dump_hits:
                    shown += 1
                    print(f"  HIT {inp.name}#{ndoc}: {txt[:120]!r}", flush=True)
            elif out:
                out.write(line)
        if out:
            out.close()
        tot_docs += ndoc
        tot_hits += nhit
        print(
            f"{inp.name}: {nhit}/{ndoc} contaminated ({100 * nhit / max(ndoc, 1):.3f}%)"
            + (f" -> {inp.with_suffix('.decontam.jsonl')}" if a.emit_clean else ""),
            flush=True,
        )
    print(
        f"=== total {tot_hits}/{tot_docs} ({100 * tot_hits / max(tot_docs, 1):.3f}%) ===",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
