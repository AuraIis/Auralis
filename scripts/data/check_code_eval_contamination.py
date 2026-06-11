#!/usr/bin/env python3
"""Decontamination check: JSONLs vs eval sets (n-gram overlap).

Supports code (HumanEval, MBPP), math (GSM8K test) and the project's German
benchmarks (GermanQuAD test, MMLU-DE test). Each eval set gets its own n-gram
bank so the report shows WHICH benchmark leaked; per-set n is chosen by text
length (13 for code/math prose, 8 for short German Q/A items).

Scans JSONLs ({"text": ...}, one doc per line) and reports contamination per
input per eval set; --emit-clean writes filtered copies; --json-out writes a
machine-readable report; --self-test plants the first problem of every bank as
a synthetic doc and exits 1 if it isn't flagged (positive control).

Eval files (fetch once):
  HumanEval:  https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz
  MBPP:       https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl
  GSM8K test: HF openai/gsm8k main/test -> {"question","answer"}
  GermanQuAD: HF deepset/germanquad test -> {"question","answers"}
  MMLU-DE:    HF alexandrainst/m_mmlu de/test -> {"instruction","option_a..d","answer"}

Run on the container:
    python3 scripts/data/check_code_eval_contamination.py \
        --inputs /workspace/v2data/data/fresh/code_multi.jsonl \
        --humaneval /tmp/HumanEval.jsonl.gz --mbpp /tmp/mbpp.jsonl \
        --gsm8k /tmp/gsm8k_test.jsonl --germanquad /tmp/germanquad_test.jsonl \
        --mmlu-de /tmp/mmlu_de_test.jsonl --self-test --json-out /tmp/contam.json

Measured 2026-06 (300-doc samples, n=13): code_multi 2.3%, opc_algorithmic
28.0% (MBPP-derived), opc_snippets 4.0%, opc_qa 6.7%.
"""
from __future__ import annotations
import argparse, gzip, json, pathlib, re, sys

NGRAM = 13          # GPT-3/PaLM decontamination standard for code/math
NGRAM_SHORT = 8     # short German Q/A items rarely span 13 words
# \w-based so German umlauts stay inside words (the old [A-Za-z_] split them).
_WORD = re.compile(r"[^\W\d]+|\d+|[^\s\w]", re.UNICODE)
_COMMENT = re.compile(r"#[^\n]*")


def words(text: str) -> list[str]:
    return _WORD.findall(_COMMENT.sub(" ", text.lower()))


def _informative(win: list[str], n: int) -> bool:
    """Drop windows dominated by digits/punctuation/1-char tokens — boilerplate
    like '$ x ^ 2 + 4 = 0' matches everything and yields false positives."""
    need = 6 if n <= 9 else 8
    return sum(len(w) > 1 and w[0].isalpha() for w in win) >= need


def ngrams(ws: list[str], n: int):
    return {hash(tuple(ws[i:i + n])) for i in range(len(ws) - n + 1)
            if _informative(ws[i:i + n], n)}


def _join(d: dict, keys: list[str]) -> str:
    parts = []
    for k in keys:
        v = d.get(k)
        if v is None:
            continue
        if isinstance(v, list):
            parts.extend(str(x) for x in v)
        else:
            parts.append(str(v))
    return "\n".join(p for p in parts if p)


# eval-set registry: cli flag -> (fields to join, default n-gram size)
EVAL_SETS = {
    "humaneval":  (["prompt", "canonical_solution"], NGRAM),
    "mbpp":       (["text", "code"], NGRAM),
    "gsm8k":      (["question", "answer"], NGRAM),
    "germanquad": (["question", "answers"], NGRAM_SHORT),
    "mmlu_de":    (["instruction", "option_a", "option_b", "option_c", "option_d"], NGRAM_SHORT),
}


def load_eval(path: pathlib.Path, fields: list[str]) -> list[str]:
    op = gzip.open if path.suffix == ".gz" else open
    probs = []
    with op(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                probs.append(_join(json.loads(line), fields))
    return probs


def build_banks(args) -> dict[str, tuple[set[int], int, int, str]]:
    """name -> (bank, n, n_problems, first_problem_text)"""
    banks = {}
    for name, (fields, n_default) in EVAL_SETS.items():
        path = getattr(args, name)
        if not path:
            continue
        n = args.ngram or n_default
        probs = load_eval(path, fields)
        bank: set[int] = set()
        for t in probs:
            bank |= ngrams(words(t), n)
        banks[name] = (bank, n, len(probs), probs[0] if probs else "")
        print(f"bank {name}: {len(probs)} problems, {len(bank):,} {n}-grams", flush=True)
    return banks


def doc_hits(txt: str, banks) -> list[str]:
    ws = words(txt)
    out = []
    for name, (bank, n, _, _) in banks.items():
        if any(hash(tuple(ws[i:i + n])) in bank for i in range(len(ws) - n + 1)):
            out.append(name)
    return out


def self_test(banks) -> bool:
    """Positive control: a doc embedding eval problem #0 must be detected."""
    ok = True
    for name, (_, _, _, first) in banks.items():
        planted = "random preamble text before the leak\n" + first + "\ntrailing filler"
        hit = name in doc_hits(planted, banks)
        print(f"self-test {name}: planted item {'DETECTED' if hit else 'MISSED'}", flush=True)
        ok &= hit
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", default="", help="comma-separated JSONLs with {'text': ...}")
    for name in EVAL_SETS:
        ap.add_argument(f"--{name.replace('_', '-')}", type=pathlib.Path, default=None)
    ap.add_argument("--ngram", type=int, default=0, help="override per-set n-gram size")
    ap.add_argument("--sample", type=int, default=0, help="check at most N docs per input")
    ap.add_argument("--emit-clean", action="store_true")
    ap.add_argument("--dump-hits", type=int, default=3)
    ap.add_argument("--json-out", type=pathlib.Path, default=None)
    ap.add_argument("--self-test", action="store_true", help="positive control then continue")
    a = ap.parse_args()

    banks = build_banks(a)
    if not banks:
        sys.exit("need at least one eval set (--humaneval/--mbpp/--gsm8k/--germanquad/--mmlu-de)")
    if a.self_test and not self_test(banks):
        print("SELF-TEST FAILED: planted eval item not detected", flush=True)
        return 1

    report = {"inputs": {}, "self_test": "pass" if a.self_test else "skipped"}
    tot_docs = tot_hits = 0
    for inp in [p for p in a.inputs.split(",") if p]:
        inp = pathlib.Path(inp)
        out = open(inp.with_suffix(".decontam.jsonl"), "w", encoding="utf-8") if a.emit_clean else None
        ndoc = nhit = shown = 0
        per_set = {k: 0 for k in banks}
        for line in open(inp, encoding="utf-8"):
            if not line.strip():
                continue
            if a.sample and ndoc >= a.sample:
                break
            ndoc += 1
            try:
                txt = json.loads(line)["text"]
            except Exception:
                continue
            hits = doc_hits(txt, banks)
            if hits:
                nhit += 1
                for h in hits:
                    per_set[h] += 1
                if shown < a.dump_hits:
                    shown += 1
                    print(f"  HIT[{','.join(hits)}] {inp.name}#{ndoc}: {txt[:110]!r}", flush=True)
            elif out:
                out.write(line)
        if out:
            out.close()
        tot_docs += ndoc; tot_hits += nhit
        rate = nhit / max(ndoc, 1)
        per_rate = {k: v / max(ndoc, 1) for k, v in per_set.items()}
        report["inputs"][inp.name] = {"docs": ndoc, "hits": nhit, "rate": rate, "per_set": per_rate}
        print(f"{inp.name}: {nhit}/{ndoc} contaminated ({100*rate:.3f}%) "
              + " ".join(f"{k}={100*v:.2f}%" for k, v in per_rate.items()), flush=True)
    if tot_docs:
        print(f"=== total {tot_hits}/{tot_docs} ({100*tot_hits/max(tot_docs,1):.3f}%) ===", flush=True)
    if a.json_out:
        a.json_out.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
