#!/usr/bin/env python3
"""Tokenizer fertility benchmark: Helix-200k vs. standard tokenizers.

Measures bytes-per-token on real German/English/code samples — a
tokenizer-independent, fair measure of how densely each tokenizer encodes text.
Fewer tokens for the same text => cheaper/faster inference per sentence and more
text per context window (conditional on equal model quality).

Code is measured both raw and with the training pipeline's --tab-indent
normalization (see scripts/data/code_format.py), since that is how the model
actually tokenizes code.

Usage:
    python scripts/eval/tokenizer_fertility.py \
        [--tokenizer tokenizer/helix_v2_tokenizer.model] \
        [--sample-dir diag/clean_audit_v1] \
        [--out diag/tokenizer_fertility.json]

Optional comparison tokenizers load only if available/reachable:
    tiktoken (o200k_base, cl100k_base) and Llama-3 via transformers.
"""
import argparse
import glob
import json
import os
import sys

TAB_WIDTH = 4
PER_FILE_BYTES = 3_000_000


def tab_indent(text, tab_width=TAB_WIDTH):
    """Convert leading space-runs to tabs (mirror of scripts/data/code_format.py)."""
    out = []
    for line in text.split("\n"):
        stripped = line.lstrip(" ")
        n = len(line) - len(stripped)
        if n >= tab_width:
            tabs, rem = divmod(n, tab_width)
            line = "\t" * tabs + " " * rem + line[n:]
        out.append(line)
    return "\n".join(out)


def load_sample(sample_dir, patterns):
    parts = []
    for pat in patterns:
        for fp in sorted(glob.glob(os.path.join(sample_dir, pat))):
            with open(fp, "rb") as f:
                parts.append(f.read(PER_FILE_BYTES).decode("utf-8", errors="ignore"))
    return "\n".join(parts)


def build_tokenizers(sp_path):
    toks = {}
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=sp_path)
    toks["Helix-200k"] = ("sp", sp, sp.get_piece_size())

    try:
        import tiktoken
        for name, enc_name in [("GPT4o-o200k", "o200k_base"), ("GPT4-cl100k", "cl100k_base")]:
            try:
                enc = tiktoken.get_encoding(enc_name)
                toks[name] = ("tk", enc, enc.n_vocab)
            except Exception as e:
                print(f"  {name} skip: {str(e)[:80]}", file=sys.stderr)
    except ImportError:
        print("  tiktoken not installed — skipping o200k/cl100k", file=sys.stderr)

    try:
        from transformers import AutoTokenizer
        t = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B")
        toks["Llama-3-128k"] = ("hf", t, t.vocab_size)
    except Exception as e:
        print(f"  Llama-3 skip: {str(e)[:80]}", file=sys.stderr)

    return toks


def count_tokens(kind, tok, text):
    if kind == "sp":
        return len(tok.encode(text, out_type=int))
    if kind == "tk":
        return len(tok.encode(text, disallowed_special=()))
    if kind == "hf":
        return len(tok.encode(text, add_special_tokens=False))
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="tokenizer/helix_v2_tokenizer.model")
    ap.add_argument("--sample-dir", default="diag/clean_audit_v1")
    ap.add_argument("--out", default="diag/tokenizer_fertility.json")
    a = ap.parse_args()

    samples = {
        "DE": load_sample(a.sample_dir, ["raw_gc.*txt", "raw_fineweb2_de.*txt", "raw_hplt_de.*txt"]),
        "EN": load_sample(a.sample_dir, ["raw_fineweb_en.*txt"]),
        "CODE-raw": load_sample(a.sample_dir, ["raw_starcoder.20k.txt"]),
    }
    samples["CODE-tab"] = tab_indent(samples["CODE-raw"])
    samples = {k: v for k, v in samples.items() if v}

    toks = build_tokenizers(a.tokenizer)
    print("tokenizers:", ", ".join(f"{n}({v:,})" for n, (_, _, v) in toks.items()))

    res = {}
    for lang, text in samples.items():
        nb = len(text.encode("utf-8"))
        res[lang] = {"bytes": nb, "toks": {}}
        print(f"\n=== {lang}: {nb / 1e6:.1f} MB")
        for name, (kind, tok, vocab) in toks.items():
            n = count_tokens(kind, tok, text)
            res[lang]["toks"][name] = {"vocab": vocab, "tokens": n, "bytes_per_tok": nb / n}
            print(f"  {name:14s} vocab={vocab:>7,} | tokens={n:>10,} | bytes/tok={nb / n:5.2f}")

    print("\n" + "=" * 60)
    print("VERDICT — Helix vs each (negative = Helix worse / more tokens)")
    print("=" * 60)
    for lang in res:
        base = res[lang]["toks"].get("Helix-200k")
        if not base:
            continue
        print(f"\n[{lang}] Helix bytes/tok={base['bytes_per_tok']:.2f}")
        for name, m in res[lang]["toks"].items():
            if name == "Helix-200k":
                continue
            adv = 100 * (m["tokens"] - base["tokens"]) / m["tokens"]
            word = "fewer" if adv > 0 else "MORE"
            print(f"   vs {name:14s}: Helix {abs(adv):5.1f}% {word} tokens")

    if "CODE-raw" in res and "CODE-tab" in res:
        hr = res["CODE-raw"]["toks"]["Helix-200k"]["tokens"]
        ht = res["CODE-tab"]["toks"]["Helix-200k"]["tokens"]
        print(f"\n[CODE] tab-indent effect on Helix: {hr:,} -> {ht:,} ({100 * (hr - ht) / hr:.1f}% reduction)")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
