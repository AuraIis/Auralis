#!/usr/bin/env python3
"""Convert corpora into JSONL ({"text": ...}) for tokenize_anneal.py (EOS-per-DOCUMENT,
newlines preserved). This fixes the phase-1 mistake where code was tokenized line-per-doc
(EOS after every line -> code seen as disconnected ~30-token fragments).

Two modes:
  --mode blocks : split a starcoder dump on <|code|>...<|endcode|>; ONE json record per
                  FILE (markers kept -> language/filename context survives, internal
                  newlines preserved).
  --mode lines  : one json record per non-empty line (for already line-per-doc sources
                  like openmath / edu-filtered prose).

    python scripts/data/blocks_to_jsonl.py --mode blocks \
        --input cleaned/code/starcoder_multi.edu.txt --output data/fresh/code_multi.jsonl
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

BEGIN = "<|code|>"
END = "<|endcode|>"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--mode", choices=["blocks", "lines"], required=True)
    ap.add_argument("--min-chars", type=int, default=1)
    a = ap.parse_args()

    a.output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with a.input.open("r", encoding="utf-8", errors="replace") as fin, \
         a.output.open("w", encoding="utf-8") as fout:
        if a.mode == "lines":
            for line in fin:
                t = line.rstrip("\n")
                if len(t) >= a.min_chars:
                    fout.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
                    n += 1
        else:  # blocks
            buf: list[str] = []
            in_block = False
            for line in fin:
                s = line.rstrip("\n")
                if s.startswith(BEGIN):
                    if in_block and buf:
                        txt = "\n".join(buf)
                        if len(txt) >= a.min_chars:
                            fout.write(json.dumps({"text": txt}, ensure_ascii=False) + "\n")
                            n += 1
                    buf = [s]
                    in_block = True
                elif in_block:
                    buf.append(s)
                    if s.strip() == END:
                        txt = "\n".join(buf)
                        if len(txt) >= a.min_chars:
                            fout.write(json.dumps({"text": txt}, ensure_ascii=False) + "\n")
                            n += 1
                        buf = []
                        in_block = False
            if in_block and buf:
                txt = "\n".join(buf)
                if len(txt) >= a.min_chars:
                    fout.write(json.dumps({"text": txt}, ensure_ascii=False) + "\n")
                    n += 1
    print(f"wrote {n:,} records -> {a.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
