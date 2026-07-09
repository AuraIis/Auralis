#!/usr/bin/env python3
"""Tokenize jsonl ({"text":...}) sources -> .bin (uint32) + .idx (int64 [offset,len])
for continued-pretraining/annealing. Preserves newlines (jsonl) unlike the line-per-doc
phase-1 tokenizer. Same on-disk format as tokenized/curated_40b."""

import argparse
import json
import pathlib
import sys

import numpy as np
import sentencepiece as spm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from code_format import TAB_WIDTH, tab_indent  # noqa: F401  (shared transform; re-exported)

REPO = pathlib.Path("/workspace/v2data")


def tok_one(name, jsonl, out_dir, sp, eos, max_len, drop_overlong=False, tab=False):
    bin_p = out_dir / f"{name}.bin"
    idx_p = out_dir / f"{name}.idx"
    ntok = ndoc = off = ndrop = 0
    with open(bin_p, "wb") as fb, open(idx_p, "wb") as fi:
        buf = []

        def flush():
            nonlocal off, ntok, ndoc, ndrop
            if not buf:
                return
            for ids in sp.encode(buf, out_type=int):
                if drop_overlong and len(ids) > max_len:
                    # truncating mid-document would cut code into invalid syntax
                    # (unterminated strings) -> drop the whole doc instead.
                    ndrop += 1
                    continue
                ids = ids[:max_len]
                if not ids:
                    continue
                ids.append(eos)
                np.asarray(ids, dtype=np.uint32).tofile(fb)
                np.asarray([off, len(ids)], dtype=np.int64).tofile(fi)
                off += len(ids)
                ntok += len(ids)
                ndoc += 1
            buf.clear()

        for line in open(jsonl, encoding="utf-8"):
            if not line.strip():
                continue
            try:
                txt = json.loads(line)["text"]
            except Exception:
                continue
            buf.append(tab_indent(txt) if tab else txt)
            if len(buf) >= 512:
                flush()
        flush()
    print(f"  {name}: {ndoc} docs, {ntok} tokens, dropped_overlong={ndrop} -> {bin_p}", flush=True)
    return ntok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sources", required=True, help="name=jsonl,name=jsonl,...")
    ap.add_argument("--tokenizer", default=str(REPO / "tokenizer/helix_v2_tokenizer.model"))
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument(
        "--drop-overlong",
        action="store_true",
        help="skip docs longer than max-len instead of truncating (use for CODE: a "
        "mid-file cut yields invalid syntax / unterminated strings)",
    )
    ap.add_argument(
        "--tab-indent",
        action="store_true",
        help="convert leading 4-space indent runs to tabs before tokenizing (use for "
        "CODE only: ~17%% fewer tokens with this tokenizer; see tab_indent())",
    )
    a = ap.parse_args()
    sp = spm.SentencePieceProcessor(model_file=a.tokenizer)
    eos = sp.eos_id()
    assert eos is not None and eos >= 0, "tokenizer needs a valid eos"
    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tot = 0
    for pair in a.sources.split(","):
        name, jsonl = pair.split("=", 1)
        tot += tok_one(name, jsonl, out, sp, eos, a.max_len, a.drop_overlong, a.tab_indent)
    print(f"=== total {tot} tokens -> {out} ===", flush=True)


if __name__ == "__main__":
    main()
