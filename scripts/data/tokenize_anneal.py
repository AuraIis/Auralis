#!/usr/bin/env python3
"""Tokenize jsonl ({"text":...}) sources -> .bin (uint32) + .idx (int64 [offset,len])
for continued-pretraining/annealing. Preserves newlines (jsonl) unlike the line-per-doc
phase-1 tokenizer. Same on-disk format as tokenized/curated_40b."""
import sys, json, argparse, pathlib, numpy as np
import sentencepiece as spm
REPO = pathlib.Path("/workspace/v2data")


def tok_one(name, jsonl, out_dir, sp, eos, max_len):
    bin_p = out_dir / f"{name}.bin"; idx_p = out_dir / f"{name}.idx"
    ntok = ndoc = off = 0
    with open(bin_p, "wb") as fb, open(idx_p, "wb") as fi:
        buf = []
        def flush():
            nonlocal off, ntok, ndoc
            if not buf: return
            for ids in sp.encode(buf, out_type=int):
                ids = ids[:max_len]
                if not ids: continue
                ids.append(eos)
                np.asarray(ids, dtype=np.uint32).tofile(fb)
                np.asarray([off, len(ids)], dtype=np.int64).tofile(fi)
                off += len(ids); ntok += len(ids); ndoc += 1
            buf.clear()
        for line in open(jsonl, encoding="utf-8"):
            if not line.strip(): continue
            try: buf.append(json.loads(line)["text"])
            except Exception: continue
            if len(buf) >= 512: flush()
        flush()
    print(f"  {name}: {ndoc} docs, {ntok} tokens -> {bin_p}", flush=True)
    return ntok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sources", required=True, help="name=jsonl,name=jsonl,...")
    ap.add_argument("--tokenizer", default=str(REPO / "tokenizer/helix_v2_tokenizer.model"))
    ap.add_argument("--max-len", type=int, default=2048)
    a = ap.parse_args()
    sp = spm.SentencePieceProcessor(model_file=a.tokenizer)
    eos = sp.eos_id()
    assert eos is not None and eos >= 0, "tokenizer needs a valid eos"
    out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    tot = 0
    for pair in a.sources.split(","):
        name, jsonl = pair.split("=", 1)
        tot += tok_one(name, jsonl, out, sp, eos, a.max_len)
    print(f"=== total {tot} tokens -> {out} ===", flush=True)


if __name__ == "__main__":
    main()
