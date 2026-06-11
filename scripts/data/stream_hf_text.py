#!/usr/bin/env python3
"""Stream a HuggingFace dataset config to a line-per-doc text file, up to a token budget.

Built for acquiring fresh, CLEAN pretraining text (FineWeb2-DE, Stack-Edu) without
holding the whole set in memory or on disk. Streams shard-by-shard, collapses each
document to ONE line (so downstream line-per-doc tooling — clean/edu-filter/tokenize —
works unchanged), and stops once the approximate token budget is hit.

RESUMABLE: on restart, counts the lines already in --output and `.skip()`s that many
documents, then appends. (skip re-streams internally, so a late-crash resume is slow
but correct.) Writes a .progress.json sidecar every --progress-every kept docs.

Token budget is APPROXIMATE (bytes / bytes-per-token) — fast, no tokenizer needed.
The real token count comes later from the tokenizer manifest. DE~5.5, code~2.85 B/tok.

Example:
  python scripts/data/stream_hf_text.py \
     --repo HuggingFaceFW/fineweb-2 --config deu_Latn \
     --output data/fresh/fineweb2_de_fresh.txt \
     --max-tokens 16e9 --bytes-per-token 5.54 --min-chars 200
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path


def count_lines(p: Path) -> int:
    if not p.exists():
        return 0
    n = 0
    with p.open("rb") as f:
        for _ in f:
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-tokens", type=float, required=True, help="approx token budget (e.g. 16e9)")
    ap.add_argument("--bytes-per-token", type=float, default=5.0)
    ap.add_argument("--text-column", default="", help="auto-detect if empty (text/content/code)")
    ap.add_argument("--jsonl", action="store_true",
                    help="write JSONL {'text':...} preserving NEWLINES (for code); default collapses to one line (prose)")
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--score-column", default="", help="optional edu/quality column to threshold on")
    ap.add_argument("--score-min", type=float, default=None)
    ap.add_argument("--progress-every", type=int, default=20000)
    a = ap.parse_args()

    from datasets import load_dataset

    a.output.parent.mkdir(parents=True, exist_ok=True)
    already = count_lines(a.output)
    print(f"[stream] resume: {already} docs already in {a.output}", flush=True)

    def fresh_stream():
        return load_dataset(a.repo, a.config, split=a.split, streaming=True)

    ds = fresh_stream()
    if not a.text_column:
        first = next(iter(ds))
        for c in ("text", "content", "code"):
            if c in first:
                a.text_column = c
                break
        if not a.text_column:
            raise SystemExit(f"no text column; keys={list(first.keys())}")
        print(f"[stream] text column = '{a.text_column}'  (example keys: {list(first.keys())})", flush=True)
        ds = fresh_stream()  # restart iterator after the peek

    if already:
        print(f"[stream] skipping {already} already-written docs (slow, re-streams)...", flush=True)
        ds = ds.skip(already)

    bytes_written = a.output.stat().st_size if a.output.exists() else 0
    budget_bytes = a.max_tokens * a.bytes_per_token
    seen = already
    kept = already
    t0 = time.monotonic()
    prog = a.output.with_suffix(a.output.suffix + ".progress.json")

    with a.output.open("a" if already else "w", encoding="utf-8") as out:
        for ex in ds:
            seen += 1
            if a.score_column and a.score_min is not None:
                sc = ex.get(a.score_column)
                if sc is None or float(sc) < a.score_min:
                    continue
            txt = ex.get(a.text_column) or ""
            if a.jsonl:
                txt = txt.strip("\n")            # preserve INTERNAL newlines (code)
            else:
                txt = " ".join(txt.split())      # collapse to ONE line (prose)
            if len(txt) < a.min_chars:
                continue
            out.write((json.dumps({"text": txt}, ensure_ascii=False) if a.jsonl else txt) + "\n")
            kept += 1
            bytes_written += len(txt.encode("utf-8")) + 1
            if kept % a.progress_every == 0:
                out.flush()
                approx_tok = bytes_written / a.bytes_per_token
                rate = (kept - already) / max(1e-9, time.monotonic() - t0)
                print(f"  seen {seen:,} | kept {kept:,} | ~{approx_tok/1e9:.2f}B tok "
                      f"| {rate:.0f} doc/s | {bytes_written/1e9:.1f} GB", flush=True)
                prog.write_text(json.dumps({"seen": seen, "kept": kept,
                                            "approx_tokens": approx_tok, "bytes": bytes_written}))
            if bytes_written >= budget_bytes:
                print("[stream] token budget reached", flush=True)
                break

    approx_tok = bytes_written / a.bytes_per_token
    prog.write_text(json.dumps({"seen": seen, "kept": kept, "approx_tokens": approx_tok,
                                "bytes": bytes_written, "done": True}))
    print(f"[stream] DONE seen {seen:,} kept {kept:,} ~{approx_tok/1e9:.2f}B tok {bytes_written/1e9:.1f} GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
