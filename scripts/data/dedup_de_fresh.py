#!/usr/bin/env python3
"""Dedup fresh German (de_fresh.jsonl) against the curated German reference (de_curated source
text), per Codex audit P2: both come from FineWeb2-DE (different crawls) -> duplicate / near-
duplicate docs -> over-repetition + possible val leakage.

Two-tier: (1) exact dedup on normalized SHA1, (2) near-dup via MinHash-LSH (word 5-shingles,
num_perm=64, Jaccard threshold 0.85). Builds the LSH over the (smaller) reference, then streams
de_fresh and drops any doc that exact- or near-matches the reference. Writes a deduped jsonl.

    python scripts/data/dedup_de_fresh.py \
        --fresh data/fresh/de_fresh.jsonl \
        --ref cleaned/edu/fineweb2_de.edu.txt cleaned/edu/fineweb2_de_v2.edu.txt cleaned/wikipedia_de.filtered.txt \
        --out data/fresh/de_fresh.dedup.jsonl
"""
from __future__ import annotations
import argparse, json, re, hashlib, time
from pathlib import Path
from datasketch import MinHash, MinHashLSH

NUM_PERM = 64
THRESH = 0.85
K = 5


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower()).strip()


def shingle_bytes(t: str):
    w = norm(t).split()
    if len(w) < K:
        s = norm(t)
        return [s.encode("utf-8")] if s else []
    return [" ".join(w[i:i + K]).encode("utf-8") for i in range(len(w) - K + 1)]


def mh(t: str) -> MinHash:
    m = MinHash(num_perm=NUM_PERM)
    sh = shingle_bytes(t)
    if sh:
        m.update_batch(sh)
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fresh", required=True, type=Path, help="de_fresh.jsonl ({'text':...})")
    ap.add_argument("--ref", required=True, nargs="+", type=Path, help="reference text files (line-per-doc)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-chars", type=int, default=200)
    a = ap.parse_args()

    lsh = MinHashLSH(threshold=THRESH, num_perm=NUM_PERM)
    exact: set[str] = set()
    t0 = time.monotonic()
    ri = 0
    for f in a.ref:
        for line in f.open(encoding="utf-8", errors="replace"):
            line = line.strip()
            if len(line) < a.min_chars:
                continue
            exact.add(hashlib.sha1(norm(line).encode("utf-8")).hexdigest())
            lsh.insert(f"r{ri}", mh(line))
            ri += 1
            if ri % 200000 == 0:
                print(f"  ref indexed {ri:,} ({ri/max(1e-9,time.monotonic()-t0):.0f}/s)", flush=True)
    print(f"[dedup] reference: {ri:,} docs indexed in {time.monotonic()-t0:.0f}s", flush=True)

    kept = d_exact = d_near = seen = 0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    t1 = time.monotonic()
    with a.fresh.open(encoding="utf-8") as fin, a.out.open("w", encoding="utf-8") as out:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            seen += 1
            try:
                t = json.loads(line)["text"]
            except Exception:
                continue
            h = hashlib.sha1(norm(t).encode("utf-8")).hexdigest()
            if h in exact:
                d_exact += 1; continue
            if lsh.query(mh(t)):
                d_near += 1; continue
            out.write(line + "\n")
            kept += 1
            if seen % 200000 == 0:
                print(f"  fresh {seen:,} | kept {kept:,} | exact-dup {d_exact:,} | near-dup {d_near:,} "
                      f"| {seen/max(1e-9,time.monotonic()-t1):.0f}/s", flush=True)

    rep = {"fresh_seen": seen, "kept": kept, "dropped_exact": d_exact, "dropped_near": d_near,
           "ref_docs": ri, "drop_pct": round(100 * (d_exact + d_near) / max(1, seen), 3)}
    a.out.with_suffix(".dedup_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"[dedup] DONE {json.dumps(rep)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
