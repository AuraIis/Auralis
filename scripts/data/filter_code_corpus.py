#!/usr/bin/env python3
"""Filter a raw code jsonl ({"text": code}) for annealing: length filter +
py_compile gate (only keep code that PARSES under Python 3 = "executor=truth";
also drops Python-2-only files) + exact dedup (sha256). Key-free, deterministic."""
import json, hashlib, argparse, pathlib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-chars", type=int, default=120)
    ap.add_argument("--max-chars", type=int, default=40000)
    a = ap.parse_args()
    seen = set()
    n = kept = drop_len = drop_compile = drop_dup = 0
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as out:
        for line in open(a.inp, encoding="utf-8"):
            if not line.strip():
                continue
            n += 1
            try:
                code = json.loads(line)["text"]
            except Exception:
                continue
            if not (a.min_chars <= len(code) <= a.max_chars):
                drop_len += 1; continue
            try:
                compile(code, "<f>", "exec")
            except Exception:
                drop_compile += 1; continue
            h = hashlib.sha256(code.encode("utf-8", "ignore")).hexdigest()
            if h in seen:
                drop_dup += 1; continue
            seen.add(h)
            out.write(json.dumps({"text": code}, ensure_ascii=False) + "\n"); kept += 1
    print(f"in {n} | KEPT {kept} ({100*kept/max(1,n):.0f}%) | drop: len={drop_len} compile={drop_compile} dup={drop_dup}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
