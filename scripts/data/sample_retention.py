#!/usr/bin/env python3
"""Sample broad doc-level retention text from large cleaned .txt corpora (one doc/line).

Used to build a BIGGER, BROADER retention set matching the foundation distribution
(iter-2 anneal): the v2 anneal proved that over-weighting a tiny 2.4M DE set caused
OVERFITTING that hurt held-out DE (+97%). Fix = more breadth from the SAME source.

Random byte-seek sampling: seek to a random offset in a (size-weighted) source file,
read the next full line as a document, dedupe by hash, until a token budget is hit.
Fast (no full read of multi-GB files) and distribution-faithful.
"""
import os, sys, json, random, hashlib, argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", required=True, help="comma-separated .txt paths (one doc per line)")
    ap.add_argument("--out", required=True, help="output jsonl ({'text':...})")
    ap.add_argument("--target-tokens", type=float, required=True, help="approx target tokens (M)")
    ap.add_argument("--bytes-per-token", type=float, default=5.4, help="DE~5.4, EN~4.8")
    ap.add_argument("--min-chars", type=int, default=250)
    ap.add_argument("--max-chars", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260608)
    a = ap.parse_args()

    files = [f for f in a.sources.split(",") if f.strip() and os.path.exists(f.strip())]
    if not files:
        print("ERROR: no existing source files", file=sys.stderr); sys.exit(1)
    sizes = {f: os.path.getsize(f) for f in files}
    total = sum(sizes.values())
    target_bytes = a.target_tokens * 1e6 * a.bytes_per_token
    rng = random.Random(a.seed)
    handles = {f: open(f, "rb") for f in files}

    seen = set()
    got_bytes = 0
    kept = 0
    tries = 0
    with open(a.out, "w", encoding="utf-8") as out:
        while got_bytes < target_bytes and tries < a.target_tokens * 1e6 / 50:
            tries += 1
            r = rng.random() * total
            acc = 0; pick = files[0]
            for f in files:
                acc += sizes[f]
                if r <= acc:
                    pick = f; break
            h = handles[pick]
            h.seek(rng.randint(0, max(0, sizes[pick] - 2)))
            h.readline()                 # discard partial line
            line = h.readline()
            if not line:
                continue
            try:
                text = line.decode("utf-8").strip()
            except UnicodeDecodeError:
                continue
            if not (a.min_chars <= len(text) <= a.max_chars):
                continue
            hh = hashlib.sha1(text.encode("utf-8")).hexdigest()
            if hh in seen:
                continue
            seen.add(hh)
            out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            got_bytes += len(line)
            kept += 1
            if kept % 5000 == 0:
                print(f"  kept {kept} docs, ~{got_bytes/1e6:.1f} MB (~{got_bytes/a.bytes_per_token/1e6:.1f}M tok)", flush=True)

    est_tok = got_bytes / a.bytes_per_token
    print(f"=== sampled {kept} docs | {got_bytes/1e6:.1f} MB | ~{est_tok/1e6:.1f}M tokens -> {a.out} ===")
    by = {}
    for f in files:
        by[os.path.basename(f)] = round(sizes[f] / total, 2)
    print(f"    source weights (by size): {by}")


if __name__ == "__main__":
    main()
