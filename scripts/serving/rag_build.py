#!/usr/bin/env python3
"""helix-rag v0 index builder: stream German Wikipedia, take title + intro, build a SQLite FTS5 (BM25) index.
Zero extra deps (sqlite3 stdlib). LIMIT=0 -> full. Smoke first with small LIMIT to validate + time it."""

import os
import sqlite3
import time

from datasets import load_dataset

DB = "/workspace/v2data/rag/dewiki.fts5.db"
LIMIT = int(os.environ.get("LIMIT", "8000"))
os.makedirs(os.path.dirname(DB), exist_ok=True)
con = sqlite3.connect(DB)
con.execute("PRAGMA journal_mode=WAL")
con.execute("DROP TABLE IF EXISTS docs")
con.execute(
    "CREATE VIRTUAL TABLE docs USING fts5(title, intro, tokenize='unicode61 remove_diacritics 2')"
)
print(f"streaming wikimedia/wikipedia 20231101.de  (LIMIT={LIMIT or 'full'}) ...", flush=True)
ds = load_dataset("wikimedia/wikipedia", "20231101.de", split="train", streaming=True)
t0 = time.time()
n = 0
skipped = 0
buf = []
for a in ds:
    title = (a.get("title") or "").strip()
    text = (a.get("text") or "").strip()
    if not title or not text:
        skipped += 1
        continue
    intro = " ".join(text.split("\n\n")[:3])[:1200].strip()  # first ~3 paragraphs, capped
    if len(intro) < 60:
        skipped += 1
        continue
    buf.append((title, intro))
    n += 1
    if len(buf) >= 2000:
        con.executemany("INSERT INTO docs(title,intro) VALUES(?,?)", buf)
        con.commit()
        buf = []
        if n % 20000 == 0:
            print(
                f"  indexed {n} | {n / (time.time() - t0):.0f}/s | {time.time() - t0:.0f}s",
                flush=True,
            )
    if LIMIT and n >= LIMIT:
        break
if buf:
    con.executemany("INSERT INTO docs(title,intro) VALUES(?,?)", buf)
    con.commit()
print(
    f"DONE: indexed {n}, skipped {skipped}, {time.time() - t0:.0f}s, rate {n / max(1, time.time() - t0):.0f}/s",
    flush=True,
)
# quick self-test query
import re


def terms(q):
    STOP = {
        "was",
        "ist",
        "eine",
        "ein",
        "der",
        "die",
        "das",
        "wer",
        "wie",
        "wo",
        "von",
        "und",
        "mir",
        "ueber",
        "kannst",
    }
    ks = [w for w in re.findall(r"[\wäöüÄÖÜß]+", q.lower()) if w not in STOP and len(w) > 1]
    return " OR ".join(ks)


for q in ["Katze", "Paris", "Hardware", "Moxthal"]:
    t = terms(q)
    try:
        r = con.execute(
            "SELECT title, bm25(docs) FROM docs WHERE docs MATCH ? ORDER BY bm25(docs) LIMIT 3",
            (t,),
        ).fetchall()
        print(f"  q={q!r:12} -> {[x[0] for x in r]}", flush=True)
    except Exception as e:
        print(f"  q={q!r} ERR {e}", flush=True)
con.close()
os._exit(0)  # skip interpreter finalization (avoids a GIL crash in the datasets streaming thread)
