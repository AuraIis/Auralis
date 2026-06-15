#!/usr/bin/env python3
"""Post-build: create an indexed exact-title lookup so the hard title rule is O(log n).
Run after rag_build.py finishes (full FTS5 index)."""
import sqlite3, time
DB = "/workspace/v2data/rag/dewiki.fts5.db"
con = sqlite3.connect(DB); t0 = time.time()
con.execute("DROP TABLE IF EXISTS titlemap")
con.execute("CREATE TABLE titlemap AS SELECT title t, intro FROM docs")
con.execute("CREATE INDEX ix_titlemap ON titlemap(t COLLATE NOCASE)")
con.commit()
n = con.execute("SELECT count(*) FROM titlemap").fetchone()[0]
print(f"titlemap: {n} rows, {time.time()-t0:.0f}s", flush=True)
for term in ["Paris", "Katze", "Hardware", "GPU", "Photosynthese", "Berlin", "Hund"]:
    r = con.execute("SELECT t FROM titlemap WHERE t = ? COLLATE NOCASE LIMIT 1", (term,)).fetchone()
    print(f"  exact-title {term:14} -> {r[0] if r else None}", flush=True)
con.close()
