#!/usr/bin/env python3
"""v0.1 redirect/alias table: download de-Wiki redirect.sql + page.sql, join to alias_title -> target_title,
keep only targets that exist in our titlemap, store aliasmap(a, target). Fixes Katze->Hauskatze etc."""

import os
import sqlite3
import time
import urllib.request

RAG = "/workspace/v2data/rag"
DB = f"{RAG}/dewiki.fts5.db"
PAGE = f"{RAG}/dewiki-latest-page.sql.gz"
REDIR = f"{RAG}/dewiki-latest-redirect.sql.gz"
BASE = "https://dumps.wikimedia.org/dewiki/latest/"


def dl(name, path):
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        print(f"  have {name} ({os.path.getsize(path) // 1024 // 1024} MB)", flush=True)
        return
    print(f"  downloading {name} ...", flush=True)
    urllib.request.urlretrieve(BASE + name, path)
    print(f"  -> {os.path.getsize(path) // 1024 // 1024} MB", flush=True)


dl("dewiki-latest-redirect.sql.gz", REDIR)
dl("dewiki-latest-page.sql.gz", PAGE)
from mwsql import Dump

t0 = time.time()
# 1) page_id -> title (underscores->spaces) for namespace-0 redirect pages
dp = Dump.from_file(PAGE)
cn = list(dp.dtypes.keys())
ci = {c: cn.index(c) for c in ("page_id", "page_namespace", "page_title", "page_is_redirect")}
id2t = {}
for r in dp.rows():
    if str(r[ci["page_namespace"]]) == "0" and str(r[ci["page_is_redirect"]]) == "1":
        id2t[str(r[ci["page_id"]])] = str(r[ci["page_title"]]).replace("_", " ")
print(f"redirect-pages: {len(id2t)}  ({time.time() - t0:.0f}s)", flush=True)
# 2) rd_from -> rd_title (target), join with id2t -> alias -> target
dr = Dump.from_file(REDIR)
cn = list(dr.dtypes.keys())
ri = {c: cn.index(c) for c in ("rd_from", "rd_namespace", "rd_title")}
alias2t = {}
for r in dr.rows():
    if str(r[ri["rd_namespace"]]) == "0":
        a = id2t.get(str(r[ri["rd_from"]]))
        if a:
            alias2t[a] = str(r[ri["rd_title"]]).replace("_", " ")
print(f"alias->target pairs: {len(alias2t)}  ({time.time() - t0:.0f}s)", flush=True)
# 3) keep only aliases whose TARGET exists in titlemap; store
con = sqlite3.connect(DB)
have = {row[0] for row in con.execute("SELECT t FROM titlemap")}
print(f"titlemap titles: {len(have)}", flush=True)
con.execute("DROP TABLE IF EXISTS aliasmap")
con.execute("CREATE TABLE aliasmap (a TEXT, target TEXT)")
rows = [(a.lower(), tgt) for a, tgt in alias2t.items() if tgt in have and a.lower() != tgt.lower()]
con.executemany("INSERT INTO aliasmap(a, target) VALUES(?,?)", rows)
con.execute("CREATE INDEX ix_alias ON aliasmap(a)")
con.commit()
print(f"aliasmap: {len(rows)} usable aliases  ({time.time() - t0:.0f}s)", flush=True)
for term in ["katze", "hund", "gpu", "auto", "pferd", "usa"]:
    r = con.execute("SELECT target FROM aliasmap WHERE a=? LIMIT 1", (term,)).fetchone()
    print(f"  alias {term:8} -> {r[0] if r else None}", flush=True)
con.close()
