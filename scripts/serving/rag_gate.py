#!/usr/bin/env python3
"""helix-rag v0 gate with SEPARATED metrics so we know retriever-vs-reader:
  retrieval_hit : right article in top-k? (checked directly against the DB)
  reader_hit    : does grounded read the answer from that context? (via the shim)
  safe_abstain  : fake term -> says nothing?
  bad_answer    : invents a non-abstain answer on a fake term?
Run after rag_finalize.py (titlemap) + shim up."""

import json
import re
import sqlite3
import urllib.request

DB = "/workspace/v2data/rag/dewiki.fts5.db"
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
RAG_STOP = {
    "was",
    "ist",
    "eine",
    "ein",
    "einen",
    "der",
    "die",
    "das",
    "wer",
    "wie",
    "wo",
    "wann",
    "warum",
    "welche",
    "welcher",
    "welches",
    "mir",
    "mich",
    "du",
    "kannst",
    "ueber",
    "über",
    "sagen",
    "erklaere",
    "erkläre",
    "kurz",
    "nenne",
    "sind",
    "den",
    "dem",
    "des",
    "von",
    "im",
    "in",
    "zu",
    "und",
    "mit",
    "auf",
    "fuer",
    "für",
    "etwas",
    "mal",
    "bitte",
    "hauptstadt",
}
ACR = {"gpu", "cpu", "ram", "vram", "ai", "ki", "llm", "api"}
_BAD = (
    "steht für",
    "begriffsklärung",
    "ist der familienname",
    "ist der name mehrerer",
    "ist der name folgender",
    "ist der name von",
    "bezeichnet:",
    "ist eine ortsbezeichnung",
    "kann sich beziehen",
    "ist eine liste",
    "ist eine begriffsklärung",
    "bezeichnet mehrere",
)


def good(ti, intro):
    if "Begriffsklärung" in ti or "(Familienname)" in ti:
        return False
    h = intro[:130].lower()
    return not any(b in h for b in _BAD)


def tcands(q):
    ws = [w for w in re.findall(r"[\wäöüÄÖÜß]+", q) if w.lower() not in RAG_STOP and len(w) > 1]
    c = [(w.upper() if w.lower() in ACR else w[0].upper() + w[1:]) for w in ws]
    if len(ws) > 1:
        c.append(" ".join((w.upper() if w.lower() in ACR else w[0].upper() + w[1:]) for w in ws))
    return c


def terms(q):
    return " OR ".join(
        w for w in re.findall(r"[\wäöüÄÖÜß]+", q.lower()) if w not in RAG_STOP and len(w) > 1
    )


def retrieve(q, k=3):
    out = []
    seen = set()
    for term in tcands(q):
        tgt = term
        try:
            a = con.execute(
                "SELECT target FROM aliasmap WHERE a=? LIMIT 1", (term.lower(),)
            ).fetchone()
            if a:
                tgt = a[0]
        except Exception:
            pass
        try:
            r = con.execute(
                "SELECT t,intro FROM titlemap WHERE t=? COLLATE NOCASE LIMIT 1", (tgt,)
            ).fetchone()
        except Exception:
            r = None
        if r and good(r[0], r[1]) and r[0].lower() not in seen:
            out.append((r[0], r[1]))
            seen.add(r[0].lower())
    if out:
        return out[:k]
    tt = terms(q)
    if tt:
        try:
            rows = con.execute(
                "SELECT title,intro FROM docs WHERE docs MATCH ? ORDER BY bm25(docs) LIMIT ?",
                (tt, k * 15),
            ).fetchall()
        except Exception:
            rows = []
        for ti, i in rows:
            if good(ti, i) and len(i) >= 80 and ti.lower() not in seen:
                out.append((ti, i))
                seen.add(ti.lower())
    return out[:k]


def ask(prompt):
    data = json.dumps(
        {"model": "helix-rag", "messages": [{"role": "user", "content": prompt}]}
    ).encode()
    req = urllib.request.Request(
        "http://localhost:11434/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    buf = ""
    with urllib.request.urlopen(req, timeout=90) as r:
        for line in r:
            line = line.decode("utf-8", "replace").strip()
            if line.startswith("data: "):
                d = line[6:].strip()
                if d == "[DONE]":
                    break
                try:
                    buf += json.loads(d)["choices"][0]["delta"].get("content", "")
                except Exception:
                    pass
    return buf


def abstains(t):
    tl = t.lower()
    return (
        "weiss es nicht" in tl
        or "weiß es nicht" in tl
        or "keinen passenden" in tl
        or "nicht hervor" in tl
        or "steht nicht im text" in tl
    )


# (query, expect_in_retrieved, expect_in_answer, is_fake)
CASES = [
    ("Was ist eine Katze?", ("katze", "säuget", "tier"), ("katze", "tier", "säuget"), False),
    ("Was ist die Hauptstadt von Frankreich?", ("paris", "frankreich"), ("paris",), False),
    ("Was ist Hardware?", ("hardware",), ("hardware", "gerät", "computer"), False),
    ("Was ist eine GPU?", ("gpu", "grafik"), ("grafik", "gpu", "chip"), False),
    (
        "Was ist Photosynthese?",
        ("photosynth", "pflanz"),
        ("pflanz", "licht", "kohlen", "sauerstoff"),
        False,
    ),
    ("Was ist Berlin?", ("berlin", "hauptstadt"), ("berlin", "hauptstadt", "stadt"), False),
    ("Was ist Moxthal?", None, None, True),
    ("Was ist ein Glaztronk?", None, None, True),
]
rh = rdh = sa = ba = nreal = nfake = 0
for q, eret, eans, fake in CASES:
    docs = retrieve(q, 3)
    blob = " ".join((t + " " + i) for t, i in docs).lower()
    ans = ask(q)
    abst = abstains(ans)
    if fake:
        nfake += 1
        sa += abst
        ba += not abst
        print(
            f"[FAKE ] {q[:30]:30s} | abstain={abst} | retrieved={[d[0] for d in docs]} -> {ans[:60]!r}"
        )
    else:
        nreal += 1
        ret_ok = any(e in blob for e in eret)
        rh += ret_ok
        read_ok = (not abst) and any(e in ans.lower() for e in eans)
        rdh += read_ok
        print(
            f"[REAL ] {q[:30]:30s} | retr={ret_ok} read={read_ok} | top={[d[0] for d in docs][:2]}\n        -> {ans[:80]!r}"
        )
print("\n=== RAG GATE (separated) ===")
print(f"  retrieval_hit : {rh}/{nreal}   reader_hit: {rdh}/{nreal}")
print(f"  safe_abstain  : {sa}/{nfake}   bad_answer: {ba}/{nfake}")
