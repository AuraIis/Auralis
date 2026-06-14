#!/usr/bin/env python3
"""Code v4 mix: code_v4 (pattern-class, x3 weight) + curated x1 + verified x1 + corrective + abstain.
Dumps train func-names (curated+verified+v4) so the gate tags seen/unseen honestly."""
import json, os, random
from collections import Counter
random.seed(60)
REPO = "/workspace/v2data"; D = REPO + "/data/training"; OUT = D + "/sft_code_v4"; os.makedirs(OUT, exist_ok=True)
def jl(p):
    out = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            try: out.append(json.loads(line))
            except Exception: pass
    return out

rows = []; funcs = set()
v4 = jl(D + "/code_v4/verified_code.jsonl")
for _ in range(3):
    for r in v4:
        if r.get("text"): rows.append({"text": r["text"], "src": "code_v4"})
for r in v4:
    if r.get("func"): funcs.add(r["func"])
cur = jl(D + "/code_curated_v1/verified_code.jsonl")
for r in cur:
    if r.get("text"): rows.append({"text": r["text"], "src": "code_curated"})
    m = r.get("meta") or {}
    if isinstance(m, dict) and m.get("funktion"): funcs.add(m["funktion"])
ver = jl(D + "/code_verified_v1/verified_code.jsonl")
for r in ver:
    if r.get("text"): rows.append({"text": r["text"], "src": "code_verified"})
    m = r.get("meta") or {}
    if isinstance(m, dict) and m.get("funktion"): funcs.add(m["funktion"])
ncode = len(rows)
corr = jl(D + "/sft_corrective/train.helix.jsonl"); random.shuffle(corr)
for r in corr[:700]: rows.append({"text": r["text"], "src": "corrective"})
ab = jl(D + "/calib_verified_v1/verified_calib.jsonl"); random.shuffle(ab)
for r in ab[:80]: rows.append({"text": r["text"], "src": "abstain"})

random.shuffle(rows)
nval = 250; val, train = rows[:nval], rows[nval:]
with open(OUT + "/train.helix.jsonl", "w", encoding="utf-8") as f:
    for r in train: f.write(json.dumps({"text": r["text"]}, ensure_ascii=False) + "\n")
with open(OUT + "/val.helix.jsonl", "w", encoding="utf-8") as f:
    for r in val: f.write(json.dumps({"text": r["text"]}, ensure_ascii=False) + "\n")
json.dump(sorted(funcs), open(REPO + "/diag/code_train_funcs.json", "w", encoding="utf-8"), ensure_ascii=False)
c = Counter(r["src"] for r in rows); tot = len(rows)
print("=== code v4 mix ===", dict(c))
print(f"code {ncode} ({100*ncode//tot}%) | total {tot} | train {len(train)} | val {len(val)} | train_funcs {len(funcs)}")
