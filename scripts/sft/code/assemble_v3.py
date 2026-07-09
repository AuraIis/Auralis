#!/usr/bin/env python3
"""Code-SFT v3 mix: curated (rendered, executor-verified) + verified (rendered from fields),
oversampled x2, + corrective chat anchor + abstain. Dumps train func-names for gate seen/unseen."""

import json
import os
import random
from collections import Counter

random.seed(50)
REPO = "/workspace/v2data"
D = REPO + "/data/training"
OUT = D + "/sft_code_v3"
os.makedirs(OUT, exist_ok=True)
SYS = "Du bist Auralis, ein hilfreicher, ehrlicher KI-Assistent."
END = "<|end|>"


def render(u, a):
    return f"<|system|>\n{SYS}\n{END}\n<|user|>\n{u}\n{END}\n<|assistant|>\n{a}\n{END}\n"


def jl(p):
    out = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


rows = []
funcs = set()
cur = jl(D + "/code_curated_v1/verified_code.jsonl")
for _ in range(2):
    for r in cur:
        if r.get("text"):
            rows.append({"text": r["text"], "src": "code_curated"})
for r in cur:
    m = r.get("meta") or {}
    if isinstance(m, dict) and m.get("funktion"):
        funcs.add(m["funktion"])
ver = jl(
    D + "/code_verified_v1/verified_code.jsonl"
)  # filtered set is already rendered (text field)
for _ in range(2):
    for r in ver:
        if r.get("text"):
            rows.append({"text": r["text"], "src": "code_verified"})
for r in ver:
    m = r.get("meta") or {}
    if isinstance(m, dict) and m.get("funktion"):
        funcs.add(m["funktion"])
ncode = len(rows)
corr = jl(D + "/sft_corrective/train.helix.jsonl")
random.shuffle(corr)
for r in corr[:900]:
    rows.append({"text": r["text"], "src": "corrective"})
ab = jl(D + "/calib_verified_v1/verified_calib.jsonl")
random.shuffle(ab)
for r in ab[:100]:
    rows.append({"text": r["text"], "src": "abstain"})

random.shuffle(rows)
nval = 250
val, train = rows[:nval], rows[nval:]
with open(OUT + "/train.helix.jsonl", "w", encoding="utf-8") as f:
    for r in train:
        f.write(json.dumps({"text": r["text"]}, ensure_ascii=False) + "\n")
with open(OUT + "/val.helix.jsonl", "w", encoding="utf-8") as f:
    for r in val:
        f.write(json.dumps({"text": r["text"]}, ensure_ascii=False) + "\n")
json.dump(
    sorted(funcs),
    open(REPO + "/diag/code_train_funcs.json", "w", encoding="utf-8"),
    ensure_ascii=False,
)
c = Counter(r["src"] for r in rows)
tot = len(rows)
print("=== code SFT v3 mix ===", dict(c))
print(
    f"code {ncode} ({100 * ncode // tot}%) | total {tot} | train {len(train)} | val {len(val)} | train_funcs {len(funcs)}"
)
