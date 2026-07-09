#!/usr/bin/env python3
"""H-v4 mix: grounded_v4 (=v3 + dense-prose/count/begin-end buckets) x2
+ grounded_v2 x1 (world-traps) + grounded_raw x1 (prose) + corrective + tool + abstain."""

import json
import os
import random
from collections import Counter

random.seed(45)
REPO = "/workspace/v2data"
D = REPO + "/data/training"
OUT = D + "/sft_grounded_v4"
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


def gtext(it):
    return render(f"{it['context'].strip()}\n\nFrage: {it['q'].strip()}", it["a"].strip())


def add_grounded(path, n, tag):
    data = jl(path)
    k = 0
    for _ in range(n):
        for it in data:
            if it.get("context") and it.get("q") and it.get("a"):
                rows.append({"text": gtext(it), "src": tag})
                k += 1
    return k


rows = []
add_grounded(D + "/sft_grounded/grounded_v4.jsonl", 2, "grounded_v4")
add_grounded(D + "/sft_grounded/grounded_v2.jsonl", 1, "grounded_v2")
add_grounded(D + "/sft_grounded/grounded_raw.jsonl", 1, "grounded_v1")
ng = sum(1 for r in rows if r["src"].startswith("grounded"))
corr = jl(D + "/sft_corrective/train.helix.jsonl")
random.shuffle(corr)
for r in corr[:1500]:
    rows.append({"text": r["text"], "src": "corrective"})
tool = jl(D + "/sft_generated/math_tool_traces.jsonl")
random.shuffle(tool)
for r in tool[:300]:
    ms = r["messages"]
    rows.append({"text": render(ms[0]["content"], ms[1]["content"]), "src": "tool"})
ab = jl(D + "/calib_verified_v1/verified_calib.jsonl")
random.shuffle(ab)
for r in ab[:150]:
    rows.append({"text": r["text"], "src": "abstain"})

random.shuffle(rows)
nval = 300
val, train = rows[:nval], rows[nval:]
with open(OUT + "/train.helix.jsonl", "w", encoding="utf-8") as f:
    for r in train:
        f.write(json.dumps({"text": r["text"]}, ensure_ascii=False) + "\n")
with open(OUT + "/val.helix.jsonl", "w", encoding="utf-8") as f:
    for r in val:
        f.write(json.dumps({"text": r["text"]}, ensure_ascii=False) + "\n")
c = Counter(r["src"] for r in rows)
tot = len(rows)
print("=== H-v4 mix ===", dict(c))
print(f"grounded {ng} ({100 * ng // tot}%) | total {tot} | train {len(train)} | val {len(val)}")
