#!/usr/bin/env python3
"""Assemble the code-LoRA training set (runs in container).

The adapter is trained on the FROZEN v2.1 chat base, so it only ADDS the code skill.
Mix: executor-verified code (the skill, oversampled) + non-code chat (so the adapter
learns code-WHEN-asked and stays transparent on normal chat, not code-on-everything)."""

import argparse
import json
import pathlib
import random
import sys

REPO = pathlib.Path("/workspace/v2data")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "data"))
from code_format import (
    tab_indent_fenced,  # pretrain code is tab-indented; normalize SFT code blocks to match
)

RES_OPEN = "<result>"


def read(p):
    p = pathlib.Path(p)
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--code", default=str(REPO / "data/training/code_verified_v1/verified_code.jsonl")
    )
    ap.add_argument("--code-rep", type=int, default=2)
    ap.add_argument("--chat", default=str(REPO / "data/training/sft_real_v1/train.helix.jsonl"))
    ap.add_argument("--n-chat", type=int, default=350)
    ap.add_argument("--out-dir", default=str(REPO / "data/training/code_lora_v1"))
    ap.add_argument("--val", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260608)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = []
    code = read(a.code)
    # tab-indent ONLY the ``` blocks (older verified_code.jsonl is 4-space); prose untouched
    rows += [
        {"text": tab_indent_fenced(r["text"]), "source": "code_verified"} for r in code
    ] * a.code_rep
    chat = [
        r
        for r in read(a.chat)
        if "<tool:" not in r.get("text", "")
        and RES_OPEN not in r.get("text", "")
        and "```" not in r.get("text", "")  # exclude code-looking chat
        and r.get("source") not in ("reasoning_de", "reasoning_en")
    ]
    rng.shuffle(chat)
    rows += [{"text": r["text"], "source": "chat"} for r in chat[: a.n_chat]]
    rng.shuffle(rows)
    nval = min(a.val, len(rows) // 15)
    val, train = rows[:nval], rows[nval:]
    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val)]:
        with open(out / f"{name}.helix.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter

    print(f"=== code-LoRA set: train {len(train)} | val {len(val)} -> {out} ===")
    print(f"    code {len(code)}x{a.code_rep} | chat {min(a.n_chat, len(chat))}")
    print("    by source:", dict(Counter(r["source"] for r in train)))


if __name__ == "__main__":
    main()
