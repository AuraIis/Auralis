#!/usr/bin/env python3
"""Assemble the BALANCED base-v2 SFT set (runs in the container).

Fixes the failures seen live on the old tool_sft_v12 (stone-word-salad, "hallo"->math,
Walhai-Einstein confabulation): mix coherent general chat (bulk, so most inputs are NOT
math) + tool-use (templated + executor-verified, balanced so it learns WHEN) + calibration
(abstain on unknowns) + grounded (answer from context, not invent). One mixed run (not a
staged chain) to avoid the sequential forgetting that plagued the old chain.
"""
import os, sys, json, random, argparse, pathlib

REPO = pathlib.Path("/workspace/v2data")
sys.path.insert(0, str(REPO / "scripts/sft"))
from gen_tool_traces import gen_tool, BASE_GENS, PHASE2_SIMPLE_REBUMP_GENS  # noqa

RES_OPEN = "<result>"


def read_helix(path):
    p = pathlib.Path(path)
    if not p.exists():
        print(f"WARN missing {path}", file=sys.stderr); return []
    out = []
    for l in open(p, encoding="utf-8"):
        l = l.strip()
        if not l:
            continue
        try:
            r = json.loads(l)
        except Exception:
            continue
        if r.get("text"):
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--general", default=str(REPO / "data/training/sft_real_v1/train.helix.jsonl"))
    ap.add_argument("--n-general", type=int, default=8000)
    ap.add_argument("--n-tool-templated", type=int, default=2000)
    ap.add_argument("--verified-math", default=str(REPO / "data/training/tool_verified_v1/verified_math.jsonl"))
    ap.add_argument("--calib", default=str(REPO / "data/training/calib_verified_v1/verified_calib.jsonl"))
    ap.add_argument("--calib-rep", type=int, default=3, help="oversample calibration (small set, key skill)")
    ap.add_argument("--grounded", nargs="*", default=[
        str(REPO / "data/training/grounded_v2/verified_grounded.jsonl"),
        str(REPO / "data/training/grounded_v1/verified_grounded.jsonl")])
    ap.add_argument("--out-dir", default=str(REPO / "data/training/basev2_sft_v1"))
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260608)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = []

    # 1) general chat = the BULK. Exclude any tool/result rows so the tool form comes only
    #    from the verified/templated traces (consistent <result> masking).
    gen = [r for r in read_helix(a.general)
           if "<tool:" not in r["text"] and RES_OPEN not in r["text"]
           and r.get("source") not in ("reasoning_de", "reasoning_en")]
    rng.shuffle(gen)
    rows += [{"text": r["text"], "source": "chat"} for r in gen[:a.n_general]]
    n_chat = min(a.n_general, len(gen))

    # 2) templated tool traces (full mode) -> learn the tool FORM at scale
    tool = gen_tool(rng, a.n_tool_templated, "full", gens=PHASE2_SIMPLE_REBUMP_GENS)
    rows += tool

    # 3) executor-verified math (natural word problems the templates can't make)
    vmath = read_helix(a.verified_math)
    rows += [{"text": r["text"], "source": "tool_math_verified"} for r in vmath]

    # 4) calibration (abstain on unknown / confident on known) — oversampled
    calib = read_helix(a.calib)
    rows += [{"text": r["text"], "source": r.get("source", "calib")} for r in calib] * a.calib_rep

    # 5) grounded (answer from context / abstain if not in context) = anti-hallucination
    n_ground = 0
    for gp in a.grounded:
        g = read_helix(gp)
        rows += [{"text": r["text"], "source": r.get("source", "grounded")} for r in g]
        n_ground += len(g)

    rng.shuffle(rows)
    nval = min(a.val, len(rows) // 20)
    val, train = rows[:nval], rows[nval:]
    out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val)]:
        with open(out / f"{name}.helix.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    comp = Counter(r["source"] for r in train)
    print(f"=== base-v2 SFT assembled: train {len(train)} | val {len(val)} -> {out} ===")
    print(f"    chat {n_chat} | tool_templated {len(tool)} | verified_math {len(vmath)} | "
          f"calib {len(calib)}x{a.calib_rep} | grounded {n_ground}")
    print("    by source (train):", dict(comp.most_common()))


if __name__ == "__main__":
    main()
