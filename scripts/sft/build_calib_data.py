#!/usr/bin/env python3
"""Build calibration (R-Tuning) SFT data from probe labels — KEY-FREE.

ABSTAIN  : invented entities (gold=None) + clearly-UNKNOWN facts (0/N correct)
           -> rotated honest "I can't answer reliably" targets.
CONFIDENT: KNOWN facts -> confident answer from gold (reinforce answering-when-known).
ANCHOR   : sample of existing general SFT QA -> retention (don't destroy answering).

SHAKY is EXCLUDED from abstain (might actually know it -> avoid the false-negative trap)."""

import argparse
import json
import pathlib
import random
import re

REPO = pathlib.Path("/workspace/v2data")
SYS = "Du bist Auralis, ein hilfreicher, ehrlicher KI-Assistent."
RES_OPEN = "<result>"

ABSTAIN_ANSWERS = [
    "Das kann ich nicht zuverlaessig beantworten.",
    "Dazu habe ich keine verlaesslichen Informationen.",
    "Das ist mir nicht sicher bekannt; ohne verlaessliche Quelle moechte ich dazu nichts behaupten.",
    "Ich bin mir da nicht sicher und moechte nichts Falsches behaupten.",
    "Dazu kann ich keine gesicherte Antwort geben.",
]


def helix(q, a):
    return f"<|system|>\n{SYS}\n<|end|>\n<|user|>\n{q}\n<|end|>\n<|assistant|>\n{a}\n<|end|>\n"


def confident_answer(q, gold, cat):
    g = gold[0]
    G = g[:1].upper() + g[1:]
    if cat == "capital":
        m = re.search(r"Hauptstadt von (.+?)\?", q)
        return f"Die Hauptstadt von {m.group(1)} ist {G}." if m else f"{G}."
    if cat == "work":
        m = re.search(r"'(.+?)'", q)
        return f"'{m.group(1)}' wurde von {G} geschrieben." if m else f"{G}."
    if cat == "element":
        m = re.search(r"Symbol (\w+)", q)
        return f"Das chemische Symbol {m.group(1)} steht fuer {G}." if m else f"{G}."
    return f"{G}."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(REPO / "data/training/calib/probe_labels_v2.jsonl"))
    ap.add_argument(
        "--anchor-src", default=str(REPO / "data/training/sft_real_v1/train.helix.jsonl")
    )
    ap.add_argument(
        "--tool-src", default=str(REPO / "data/training/tool_sft_v12_full/train.helix.jsonl")
    )
    ap.add_argument("--n-anchor", type=int, default=500)
    ap.add_argument(
        "--n-tool", type=int, default=400, help="tool traces mixed in -> preserve dispatch"
    )
    ap.add_argument(
        "--abstain-unknown",
        action="store_true",
        help="also abstain on UNKNOWN real facts (v1 did this -> over-refusal leak; off in v2)",
    )
    ap.add_argument("--out-dir", default=str(REPO / "data/training/calib_sft_v2"))
    ap.add_argument("--val", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260607)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    labels = [json.loads(l) for l in open(a.labels, encoding="utf-8") if l.strip()]

    rows = []
    n_abstain = n_confident = 0
    for r in labels:
        q, gold, cat, label = r["q"], r["gold"], r["cat"], r["label"]
        if cat == "invented" or (a.abstain_unknown and gold is not None and label == "UNKNOWN"):
            ans = rng.choice(ABSTAIN_ANSWERS)
            rows.append({"text": helix(q, ans), "source": "calib_abstain"})
            n_abstain += 1
        elif gold is not None and label == "KNOWN":
            rows.append(
                {"text": helix(q, confident_answer(q, gold, cat)), "source": "calib_confident"}
            )
            n_confident += 1
        # SHAKY -> skip (uncertain whether it knows)

    # tool traces -> preserve math dispatch (v1 forgot it: 12+15 -> abstain)
    tool = []
    tp = pathlib.Path(a.tool_src)
    if tp.exists():
        for l in open(tp, encoding="utf-8"):
            try:
                rr = json.loads(l)
            except Exception:
                continue
            if rr.get("source") == "tool_math":
                tool.append({"text": rr["text"], "source": "tool"})
        rng.shuffle(tool)
        tool = tool[: a.n_tool]
    rows += tool

    # retention anchor: general SFT QA (no tool/result)
    anchor = []
    for l in open(a.anchor_src, encoding="utf-8"):
        try:
            rr = json.loads(l)
        except Exception:
            continue
        if rr.get("source") in ("reasoning_de", "reasoning_en"):
            continue
        if "<tool:" in rr.get("text", "") or RES_OPEN in rr.get("text", ""):
            continue
        anchor.append({"text": rr["text"], "source": "anchor"})
    rng.shuffle(anchor)
    anchor = anchor[: a.n_anchor]
    rows += anchor

    rng.shuffle(rows)
    nval = min(a.val, len(rows) // 10)
    val, train = rows[:nval], rows[nval:]
    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val)]:
        with open(out / f"{name}.helix.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"=== calib-SFT: train {len(train)} | val {len(val)} ===")
    print(
        f"    abstain {n_abstain} | confident {n_confident} | tool {len(tool)} | anchor {len(anchor)}"
    )
    print(f"    abstain-ratio ~{n_abstain / len(rows):.0%}  -> {out}")
    print("\n=== SAMPLES ===")
    for src in ("calib_abstain", "calib_confident"):
        ex = next((r for r in train if r["source"] == src), None)
        if ex:
            print("---", src, "---")
            u = ex["text"].split("<|user|>")[1].split("<|end|>")[0].strip()
            asst = ex["text"].split("<|assistant|>")[1].split("<|end|>")[0].strip()
            print("Q:", u, "\nA:", asst)


if __name__ == "__main__":
    main()
