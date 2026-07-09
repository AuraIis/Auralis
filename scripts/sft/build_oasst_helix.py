#!/usr/bin/env python3
"""Download OASST1 + convert German & English best-reply pairs to Helix SFT format."""

import gzip
import json
import pathlib
import urllib.request

URL = "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/2023-04-12_oasst_ready.trees.jsonl.gz"
WORK = pathlib.Path("/workspace/v2data/data/training/oasst")
GZ = WORK / "oasst_trees.jsonl.gz"


def download():
    WORK.mkdir(parents=True, exist_ok=True)
    if GZ.exists() and GZ.stat().st_size > 1_000_000:
        print("schon vorhanden:", GZ, GZ.stat().st_size, "bytes")
        return
    print("lade OASST1 ...", flush=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as r, open(GZ, "wb") as f:
        f.write(r.read())
    print("geladen:", GZ.stat().st_size, "bytes")


def best_reply(msg):
    cands = [
        m for m in msg.get("replies", []) if m.get("role") == "assistant" and not m.get("deleted")
    ]
    if not cands:
        return None
    ranked = [m for m in cands if m.get("rank") is not None]
    return min(ranked, key=lambda m: m["rank"]) if ranked else cands[0]


def helix(user, asst):
    return (
        f"<|system|>\nDu bist Auralis, ein hilfreicher, ehrlicher KI-Assistent.\n<|end|>\n"
        f"<|user|>\n{user.strip()}\n<|end|>\n<|assistant|>\n{asst.strip()}\n<|end|>\n"
    )


def main():
    download()
    de = []
    en = []
    n = 0
    with gzip.open(GZ, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            n += 1
            prompt = t.get("prompt") or t
            if prompt.get("role") != "prompter" or prompt.get("deleted"):
                continue
            lang = prompt.get("lang")
            if lang not in ("de", "en"):
                continue
            a = best_reply(prompt)
            if not a:
                continue
            u = (prompt.get("text") or "").strip()
            ans = (a.get("text") or "").strip()
            if not (10 <= len(u) <= 1500 and 10 <= len(ans) <= 4000):
                continue
            row = {
                "text": helix(u, ans),
                "category": f"oasst_{lang}",
                "question": u,
                "source": "oasst1",
            }
            (de if lang == "de" else en).append(row)
    for name, rows in [("oasst_de.helix.jsonl", de), ("oasst_en.helix.jsonl", en)]:
        with open(WORK / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"trees: {n}  ->  DE: {len(de)}  EN: {len(en)}")
    if de:
        print("DE-Beispiel:", de[0]["question"][:90])


if __name__ == "__main__":
    main()
