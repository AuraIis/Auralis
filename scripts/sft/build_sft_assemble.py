#!/usr/bin/env python3
"""Assemble final SFT dataset: merge all sources, normalize to canonical Helix
turn format, dedup, decontaminate vs eval probes, shuffle, split train/val."""
import json, re, random, sys, pathlib
from collections import Counter

BASE="/workspace/v2data/data/training"
OUT=pathlib.Path(BASE)/"sft_real_v1"
SYS="Du bist Auralis, ein hilfreicher, ehrlicher KI-Assistent."

SOURCES=[
 (f"{BASE}/sft_real_v1/answers_verified.jsonl","gen", 0),
 (f"{BASE}/sft_clean_de_v1/train.helix.jsonl","sft_clean", 0),
 (f"{BASE}/helix_sft_de_booster_v1/helix_sft_de_booster_v1.jsonl","booster", 0),
 (f"{BASE}/oasst/oasst_de.helix.jsonl","oasst_de", 0),
 (f"{BASE}/oasst/oasst_en.helix.jsonl","oasst_en", 0),
 (f"{BASE}/reasoning/de_reasoning_verified.jsonl","reasoning_de", 0),
 (f"{BASE}/reasoning/gsm8k_en.helix.jsonl","reasoning_en", 0),
]

U_RE=re.compile(r"<\|user\|>\s*(.*?)\s*<\|(?:assistant|end)\|>", re.DOTALL)
A_RE=re.compile(r"<\|assistant\|>\s*(.*?)\s*<\|end\|>", re.DOTALL)

def norm(s): return re.sub(r"\s+"," ", re.sub(r"[^\w\s]","",s.casefold())).strip()

# Eval probes kept OUT of training (decontamination)
EVAL=[ "Was ist die Hauptstadt von Deutschland?","Ist Bonn heute die Hauptstadt von Deutschland?",
 "Schrieb Goethe Mein Kampf?","Wer hat Faust geschrieben?","What is the capital of Germany?",
 "Erkläre kurz, was Wasser ist.","Schreibe einen einfachen deutschen Satz über Wasser.",
 "Was ist die Hauptstadt von Österreich?","Was ist ein Vulkan?","Wer hat die Glühbirne erfunden?",
 "Was ist schwerer, 1 kg Eisen oder 1 kg Federn?","Ist Pluto ein Planet?","Wie viele Tage hat ein Schaltjahr?",
 "What is the capital of Spain?","Nenne drei Bundesländer in Deutschland.","Was ist 12 plus 15?",
 "Warum muss man Zähne putzen?","Wer hat die Glühbirne erfunden","Was ist die Hauptstadt von Italien?",
 "Wo liegt Deutschland?","Welches ist das bevölkerungsreichste Land Europas?" ]
EVAL_N=set(norm(x) for x in EVAL)

def extract(text):
    mu=U_RE.search(text); ma=A_RE.search(text)
    if not mu or not ma: return None
    u=mu.group(1).strip(); a=ma.group(1).strip()
    if len(u)<3 or len(a)<2: return None
    return u,a

def render(u,a): return f"<|system|>\n{SYS}\n<|end|>\n<|user|>\n{u}\n<|end|>\n<|assistant|>\n{a}\n<|end|>\n"

def main():
    rng=random.Random(20260605)
    rows=[]; seen=set(); per=Counter(); decon=0; dup=0; bad=0
    for path,src,cap in SOURCES:
        p=pathlib.Path(path)
        if not p.exists(): print(f"WARN fehlt: {path}",file=sys.stderr); continue
        n=0
        for line in open(p,encoding="utf-8"):
            line=line.strip()
            if not line: continue
            try: r=json.loads(line)
            except Exception: continue
            ex=extract(r.get("text",""))
            if not ex: bad+=1; continue
            u,a=ex; un=norm(u)
            if un in EVAL_N: decon+=1; continue
            if un in seen: dup+=1; continue
            seen.add(un)
            rows.append({"text":render(u,a),"source":src}); per[src]+=1; n+=1
            if cap and n>=cap: break
    rng.shuffle(rows)
    nval=min(400, len(rows)//50)
    val=rows[:nval]; train=rows[nval:]
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT/"train.helix.jsonl","w",encoding="utf-8") as f:
        for r in train: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    with open(OUT/"val.helix.jsonl","w",encoding="utf-8") as f:
        for r in val: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    print(f"=== assembled: train {len(train)} | val {len(val)} ===")
    print("pro Quelle:", dict(per))
    print(f"deduped: {dup} | decontaminated(eval): {decon} | unparsable: {bad}")
    (OUT/"assemble_manifest.json").write_text(json.dumps(
        {"train":len(train),"val":len(val),"per_source":dict(per),"dedup":dup,"decontam":decon,"bad":bad},
        ensure_ascii=False, indent=2))

if __name__=="__main__":
    main()
