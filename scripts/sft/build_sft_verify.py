#!/usr/bin/env python3
"""Premium verify-pass for generated SFT data.

A stronger 'judge' model checks each answer against hard reject rules + factual
correctness. On reject it returns a CORRECTED answer (same Mix style), so we keep
the example with the fix instead of discarding it.

Reject rules (from project spec):
  1 claims to live-check / fetch / 'give me a moment'
  2 current/changeable number/version/price WITHOUT staleness caveat
  3 definition-dependent question WITHOUT a ❗ clarification of variants
  4 medical/legal/financial WITHOUT risk/caution hint
  5 false premise NOT corrected
  6 unnecessary marker on a trivial stable fact
  7 factually WRONG

Key from env OPENROUTER_API_KEY. Out: Helix `{"text": ...}` jsonl (accepted+fixed).
"""
import argparse, json, os, re, sys, time, urllib.request, pathlib
from collections import Counter

JUDGE = (
"Du bist ein strenger Pruefer fuer SFT-Trainingsdaten eines deutschen KI-Assistenten OHNE Tools/Websuche.\n"
"Bewerte die ANTWORT auf die FRAGE nach diesen REJECT-Regeln:\n"
"1 = behauptet live zu pruefen/nachzuschlagen oder sagt 'gib mir einen Moment'\n"
"2 = nennt aktuelle/veraenderliche Zahl/Version/Preis OHNE Veraltungs-Hinweis\n"
"3 = definitionsabhaengige Frage (groesste/beste/reichste/bevoelkerungsreichste/'in Europa'...) OHNE ❗-Klaerung der Varianten\n"
"4 = medizinisch/rechtlich/finanziell OHNE Risiko-/Vorsicht-Hinweis\n"
"5 = falsche Praemisse NICHT korrigiert\n"
"6 = unnoetiger Marker bei trivial stabilem Fakt\n"
"7 = faktisch FALSCH\n"
"Wenn KEINE Regel verletzt ist und die Antwort korrekt ist: verdict='accept'.\n"
"Sonst verdict='reject' UND liefere eine KORRIGIERTE Antwort im Mix-Stil: Prosa als Standard; "
"Marker (🔧 aktuelle Quelle / ⚠️ Vorsicht / ❗ definitionsabhaengig) NUR bei echtem Bedarf; "
"bei aktuellen Fakten best-effort-Wissen + Veraltungs-Hinweis OHNE Tool-Behauptung; stabile Fakten direkt.\n"
"Antworte AUSSCHLIESSLICH als JSON: {\"verdict\":\"accept|reject\",\"rule\":<0-7>,\"fixed\":\"<korrigierte Antwort oder leer>\"}"
)

def call(model, q, ans, key, retries=3):
    body=json.dumps({"model":model,"temperature":0,
        "messages":[{"role":"system","content":JUDGE},
                    {"role":"user","content":f"FRAGE: {q}\nANTWORT: {ans}"}],
        "max_tokens":700}).encode()
    for a in range(retries):
        try:
            req=urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",data=body,
                headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"})
            r=json.load(urllib.request.urlopen(req,timeout=120))
            return r["choices"][0]["message"]["content"], r.get("usage",{})
        except Exception as e:
            if a==retries-1: raise
            time.sleep(2*(a+1))

def parse_json(s):
    i=s.find("{"); j=s.rfind("}")
    if i<0 or j<0: return None
    try: return json.loads(s[i:j+1])
    except Exception: return None

def ans_of(text):
    return text.split("<|assistant|>\n",1)[1].rsplit("\n<|end|>",1)[0]

def wrap(q, ans):
    return (f"<|system|>\nDu bist Auralis, ein hilfreicher, ehrlicher KI-Assistent.\n<|end|>\n"
            f"<|user|>\n{q}\n<|end|>\n<|assistant|>\n{ans}\n<|end|>\n")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="openai/gpt-4o")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--sample-rate", type=float, default=0.3, help="verify fraction of non-critical rows")
    ap.add_argument("--always", default="de_factual,de_current,de_correction,de_definition,en_factual",
                    help="categories always verified 100%")
    args=ap.parse_args()
    key=os.environ.get("OPENROUTER_API_KEY")
    if not key: print("OPENROUTER_API_KEY fehlt",file=sys.stderr); sys.exit(1)

    import random, threading, concurrent.futures
    rows=[json.loads(l) for l in open(args.inp,encoding="utf-8") if l.strip()]
    always=set(c for c in args.always.split(",") if c)
    rng=random.Random(42)
    to_verify=[]; skipped=[]
    for r in rows:
        cat=r.get("category","")
        (to_verify if (cat in always or rng.random()<args.sample_rate) else skipped).append(r)
    kept=list(skipped); stats=Counter(); stats["accept(skipped)"]=len(skipped)
    tot=0; parsefail=0; done=0; lock=threading.Lock()
    def process(r):
        nonlocal tot,parsefail,done
        q=r.get("question") or r["text"].split("<|user|>\n",1)[1].split("\n<|end|>",1)[0]
        ans=ans_of(r["text"])
        try: raw,us=call(args.model,q,ans,key)
        except Exception:
            with lock: stats["error"]+=1; kept.append(r); done+=1
            return
        v=parse_json(raw)
        with lock:
            tot+=us.get("total_tokens",0); done+=1; d=done
            if not v:
                parsefail+=1; stats["accept(parsefail)"]+=1; kept.append(r)
            elif v.get("verdict")=="reject" and str(v.get("fixed","")).strip():
                r2=dict(r); r2["text"]=wrap(q,str(v["fixed"]).strip()); r2["corrected"]=True; r2["rule"]=v.get("rule")
                kept.append(r2); stats[f"reject->fixed(rule{v.get('rule')})"]+=1
            elif v.get("verdict")=="reject":
                stats[f"reject->drop(rule{v.get('rule')})"]+=1
            else:
                stats["accept"]+=1; kept.append(r)
        if d%50==0: print(f"  verify {d}/{len(to_verify)} ({tot} tok)",file=sys.stderr)
    print(f"verify: {len(to_verify)} geprueft, {len(skipped)} als-ist behalten",file=sys.stderr)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(process, to_verify))

    p=pathlib.Path(args.out); p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,"w",encoding="utf-8") as f:
        for r in kept: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    print(f"\n=== verify fertig: {len(rows)} -> {len(kept)} behalten | {tot} tok | parsefail={parsefail} ===")
    for k,v in stats.most_common(): print(f"  {k}: {v}")
    print("->", args.out)

if __name__=="__main__":
    main()
