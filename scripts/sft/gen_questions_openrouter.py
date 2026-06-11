#!/usr/bin/env python3
"""Stage 0: generate DIVERSE SFT questions via teacher, dedup, write category<TAB>question.

Topic-anchored so each call explores a different area -> high diversity, low overlap.
Special categories (current/correction/fictional) use tailored prompts.
Key from env OPENROUTER_API_KEY. Output: questions tsv.
"""
import argparse, json, os, re, sys, threading, concurrent.futures, urllib.request, pathlib

URL="https://openrouter.ai/api/v1/chat/completions"

TOPICS_DE = [
 "Geografie","Laender und Hauptstaedte","Fluesse und Berge","Geschichte (Antike)","Geschichte (Mittelalter)",
 "Geschichte (Neuzeit)","deutsche Geschichte","Physik","Chemie","Biologie","Astronomie und Weltall",
 "Mathematik-Grundlagen","Technik und Maschinen","Computer und Software","Internet und Netzwerke",
 "Gesundheit allgemein","Ernaehrung","menschlicher Koerper","Sport","Musik","Kunst","Literatur","Film und Serien",
 "deutsche Sprache und Grammatik","Wirtschaft-Grundlagen","Politik-Grundlagen","Recht-Grundlagen (allgemein)",
 "Haushalt und Alltag","Reisen","Natur und Umwelt","Tiere","Pflanzen","Wetter und Klima","Kochen und Backen",
 "Garten","Auto und Verkehr","Handwerk und Reparatur","persoenliche Finanzen (allgemein)","Bildung und Lernen",
 "Psychologie im Alltag","Philosophie-Grundlagen","Religionen und Kulturen","Erfindungen","beruehmte Personen",
 "Energie und Strom","Materialien und Stoffe","Zeit und Kalender","Masse und Einheiten","Farben und Licht",
 "Wasser und Meere","Vulkane und Erdbeben","Mikroorganismen","Genetik-Grundlagen","Wirtschaftsbegriffe",
 "Programmierkonzepte (allgemein)","Datensicherheit","Logik und Raetsel","Mathe im Alltag","Statistik-Grundlagen",
]

def cfg():
    return {
     "de_factual":  dict(lang="de", n=5000, per=18, topics=TOPICS_DE,
        instr="kurze, eindeutig beantwortbare Faktenfragen (eine klare richtige Antwort)"),
     "de_explain":  dict(lang="de", n=3000, per=15, topics=TOPICS_DE,
        instr="Erklaer-Fragen wie 'Was ist ...?', 'Wie funktioniert ...?', 'Warum ...?'"),
     "de_everyday": dict(lang="de", n=2000, per=15, topics=TOPICS_DE,
        instr="praktische Alltagsfragen, die jemand einem hilfreichen Assistenten stellt"),
     "de_reasoning":dict(lang="de", n=1200, per=12, topics=["Logik","Mathe im Alltag","Vergleiche","Schlussfolgern","Zeit/Rechnen"],
        instr="einfache 1-2-Schritt-Logik- oder Rechenaufgaben in Worten (mit eindeutiger Loesung)"),
     "de_current":  dict(lang="de", n=1500, per=15, topics=["aktuelle Zahlen","Preise","Versionen","Amtsinhaber","Gesetzeslage","Wetter","Verfuegbarkeit"],
        instr="Fragen nach AKTUELLEN/veraenderlichen Fakten (Preise, Versionen, aktuelle Amtstraeger, aktuelle Statistiken)"),
     "de_correction":dict(lang="de", n=1500, per=12, topics=TOPICS_DE,
        instr="Ja/Nein- oder Behauptungs-Fragen, die eine FALSCHE Praemisse oder einen verbreiteten Irrtum enthalten (zum Korrigieren)"),
     "de_definition":dict(lang="de", n=800, per=10, topics=["Superlative","Vergleiche","Rankings","mehrdeutige Begriffe"],
        instr="Fragen, deren Antwort von der Definition abhaengt (groesste/beste/reichste, je nach Zaehlweise)"),
     "en_factual":  dict(lang="en", n=2500, per=18, topics=TOPICS_DE,
        instr="short, clearly answerable factual questions (one correct answer)"),
     "en_explain":  dict(lang="en", n=1500, per=15, topics=TOPICS_DE,
        instr="explanation questions like 'What is ...?', 'How does ... work?'"),
     "en_everyday": dict(lang="en", n=1000, per=15, topics=TOPICS_DE,
        instr="practical everyday questions a person asks a helpful assistant"),
    }

def build_prompt(c, topic):
    lang = "deutsche" if c["lang"]=="de" else "English"
    langword = "auf Deutsch" if c["lang"]=="de" else "in English"
    return (f"Erzeuge {c['per']} {lang} Nutzer-Fragen {langword} zum Bereich \"{topic}\": {c['instr']}.\n"
            f"Regeln: jede Frage in EINER Zeile, KEINE Nummerierung, keine Antworten, keine Einleitung, "
            f"moeglichst unterschiedlich, natuerlich formuliert.")

def norm(s): return re.sub(r"\s+"," ",s.strip().lower())

def call(model, prompt, key, retries=3):
    body=json.dumps({"model":model,"temperature":1.0,
        "messages":[{"role":"user","content":prompt}],"max_tokens":1200}).encode()
    import time
    for a in range(retries):
        try:
            req=urllib.request.Request(URL,data=body,headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"})
            r=json.load(urllib.request.urlopen(req,timeout=120))
            return r["choices"][0]["message"]["content"]
        except Exception:
            if a==retries-1: raise
            time.sleep(2*(a+1))

def extract(text):
    out=[]
    for ln in text.splitlines():
        ln=re.sub(r"^\s*[\d\.\)\-\*•]+\s*","",ln).strip()
        ln=ln.strip(' "')
        if 8<=len(ln)<=240 and ln.endswith(("?",".",":")) or (8<=len(ln)<=240 and "?" in ln):
            out.append(ln)
        elif 8<=len(ln)<=240 and ln:
            out.append(ln)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="deepseek/deepseek-chat")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only", default=None, help="comma list of categories to run")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply all targets (for test runs)")
    args=ap.parse_args()
    key=os.environ.get("OPENROUTER_API_KEY");
    if not key: print("OPENROUTER_API_KEY fehlt",file=sys.stderr); sys.exit(1)
    C=cfg()
    if args.only: C={k:v for k,v in C.items() if k in args.only.split(",")}

    seen=set(); rows=[]; lock=threading.Lock()
    # build work items: (cat, cfg, topic) repeated enough to hit target
    jobs=[]
    for cat,c in C.items():
        target=int(c["n"]*args.scale)
        c=dict(c); c["target"]=target
        # how many calls: target/per * 1.6 buffer, spread over topics
        ncalls=max(1, int(target/c["per"]*1.7))
        tps=c["topics"]
        for i in range(ncalls):
            jobs.append((cat,c,tps[i%len(tps)]))
    catcount={k:0 for k in C}
    def work(cat,c,topic):
        if catcount[cat] >= c["target"]: return
        try: txt=call(args.model, build_prompt(c,topic), key)
        except Exception as e: print("FEHLER",cat,repr(e)[:100],file=sys.stderr); return
        for q in extract(txt):
            k=norm(q)
            with lock:
                if catcount[cat]>=c["target"]: break
                if k in seen: continue
                seen.add(k); rows.append((cat,q)); catcount[cat]+=1
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs=[ex.submit(work,*j) for j in jobs]
        for n,_ in enumerate(concurrent.futures.as_completed(futs),1):
            if n%50==0: print(f"  calls {n}/{len(jobs)} | fragen {len(rows)}",file=sys.stderr)
    p=pathlib.Path(args.out); p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,"w",encoding="utf-8") as f:
        for cat,q in rows: f.write(f"{cat}\t{q}\n")
    from collections import Counter
    print(f"\n=== {len(rows)} Fragen -> {args.out} ===")
    for k,v in Counter(c for c,_ in rows).most_common(): print(f"  {k}: {v}")

if __name__=="__main__":
    main()
