#!/usr/bin/env python3
"""Generate native German math/logic reasoning (step-by-step) for SFT.
Teacher produces AUFGABE + LOESUNG pairs; later verified by build_sft_verify.py."""
import argparse, json, os, re, sys, threading, concurrent.futures, urllib.request, pathlib, time

URL = "https://openrouter.ai/api/v1/chat/completions"
SYS = ("Du erstellst deutsche Mathe-Textaufgaben mit Schritt-fuer-Schritt-Loesung fuer KI-Training.\n"
"Regeln: RICHTIG rechnen (das ist entscheidend); die Loesung zeigt den Rechenweg in kurzen klaren Schritten "
"und endet mit 'Die Antwort ist X.'; natuerliches Deutsch; eindeutige Zahlen-Antwort; nicht zu lang.\n"
"Gib pro Aufgabe EXAKT dieses Format:\nAUFGABE: <eine Frage>\nLOESUNG: <Schritte> Die Antwort ist X.\n---")
TOPICS = ["Einkauf und Geld", "Prozentrechnung", "Geschwindigkeit und Strecke", "Zeit und Alter",
 "Mengen aufteilen und verteilen", "einfache Geometrie (Flaeche, Umfang)", "Mischungen und Verhaeltnisse",
 "Durchschnitt berechnen", "Bruchrechnung im Alltag", "einfache Logik und Schlussfolgern",
 "Zinsen einfach", "Einheiten umrechnen", "Rabatt und Aufschlag", "Anzahl und Stueckzahl"]

def call(model, prompt, key, retries=3):
    body = json.dumps({"model": model, "temperature": 0.8,
        "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": prompt}],
        "max_tokens": 2000}).encode()
    for a in range(retries):
        try:
            req = urllib.request.Request(URL, data=body, headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(req, timeout=120))["choices"][0]["message"]["content"]
        except Exception:
            if a == retries - 1: raise
            time.sleep(2 * (a + 1))

PAT = re.compile(r"AUFGABE:\s*(.+?)\s*L[OÖ]ESUNG:\s*(.+?)(?=(?:\n\s*\d+[\.\)]?\s*)?AUFGABE:|\Z)",
                 re.DOTALL | re.IGNORECASE)

def parse(txt):
    out = []
    for m in PAT.finditer(txt):
        q = re.sub(r"^\d+[\.\)]\s*", "", re.sub(r"\s+", " ", m.group(1).strip()))
        a = re.sub(r"\s*-{2,}\s*$", "", re.sub(r"\n{2,}", " ", m.group(2).strip())).strip()
        # merge-guard: a valid single solution contains 'Antwort ist' exactly once
        if 15 <= len(q) <= 600 and 15 <= len(a) <= 1200 and a.count("Antwort ist") == 1:
            out.append((q, a))
    return out

def helix(q, a):
    return (f"<|system|>\nDu bist Auralis, ein hilfreicher, ehrlicher KI-Assistent.\n<|end|>\n"
            f"<|user|>\n{q}\n<|end|>\n<|assistant|>\n{a}\n<|end|>\n")

def norm(s): return re.sub(r"\s+", " ", s.lower()).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="deepseek/deepseek-chat")
    ap.add_argument("--target", type=int, default=2500)
    ap.add_argument("--per", type=int, default=8)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key: print("OPENROUTER_API_KEY fehlt", file=sys.stderr); sys.exit(1)

    ncalls = int(args.target / args.per * 1.8)
    jobs = [TOPICS[i % len(TOPICS)] for i in range(ncalls)]
    rows = []; seen = set(); lock = threading.Lock(); done = 0
    def work(topic):
        nonlocal done
        prompt = (f"Erzeuge {args.per} deutsche Mathe-Textaufgaben zum Bereich \"{topic}\" mit Schritt-fuer-Schritt-Loesung. "
                  f"WICHTIG: variiere Zahlen, Namen, Kontexte UND die Rechenart stark — keine zwei Aufgaben duerfen sich aehneln "
                  f"(nicht nur Substantive tauschen). Unterschiedliche Schwierigkeit (1 bis 3 Rechenschritte).")
        try: txt = call(args.model, prompt, key)
        except Exception as e: print("FEHLER", topic, repr(e)[:90], file=sys.stderr); return
        for q, a in parse(txt):
            k = norm(q)
            with lock:
                if len(rows) >= args.target or k in seen: continue
                seen.add(k); rows.append({"text": helix(q, a), "category": "reasoning_math_de", "question": q, "source": "gen_de"})
        with lock:
            done += 1
            if done % 25 == 0: print(f"  calls {done}/{len(jobs)} | examples {len(rows)}", file=sys.stderr)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, jobs))
    p = pathlib.Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows[:args.target]: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n=== {min(len(rows),args.target)} DE-Reasoning -> {args.out} ===")

if __name__ == "__main__":
    main()
