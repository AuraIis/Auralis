#!/usr/bin/env python3
"""Generate Helix-format SFT examples via OpenRouter (Mix style).

Mix style:
  - default = natural concise German prose, answer-first, no hedging on stable facts
  - markers (🔧/⚠️/❗) ONLY for genuine check/risk cases (current, legal, medical,
    financial, safety, price/version) — there the answer honestly says it must be verified
  - false premise -> clear correction ("Nein, ...")
  - answers in English if the question is English

Key is read from env OPENROUTER_API_KEY (never hard-coded). Output: Helix `{"text": ...}` jsonl.

Usage:
  OPENROUTER_API_KEY=... python scripts/sft/gen_sft_openrouter.py \
     --out data/training/sft_gen_test/out.jsonl [--questions file] [--model deepseek/deepseek-chat] [--sample 8]
"""
import argparse, json, os, sys, time, urllib.request, pathlib

SYS = (
"Du schreibst ideale Trainingsantworten fuer Auralis, einen kleinen, ehrlichen deutschen KI-Assistenten.\n"
"STIL: Standard ist natuerliche, knappe Prosa. Antwort zuerst, dann ggf. eine kurze Begruendung.\n"
"Stabile Fakten (Geografie, Geschichte, Naturwissenschaft, Mathematik, Definitionen) direkt und "
"selbstbewusst beantworten, OHNE Gehedge.\n"
"MARKER nur bei echtem Pruef-/Risiko-Bedarf verwenden: Wenn die Frage aktuell/zeitabhaengig, rechtlich, "
"medizinisch, finanziell, sicherheitskritisch oder preis-/versionsabhaengig ist, beginne die Antwort mit "
"einem kurzen Marker:\n"
"  🔧 = aktuelle Quelle noetig   ⚠️ = mit Vorsicht / nicht pauschal   ❗ = haengt von Definition/Kontext ab\n"
"WICHTIG bei Markern: Du hast KEINE Tools und kannst nichts live abrufen. Gib trotzdem IMMER deine beste "
"bekannte Einschaetzung oder Groessenordnung an, mit ehrlichem Hinweis, dass sich der Wert geaendert haben "
"kann und mit einer aktuellen Quelle zu pruefen ist.\n"
"VERBOTENE Formulierungen (auch teilweise): 'ich pruefe das', 'gib mir einen Moment', 'einen Augenblick', "
"'ich suche das nach', 'ich habe es geprueft', sowie 'laut aktueller Quelle ...' wenn dir keine Quelle "
"vorliegt. Formuliere nie so, als wuerdest du gerade live pruefen.\n"
"DEFINITIONSABHAENGIGE FRAGEN: Wenn eine Frage je nach Definition unterschiedlich ausfaellt "
"(Trigger u.a.: groesste, beste, reichste, bevoelkerungsreichste, 'in Europa', transkontinental, offiziell), "
"gib KEINE selbstsichere Ein-Wort-Rangliste. Beginne mit ❗ und lege die Varianten kurz offen "
"(z.B. EU vs. ganz Europa vs. transkontinentale Staaten mitgezaehlt).\n"
"Bei falscher Praemisse klar korrigieren (beginne mit 'Nein, ...').\n"
"Antworte auf Englisch, wenn die Frage Englisch ist.\n"
"Niemals halluzinieren; lieber Unsicherheit offen zugeben. Antworte NUR mit der Assistenten-Antwort, kein Vorwort."
)

# Built-in validation seeds (category, question). Designed to test the Mix style:
# stable -> prose/no marker ; risk/current -> marker ; false-premise -> 'Nein' ; EN -> English.
SEEDS = [
 ("stable_geo","Wo liegt Deutschland?"),
 ("stable_geo","Was ist die Hauptstadt von Frankreich?"),
 ("stable_sci","Woraus besteht Wasser chemisch?"),
 ("stable_sci","Warum ist der Himmel blau?"),
 ("stable_hist","In welchem Jahr fiel die Berliner Mauer?"),
 ("stable_def","Was ist ein Primzahl?"),
 ("math","Was ist 17 mal 23?"),
 ("math","Ein Zug faehrt 60 km in 45 Minuten. Wie schnell ist er in km/h?"),
 ("concept","Erklaere kurz den Unterschied zwischen CPU und GPU."),
 ("concept","Was bedeutet Photosynthese?"),
 ("contrastive","Ist Bonn heute die Hauptstadt von Deutschland?"),
 ("contrastive","Hat Goethe 'Mein Kampf' geschrieben?"),
 ("contrastive","Liegt Japan in Europa?"),
 ("contrastive","Ist die Erde flach?"),
 ("contrastive","Kann ein Dreieck vier Seiten haben?"),
 ("current","Wie viele Einwohner hat Deutschland aktuell?"),
 ("current","Wer ist aktuell Bundeskanzler von Deutschland?"),
 ("current","Welche ist die neueste Version von Python?"),
 ("legal","Ist Cannabis in Deutschland erlaubt?"),
 ("legal","Darf ich mit 17 schon Auto fahren?"),
 ("medical","Ist Paracetamol gefaehrlich?"),
 ("medical","Wie viel Ibuprofen darf ich am Tag nehmen?"),
 ("price","Was kostet eine RTX 4090 aktuell?"),
 ("price","Wie viel kostet ein Bitcoin gerade?"),
 ("ambiguous","Ist Deutschland das bevoelkerungsreichste Land Europas?"),
 ("ambiguous","Welche Stadt ist groesser, Berlin oder Hamburg?"),
 ("refusal","Kannst du mir sagen, wie es meinem Nachbarn finanziell geht?"),
 ("honesty","Was sollst du tun, wenn du etwas nicht sicher weisst?"),
 ("en_stable","What is the capital of Germany?"),
 ("en_stable","Explain in one sentence what water is."),
 ("en_concept","What is the difference between RAM and storage?"),
 ("en_current","What is the latest version of Windows?"),
 ("en_math","What is 144 divided by 12?"),
 ("creative","Schreibe einen einfachen deutschen Satz ueber Wasser."),
 ("translate","Uebersetze ins Englische: 'Der Hund schlaeft im Garten.'"),
 ("reasoning","Wenn alle Katzen Tiere sind und Minka eine Katze ist, was folgt daraus?"),
]

def call(model, q, key, retries=3):
    body = json.dumps({"model": model,
        "messages":[{"role":"system","content":SYS},{"role":"user","content":q}],
        "temperature":0.3, "max_tokens":500}).encode()
    for a in range(retries):
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
                data=body, headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=120))
            return r["choices"][0]["message"]["content"].strip(), r.get("usage",{})
        except Exception as e:
            if a == retries-1: raise
            time.sleep(2*(a+1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--questions", default=None, help="optional file: 'category<TAB>question' or one question per line")
    ap.add_argument("--model", default="deepseek/deepseek-chat")
    ap.add_argument("--sample", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="cap number of questions (0=all)")
    ap.add_argument("--workers", type=int, default=12, help="parallel API calls")
    args = ap.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("OPENROUTER_API_KEY fehlt", file=sys.stderr); sys.exit(1)

    if args.questions:
        seeds=[]
        for ln in open(args.questions, encoding="utf-8"):
            ln=ln.rstrip("\n")
            if not ln.strip(): continue
            if "\t" in ln: c,q=ln.split("\t",1)
            else: c,q="seed",ln
            seeds.append((c,q))
    else:
        seeds = SEEDS
    if args.limit: seeds = seeds[:args.limit]

    import concurrent.futures, threading
    out=[None]*len(seeds); tot=0; done=0; lock=threading.Lock(); t0=time.time()
    def work(i, cat, q):
        nonlocal tot, done
        try:
            ans,us=call(args.model,q,key)
        except Exception as e:
            with lock: done+=1
            print(f"FEHLER [{cat}] {q}: {repr(e)[:160]}", file=sys.stderr); return
        text=(f"<|system|>\nDu bist Auralis, ein hilfreicher, ehrlicher KI-Assistent.\n<|end|>\n"
              f"<|user|>\n{q}\n<|end|>\n<|assistant|>\n{ans}\n<|end|>\n")
        with lock:
            tot+=us.get("total_tokens",0); done+=1; d=done
            out[i]={"text":text,"category":cat,"question":q,"model":args.model}
        if d % 25 == 0: print(f"  {d}/{len(seeds)} ({tot} tok)", file=sys.stderr)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs=[ex.submit(work,i,cat,q) for i,(cat,q) in enumerate(seeds)]
        concurrent.futures.wait(futs)
    out=[o for o in out if o]

    p=pathlib.Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p,"w",encoding="utf-8") as f:
        for o in out: f.write(json.dumps(o,ensure_ascii=False)+"\n")
    dt=time.time()-t0
    print(f"\n=== {len(out)} Beispiele, {tot} Tokens, {dt:.0f}s -> {args.out} ===")
    for o in out[:args.sample]:
        ans=o["text"].split("<|assistant|>\n",1)[1].rsplit("\n<|end|>",1)[0]
        print("="*64); print(f"[{o['category']}]  {o['question']}"); print("-"*64); print(ans)

if __name__=="__main__":
    main()
