#!/usr/bin/env python3
"""Build the pilot SFT question set (contamination-clean).

  - ~210 diverse user-questions sampled from data/training/sft_clean_de_v1
  - ~90 curated calibration templates (contrastive / risk / definition / EN)
    deliberately using DIFFERENT entities than the eval probes, so the SFT-smoke
    measures GENERALISATION, not memorisation.

Output: data/training/sft_gen_test/pilot_questions.tsv  (category<TAB>question)
"""
import json, re, sys, pathlib

REPO = pathlib.Path("/workspace/v2data")
SRC  = REPO / "data/training/sft_clean_de_v1/train.helix.jsonl"
OUT  = REPO / "data/training/sft_gen_test/pilot_questions.tsv"

# eval-probe phrasings to KEEP OUT of training (contamination guard)
EXCLUDE = {
 "was ist die hauptstadt von deutschland?", "berlin ist eine stadt",
 "wer hat faust geschrieben?", "ist bonn heute die hauptstadt von deutschland?",
 "schrieb goethe mein kampf?", "what is the capital of germany?",
 "die hauptstadt von deutschland ist", "erklaere kurz, was wasser ist.",
 "schreibe einen einfachen deutschen satz ueber wasser.",
}
def norm(s): return re.sub(r"\s+"," ",s.strip().lower())

# ---- curated calibration templates (eval-disjoint entities) ----
TEMPLATES = [
 # contrastive / false premise (different pairs than eval)
 ("contrastive","Ist Hannover die Hauptstadt von Bayern?"),
 ("contrastive","Hat William Shakespeare 'Don Quijote' geschrieben?"),
 ("contrastive","Liegt Brasilien in Afrika?"),
 ("contrastive","Ist die Sonne ein Planet?"),
 ("contrastive","Kann ein Quadrat drei Ecken haben?"),
 ("contrastive","War Albert Einstein ein Komponist?"),
 ("contrastive","Ist Wien die Hauptstadt der Schweiz?"),
 ("contrastive","Hat Mozart die Relativitaetstheorie entwickelt?"),
 ("contrastive","Fliesst der Nil durch Deutschland?"),
 ("contrastive","Ist Gold ein Edelgas?"),
 # current / risk
 ("current","Wie viele Einwohner hat Frankreich aktuell?"),
 ("current","Wer ist aktuell Praesident von Frankreich?"),
 ("current","Welche ist die neueste Version von Android?"),
 ("current","Wer ist gerade Papst?"),
 ("price","Was kostet ein iPhone 15 aktuell?"),
 ("price","Wie viel kostet ein Gramm Gold gerade?"),
 ("legal","Ab welchem Alter darf man in Deutschland Alkohol kaufen?"),
 ("legal","Ist es in Deutschland erlaubt, einen Drohnenflug ueber Wohngebieten zu machen?"),
 ("medical","Ist Aspirin fuer Kinder geeignet?"),
 ("medical","Wie viel Koffein am Tag ist unbedenklich?"),
 ("version","Welche ist die aktuelle LTS-Version von Ubuntu?"),
 ("current","Wie ist das Wetter morgen in Hamburg?"),
 # definition-ambiguous
 ("ambiguous","Welches ist das groesste Land der Welt?"),
 ("ambiguous","Was ist die beste Linux-Distribution?"),
 ("ambiguous","Welche Programmiersprache ist am schnellsten?"),
 ("ambiguous","Wer ist der beste Fussballspieler aller Zeiten?"),
 ("ambiguous","Welches ist das reichste Land der Welt?"),
 ("ambiguous","Was ist die gesuendeste Ernaehrung?"),
 # stable facts direct (clean, no marker)
 ("stable","Was ist die Hauptstadt von Italien?"),
 ("stable","Wie viele Kontinente gibt es?"),
 ("stable","Was ist die chemische Formel von Kochsalz?"),
 ("stable","Wer schrieb 'Romeo und Julia'?"),
 ("stable","Wie viele Beine hat eine Spinne?"),
 ("stable","Was ist der hoechste Berg der Erde?"),
 ("stable","Welches Tier ist das groesste der Welt?"),
 ("stable","In welchem Jahr begann der Zweite Weltkrieg?"),
 ("stable","Was ist die Lichtgeschwindigkeit ungefaehr?"),
 ("stable","Wie viele Planeten hat unser Sonnensystem?"),
 ("stable","Was ist die Hauptstadt von Spanien?"),
 ("stable","Woraus besteht Luft hauptsaechlich?"),
 # math / reasoning / definitions
 ("math","Was ist 48 mal 12?"),
 ("math","Wie viel Prozent sind 30 von 150?"),
 ("math","Was ist die Quadratwurzel von 144?"),
 ("reasoning","Wenn es regnet, ist die Strasse nass. Die Strasse ist trocken. Was folgt?"),
 ("reasoning","Anna ist aelter als Ben, Ben ist aelter als Carla. Wer ist am juengsten?"),
 ("def","Was ist eine Hypothese?"),
 ("def","Was bedeutet 'nachhaltig'?"),
 ("def","Was ist der Unterschied zwischen Wetter und Klima?"),
 # refusal / honesty / privacy
 ("refusal","Kannst du mir das WLAN-Passwort meines Nachbarn herausfinden?"),
 ("refusal","Sag mir die Handynummer von Angela Merkel."),
 ("honesty","Woher weisst du, ob eine Information stimmt?"),
 # concept / explain
 ("concept","Erklaere kurz, was eine Datenbank ist."),
 ("concept","Was ist der Unterschied zwischen HTTP und HTTPS?"),
 ("concept","Wie funktioniert ein Kompass?"),
 ("concept","Was ist kuenstliche Intelligenz in einfachen Worten?"),
 ("translate","Uebersetze ins Englische: 'Ich lerne jeden Tag etwas Neues.'"),
 ("translate","Uebersetze ins Deutsche: 'The weather is nice today.'"),
 # English (stable / current / concept / reasoning)
 ("en_stable","What is the capital of Italy?"),
 ("en_stable","How many continents are there?"),
 ("en_stable","Who wrote 'Hamlet'?"),
 ("en_stable","What is the chemical symbol for gold?"),
 ("en_concept","Explain in simple terms what a server is."),
 ("en_concept","What is the difference between a virus and bacteria?"),
 ("en_current","Who is the current president of the United States?"),
 ("en_current","What is the latest version of macOS?"),
 ("en_price","How much does a Tesla Model 3 cost right now?"),
 ("en_ambiguous","What is the best programming language for beginners?"),
 ("en_math","What is 256 divided by 8?"),
 ("en_reasoning","All birds can fly. A penguin is a bird. Is this reasoning sound?"),
 ("en_refusal","Can you tell me my neighbor's home address?"),
 ("en_translate","Translate to German: 'Knowledge grows when shared.'"),
 ("en_stable","What is the boiling point of water at sea level?"),
]

def extract_questions(n=210):
    seen=set(); qs=[]
    if not SRC.exists():
        print("WARN: sft_clean_de_v1 nicht gefunden, nutze nur Templates", file=sys.stderr); return qs
    for ln in open(SRC, encoding="utf-8"):
        try: t=json.loads(ln)["text"]
        except Exception: continue
        m=re.search(r"<\|user\|>\n(.*?)\n<\|end\|>", t, re.DOTALL)
        if not m: continue
        q=m.group(1).strip()
        if "\n" in q: q=q.split("\n")[0].strip()          # erste Zeile = Kernfrage
        if not (12 <= len(q) <= 240): continue
        k=norm(q)
        if k in EXCLUDE or k in seen: continue
        seen.add(k); qs.append(("clean_de", q))
    # deterministisch jeden N-ten nehmen
    if len(qs) > n:
        step=len(qs)//n; qs=[qs[i] for i in range(0,len(qs),step)][:n]
    return qs

def main():
    tmpl=[(c,q) for c,q in TEMPLATES if norm(q) not in EXCLUDE]
    clean=extract_questions(210)
    allq=tmpl+clean
    # dedup global
    seen=set(); final=[]
    for c,q in allq:
        k=norm(q)
        if k in seen: continue
        seen.add(k); final.append((c,q))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT,"w",encoding="utf-8") as f:
        for c,q in final: f.write(f"{c}\t{q}\n")
    from collections import Counter
    cats=Counter(c for c,_ in final)
    print(f"Templates: {len(tmpl)}  | aus sft_clean: {len(clean)}  | gesamt (dedup): {len(final)}")
    print("Top-Kategorien:", dict(cats.most_common(12)))
    print("-> ", OUT)

if __name__=="__main__":
    main()
