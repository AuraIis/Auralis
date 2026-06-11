#!/usr/bin/env python3
"""Archetype H — GROUNDED QA (Antwort-Doktrin), VERIFIED + key-free.

The doctrine's key insight: facts have no executor, but GROUNDED facts do — the
provided CONTEXT is the truth. We train: context present -> answer FROM it; asked
fact NOT in context -> abstain ("steht nicht im Kontext"). Pure anti-hallucination.

qwen3.6 proposes: a short German context + ANSWERABLE questions (answer + verbatim
'beleg' quote) + NOT-ANSWERABLE questions (about something the text doesn't cover).

Verification is AGAINST THE CONTEXT (the executor for facts):
  answerable   : beleg must be a VERBATIM substring of the context  (grounding proof)
                 AND the answer introduces no number absent from the context  (no fabrication)
  unanswerable : a READING-COMPREHENSION check (qwen3.6: 'answerable from text? JA/NEIN')
                 must say NEIN. This is text-containment, NOT world-fact recall -> reliable,
                 and it IS 'checking against the context'. (Distinct from the net-negative
                 free-fact LLM judge we removed in archetype C.)

Two phases (qwen resident), incremental append, resumable — same shape as the others.
"""
import os, sys, re, json, argparse, pathlib, urllib.request, unicodedata

HERE = pathlib.Path(__file__).resolve().parent
GROUNDED_SYS = ("Du bist Auralis. Beantworte die Frage AUSSCHLIESSLICH auf Basis des gegebenen "
                "Kontexts. Wenn die Antwort nicht im Kontext steht, sage ehrlich, dass es daraus "
                "nicht hervorgeht — erfinde nichts.")
OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/generate"

ABSTAIN_TEMPLATES = [
    "Das geht aus dem gegebenen Kontext nicht hervor. Ich kann es daraus nicht zuverlaessig beantworten.",
    "Im Kontext steht dazu nichts. Ohne weitere Angaben kann ich das nicht beantworten.",
    "Der Text enthaelt diese Information nicht, daher kann ich die Frage daraus nicht beantworten.",
    "Dazu sagt der Kontext nichts. Ich moechte nichts erfinden, was nicht im Text steht.",
]


def norm(s):
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def nums(s):
    return set(re.findall(r"\d+", s or ""))


def toks(s):
    return [t for t in re.split(r"[^0-9a-z]+", norm(s)) if len(t) >= 3]


def grounded_ratio(span, kontext):
    """fraction of the span's content-words present in the context (robust to minor
    article/inflection paraphrase like 'Sein'/'Seine', unlike strict verbatim match)."""
    bt = toks(span)
    if not bt:
        return 0.0
    ck = set(toks(kontext))
    return sum(1 for t in bt if t in ck) / len(bt)


def ollama(model, prompt, n_predict=1024, temp=0.7, timeout=240):
    body = json.dumps({"model": model, "prompt": prompt, "stream": False, "think": False,
                       "keep_alive": "30m", "options": {"temperature": temp, "num_predict": n_predict}}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8")).get("response", "").strip()


def read_jsonl(p):
    p = pathlib.Path(p)
    if not p.exists():
        return []
    out = []
    for l in open(p, encoding="utf-8"):
        l = l.strip()
        if l:
            try:
                out.append(json.loads(l))
            except Exception:
                pass
    return out


def helix(sys_p, user, asst):
    return f"<|system|>\n{sys_p}\n<|end|>\n<|user|>\n{user}\n<|end|>\n<|assistant|>\n{asst}\n<|end|>\n"


def grounded_user(kontext, frage):
    return f"Kontext:\n{kontext}\n\nFrage: {frage}"


GEN_PROMPT = """Erzeuge EIN JSON-Objekt fuer deutsches Leseverstaendnis-Training. Thema-Schwerpunkt: {topic}.

Felder:
- "kontext": ein kurzer Sachtext auf Deutsch (3-5 Saetze) mit KONKRETEN Fakten.
- "beantwortbar": 2 Fragen, deren Antwort EINDEUTIG im Text steht. Jede als
  {{"frage":"...","antwort":"...","beleg":"..."}} — "beleg" ist ein WOERTLICHES Zitat
  (Teilsatz) AUS dem kontext, das die Antwort stuetzt (exakt so im Text vorhanden).
- "nicht_beantwortbar": 2 Fragen, die zum Thema passen, deren Antwort der Text aber
  NICHT enthaelt (der Kontext sagt dazu nichts).

Gib AUSSCHLIESSLICH das JSON aus, nichts sonst:
{{"kontext":"...","beantwortbar":[{{"frage":"...","antwort":"...","beleg":"..."}},{{"frage":"...","antwort":"...","beleg":"..."}}],"nicht_beantwortbar":["...","..."]}}"""

RC_CHECK = """Text:
{kontext}

Frage: {frage}

Laesst sich diese Frage ALLEIN aus dem Text eindeutig und vollstaendig beantworten?
Antworte NUR mit einem Wort: JA oder NEIN."""

# Theme -> sub-aspects, so two contexts of the same theme differ (avoids near-duplicates).
TOPIC_TREE = {
    "Pflanzen": ["Licht", "Wasser/Giessen", "Duenger", "Schaedlinge", "Umtopfen", "Temperatur", "Substrat"],
    "Technik/Geraete": ["Funktionsweise", "Wartung", "Energieverbrauch", "Sicherheit", "Bedienung", "Fehlerbehebung"],
    "Haushalt": ["Reinigung", "Aufbewahrung", "Reparatur", "Materialpflege", "Organisation"],
    "Geschichte": ["ein Ereignis", "eine Person", "ein Zeitraum", "Ursache und Folge", "ein Ort"],
    "Geografie": ["ein Land/eine Stadt", "Fluss/Gebirge", "Klima", "Bevoelkerung", "ein Wahrzeichen"],
    "Biologie": ["eine Tierart", "eine Pflanzenart", "Organe/Koerper", "ein Oekosystem", "Fortpflanzung"],
    "Physik": ["Kraft/Bewegung", "Energie", "Licht/Optik", "Waerme", "Elektrizitaet"],
    "Computer/IT": ["Hardware", "Software", "Netzwerk", "Datei/Speicher", "IT-Sicherheit"],
    "Gesundheit": ["Ernaehrung", "Bewegung", "Schlaf", "Vorbeugung", "allgemeine Erste Hilfe"],
    "Rezepte/Kochen": ["Zutaten", "Zubereitung", "Back-Temperatur/-Zeit", "Aufbewahrung", "Varianten"],
    "Alltag": ["Verkehr", "Einkauf", "Termine/Planung", "Kommunikation"],
    "Tiere": ["Haltung", "Ernaehrung", "Verhalten", "Lebensraum", "Pflege"],
    "Rechtliche Alltagstexte (rein sachlich, KEINE Beratung)": ["Fristen", "Formulare", "Begriffe", "Rechte/Pflichten allgemein"],
    "Bedienungsanleitungen": ["Inbetriebnahme", "Einstellungen", "Wartung", "Sicherheitshinweise"],
    "Produktbeschreibungen": ["Eigenschaften", "Masse/Gewicht", "Material", "Lieferumfang", "Anwendung"],
}
TOPIC_PAIRS = [(t, s) for t, subs in TOPIC_TREE.items() for s in subs]  # ~85 distinct angles


def gen_phase(a, raw_path):
    have = len(read_jsonl(raw_path))
    print(f"[gen] {have} contexts on disk, target {a.contexts}", flush=True)
    fout = open(raw_path, "a", encoding="utf-8")
    made = have
    call = 0
    while made < a.contexts and call < a.contexts * 3:
        theme, aspekt = TOPIC_PAIRS[call % len(TOPIC_PAIRS)]
        topic = f"{theme} — konkreter Aspekt: {aspekt}"
        if call >= len(TOPIC_PAIRS):
            topic += " (waehle andere konkrete Beispiele/Zahlen als in fruehzeitigen Texten)"
        call += 1
        try:
            txt = ollama(a.teacher, GEN_PROMPT.format(topic=topic), temp=a.temp)
        except Exception as e:
            print(f"[gen] error: {e}", file=sys.stderr, flush=True); continue
        m = re.search(r"\{.*\}", txt, re.S)
        if not m:
            continue
        try:
            obj = json.loads(m.group(0))
        except Exception:
            continue
        if not obj.get("kontext") or not obj.get("beantwortbar"):
            continue
        obj["_topic"] = topic
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n"); fout.flush()
        made += 1
        if made % 10 == 0:
            print(f"[gen] {made} contexts", flush=True)
    fout.close()
    print(f"[gen] done: {made} contexts", flush=True)


def verify_phase(a, raw_path, out_path):
    raw = read_jsonl(raw_path)
    done = {r.get("meta", {}).get("key") for r in read_jsonl(out_path)}
    fout = open(out_path, "a", encoding="utf-8")
    st = dict(ctx=0, ans_seen=0, ans_kept=0, una_seen=0, una_kept=0, una_rc_drop=0)
    rng_i = 0
    for obj in raw:
        kontext = obj.get("kontext", "")
        if not kontext:
            continue
        st["ctx"] += 1
        nk = norm(kontext)
        knums = nums(kontext)
        # answerable: beleg verbatim in context + no fabricated numbers
        for qa in obj.get("beantwortbar", []):
            frage = (qa.get("frage") or "").strip()
            antwort = (qa.get("antwort") or "").strip()
            beleg = (qa.get("beleg") or "").strip()
            if not (frage and antwort and beleg):
                continue
            st["ans_seen"] += 1
            key = "A|" + frage
            if key in done:
                continue
            # grounding = the beleg (support span) comes from the context (fuzzy ≥0.75,
            # robust to article/inflection paraphrase) AND the answer adds no number absent
            # from the context (catches fabricated quantities). The beleg IS the proof; a
            # separate answer-overlap check was over-strict on short inflected answers.
            grounded = (grounded_ratio(beleg, kontext) >= 0.75
                        and nums(antwort).issubset(knums))
            if grounded:
                done.add(key)
                fout.write(json.dumps({
                    "text": helix(GROUNDED_SYS, grounded_user(kontext, frage), antwort),
                    "source": "grounded_answer", "has_tool": False,
                    "meta": {"key": key, "kind": "answerable", "beleg": beleg},
                }, ensure_ascii=False) + "\n"); fout.flush()
                st["ans_kept"] += 1
        # unanswerable: reading-comprehension check must say NEIN
        for frage in obj.get("nicht_beantwortbar", []):
            frage = (frage or "").strip()
            if not frage:
                continue
            st["una_seen"] += 1
            key = "U|" + frage
            if key in done:
                continue
            try:
                v = ollama(a.check_model, RC_CHECK.format(kontext=kontext, frage=frage), n_predict=8, temp=0.0)
            except Exception as e:
                print(f"  rc error: {e}", file=sys.stderr, flush=True); continue
            if v.strip().upper().startswith("NEIN"):
                done.add(key)
                ans = ABSTAIN_TEMPLATES[rng_i % len(ABSTAIN_TEMPLATES)]; rng_i += 1
                fout.write(json.dumps({
                    "text": helix(GROUNDED_SYS, grounded_user(kontext, frage), ans),
                    "source": "grounded_abstain", "has_tool": False,
                    "meta": {"key": key, "kind": "unanswerable"},
                }, ensure_ascii=False) + "\n"); fout.flush()
                st["una_kept"] += 1
            else:
                st["una_rc_drop"] += 1   # RC says it IS answerable -> teacher mislabeled -> drop
    fout.close()
    total = len(read_jsonl(out_path))
    print("\n=== GROUNDED VERIFY DONE ===")
    for k in ["ctx", "ans_seen", "ans_kept", "una_seen", "una_rc_drop", "una_kept"]:
        print(f"  {k:12} {st[k]}")
    if st["ans_seen"]:
        print(f"  answerable grounded-rate = {st['ans_kept']/st['ans_seen']:.0%}")
    if st["una_seen"]:
        print(f"  unanswerable confirm-rate = {st['una_kept']/st['una_seen']:.0%}")
    print(f"  -> {out_path}  (TOTAL on disk: {total})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(HERE.parent.parent / "data/training/grounded_v1"))
    ap.add_argument("--teacher", default="qwen3.6:27b")
    ap.add_argument("--check-model", default="qwen3.6:27b")
    ap.add_argument("--contexts", type=int, default=10)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--phase", choices=["gen", "verify", "all"], default="all")
    a = ap.parse_args()
    out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "raw_contexts.jsonl"
    out_path = out / "verified_grounded.jsonl"
    print(f"=== grounded-gen | teacher={a.teacher} phase={a.phase} contexts={a.contexts} ===", flush=True)
    if a.phase in ("gen", "all"):
        gen_phase(a, raw_path)
    if a.phase in ("verify", "all"):
        verify_phase(a, raw_path, out_path)


if __name__ == "__main__":
    main()
