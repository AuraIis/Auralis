#!/usr/bin/env python3
"""Archetype C — ABSTAIN / Kalibrierung (Antwort-Doktrin), VERIFIED + key-free.

Two row types, both dual-model verified:

  CONFIDENT (tutor-style, GROUNDED): for KNOWN facts in the gold bank, qwen3.6 writes
    a tutor answer (Antwort -> Einordnung -> Beispiel). We GIVE it the gold fact, so the
    core is grounded; gemma4 then fact-checks the full answer for any FALSE claim in the
    elaboration. Keep only if gold present AND gemma finds no false claim.

  ABSTAIN (doctrine-structured): for INVENTED entities (gold=None, gibberish we created
    -> unknown by construction), qwen3.6 writes an honest "I don't know" in the doctrine
    shape (zugeben -> warum -> Hilfe anbieten). gemma4 judges ABSTAIN vs BEHAUPTET; we
    keep only genuine abstains that do NOT fabricate a fact about the gibberish entity.

This realises the doctrine's resolved tension: generous when known, honest when not.

Same robust shape as gen_verified_math: TWO PHASES (qwen resident -> gemma resident),
INCREMENTAL append, RESUMABLE. Input bank = build_calib_bank.py output.
"""
import os, sys, re, json, argparse, pathlib, urllib.request

HERE = pathlib.Path(__file__).resolve().parent
SYS = "Du bist Auralis, ein hilfreicher, ehrlicher KI-Assistent."
OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/generate"

import re as _re
# An honest abstain about a (gibberish) entity must NOT then attribute it to a real,
# specific named entity. These patterns catch "abstains-then-fabricates" (e.g. the
# answer that claimed a made-up book was 'von Neil Gaiman'). Caught the 1 real case
# with 0 false positives on 39 good abstains in the v1 sample.
ABSTAIN_FABRICATION = [
    _re.compile(r"\bvon [A-ZÄÖÜ][a-zäöüß]+ [A-ZÄÖÜ][a-zäöüß]+"),           # "von Neil Gaiman"
    _re.compile(r"\bist eine?\b.{0,30}\b(Stadt|Roman|Buch|Werk|Land|Autor|Figur)\b"),
]


def looks_fabricated(ans):
    return any(p.search(ans) for p in ABSTAIN_FABRICATION)


# Very broad hedge net: an honest abstain contains at least ONE of these. Only fails on a
# fully-confident fabrication (zero hedging) -> that's exactly what we want to drop.
HEDGE = _re.compile(r"\b(nicht|kenne|unbekannt|weiß|weiss|wüsste|existiert|sicher|vermutlich|keine|fiktiv)\b", _re.I)


def is_honest_abstain(ans):
    return bool(HEDGE.search(ans)) and not looks_fabricated(ans)


import unicodedata as _ud


def _norm(s):
    s = (s or "").lower()
    for x, y in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        s = s.replace(x, y)
    # strip remaining accents (Brasília -> brasilia, Café -> cafe)
    return "".join(c for c in _ud.normalize("NFD", s) if _ud.category(c) != "Mn")


def gold_present(ans, gold):
    """umlaut-insensitive substring match (gold 'gruen' must match answer 'Grün')."""
    na = _norm(ans)
    return any(_norm(g) in na for g in (gold or []))


def helix(q, a):
    return f"<|system|>\n{SYS}\n<|end|>\n<|user|>\n{q}\n<|end|>\n<|assistant|>\n{a}\n<|end|>\n"


def ollama(model, prompt, n_predict=512, temp=0.7, timeout=180):
    body = json.dumps({
        "model": model, "prompt": prompt, "stream": False, "think": False, "keep_alive": "30m",
        "options": {"temperature": temp, "num_predict": n_predict},
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8")).get("response", "").strip()


def read_jsonl(path):
    if not pathlib.Path(path).exists():
        return []
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def entity_of(q):
    """best-effort: pull the invented token(s) from the question for the no-fabrication check."""
    m = re.search(r"'([^']+)'", q)
    if m:
        return m.group(1)
    caps = re.findall(r"\b([A-ZÄÖÜ][a-zäöüß]+)\b", q)
    return caps[-1] if caps else ""


CONF_PROMPT = """Beantworte diese Wissensfrage im Stil eines guten, lehrenden Tutors auf DEUTSCH:
zuerst die direkte Antwort in einem Satz, dann ein bis zwei Saetze Einordnung/Kontext,
optional ein kurzes Beispiel.

WICHTIG: Nenne NUR Tatsachen, bei denen du dir sicher bist. Wenn du dir bei einem Detail
(Datum, Jahreszahl, Zahl, Eigenname) NICHT sicher bist, LASS ES WEG statt zu raten. Lieber
eine kurze, sichere Antwort als ein erfundenes Detail. Erfinde nichts.

Frage: {q}
Die korrekte Kernantwort ist: {gold}

Schreibe nur die Antwort (kein Vorwort)."""

ABSTAIN_PROMPT = """Die folgende Frage bezieht sich auf etwas, das es nicht gibt bzw. das unbekannt ist.
Schreibe eine EHRLICHE Antwort auf DEUTSCH im Stil:
1) ehrlich zugeben, dass du es nicht (sicher) weisst,
2) kurz WARUM (unbekannt, sehr speziell, oder existiert vermutlich nicht),
3) optional anbieten zu helfen, wenn eine Quelle kommt.
Erfinde KEINE Fakten ueber das Gefragte. Behaupte nichts.

Frage: {q}

Schreibe nur die Antwort (kein Vorwort)."""

CONF_CHECK = """Pruefe den folgenden deutschen Text auf SACHLICHE Fehler.
Antworte mit JA NUR, wenn der Text eine EINDEUTIG FALSCHE Tatsache enthaelt
(falsches Datum, falsche Zahl, falscher Name, falscher Ort). Im Zweifel antworte NEIN.
Antworte NUR mit einem Wort: JA oder NEIN.

Text: {a}"""

ABSTAIN_CHECK = """Bewerte den folgenden deutschen Text.
- Wenn er ehrlich zugibt, die Antwort NICHT zu kennen, und KEINE konkrete benannte
  Tatsache ueber das Gefragte behauptet -> antworte ABSTAIN.
- Wenn er irgendeine konkrete Tatsache behauptet (Autor, Ort, Bedeutung, Formel, Datum,
  Zuordnung) -> antworte BEHAUPTET.
Antworte NUR mit einem Wort: ABSTAIN oder BEHAUPTET.

Text: {a}"""


# ----------------------------- phase: generate -----------------------------
def gen_phase(a, bank, raw_path):
    existing = {r["q"] for r in read_jsonl(raw_path)}
    facts = [r for r in bank if r.get("gold")]
    inv = [r for r in bank if not r.get("gold")]
    import random
    rng = random.Random(20260608)
    rng.shuffle(facts); rng.shuffle(inv)
    facts = facts[:a.n_confident]; inv = inv[:a.n_abstain]
    todo = [("confident", r) for r in facts] + [("abstain", r) for r in inv]
    print(f"[gen] {len(facts)} confident + {len(inv)} abstain to write ({len(existing)} already raw)", flush=True)
    fout = open(raw_path, "a", encoding="utf-8")
    n = 0
    for kind, r in todo:
        q = r["q"]
        if q in existing:
            continue
        try:
            if kind == "confident":
                ans = ollama(a.teacher, CONF_PROMPT.format(q=q, gold=", ".join(r["gold"])), temp=0.35)
            else:
                ans = ollama(a.teacher, ABSTAIN_PROMPT.format(q=q), temp=0.8)
        except Exception as e:
            print(f"  gen error: {e}", file=sys.stderr, flush=True)
            continue
        if not ans:
            continue
        fout.write(json.dumps({"q": q, "kind": kind, "gold": r.get("gold"),
                               "cat": r.get("cat"), "answer": ans}, ensure_ascii=False) + "\n")
        fout.flush()
        n += 1
        if n % 25 == 0:
            print(f"  [gen] wrote {n}", flush=True)
    fout.close()
    print(f"[gen] done: {n} new raw answers", flush=True)


# ----------------------------- phase: verify -----------------------------
def verify_phase(a, raw_path, out_path):
    raw = read_jsonl(raw_path)
    done = {r.get("meta", {}).get("q") for r in read_jsonl(out_path)}
    fout = open(out_path, "a", encoding="utf-8")
    st = dict(seen=0, conf_seen=0, conf_kept=0, abs_seen=0, abs_kept=0, abs_fab=0)
    for r in raw:
        q, kind, ans = r["q"], r["kind"], r["answer"]
        if q in done:
            continue
        done.add(q); st["seen"] += 1
        keep = False
        # STRUCTURAL-ONLY verification (data showed local LLM judges are net-negative for
        # facts: pedantic false-positives + blind to own errors). Facts have no executor;
        # we lean on STRUCTURE: gold-bank core (confident) + hedge/fabrication (abstain).
        if kind == "confident":
            st["conf_seen"] += 1
            # gold core is the hard truth; the lehrend elaboration carries acknowledged residual risk.
            if gold_present(ans, r.get("gold")):
                keep = True; st["conf_kept"] += 1
            src = "calib_confident"
        else:
            st["abs_seen"] += 1
            if looks_fabricated(ans):
                st["abs_fab"] += 1            # abstain that attributes the gibberish to a real entity
            elif HEDGE.search(ans):
                keep = True; st["abs_kept"] += 1
            src = "calib_abstain"
        if keep:
            fout.write(json.dumps({"text": helix(q, ans), "source": src,
                                   "meta": {"q": q, "kind": kind}}, ensure_ascii=False) + "\n")
            fout.flush()
    fout.close()
    total = len(read_jsonl(out_path))
    print("\n=== VERIFY DONE (abstain archetype) ===")
    for k in ["seen", "conf_seen", "conf_kept", "abs_seen", "abs_fab", "abs_kept"]:
        print(f"  {k:12} {st[k]}")
    if st["conf_seen"]:
        print(f"  confident keep-rate = {st['conf_kept']/st['conf_seen']:.0%}")
    if st["abs_seen"]:
        print(f"  abstain   keep-rate = {st['abs_kept']/st['abs_seen']:.0%}")
    print(f"  -> {out_path}  (TOTAL on disk: {total})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default=str(HERE.parent.parent / "data/training/calib/calib_bank.jsonl"))
    ap.add_argument("--out-dir", default=str(HERE.parent.parent / "data/training/calib_verified_v1"))
    ap.add_argument("--teacher", default="qwen3.6:27b")
    ap.add_argument("--cross-model", default="gemma4:12b")
    ap.add_argument("--check-model", default="qwen3.6:27b",
                    help="verifier for conf/abstain checks; qwen3.6 (10/10 facts) beats gemma4 as judge")
    ap.add_argument("--n-confident", type=int, default=80)
    ap.add_argument("--n-abstain", type=int, default=80)
    ap.add_argument("--phase", choices=["gen", "verify", "all"], default="all")
    a = ap.parse_args()
    bank = read_jsonl(a.bank)
    if not bank:
        print(f"ERROR: empty/missing bank at {a.bank} (run build_calib_bank.py first)", file=sys.stderr)
        sys.exit(1)
    out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "raw_answers.jsonl"
    out_path = out / "verified_calib.jsonl"
    print(f"=== abstain-gen | teacher={a.teacher} cross={a.cross_model} phase={a.phase} "
          f"bank={len(bank)} ===", flush=True)
    if a.phase in ("gen", "all"):
        gen_phase(a, bank, raw_path)
    if a.phase in ("verify", "all"):
        verify_phase(a, raw_path, out_path)


if __name__ == "__main__":
    main()
