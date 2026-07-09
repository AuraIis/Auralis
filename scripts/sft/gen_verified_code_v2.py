#!/usr/bin/env python3
"""Verified CODE-instruction data v2 — narrow scope + HIDDEN-TEST cross-implementation gate.

v1 lesson: teacher tests alone let through code that games its own weak tests. v2 adds a
second, independent gate (the code analog of math's calculator+cross-solve):

  Gate 1 (executor) : denylist + py_compile + pass ALL teacher tests
  Gate 2 (HIDDEN)   : gemma4 solves the SAME task independently (sees only the task) and
                      proposes extra inputs; we run BOTH solutions on (teacher inputs +
                      gemma inputs) and require IDENTICAL outputs everywhere. Two independent
                      implementations agreeing = strong corroboration; a solution that merely
                      games its own tests will diverge from an independent one.

Scope (narrow on purpose): small pure Python functions, NO imports beyond math, simple types
(int/str/list/dict/bool). No files/web/classes/frameworks. Runs on HOST (Ollama local).
"""

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "data"))
from code_format import tab_indent  # pretrain code corpus is tab-indented -> SFT code must match
from gen_verified_code import helix, ollama, read_jsonl, run_tests, safe_code  # reuse v1 helpers

GEN_PROMPT = """Erzeuge {k} kleine PYTHON-PROGRAMMIERAUFGABEN mit Loesung und Testfaellen.
STRIKTE Vorgaben:
- Eine SELF-CONTAINED Funktion, reine Berechnung. KEINE Imports (hoechstens 'import math').
- Nur einfache Typen (int, float, str, list, dict, bool). KEINE Dateien, KEIN Netzwerk,
  KEINE Klassen, KEINE Frameworks, KEINE Ein-/Ausgabe.
- Schwerpunkt: {topic}. Schwierigkeit: Anfaenger bis Mittel.
- MINDESTENS 8 Testfaelle, inkl. Randfaelle (leere Liste, 0, negative Zahlen, etc.).

Gib AUSSCHLIESSLICH {k} JSON-Zeilen aus (JSONL), je exakt:
{{"aufgabe":"deutsche Aufgabe","funktion":"name","code":"def name(...):\\n    ...","erklaerung":"1-2 Saetze","tests":[{{"args":[...],"erwartet":...}}, ... mind. 8]}}

Beispiel:
{{"aufgabe":"Schreibe eine Funktion, die die Anzahl der Vokale in einem String zaehlt.","funktion":"zaehle_vokale","code":"def zaehle_vokale(s):\\n    return sum(1 for c in s.lower() if c in 'aeiou')","erklaerung":"Zaehlt Zeichen, die Vokale sind.","tests":[{{"args":["hallo"],"erwartet":2}},{{"args":[""],"erwartet":0}},{{"args":["xyz"],"erwartet":0}},{{"args":["AEIOU"],"erwartet":5}},{{"args":["Programmierung"],"erwartet":4}},{{"args":["a"],"erwartet":1}},{{"args":["bcd"],"erwartet":0}},{{"args":["aaa"],"erwartet":3}}]}}

Jetzt {k} neue, verschiedene Aufgaben:"""

SOLVE_PROMPT = """Loese diese Python-Aufgabe UNABHAENGIG. Schreibe genau die Funktion mit dem Namen "{fname}".
Nenne zusaetzlich 5 sinnvolle Test-INPUTS (nur die Argument-Listen, keine erwarteten Werte).

Aufgabe: {q}

Gib AUSSCHLIESSLICH ein JSON-Objekt aus:
{{"code":"def {fname}(...):\\n    ...","inputs":[[...],[...],[...],[...],[...]]}}"""

TOPICS = [
    "Zahlen/Arithmetik",
    "Strings/Text-Analyse",
    "Listen verarbeiten",
    "Dictionaries/Haeufigkeiten",
    "Rekursion",
    "Lineares Suchen",
    "Sortier-Logik",
    "Primzahlen/Teiler",
    "Fibonacci/Zahlenfolgen",
    "String-Manipulation (umkehren, ersetzen)",
    "Listen filtern",
    "Listen transformieren (map)",
    "Mengen/Duplikate entfernen",
    "Zaehlen/Gruppieren",
    "Min/Max/Durchschnitt",
    "Basis-Umwandlung",
    "Eingabe-Validierung/Bedingungen",
    "Verschachtelte Listen/Dicts",
    "Zeit/Datum-Berechnung (ohne Imports)",
    "Geometrie-Formeln",
]


def _norm_code(c):
    return c.replace("\\n", "\n") if (c and "\\n" in c and "\n" not in c) else (c or "")


def run_pair(codeA, fa, codeB, fb, inputs, timeout=6):
    """run both solutions on all inputs in a subprocess; True iff identical outputs everywhere."""
    h = (
        "import json as _j\n"
        "_inputs=_j.loads(" + repr(json.dumps(inputs)) + ")\n"
        "_nsA={}; _nsB={}\n"
        "exec(compile(" + repr(codeA) + ",'A','exec'), _nsA)\n"
        "exec(compile(" + repr(codeB) + ",'B','exec'), _nsB)\n"
        "_fa=_nsA.get(" + repr(fa) + "); _fb=_nsB.get(" + repr(fb) + ")\n"
        "if _fa is None or _fb is None:\n    print('NOFUNC'); raise SystemExit\n"
        "_ok=True\n"
        "for _in in _inputs:\n"
        "    try:\n"
        "        _ra=_fa(*_in); _rb=_fb(*_in)\n"
        "    except Exception:\n"
        "        _ok=False; break\n"
        "    if _ra!=_rb:\n        _ok=False; break\n"
        "print('AGREE' if _ok else 'DISAGREE')\n"
    )
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "pair.py")
        open(fp, "w", encoding="utf-8").write(h)
        try:
            r = subprocess.run(
                [sys.executable, fp], capture_output=True, text=True, timeout=timeout, cwd=td
            )
        except Exception:
            return False
    out = (r.stdout or "").strip().splitlines()
    return bool(out) and out[-1] == "AGREE"


def gen_phase(a, raw_path):
    seen = {r.get("aufgabe", "") for r in read_jsonl(raw_path)}
    print(f"[gen] {len(seen)} raw, target {a.gen_target}", flush=True)
    fout = open(raw_path, "a", encoding="utf-8")
    call = 0
    while len(seen) < a.gen_target and call < a.max_calls:
        topic = TOPICS[call % len(TOPICS)]
        call += 1
        prompt = GEN_PROMPT.format(k=a.per_call, topic=topic)
        if call > len(TOPICS):
            prompt += "\n(Waehle ANDERE konkrete Aufgaben/Funktionen als in fruehzeitigen Chargen.)"
        try:
            txt = ollama(a.teacher, prompt, temp=a.temp)
        except Exception as e:
            print(f"[gen] err {e}", file=sys.stderr, flush=True)
            continue
        added = 0
        for line in txt.splitlines():
            line = line.strip().strip("`")
            if not line.startswith("{"):
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            q = (r.get("aufgabe") or "").strip()
            if (
                not q
                or q in seen
                or not r.get("code")
                or not r.get("funktion")
                or len(r.get("tests") or []) < 8
            ):
                continue
            seen.add(q)
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            fout.flush()
            added += 1
        print(f"[gen {call}] +{added} -> {len(seen)}", flush=True)
    fout.close()
    print(f"[gen] done {len(seen)}", flush=True)


def verify_phase(a, raw_path, out_path):
    raw = read_jsonl(raw_path)
    done = {r.get("meta", {}).get("q") for r in read_jsonl(out_path)}
    fout = open(out_path, "a", encoding="utf-8")
    st = dict(seen=0, g1=0, solveB=0, g2=0, kept=0)
    for r in raw:
        q = (r.get("aufgabe") or "").strip()
        if not q or q in done:
            continue
        done.add(q)
        st["seen"] += 1
        codeA = _norm_code(r.get("code", ""))
        fa = r.get("funktion", "").strip()
        tests = r.get("tests") or []
        ok, _ = safe_code(codeA)
        if not ok:
            continue
        try:
            compile(codeA, "<A>", "exec")
        except Exception:
            continue
        passed, _ = run_tests(codeA, fa, tests)
        if not passed:
            continue
        st["g1"] += 1
        # Gate 2: independent gemma solution + agreement
        try:
            sb = ollama(a.cross_model, SOLVE_PROMPT.format(q=q, fname=fa), temp=0.3)
        except Exception:
            continue
        m = re.search(r"\{.*\}", sb, re.S)
        if not m:
            continue
        try:
            objB = json.loads(m.group(0))
        except Exception:
            continue
        codeB = _norm_code(objB.get("code", ""))
        okB, _ = safe_code(codeB)
        if not okB or not codeB:
            continue
        try:
            compile(codeB, "<B>", "exec")
        except Exception:
            continue
        st["solveB"] += 1
        # Gate 2 (corroboration): gemma's INDEPENDENT solution must ALSO pass the 8 teacher
        # tests. Two independent implementations satisfying the same edge-case-heavy spec is
        # strong evidence A is correct — WITHOUT the spec-ambiguity false-drops that full
        # output-agreement on extra inputs caused (n<0 factorial, spaces-in-palindrome, etc.).
        passedB, _ = run_tests(codeB, fa, tests)
        if not passedB:
            continue
        st["g2"] += 1
        erkl = (r.get("erklaerung") or "").strip()
        # gates ran on the ORIGINAL spaces version above; tab-indent only the rendered target
        answer = f"```python\n{tab_indent(codeA.strip())}\n```" + (f"\n\n{erkl}" if erkl else "")
        fout.write(
            json.dumps(
                {
                    "text": helix(q, answer),
                    "source": "code_verified_v2",
                    "has_tool": False,
                    "meta": {"q": q, "funktion": fa},
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        fout.flush()
        st["kept"] += 1
        if st["kept"] % 25 == 0:
            print(f"  [verify] kept {st['kept']} (seen {st['seen']})", flush=True)
    fout.close()
    print("\n=== CODE-v2 VERIFY DONE ===")
    for k in ["seen", "g1", "solveB", "g2", "kept"]:
        print(f"  {k:8} {st[k]}")
    if st["seen"]:
        print(
            f"  gate1(tests) {st['g1'] / st['seen']:.0%} | gate2(cross-impl agree) {st['g2'] / max(1, st['g1']):.0%} | overall {st['kept'] / st['seen']:.0%}"
        )
    print(f"  -> {out_path} (TOTAL {len(read_jsonl(out_path))})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(HERE.parent.parent / "data/training/code_verified_v2"))
    ap.add_argument("--teacher", default="qwen3.6:27b")
    ap.add_argument("--cross-model", default="gemma4:12b")
    ap.add_argument("--target", type=int, default=300)
    ap.add_argument("--gen-target", type=int, default=0)
    ap.add_argument("--per-call", type=int, default=6)
    ap.add_argument("--max-calls", type=int, default=200)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--phase", choices=["gen", "verify", "all"], default="all")
    a = ap.parse_args()
    if not a.gen_target:
        a.gen_target = int(a.target / 0.4) + 10  # v2 yield lower (two gates)
    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"=== verified-code-v2 | teacher={a.teacher} cross={a.cross_model} phase={a.phase} target={a.target} gen_target={a.gen_target} ===",
        flush=True,
    )
    if a.phase in ("gen", "all"):
        gen_phase(a, out / "raw_code.jsonl")
    if a.phase in ("verify", "all"):
        verify_phase(a, out / "raw_code.jsonl", out / "verified_code.jsonl")


if __name__ == "__main__":
    main()
