#!/usr/bin/env python3
"""Verified CODE-instruction data — LOCAL teacher + EXECUTOR ground-truth.

Third executor axis (after math=calculator, grounded=context): for code, the EXECUTOR
is py_compile + actually RUNNING the function against test cases. qwen3.6 proposes a
German coding task + a self-contained Python function + test cases; we compile and run
it, and keep ONLY solutions that compile AND pass every test. No hallucinated code gets
through — the interpreter is the judge.

Scope (v1): self-contained ALGORITHMIC functions (no I/O, no network, no imports beyond a
safe stdlib subset). Safety: a denylist rejects dangerous code BEFORE execution, and each
solution runs in a fresh subprocess with a hard timeout (benign teacher code, controlled box
-> adequate "Stufe 4 light"; a full nsjail would be needed for adversarial code).

Runs on the HOST (Ollama is local). Two-phase / incremental / resumable like the others.
"""

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "data"))
from tool_harness import TOOL_OPEN  # noqa (unused, keeps format module loaded)
from code_format import tab_indent  # pretrain code corpus is tab-indented -> SFT code must match

SYS = "Du bist Auralis, ein hilfreicher, ehrlicher KI-Assistent."
OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/generate"

# reject BEFORE executing — self-contained algorithmic tasks need none of these
DENY = [
    "import os",
    "import sys",
    "import subprocess",
    "import socket",
    "import shutil",
    "import requests",
    "import urllib",
    "import importlib",
    "from os",
    "from sys",
    "open(",
    "eval(",
    "exec(",
    "__import__",
    "input(",
    "globals(",
    "compile(",
    "os.",
    "sys.",
    "subprocess",
    "pathlib",
    "shutil",
    "remove(",
    "rmdir",
    "system(",
    "popen",
    "fork",
    "while True",
    "while 1",
]
ALLOW_IMPORT = re.compile(
    r"^\s*(import (math|re|itertools|functools|collections|heapq|bisect|string|typing)\b|from (math|typing|collections|itertools|functools) import )"
)


def helix(q, a):
    return f"<|system|>\n{SYS}\n<|end|>\n<|user|>\n{q}\n<|end|>\n<|assistant|>\n{a}\n<|end|>\n"


def ollama(model, prompt, n_predict=1400, temp=0.6, timeout=240):
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "keep_alive": "30m",
            "options": {"temperature": temp, "num_predict": n_predict},
        }
    ).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8")).get("response", "").strip()


def read_jsonl(p):
    p = pathlib.Path(p)
    if not p.exists():
        return []
    out = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def safe_code(code):
    """reject obviously-dangerous code; allow only a safe stdlib subset of imports."""
    low = code.lower()
    for d in DENY:
        if d in low:
            return False, f"denylist:{d}"
    for ln in code.splitlines():
        s = ln.strip()
        if s.startswith("import ") or s.startswith("from "):
            if not ALLOW_IMPORT.match(ln):
                return False, f"import not allowed: {s}"
    return True, "ok"


def run_tests(code, fname, tests, timeout=5):
    """execute the function against tests in a fresh subprocess; return (ok, detail)."""
    # embed tests as a JSON STRING parsed by json.loads inside the harness, so JSON
    # true/false/null become Python True/False/None (embedding the raw json.dumps as
    # source would NameError on booleans -> false-dropped every predicate function).
    harness = code + "\n\nimport json as _json\n_T=_json.loads(" + repr(json.dumps(tests)) + ")\n"
    harness += (
        "_ok=True\n"
        "for _t in _T:\n"
        "    try:\n"
        f"        _r={fname}(*_t['args'])\n"
        "    except Exception as _e:\n"
        "        print('ERR', type(_e).__name__); _ok=False; break\n"
        "    if _r != _t['erwartet']:\n"
        "        print('MISMATCH', _r, _t['erwartet']); _ok=False; break\n"
        "print('ALLPASS' if _ok else 'FAIL')\n"
    )
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "sol.py")
        with open(fp, "w", encoding="utf-8") as f:
            f.write(harness)
        try:
            r = subprocess.run(
                [sys.executable, fp], capture_output=True, text=True, timeout=timeout, cwd=td
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except Exception as e:
            return False, f"runerr:{type(e).__name__}"
    out = (r.stdout or "").strip().splitlines()
    return (bool(out) and out[-1] == "ALLPASS"), (out[-1] if out else f"rc={r.returncode}")


GEN_PROMPT = """Erzeuge {k} kleine PYTHON-PROGRAMMIERAUFGABEN mit Loesung und Testfaellen.
Jede Aufgabe: eine SELF-CONTAINED Funktion (reine Berechnung, KEINE Ein-/Ausgabe, KEIN Datei-/
Netzwerkzugriff, KEINE Imports ausser math/re/itertools/functools/collections/typing).
Schwerpunkt: {topic}.

Gib AUSSCHLIESSLICH {k} JSON-Zeilen aus (JSONL), je exakt dieses Schema:
{{"aufgabe":"deutsche Aufgabenstellung","funktion":"name","code":"def name(...):\\n    ...","erklaerung":"1-2 Saetze","tests":[{{"args":[...],"erwartet":...}}]}}

- "code": die komplette Funktionsdefinition als String (mit \\n fuer Zeilenumbrueche, 4 Leerzeichen Einrueckung).
- "tests": mindestens 3 Testfaelle; "args" ist die Argumentliste, "erwartet" der korrekte Rueckgabewert.
- Die Funktion MUSS fuer alle Testfaelle das korrekte Ergebnis liefern.

Beispiel:
{{"aufgabe":"Schreibe eine Funktion, die prueft ob eine Zahl eine Primzahl ist.","funktion":"ist_primzahl","code":"def ist_primzahl(n):\\n    if n < 2:\\n        return False\\n    for t in range(2, int(n**0.5)+1):\\n        if n % t == 0:\\n            return False\\n    return True","erklaerung":"Testet Teiler bis zur Wurzel von n.","tests":[{{"args":[7],"erwartet":true}},{{"args":[1],"erwartet":false}},{{"args":[9],"erwartet":false}}]}}

Jetzt {k} neue, verschiedene Aufgaben:"""

TOPICS = [
    "Zahlen/Arithmetik",
    "Strings/Text",
    "Listen/Sortieren",
    "Schleifen/Zaehlen",
    "Rekursion",
    "Mathe (ggT, Fakultaet, Fibonacci)",
    "Suchen/Filtern",
    "Dictionaries/Zaehlen",
]


def gen_phase(a, raw_path):
    seen = {r.get("aufgabe", "") for r in read_jsonl(raw_path)}
    print(f"[gen] {len(seen)} raw on disk, target {a.gen_target}", flush=True)
    fout = open(raw_path, "a", encoding="utf-8")
    call = 0
    while len(seen) < a.gen_target and call < a.max_calls:
        topic = TOPICS[call % len(TOPICS)]
        call += 1
        try:
            txt = ollama(a.teacher, GEN_PROMPT.format(k=a.per_call, topic=topic), temp=a.temp)
        except Exception as e:
            print(f"[gen] error: {e}", file=sys.stderr, flush=True)
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
                or not r.get("tests")
            ):
                continue
            seen.add(q)
            r["_topic"] = topic
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            fout.flush()
            added += 1
        print(f"[gen call {call}] +{added} -> {len(seen)}", flush=True)
    fout.close()
    print(f"[gen] done: {len(seen)} raw", flush=True)


def verify_phase(a, raw_path, out_path):
    raw = read_jsonl(raw_path)
    done = {r.get("meta", {}).get("q") for r in read_jsonl(out_path)}
    fout = open(out_path, "a", encoding="utf-8")
    st = dict(seen=0, safe=0, compile=0, pass_=0, kept=0)
    for r in raw:
        q = (r.get("aufgabe") or "").strip()
        if not q or q in done:
            continue
        done.add(q)
        st["seen"] += 1
        code = (
            (r.get("code") or "").replace("\\n", "\n")
            if "\\n" in r.get("code", "")
            else r.get("code", "")
        )
        fname = r.get("funktion", "").strip()
        tests = r.get("tests") or []
        ok, _ = safe_code(code)
        if not ok:
            continue
        st["safe"] += 1
        try:
            compile(code, "<sol>", "exec")
        except Exception:
            continue
        st["compile"] += 1
        passed, _ = run_tests(code, fname, tests)
        if not passed:
            continue
        st["pass_"] += 1
        erkl = (r.get("erklaerung") or "").strip()
        # verification ran on the ORIGINAL spaces version above; tab-indent only the rendered target
        answer = f"```python\n{tab_indent(code.strip())}\n```" + (f"\n\n{erkl}" if erkl else "")
        fout.write(
            json.dumps(
                {
                    "text": helix(q, answer),
                    "source": "code_verified",
                    "has_tool": False,
                    "meta": {"q": q, "funktion": fname},
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        fout.flush()
        st["kept"] += 1
        if st["kept"] % 20 == 0:
            print(f"  [verify] kept {st['kept']} (seen {st['seen']})", flush=True)
    fout.close()
    total = len(read_jsonl(out_path))
    print("\n=== CODE VERIFY DONE ===")
    for k in ["seen", "safe", "compile", "pass_", "kept"]:
        print(f"  {k:9} {st[k]}")
    if st["seen"]:
        print(
            f"  compile-rate {st['compile'] / st['seen']:.0%} | pass-rate {st['pass_'] / max(1, st['compile']):.0%} | overall kept {st['kept'] / st['seen']:.0%}"
        )
    print(f"  -> {out_path}  (TOTAL on disk: {total})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(HERE.parent.parent / "data/training/code_verified_v1"))
    ap.add_argument("--teacher", default="qwen3.6:27b")
    ap.add_argument("--target", type=int, default=200, help="verified solutions wanted")
    ap.add_argument("--gen-target", type=int, default=0)
    ap.add_argument("--per-call", type=int, default=6)
    ap.add_argument("--max-calls", type=int, default=80)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--phase", choices=["gen", "verify", "all"], default="all")
    a = ap.parse_args()
    if not a.gen_target:
        a.gen_target = int(a.target / 0.6) + 10  # code yield is lower than math
    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "raw_code.jsonl"
    out_path = out / "verified_code.jsonl"
    print(
        f"=== verified-code | teacher={a.teacher} phase={a.phase} target={a.target} gen_target={a.gen_target} ===",
        flush=True,
    )
    if a.phase in ("gen", "all"):
        gen_phase(a, raw_path)
    if a.phase in ("verify", "all"):
        verify_phase(a, raw_path, out_path)


if __name__ == "__main__":
    main()
