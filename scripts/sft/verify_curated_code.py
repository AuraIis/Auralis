#!/usr/bin/env python3
"""Verify an EXTERNAL curated code dataset through an executor gate (compile + run vs tests).
Schema: {task, function_name, solution, tests:[{input, expected}], explanation, ...}.
'curated' != 'verified' -> keep only solutions that compile AND pass all their tests.
Self-contained (no imports of sibling modules, to avoid UNC path issues on the host)."""
import os, re, sys, json, argparse, pathlib, tempfile, subprocess

SYS = "Du bist Auralis, ein hilfreicher, ehrlicher KI-Assistent."
DENY = ["import os", "import sys", "import subprocess", "import socket", "import shutil",
        "import requests", "import urllib", "import importlib", "from os", "from sys",
        "open(", "eval(", "exec(", "__import__", "input(", "globals(", "compile(",
        "os.", "sys.", "subprocess", "shutil", "system(", "popen", "fork", "while true", "while 1"]
ALLOW_IMPORT = re.compile(r"^\s*(import (math|re|itertools|functools|collections|heapq|bisect|string|typing)\b|from (math|typing|collections|itertools|functools) import )")


def helix(q, a):
    return f"<|system|>\n{SYS}\n<|end|>\n<|user|>\n{q}\n<|end|>\n<|assistant|>\n{a}\n<|end|>\n"


def safe_code(code):
    low = code.lower()
    for d in DENY:
        if d in low:
            return False, d
    for ln in code.splitlines():
        s = ln.strip()
        if s.startswith("import ") or s.startswith("from "):
            if not ALLOW_IMPORT.match(ln):
                return False, "import"
    return True, "ok"


def run_tests(code, fname, tests, timeout=6):
    harness = code + "\n\nimport json as _j\n_T=_j.loads(" + repr(json.dumps(tests)) + ")\n" + (
        "_ok=True\n"
        "for _t in _T:\n"
        "    try:\n"
        f"        _r={fname}(*_t['args'])\n"
        "    except Exception:\n        print('ERR'); _ok=False; break\n"
        "    if _r != _t['erwartet']:\n        print('MISMATCH'); _ok=False; break\n"
        "print('ALLPASS' if _ok else 'FAIL')\n")
    with tempfile.TemporaryDirectory() as td:
        fp = os.path.join(td, "s.py")
        open(fp, "w", encoding="utf-8").write(harness)
        try:
            r = subprocess.run([sys.executable, fp], capture_output=True, text=True, timeout=timeout, cwd=td)
        except Exception:
            return False
    out = (r.stdout or "").strip().splitlines()
    return bool(out) and out[-1] == "ALLPASS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.src, encoding="utf-8") if l.strip()]
    st = dict(seen=0, fields=0, safe=0, compile=0, passed=0)
    out = []
    for r in rows:
        st["seen"] += 1
        task = (r.get("task") or "").strip(); code = (r.get("solution") or "").strip(); fn = (r.get("function_name") or "").strip()
        tests = [{"args": t.get("input"), "erwartet": t.get("expected")}
                 for t in (r.get("tests") or []) if "input" in t and "expected" in t]
        if not (task and code and fn and tests):
            continue
        st["fields"] += 1
        ok, _ = safe_code(code)
        if not ok:
            continue
        st["safe"] += 1
        try:
            compile(code, "<s>", "exec")
        except Exception:
            continue
        st["compile"] += 1
        if not run_tests(code, fn, tests):
            continue
        st["passed"] += 1
        expl = (r.get("explanation") or "").strip()
        answer = f"```python\n{code}\n```" + (f"\n\n{expl}" if expl else "")
        out.append({"text": helix(task, answer), "source": "code_curated_verified",
                    "meta": {"q": task, "funktion": fn, "ntests": len(tests), "difficulty": r.get("difficulty")}})
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("=== curated-code verify ===")
    for k in ["seen", "fields", "safe", "compile", "passed"]:
        print(f"  {k:9} {st[k]}")
    print(f"  yield = {st['passed']/max(1,st['seen']):.0%} | avg tests/task = {sum(r['meta']['ntests'] for r in out)/max(1,len(out)):.1f}")
    print(f"  -> {a.out}  ({len(out)} verified)")


if __name__ == "__main__":
    main()
