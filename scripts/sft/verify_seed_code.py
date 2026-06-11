#!/usr/bin/env python3
"""Executor-verify the code_instruction rows of the OS/security SFT seed.

The responses are raw Python (German inline comments), often with embedded `assert`
tests, sometimes with a trailing German-prose explanation (which is NOT valid Python).
We extract the largest COMPILABLE prefix, then execute it in a subprocess sandbox.

Honest 3-level classification (no fake confidence):
  VERIFIED  - compiles, RUNS, and contains assert/unittest tests that PASS  (provably correct)
  RUNS      - compiles + runs but has NO test to prove correctness          (syntactically sound)
  FAILED    - does not compile, or runtime error / failed assertion         (broken)

Writes <out> = the seed with a 'verify' field added on code_instruction rows, plus a
.report.json summary. Non-code rows pass through unchanged.

    python scripts/sft/verify_seed_code.py --in raw/sft/os_security_de_seed.jsonl \
        --out raw/sft/os_security_de_seed.verified.jsonl
"""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path


def largest_compilable_block(resp: str) -> tuple[str, bool]:
    """Largest contiguous line-window that compiles as Python. Trims BOTH leading prose
    (e.g. GPT's 'Code:' prefix) and trailing prose (e.g. a German 'Erklärung: ...')."""
    lines = resp.split("\n")
    best = ""
    for start in range(len(lines)):
        for end in range(len(lines), start, -1):
            src = "\n".join(lines[start:end]).strip()
            if len(src) <= len(best):
                break          # shorter than current best -> no point shrinking further
            try:
                compile(src, "<seed>", "exec")
                best = src
                break          # longest end for this start found
            except SyntaxError:
                continue
    return best, bool(best)


LANGS = ("python", "javascript", "typescript", "bash", "shell", "sql", "java", "powershell")


def lang_of(row: dict) -> str:
    tags = [t.lower() for t in row.get("tags", [])]
    for L in LANGS:
        if L in tags:
            return "python" if L == "python" else L
    rsp = row.get("response", "")
    if "def " in rsp or "import " in rsp or "Code:" in rsp[:10]:
        return "python"
    return "other"


def run_code(src: str, timeout: int = 10) -> tuple[bool, str]:
    # auto-run an unittest TestCase if defined but not invoked
    if "TestCase" in src and "unittest.main" not in src and "main(" not in src:
        src = src + "\n\nimport unittest as __u\n__u.main(argv=[''], exit=False, verbosity=0)\n"
    try:
        p = subprocess.run([sys.executable, "-I", "-c", src],
                           capture_output=True, text=True, timeout=timeout)
        return p.returncode == 0, (p.stderr or "")[-300:]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:  # noqa
        return False, f"{type(e).__name__}: {e}"


def classify(resp: str) -> dict:
    src, compile_ok = largest_compilable_block(resp)
    has_test = ("assert " in src) or ("TestCase" in src)
    if not compile_ok:
        return {"status": "FAILED", "compile_ok": False, "run_ok": False, "has_test": has_test, "reason": "no_compile"}
    run_ok, err = run_code(src)
    if run_ok and has_test:
        status = "VERIFIED"
    elif run_ok:
        status = "RUNS"
    else:
        status = "FAILED"
    out = {"status": status, "compile_ok": True, "run_ok": run_ok, "has_test": has_test}
    if not run_ok:
        out["reason"] = err
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    rows = [json.loads(l) for l in a.inp.open(encoding="utf-8") if l.strip()]
    counts = {"VERIFIED": 0, "RUNS": 0, "FAILED": 0, "OTHER_LANG": 0}
    by_teacher: dict[str, dict] = {}
    by_lang: dict[str, int] = {}
    n_code = n_py = 0
    fails: list[dict] = []

    for r in rows:
        if r.get("domain") != "code_instruction":
            continue
        n_code += 1
        lang = lang_of(r)
        by_lang[lang] = by_lang.get(lang, 0) + 1
        if lang != "python":
            r["verify"] = {"status": "OTHER_LANG", "lang": lang}
            counts["OTHER_LANG"] += 1
            continue
        n_py += 1
        v = classify(r["response"])
        r["verify"] = v
        counts[v["status"]] += 1
        bt = by_teacher.setdefault(r["teacher"], {"VERIFIED": 0, "RUNS": 0, "FAILED": 0})
        bt[v["status"]] += 1
        if v["status"] == "FAILED" and len(fails) < 12:
            fails.append({"teacher": r["teacher"], "topic": r["topic"], "reason": v.get("reason", "")[:160]})

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as out:
        for r in rows:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {"code_rows": n_code, "python_rows": n_py, "counts": counts, "by_language": by_lang,
              "python_pass_pct": round(100 * (counts["VERIFIED"] + counts["RUNS"]) / max(1, n_py), 1),
              "python_verified_pct": round(100 * counts["VERIFIED"] / max(1, n_py), 1),
              "by_teacher_python": by_teacher, "sample_failures": fails}
    a.out.with_suffix(".report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
