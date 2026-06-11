#!/usr/bin/env python3
"""Syntax-check the NON-Python code_instruction rows of the SFT seed (JS/TS/Bash/PowerShell).

The Python rows are executor-verified by verify_seed_code.py. This does a SYNTAX-only check
(no execution — these snippets reference context we don't have) per language:
  javascript -> node --check
  typescript -> node --check (Node 24 strips type annotations)
  powershell -> [PSParser]::Tokenize error count (via pwsh)
  bash       -> emitted to a folder for `bash -n` (run where bash exists, e.g. the container)

Extracts the code by stripping a leading 'Code:'/'Lösung:' marker and a trailing German
'Erklärung:'/'Hinweis:' explanation. Reports syntax_ok / syntax_error / no_checker.

    python scripts/sft/verify_seed_nonpy.py --in <verified.jsonl> --bash-out <dir>
"""
from __future__ import annotations
import argparse, json, re, subprocess, tempfile, os, shutil
from pathlib import Path

LEAD = re.compile(r"^(Code|L[oö]sung|Beispiel|Antwort)\s*:?\s*$", re.I)
TRAIL = re.compile(r"^(Erkl[äa]rung|Erl[äa]uterung|Hinweis|Anmerkung|Erg[äa]nzung)\s*:", re.I)


def extract(resp: str) -> str:
    lines = resp.split("\n")
    while lines and LEAD.match(lines[0].strip()):
        lines.pop(0)
    out = []
    for ln in lines:
        if TRAIL.match(ln.strip()):
            break
        out.append(ln)
    return "\n".join(out).strip()


def check_node(code: str, suffix: str) -> tuple[bool, str]:
    node = shutil.which("node")
    if not node:
        return None, "no node"
    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8") as f:
        f.write(code); path = f.name
    try:
        p = subprocess.run([node, "--check", path], capture_output=True, text=True, timeout=15)
        return p.returncode == 0, (p.stderr or "")[-200:]
    except Exception as e:  # noqa
        return False, f"{type(e).__name__}"
    finally:
        os.unlink(path)


def check_pwsh(code: str) -> tuple[bool, str]:
    pwsh = shutil.which("pwsh") or shutil.which("powershell")
    if not pwsh:
        return None, "no pwsh"
    with tempfile.NamedTemporaryFile("w", suffix=".ps1", delete=False, encoding="utf-8") as f:
        f.write(code); path = f.name
    ps = (f"$e=$null;[void][System.Management.Automation.PSParser]::Tokenize("
          f"(Get-Content -Raw '{path}'),[ref]$e);if($e){{$e[0].Message;exit 1}}else{{exit 0}}")
    try:
        p = subprocess.run([pwsh, "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=20)
        return p.returncode == 0, (p.stdout or p.stderr or "")[-200:]
    except Exception as e:  # noqa
        return False, f"{type(e).__name__}"
    finally:
        os.unlink(path)


def lang_of(row: dict) -> str:
    tags = [t.lower() for t in row.get("tags", [])]
    for L in ("javascript", "typescript", "bash", "shell", "powershell"):
        if L in tags:
            return "bash" if L == "shell" else L
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--bash-out", type=Path, default=None, help="dir to emit bash snippets for `bash -n`")
    a = ap.parse_args()

    rows = [json.loads(l) for l in a.inp.open(encoding="utf-8") if l.strip()]
    res: dict[str, dict] = {}
    fails: list[dict] = []
    bash_n = 0
    if a.bash_out:
        a.bash_out.mkdir(parents=True, exist_ok=True)

    for r in rows:
        if r.get("domain") != "code_instruction" or r.get("verify", {}).get("status") != "OTHER_LANG":
            continue
        lang = lang_of(r)
        code = extract(r["response"])
        st = res.setdefault(lang, {"ok": 0, "err": 0, "nocheck": 0})
        if lang in ("javascript", "typescript"):
            ok, msg = check_node(code, ".js" if lang == "javascript" else ".ts")
        elif lang == "powershell":
            ok, msg = check_pwsh(code)
        elif lang == "bash":
            if a.bash_out:
                (a.bash_out / f"bash_{bash_n:03d}.sh").write_text(code, encoding="utf-8")
                bash_n += 1
            ok, msg = None, "deferred to container bash -n"
        else:
            ok, msg = None, "unknown lang"
        if ok is True:
            st["ok"] += 1
        elif ok is False:
            st["err"] += 1
            if len(fails) < 12:
                fails.append({"lang": lang, "teacher": r["teacher"], "topic": r["topic"], "msg": msg[:140]})
        else:
            st["nocheck"] += 1

    report = {"by_language": res, "bash_emitted": bash_n, "sample_syntax_errors": fails}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
