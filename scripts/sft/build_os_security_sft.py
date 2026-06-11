#!/usr/bin/env python3
"""Merge + normalize the German OS-troubleshooting / defensive-security seed files
(multi-teacher: GPT / qwen / Claude) into ONE schema-unified, deduplicated SFT seed.

These are SFT-stage data (instruction -> response, some with diagnostic command traces),
NOT pretraining corpus. The three teachers answer the SAME topic lists, so cross-teacher
variations are KEPT (phrasing diversity); only EXACT-duplicate responses are dropped.

Unified record:
  {id, teacher, domain, os, topic, difficulty, task_type, instruction, context,
   trace?, response, tags, learning?, risk_level?}

    python scripts/sft/build_os_security_sft.py --in-dir <dir> --out <out.jsonl>
"""
from __future__ import annotations
import argparse, json, glob, os, re
from pathlib import Path

# filename substring -> teacher
TEACHERS = [("gpt", "gpt"), ("qwen", "qwen"), ("claude", "claude")]


def teacher_of(fn: str, default: str = "unknown") -> str:
    low = fn.lower()
    for needle, name in TEACHERS:
        if needle in low:
            return name
    return default      # untagged files (e.g. 'defensive Systemsicherheit.Jsonl') -> --default-teacher


def domain_of(fn: str, rec: dict) -> str:
    d = rec.get("domain")
    if d:
        return d
    low = fn.lower()
    if "security" in low or "systemsicherheit" in low:
        return "defensive_system_security"
    return "os_troubleshooting"


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def normalize(rec: dict, teacher: str, domain: str, idx: int) -> dict | None:
    instr = (rec.get("instruction") or "").strip()
    # agent_shell_trace records carry the answer in 'final_response' (+ a rich trace)
    resp = (rec.get("response") or rec.get("final_response") or "").strip()
    if len(instr) < 5 or len(resp) < 40:        # drop empty/stub
        return None
    out = {
        "id": f"{domain}_{teacher}_{idx:06d}",
        "orig_id": rec.get("id"),
        "teacher": teacher,
        "domain": domain,
        "os": rec.get("os", ""),
        "topic": rec.get("topic", ""),
        "difficulty": rec.get("difficulty", ""),
        "task_type": rec.get("task_type") or ("troubleshooting" if domain == "os_troubleshooting" else ""),
        "instruction": instr,
        "context": (rec.get("context") or "").strip(),
        "response": resp,
        "tags": rec.get("tags", []),
        "learning": rec.get("expected_learning") or rec.get("expected_outcome") or "",
    }
    if rec.get("trace"):
        out["trace"] = rec["trace"]
    if rec.get("risk_level"):
        out["risk_level"] = rec["risk_level"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", required=True, help="dir containing the seed *.jsonl/*.Jsonl files")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--patterns", nargs="+",
                    default=["*OS-Troubleshooting*", "*Systemsicherheit*", "*systemsicherheit*"])
    ap.add_argument("--default-teacher", default="unknown",
                    help="teacher for files without a GPT/qwen/Claude marker in the name")
    a = ap.parse_args()

    files: list[str] = []
    for pat in a.patterns:
        for ext in (".jsonl", ".Jsonl", ".JSONL"):
            files.extend(glob.glob(os.path.join(a.in_dir, pat + ext)))
    files = sorted(set(files))
    if not files:
        raise SystemExit(f"no seed files matched in {a.in_dir}")

    seen_resp: set[str] = set()
    rows: list[dict] = []
    per_file: dict[str, int] = {}
    dropped_dup = dropped_stub = 0
    by_teacher: dict[str, int] = {}
    by_domain: dict[str, int] = {}
    by_os: dict[str, int] = {}
    with_trace = 0

    for fp in files:
        fn = os.path.basename(fp)
        teacher = teacher_of(fn, a.default_teacher)
        kept_here = 0
        with open(fp, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                dom = domain_of(fn, rec)
                norm = normalize(rec, teacher, dom, len(rows) + 1)
                if norm is None:
                    dropped_stub += 1
                    continue
                key = norm_ws(norm["response"])
                if key in seen_resp:        # EXACT-dup response (rare across teachers) -> drop
                    dropped_dup += 1
                    continue
                seen_resp.add(key)
                rows.append(norm)
                kept_here += 1
                by_teacher[teacher] = by_teacher.get(teacher, 0) + 1
                by_domain[dom] = by_domain.get(dom, 0) + 1
                by_os[norm["os"]] = by_os.get(norm["os"], 0) + 1
                if "trace" in norm:
                    with_trace += 1
        per_file[fn] = kept_here

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as out:
        for r in rows:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "out": str(a.out), "files": per_file, "total_kept": len(rows),
        "dropped_exact_dup": dropped_dup, "dropped_stub": dropped_stub,
        "by_teacher": by_teacher, "by_domain": by_domain, "by_os": by_os,
        "with_command_trace": with_trace,
    }
    a.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nwrote {len(rows)} normalized SFT rows -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
