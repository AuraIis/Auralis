#!/usr/bin/env python3
"""Audit current cleaning/filter quality on fixed samples (diag-only)."""
from __future__ import annotations
import json, re, sys, argparse
from pathlib import Path
from collections import Counter

sys.path.insert(0, "/workspace/v2data")
from scripts.data.filter_quality import _passes, BOILERPLATE_PATTERNS  # noqa
from scripts.data.filter_code_quality import analyze, AUTOGEN, TAG, BEGIN, END, FILENAME  # noqa

import fasttext
_FT = fasttext.load_model("/workspace/v2data/models/lid.176.ftz")

def lang_of(text):
    t = " ".join(text[:1200].split())
    if not t:
        return "??", 0.0
    labels = _FT.f.predict(t + "\n", 1, 0.0, "strict")
    if not labels:
        return "??", 0.0
    p, lab = labels[0]
    return lab.replace("__label__", ""), min(p, 1.0)

EMAIL = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?\d{4}){4,7}\b")
HTML_TAG = re.compile(r"</?(?:div|span|p|a|li|ul|td|tr|img|br|h[1-6])\b[^>]*>", re.I)
HTML_ENT = re.compile(r"&(?:nbsp|amp|quot|lt|gt|#\d+);")
MOJI = ("â€™", "â€œ", "Ã¼", "Ã¶", "Ã¤", "ÃŸ", "Â ", "�")
PY2 = re.compile(r"^\s*print\s+[^(\s]|^\s*except\s+\w+\s*,\s*\w+\s*:", re.M)

def doc_junk(text, expect_lang):
    tags = []
    lang, conf = lang_of(text)
    if expect_lang and lang != expect_lang and conf >= 0.65 and len(text) >= 200:
        tags.append(f"lang_{lang}")
    if any(m in text for m in MOJI):
        tags.append("mojibake")
    if len(EMAIL.findall(text)) >= 1:
        tags.append("email")
    if IBAN.search(text):
        tags.append("iban")
    if len(HTML_TAG.findall(text)) >= 3 or len(HTML_ENT.findall(text)) >= 5:
        tags.append("html")
    low = text.lower()
    if any(p in low for p in BOILERPLATE_PATTERNS):
        tags.append("boilerplate")
    parts = [p.strip() for p in re.split(r"[\n]| {3,}", text) if len(p.strip()) > 40]
    if len(parts) >= 5:
        c = Counter(parts)
        dup = sum(v - 1 for v in c.values() if v > 1)
        if dup / len(parts) > 0.3:
            tags.append("dup_paragraphs")
    toks = text.split()
    if len(toks) >= 50 and 1 - len(set(toks)) / len(toks) > 0.7:
        tags.append("repetitive")
    return tags

def audit_prose(path, expect_lang, is_jsonl, min_len, n_show=3):
    res = Counter(); junk_kept = Counter(); examples = {}
    drop_good = 0; drop_good_ex = []
    n = 0
    for line in path.open(encoding="utf-8", errors="replace"):
        if is_jsonl:
            try:
                text = json.loads(line)["text"]
            except Exception:
                continue
        else:
            text = line.rstrip("\n")
        n += 1
        reason = _passes(text, min_length=min_len, max_length=100_000,
                         preserve_newlines=False, allow_mojibake=False)
        if reason:
            res[f"drop:{reason}"] += 1
            if reason in ("boilerplate", "repetitive", "symbol_dense", "mojibake") and len(text) > 300:
                lang, conf = lang_of(text)
                if lang == expect_lang and conf > 0.8:
                    drop_good += 1
                    if len(drop_good_ex) < n_show:
                        drop_good_ex.append((reason, text[:300]))
        else:
            res["kept"] += 1
            for t in doc_junk(text, expect_lang):
                junk_kept[t] += 1
                if t not in examples:
                    examples[t] = text[:300]
    return {"docs": n, "result": dict(res), "junk_in_kept": dict(junk_kept),
            "candidate_false_drops": drop_good, "false_drop_examples": drop_good_ex,
            "junk_examples": examples}

def audit_code(path, threshold=3):
    res = Counter(); junk_kept = Counter(); examples = {}
    n = 0
    for line in path.open(encoding="utf-8", errors="replace"):
        try:
            text = json.loads(line)["text"]
        except Exception:
            continue
        n += 1
        body = [ln for ln in text.split("\n")
                if not ln.startswith(BEGIN) and not ln.startswith(FILENAME) and ln.strip() != END]
        code = "\n".join(body)
        m = TAG.match(text.split("\n", 1)[0])
        lang = m.group(1).lower() if m else "python"
        keep, score, reason = analyze(code, lang, threshold)
        if not keep:
            res[f"drop:{reason}"] += 1
            continue
        res["kept"] += 1
        mm = AUTOGEN.search(code[4000:])
        if mm:
            junk_kept["autogen_after_4k"] += 1
            if "autogen_after_4k" not in examples:
                i = mm.start() + 4000
                examples["autogen_after_4k"] = code[max(0, i-100):i+150]
        if lang == "python" and PY2.search(code):
            junk_kept["py2_marker"] += 1
            examples.setdefault("py2_marker", code[:300])
        lines = code.split("\n")
        if len(lines) > 3 and sum(len(l) > 500 for l in lines) > len(lines) * 0.3:
            junk_kept["long_line_heavy"] += 1
        if any(m_ in code for m_ in MOJI):
            junk_kept["mojibake"] += 1
            examples.setdefault("mojibake", code[:300])
    return {"docs": n, "result": dict(res), "junk_in_kept": dict(junk_kept), "junk_examples": examples}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    D = Path("/workspace/v2data/diag/clean_audit_v1")
    out = {}
    jobs = [
        ("de_fresh", D/"de_fresh.10k.jsonl", "de", True, 300),
        ("gc_edu", D/"gc_edu.10k.jsonl", "de", True, 300),
        ("se_os", D/"se_os.10k.jsonl", "en", True, 200),
        ("raw_fineweb2_de", D/"raw_fineweb2_de.10k.txt", "de", False, 300),
        ("raw_hplt_de", D/"raw_hplt_de.10k.txt", "de", False, 300),
        ("raw_gc", D/"raw_gc.10k.txt", "de", False, 300),
        ("raw_fineweb_en", D/"raw_fineweb_en.10k.txt", "en", False, 200),
    ]
    for name, p, lang, jl, ml in jobs:
        if p.exists():
            print(f"== prose {name}", flush=True)
            out[name] = audit_prose(p, lang, jl, ml)
    for name, p in [("code_multi", D/"code_multi.10k.jsonl"), ("opc_snippets", D/"opc_snippets.5k.jsonl"),
                    ("opc_qa", D/"opc_qa.5k.jsonl"), ("opc_algo", D/"opc_algo.5k.jsonl")]:
        if p.exists():
            print(f"== code {name}", flush=True)
            out[name] = audit_code(p)
    a.out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {a.out}")
