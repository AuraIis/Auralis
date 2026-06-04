"""Rigorous fact-recall probe for Helix checkpoints (multi-distractor + top-k).

Why this exists: a single greedy generation ("capital = Munich→Berlin→Munich") is too
fragile — it confuses DECODING drift with missing KNOWLEDGE. This measures fact recall
as a contrastive margin over a battery spanning several categories, each with MULTIPLE
wrong candidates:

    margin = NLL(best/hardest wrong) - NLL(correct)      (per-token mean)
    ok     = margin > 0   →  correct is rank-1 vs ALL distractors (real preference)

Plus a TOP-K check: after the fact prompt, does the correct answer's first token even
appear among the top tokens? Reports accuracy + mean margin + top-k hit-rate, overall,
per language, and per category. Chance (1 of k candidates) is well below 100%.

Read-only. Run on the Blackwell (needs the kernels).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import sentencepiece as spm
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

if torch.cuda.is_available():
    try:
        import mamba_ssm  # noqa: F401
        os.environ.setdefault("AURALIS_USE_MAMBA_KERNEL", "1")
        os.environ.setdefault("AURALIS_USE_GLA_KERNEL", "1")
        os.environ.setdefault("AURALIS_USE_FLASH_ATTN", "1")
    except Exception:
        pass

from auralis.model import build_model  # noqa: E402

# (id, lang, category, prompt, correct, [wrong, wrong, ...])
FACT_PROBES = [
    # ---- Geografie ----
    ("de_cap_de", "de", "geo", "Die Hauptstadt von Deutschland ist", " Berlin.", [" München.", " Hamburg.", " Wien.", " Paris."]),
    ("de_cap_fr", "de", "geo", "Die Hauptstadt von Frankreich ist", " Paris.", [" Lyon.", " Marseille.", " Berlin.", " Rom."]),
    ("de_cap_it", "de", "geo", "Die Hauptstadt von Italien ist", " Rom.", [" Mailand.", " Neapel.", " Venedig.", " Madrid."]),
    ("de_cap_es", "de", "geo", "Die Hauptstadt von Spanien ist", " Madrid.", [" Barcelona.", " Sevilla.", " Lissabon.", " Rom."]),
    ("en_cap_de", "en", "geo", "The capital of Germany is", " Berlin.", [" Munich.", " Hamburg.", " Frankfurt.", " Vienna."]),
    ("en_cap_jp", "en", "geo", "The capital of Japan is", " Tokyo.", [" Osaka.", " Kyoto.", " Beijing.", " Seoul."]),
    ("en_cap_uk", "en", "geo", "The capital of the United Kingdom is", " London.", [" Manchester.", " Birmingham.", " Dublin.", " Paris."]),
    # ---- Wissenschaft ----
    ("de_boil",   "de", "sci", "Wasser siedet bei Normaldruck bei", " 100 Grad Celsius.", [" 50 Grad Celsius.", " 0 Grad Celsius.", " 200 Grad Celsius."]),
    ("de_center", "de", "sci", "Der Mittelpunkt unseres Sonnensystems ist die", " Sonne.", [" Erde.", " Milchstraße.", " Galaxie."]),
    ("de_gold",   "de", "sci", "Das chemische Symbol für Gold ist", " Au.", [" Ag.", " Go.", " Fe.", " Gd."]),
    ("de_sun",    "de", "sci", "Die Sonne ist ein", " Stern.", [" Planet.", " Mond.", " Komet."]),
    ("en_boil",   "en", "sci", "Water boils at normal pressure at", " 100 degrees Celsius.", [" 50 degrees Celsius.", " 0 degrees Celsius.", " 200 degrees Celsius."]),
    ("en_jupiter","en", "sci", "The largest planet in our solar system is", " Jupiter.", [" Mars.", " Saturn.", " Earth."]),
    ("en_gold",   "en", "sci", "The chemical symbol for gold is", " Au.", [" Ag.", " Go.", " Fe."]),
    # ---- Geschichte ----
    ("de_brd",    "de", "hist", "Die Bundesrepublik Deutschland wurde gegründet im Jahr", " 1949.", [" 1939.", " 1989.", " 1918."]),
    ("de_mauer",  "de", "hist", "Die Berliner Mauer fiel im Jahr", " 1989.", [" 1949.", " 1961.", " 1979."]),
    ("de_faust",  "de", "hist", "Das Drama Faust wurde geschrieben von", " Goethe.", [" Schiller.", " Lessing.", " Brecht."]),
    ("de_relativ","de", "hist", "Die Relativitätstheorie stammt von Albert", " Einstein.", [" Newton.", " Bohr.", " Planck."]),
    ("en_moon",   "en", "hist", "The first moon landing happened in the year", " 1969.", [" 1959.", " 1979.", " 1989."]),
    ("en_romeo",  "en", "hist", "Romeo and Juliet was written by", " Shakespeare.", [" Dickens.", " Milton.", " Chaucer."]),
    ("en_relativ","en", "hist", "The theory of relativity was developed by Albert", " Einstein.", [" Newton.", " Bohr.", " Darwin."]),
    # ---- Sprache / Übersetzung ----
    ("de2en_dog", "de", "lang", "Das englische Wort für Hund ist", " dog.", [" cat.", " house.", " bird."]),
    ("de2en_cat", "de", "lang", "Das englische Wort für Katze ist", " cat.", [" dog.", " mouse.", " fish."]),
    ("en2de_water","en", "lang", "The German word for water is", " Wasser.", [" Feuer.", " Brot.", " Haus."]),
    ("en2de_house","en", "lang", "The German word for house is", " Haus.", [" Hund.", " Baum.", " Auto."]),
    # ---- Code ----
    ("code_print","en", "code", "In Python, to print text to the console you use the function", " print().", [" echo().", " println().", " printf()."]),
    ("code_comment","en","code", "In Python, a single-line comment starts with the symbol", " #.", [" //.", " /*.", " --."]),
    ("code_json", "en", "code", "JSON stores data as collections of key-value", " pairs.", [" tables.", " columns.", " rows."]),
    ("code_list", "en", "code", "In Python, an empty list is written as", " [].", [" {}.", " ().", " <>."]),
]


def continuation_nll(model, sp, device, prompt: str, continuation: str):
    """Return (mean per-token NLL of continuation, first continuation token id)."""
    prompt_ids = sp.EncodeAsIds(prompt)
    full_ids = sp.EncodeAsIds(prompt + continuation)
    cont_start = 0
    for a, b in zip(prompt_ids, full_ids):
        if a != b:
            break
        cont_start += 1
    if cont_start >= len(full_ids):
        return float("inf"), None
    first_tok = full_ids[cont_start]
    x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor(full_ids[1:], dtype=torch.long, device=device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(input_ids=x)["logits"][0].float()
    logp = torch.log_softmax(logits, dim=-1)
    losses = [float(-logp[pos - 1, int(y[pos - 1])].item()) for pos in range(max(1, cont_start), len(full_ids))]
    return sum(losses) / max(1, len(losses)), first_tok


def topk_rank(model, sp, device, prompt: str, first_tok: int, k: int = 10):
    """Rank of `first_tok` among next-token logits after the prompt (0 = top). None if > big."""
    ids = sp.EncodeAsIds(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(input_ids=x)["logits"][0, -1].float()
    order = torch.argsort(logits, descending=True)
    pos = (order == first_tok).nonzero(as_tuple=True)[0]
    rank = int(pos.item()) if pos.numel() else 10**9
    return rank, rank < k


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    model = build_model(args.model_config).to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = {k.replace("_orig_mod.", ""): v for k, v in payload["model"].items()}
    missing, extra = model.load_state_dict(state, strict=False)
    if missing or extra:
        raise RuntimeError(f"state mismatch missing={len(missing)} extra={len(extra)}")
    model.eval()
    step = (payload.get("state") or {}).get("step")

    rows = []
    for pid, lang, cat, prompt, correct, wrongs in FACT_PROBES:
        nll_c, first_c = continuation_nll(model, sp, device, prompt, correct)
        nll_wrongs = [continuation_nll(model, sp, device, prompt, w)[0] for w in wrongs]
        best_wrong = min(nll_wrongs)                      # hardest distractor
        margin = best_wrong - nll_c                        # >0 => correct beats ALL wrongs
        rank, hit = topk_rank(model, sp, device, prompt, first_c, args.topk) if first_c is not None else (10**9, False)
        rows.append({"id": pid, "lang": lang, "cat": cat, "correct": correct.strip(),
                     "n_wrong": len(wrongs), "nll_correct": round(nll_c, 4),
                     "best_wrong_nll": round(best_wrong, 4), "margin": round(margin, 4),
                     "ok": bool(margin > 0), "topk_rank": rank, "topk_hit": bool(hit)})

    def agg(sub):
        n = len(sub)
        return (n,
                sum(r["ok"] for r in sub) / max(1, n),
                sum(r["margin"] for r in sub) / max(1, n),
                sum(r["topk_hit"] for r in sub) / max(1, n))

    n, acc, mm, hk = agg(rows)
    print(f"=== FACT RECALL (multi-distractor) — step {step} ===")
    print(f"{'id':14s}{'cat':6s}{'margin':>8s}  ok  top{args.topk}  correct")
    for r in sorted(rows, key=lambda x: x["margin"]):
        print(f"{r['id']:14s}{r['cat']:6s}{r['margin']:>8.3f}  {'✓' if r['ok'] else '✗'}   "
              f"{('#'+str(r['topk_rank'])) if r['topk_hit'] else '—':>4s}  {r['correct']}")
    print("-" * 64)
    for label, sub in [("OVERALL", rows), ("GERMAN", [r for r in rows if r['lang']=='de']),
                       ("ENGLISH", [r for r in rows if r['lang']=='en'])]:
        nn, aa, m, h = agg(sub)
        print(f"{label:8s}: acc={aa*100:5.1f}%  mean_margin={m:+.3f}  top{args.topk}_hit={h*100:5.1f}%  (n={nn})")
    print("by category:")
    for c in sorted({r['cat'] for r in rows}):
        nn, aa, m, h = agg([r for r in rows if r['cat']==c])
        print(f"  {c:6s}: acc={aa*100:5.1f}%  margin={m:+.3f}  top{args.topk}={h*100:4.0f}%  (n={nn})")
    verdict = ("ANCHORED" if acc >= 0.75 else "PARTIAL" if acc >= 0.55 else "NOT ANCHORED (≈guessing)")
    print(f"VERDICT : {verdict}  (multi-distractor is harder than 2-way)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "step": step, "checkpoint": str(args.checkpoint),
            "overall": {"n": n, "accuracy": acc, "mean_margin": mm, "topk_hit": hk, "k": args.topk},
            "by_category": {c: dict(zip(["n", "accuracy", "mean_margin", "topk_hit"],
                            agg([r for r in rows if r['cat']==c]))) for c in sorted({r['cat'] for r in rows})},
            "verdict": verdict, "probes": rows,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
