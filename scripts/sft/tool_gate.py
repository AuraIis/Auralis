#!/usr/bin/env python3
"""DUAL tool-gate — judges a tool-SFT checkpoint by DISPATCH BEHAVIOUR, not val_loss.

MATH probes (expect a tool-call):
  tool_rate   : emitted <tool:python>…</tool>
  parse_rate  : the call is parsebar + runs in the safe calculator
  correct_rate: executor(model-expr) == ground truth
  fake_rate   : model wrote its own <result> (BAD — stop-sequence failed)
NON-MATH probes (expect NO tool-call):
  false_tool  : emitted a tool-call on a plain fact question (BAD)

Composite score = correct_rate - false_tool_rate  (high math recall, no false calls).
Builds the model ONCE and swaps checkpoint weights -> compares many ckpts fast."""
import os, sys, argparse, pathlib, random
from collections import defaultdict

REPO = pathlib.Path("/workspace/v2data")
for p in (REPO / "scripts/sft", REPO, REPO / "src"):
    sys.path.insert(0, str(p))
os.environ.setdefault("AURALIS_USE_MAMBA_KERNEL", "1")
from tool_harness import safe_calc, extract_code, gen_until, run_with_tools, TOOL_OPEN, TOOL_CLOSE, RES_OPEN  # noqa
import gen_tool_traces as G  # noqa

SYS = "Du bist Auralis, ein hilfreicher, ehrlicher KI-Assistent."
NONMATH = [
    # --- Fakten (DE) ---
    "Was ist die Hauptstadt von Oesterreich?", "Was ist die Hauptstadt von Italien?",
    "Was ist die Hauptstadt von Frankreich?", "Was ist die Hauptstadt von Spanien?",
    "Was ist die Hauptstadt von Deutschland?", "Wer schrieb Faust?", "Wer malte die Mona Lisa?",
    "Wer war Albert Einstein?", "Was ist die Donau?", "Was ist Photosynthese?",
    "Welche Farbe hat der Himmel?", "Was ist ein Saeugetier?", "Was ist die Hauptstadt von Polen?",
    # --- Ja/Nein ---
    "Ist Pluto ein Planet?", "Ist die Erde flach?", "Ist Berlin eine Stadt?",
    "Ist ein Wal ein Fisch?", "Ist Gold ein Metall?", "Ist die Sonne ein Stern?",
    "Kann ein Pinguin fliegen?", "Ist Wasser bei Zimmertemperatur fluessig?",
    # --- Erklaerfragen ---
    "Erklaere kurz, was ein Vulkan ist.", "Erklaere kurz, was Wasser ist.",
    "Erklaere kurz, wie ein Regenbogen entsteht.", "Erklaere kurz, was ein Computer ist.",
    "Erklaere kurz, was Demokratie bedeutet.", "Warum muss man Zaehne putzen?",
    "Warum ist der Himmel blau?", "Wie funktioniert ein Fahrrad grob?",
    # --- Alltag ---
    "Wie kocht man Reis?", "Was kann man bei Regen drinnen machen?", "Wie pflanzt man einen Baum?",
    "Was zieht man im Winter an?", "Wie haelt man sich gesund?",
    # --- numerisch, aber KEINE Rechnung (darf KEIN Tool ausloesen) ---
    "Wie viele Bundeslaender hat Deutschland?", "Wie viele Planeten hat unser Sonnensystem?",
    "Wie viele Kontinente gibt es?", "In welchem Jahr fiel die Berliner Mauer?",
    "Wie viele Beine hat eine Spinne?",
    # --- English ---
    "What is the capital of Spain?", "What is the capital of Japan?",
    "Who wrote Romeo and Juliet?", "What is a volcano?", "Is the moon a planet?",
    "Why is the sky blue?", "How do you boil an egg?", "What is gravity?",
    "Is a tomato a fruit?", "Explain briefly what water is.",
    "Who painted the Mona Lisa?", "What is the largest ocean?",
]


BUCKET = {
    "g_add": "simple", "g_sub": "simple", "g_mul": "simple", "g_div": "simple",
    "g_square": "simple", "g_sqrt": "simple",
    "g_percent": "percent", "g_discount_price": "percent", "g_discount_amount": "percent",
    "g_markup_price": "percent",
    "g_hours_min": "time_unit", "g_days_hours": "time_unit", "g_min_sec": "time_unit", "g_km_m": "time_unit",
    "g_speed_dist": "speed", "g_dist_time": "speed",
    "g_word_total": "word", "g_word_change": "word", "g_price_total": "word",
    "g_recipe_scale": "word", "g_fraction": "word", "g_avg": "word",
    "g_en_mul": "english", "g_en_pct": "english",
}


def math_probes(n, seed=999):
    rng = random.Random(seed)
    pool = [g for g, w in G.GENS for _ in range(w)]
    out, seen = [], set()
    while len(out) < n:
        g = rng.choice(pool)
        q, expr, _ = g(rng)
        if q in seen:
            continue
        ok, res = safe_calc(f"print({expr})")
        if not ok:
            continue
        seen.add(q); out.append((q, res, BUCKET.get(g.__name__, "other")))
    return out


def main():
    import torch
    import sentencepiece as spm
    from auralis.model import build_model
    from auralis.tokenizer.chat_template import build_inference_prompt
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", required=True, help="comma-separated .pt paths")
    ap.add_argument("--model-config", default=str(REPO / "configs/model/helix_v2_1b.yaml"))
    ap.add_argument("--tokenizer", default=str(REPO / "tokenizer/helix_v2_tokenizer.model"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument(
        "--mode",
        choices=["call_only", "full"],
        default="call_only",
        help="call_only scores tool dispatch; full runs the harness and scores result usage + final numeric answer.",
    )
    a = ap.parse_args()
    device = torch.device("cuda")
    sp = spm.SentencePieceProcessor(model_file=a.tokenizer)
    model = build_model(a.model_config).to(device).eval()
    probes = math_probes(a.n)
    N, M = len(probes), len(NONMATH)
    print(f"=== TOOL-GATE | math probes={N} | non-math probes={M} ===")

    def ask(q):
        prompt = build_inference_prompt([{"role": "user", "content": q}], default_system=SYS)
        text, _ = gen_until(model, sp, prompt, [], device, max_new=64)
        return text

    best = (None, -9)
    for ckpt in a.checkpoints.split(","):
        ckpt = ckpt.strip()
        if not ckpt:
            continue
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        state = payload.get("model", payload.get("state_dict", payload))
        model.load_state_dict(state, strict=False)
        em = pa = co = fk = used = ansmatch = 0
        bkt = defaultdict(lambda: [0, 0])
        for q, truth, bucket in probes:
            if a.mode == "full":
                prompt = build_inference_prompt([{"role": "user", "content": q}], default_system=SYS)
                t = run_with_tools(model, sp, prompt, device, verbose=False, max_tool_calls=1)
            else:
                t = ask(q)
            has = (TOOL_OPEN in t and TOOL_CLOSE in t)
            if has:
                em += 1
            code = extract_code(t)
            ok, res = safe_calc(code) if code else (False, "")
            bkt[bucket][1] += 1
            if ok:
                pa += 1
                if res == truth:
                    co += 1
                    bkt[bucket][0] += 1
            if (a.mode == "call_only" and RES_OPEN in t) or (a.mode == "full" and t.count(RES_OPEN) != 1):
                fk += 1
            after_result = t.split("</result>", 1)[1] if "</result>" in t else t
            if a.mode == "full" and res == truth and truth in after_result:
                used += 1
            if a.mode == "full" and truth in after_result:
                ansmatch += 1
        ft = sum(1 for q in NONMATH if TOOL_OPEN in ask(q))
        tool_r, parse_r, corr_r, fake_r, false_r = em / N, pa / N, co / N, fk / N, ft / M
        score = corr_r - false_r
        name = pathlib.Path(ckpt).name
        print(f"\n## {name}")
        print(f"  MATH  tool_rate={tool_r:.0%}  parse_rate={parse_r:.0%}  correct_rate={corr_r:.0%}  fake_result={fake_r:.0%}")
        if a.mode == "full":
            print(f"  E2E   result_usage_rate={used/N:.0%}  answer_numeric_match={ansmatch/N:.0%}")
        print(f"  NONM  false_tool_rate={false_r:.0%}")
        print(f"  SCORE (correct - false_tool) = {score:.3f}")
        bd = "  ".join(f"{k}:{v[0]}/{v[1]}" for k, v in sorted(bkt.items()))
        print(f"  BY-TYPE (correct/total)  {bd}")
        if score > best[1]:
            best = (name, score)
    print(f"\n=== BEST by gate: {best[0]} (score {best[1]:.3f}) ===")


if __name__ == "__main__":
    main()
