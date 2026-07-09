#!/usr/bin/env python3
"""Generate tool-use MATH SFT traces — SELF-GENERATING (the safe calculator is the
ground truth; no teacher/LLM/key needed). Mixes in non-tool QA from the existing
SFT set so the model learns WHEN to call the tool (math) vs answer directly (facts).

Trace format (single assistant turn):
  <tool:python>
  print(<expr>)
  </tool>
  <result>            <-- MUST be loss-masked by the trainer (harness injects this)
  <executor output>
  </result>
  <final natural-language answer using the result>

The non-tool QA is sampled from sources != reasoning_* (so we don't teach
"answer math directly", which would contradict tool-use)."""

import argparse
import json
import pathlib
import random
import sys

REPO = pathlib.Path("/workspace/v2data")
sys.path.insert(0, str(REPO / "scripts/sft"))
sys.path.insert(0, str(REPO))
from tool_harness import safe_calc, TOOL_OPEN, TOOL_CLOSE, RES_OPEN, RES_CLOSE  # noqa

SYS = "Du bist Auralis, ein hilfreicher, ehrlicher KI-Assistent."


def helix(q, a):
    return f"<|system|>\n{SYS}\n<|end|>\n<|user|>\n{q}\n<|end|>\n<|assistant|>\n{a}\n<|end|>\n"


def make_trace(q, expr, ans_tmpl, mode="call_only"):
    ok, res = safe_calc(f"print({expr})")
    if not ok:
        return None  # only emit traces whose call the executor can actually run
    if mode == "call_only":
        # Phase 1: learn ONLY to emit the call and stop. No <result>, no answer.
        a = f"{TOOL_OPEN}\nprint({expr})\n{TOOL_CLOSE}"
    else:
        # Phase 2: full trace. Trainer MUST loss-mask the <result>...</result> block.
        a = (
            f"{TOOL_OPEN}\nprint({expr})\n{TOOL_CLOSE}\n{RES_OPEN}\n{res}\n{RES_CLOSE}\n"
            + ans_tmpl.format(res=res)
        )
    return helix(q, a)


# ---- problem generators: each returns (question, expr, answer_template) ----
def g_add(r):
    a, b = r.randint(7, 999), r.randint(7, 999)
    q = r.choice([f"Was ist {a} plus {b}?", f"Wie viel ist {a} + {b}?", f"Berechne {a} plus {b}."])
    return q, f"{a} + {b}", r.choice([f"{a} plus {b} ergibt {{res}}.", "Das ergibt {res}."])


def g_sub(r):
    a = r.randint(50, 999)
    b = r.randint(7, a - 1)
    q = r.choice(
        [f"Was ist {a} minus {b}?", f"Berechne {a} - {b}.", f"Wie viel ist {a} weniger {b}?"]
    )
    return q, f"{a} - {b}", r.choice([f"{a} minus {b} ergibt {{res}}.", "Das ergibt {res}."])


def g_mul(r):
    a, b = r.randint(3, 99), r.randint(3, 99)
    q = r.choice([f"Was ist {a} mal {b}?", f"Berechne {a} × {b}.", f"Wie viel ist {a} mal {b}?"])
    return q, f"{a} * {b}", r.choice([f"{a} mal {b} ergibt {{res}}.", "Das Produkt ist {res}."])


def g_div(r):
    b = r.randint(2, 25)
    k = r.randint(2, 40)
    a = b * k
    q = r.choice(
        [f"Was ist {a} geteilt durch {b}?", f"Berechne {a} ÷ {b}.", f"Wie viel ist {a} / {b}?"]
    )
    return (
        q,
        f"{a} / {b}",
        r.choice([f"{a} geteilt durch {b} ergibt {{res}}.", "Das ergibt {res}."]),
    )


def g_percent(r):
    p = r.choice([5, 10, 12, 15, 20, 25, 30, 40, 50, 75])
    n = r.randint(2, 40) * 20
    q = r.choice(
        [
            f"Was sind {p}% von {n}?",
            f"Berechne {p} Prozent von {n}.",
            f"Wie viel sind {p}% von {n}?",
        ]
    )
    return q, f"{n} * {p} / 100", r.choice([f"{p}% von {n} sind {{res}}.", "Das sind {res}."])


def g_hours_min(r):
    h = r.randint(1, 9)
    m = r.randint(1, 59)
    q = f"Wie viele Minuten sind {h} Stunden und {m} Minuten?"
    return q, f"{h} * 60 + {m}", "Das sind {res} Minuten."


def g_days_hours(r):
    d = r.randint(2, 30)
    q = r.choice([f"Wie viele Stunden hat {d} Tage?", f"Wie viele Stunden sind {d} Tage?"])
    return q, f"{d} * 24", "Das sind {res} Stunden."


def g_min_sec(r):
    m = r.randint(2, 90)
    q = f"Wie viele Sekunden sind {m} Minuten?"
    return q, f"{m} * 60", "Das sind {res} Sekunden."


def g_km_m(r):
    k = r.randint(2, 50)
    q = f"Wie viele Meter sind {k} Kilometer?"
    return q, f"{k} * 1000", "Das sind {res} Meter."


def g_avg(r):
    nums = [r.randint(2, 100) for _ in range(r.choice([3, 4]))]
    s = " + ".join(str(x) for x in nums)
    q = f"Was ist der Durchschnitt von {', '.join(str(x) for x in nums)}?"
    return q, f"({s}) / {len(nums)}", "Der Durchschnitt ist {res}."


def g_square(r):
    a = r.randint(4, 40)
    q = r.choice([f"Was ist {a} hoch 2?", f"Was ist {a} zum Quadrat?"])
    return q, f"{a} ** 2", r.choice([f"{a} hoch 2 ergibt {{res}}.", "Das ergibt {res}."])


def g_sqrt(r):
    a = r.randint(4, 40)
    n = a * a
    q = f"Was ist die Quadratwurzel von {n}?"
    return q, f"sqrt({n})", "Die Quadratwurzel von %d ist {res}." % n


def g_word_total(r):
    x = r.randint(3, 40)
    y = r.randint(2, 30)
    item = r.choice(["Kisten", "Pakete", "Regale", "Tüten"])
    thing = r.choice(["Äpfel", "Bücher", "Flaschen", "Stifte"])
    q = f"Ein Laden hat {x} {item} mit je {y} {thing}. Wie viele {thing} sind das insgesamt?"
    return q, f"{x} * {y}", "Insgesamt sind das {res} %s." % thing


def g_word_change(r):
    n = r.randint(2, 9)
    p = r.randint(2, 15)
    bill = ((n * p) // 10 + 1) * 10 + r.choice([0, 10])
    q = f"Anna kauft {n} Hefte zu je {p} Euro und zahlt mit {bill} Euro. Wie viel Wechselgeld bekommt sie?"
    return q, f"{bill} - {n} * {p}", "Sie bekommt {res} Euro Wechselgeld."


def g_en_mul(r):
    a, b = r.randint(3, 99), r.randint(3, 99)
    q = r.choice([f"What is {a} times {b}?", f"Calculate {a} * {b}."])
    return q, f"{a} * {b}", r.choice([f"{a} times {b} equals {{res}}.", "The product is {res}."])


def g_en_pct(r):
    p = r.choice([5, 10, 15, 20, 25, 50])
    n = r.randint(2, 40) * 20
    q = f"What is {p}% of {n}?"
    return q, f"{n} * {p} / 100", "{res}."


# ---- harder TRANSLATION problems (where correct_rate is weak: problem -> formula) ----
def g_discount_price(r):
    B = r.randint(2, 40) * 10
    P = r.choice([10, 15, 20, 25, 30, 50])
    q = f"Ein Artikel kostet {B} Euro. Mit {P}% Rabatt - was kostet er dann?"
    return q, f"{B} - {B} * {P} / 100", "Er kostet dann {res} Euro."


def g_discount_amount(r):
    B = r.randint(2, 40) * 10
    P = r.choice([10, 15, 20, 25, 30, 50])
    q = f"Wie viel Rabatt sind {P}% auf {B} Euro?"
    return q, f"{B} * {P} / 100", "Der Rabatt betraegt {res} Euro."


def g_markup_price(r):
    B = r.randint(2, 40) * 10
    P = r.choice([5, 10, 15, 20, 25])
    q = f"Ein Preis von {B} Euro steigt um {P}%. Wie hoch ist der neue Preis?"
    return q, f"{B} + {B} * {P} / 100", "Der neue Preis ist {res} Euro."


def g_speed_dist(r):
    S = r.choice([40, 50, 60, 80, 90, 100, 120])
    T = r.randint(2, 5)
    q = f"Ein Zug faehrt {S} km/h fuer {T} Stunden. Wie weit kommt er?"
    return q, f"{S} * {T}", "Er kommt {res} km weit."


def g_dist_time(r):
    S = r.choice([40, 50, 60, 80, 100])
    T = r.randint(2, 5)
    D = S * T
    q = f"Ein Auto faehrt {D} km mit {S} km/h. Wie viele Stunden braucht es?"
    return q, f"{D} / {S}", "Es braucht {res} Stunden."


def g_price_total(r):
    n = r.randint(2, 12)
    p = r.randint(2, 40)
    q = f"{n} Artikel kosten je {p} Euro. Was kostet alles zusammen?"
    return q, f"{n} * {p}", "Zusammen kostet es {res} Euro."


def g_fraction(r):
    d = r.choice([2, 3, 4, 5])
    k = r.randint(2, 20)
    n = d * k
    name = {2: "die Haelfte", 3: "ein Drittel", 4: "ein Viertel", 5: "ein Fuenftel"}[d]
    q = f"Was ist {name} von {n}?"
    return q, f"{n} / {d}", (name + " von %d ist {res}." % n)


def g_recipe_scale(r):
    per = r.choice([2, 3, 4])
    g = r.randint(2, 20) * 10
    t = r.randint(2, 8)
    q = f"Ein Rezept fuer {per} Personen braucht {g} Gramm Mehl. Wie viel fuer {t} Personen?"
    return q, f"{g} * {t} / {per}", "Man braucht {res} Gramm."


BASE_GENS = [
    (g_add, 3),
    (g_sub, 3),
    (g_mul, 4),
    (g_div, 3),
    (g_percent, 8),
    (g_discount_price, 8),
    (g_discount_amount, 6),
    (g_markup_price, 6),
    (g_hours_min, 6),
    (g_days_hours, 4),
    (g_min_sec, 4),
    (g_km_m, 3),
    (g_speed_dist, 7),
    (g_dist_time, 6),
    (g_word_total, 6),
    (g_word_change, 6),
    (g_price_total, 6),
    (g_recipe_scale, 6),
    (g_fraction, 5),
    (g_avg, 5),
    (g_square, 3),
    (g_sqrt, 3),
    (g_en_mul, 3),
    (g_en_pct, 4),
]


PHASE2_SIMPLE_REBUMP_GENS = [
    (g_add, 6),
    (g_sub, 6),
    (g_mul, 7),
    (g_div, 6),
    (g_percent, 8),
    (g_discount_price, 8),
    (g_discount_amount, 6),
    (g_markup_price, 6),
    (g_hours_min, 6),
    (g_days_hours, 4),
    (g_min_sec, 4),
    (g_km_m, 3),
    (g_speed_dist, 7),
    (g_dist_time, 6),
    (g_word_total, 6),
    (g_word_change, 6),
    (g_price_total, 6),
    (g_recipe_scale, 6),
    (g_fraction, 5),
    (g_avg, 5),
    (g_square, 6),
    (g_sqrt, 6),
    (g_en_mul, 3),
    (g_en_pct, 4),
]

# Back-compat: tool_gate imports GENS for the held-out probe distribution.
GENS = BASE_GENS


def gen_tool(rng, n, mode, gens=None):
    gens = gens or BASE_GENS
    pool = [g for g, w in gens for _ in range(w)]
    rows, seen, tries = [], set(), 0
    while len(rows) < n and tries < n * 40:
        tries += 1
        q, expr, tmpl = rng.choice(pool)(rng)
        if q in seen:
            continue
        t = make_trace(q, expr, tmpl, mode)
        if t is None:
            continue
        seen.add(q)
        rows.append({"text": t, "source": "tool_math", "has_tool": True})
    return rows


def sample_qa(rng, n, path):
    p = pathlib.Path(path)
    if not p.exists():
        print(f"WARN no QA source at {path}", file=sys.stderr)
        return []
    cand = []
    for line in open(p, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("source") in ("reasoning_de", "reasoning_en"):
            continue  # exclude math-direct so we don't contradict tool-use
        if "<tool:" in r.get("text", "") or RES_OPEN in r.get("text", ""):
            continue
        cand.append({"text": r["text"], "source": "qa_nontool", "has_tool": False})
    rng.shuffle(cand)
    return cand[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO / "data/training/tool_sft_v1"))
    ap.add_argument("--n-tool", type=int, default=1500)
    ap.add_argument("--n-qa", type=int, default=3500)
    ap.add_argument("--qa-src", default=str(REPO / "data/training/sft_real_v1/train.helix.jsonl"))
    ap.add_argument("--val", type=int, default=200)
    ap.add_argument("--mode", choices=["call_only", "full"], default="call_only")
    ap.add_argument(
        "--simple-rebump",
        action="store_true",
        help="Increase simple/square/sqrt trace weight for Phase 2 after the Phase-1.1 simple-bucket dip.",
    )
    ap.add_argument("--seed", type=int, default=20260606)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    gens = PHASE2_SIMPLE_REBUMP_GENS if a.simple_rebump else BASE_GENS
    tool = gen_tool(rng, a.n_tool, a.mode, gens=gens)
    qa = sample_qa(rng, a.n_qa, a.qa_src)
    rows = tool + qa
    rng.shuffle(rows)
    nval = min(a.val, len(rows) // 20)
    val, train = rows[:nval], rows[nval:]
    out = pathlib.Path(a.out_dir + ("_" + a.mode))
    out.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val)]:
        with open(out / f"{name}.helix.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    nt = sum(r["has_tool"] for r in train)
    print(
        f"=== tool-SFT [{a.mode}]: train {len(train)} (tool {nt} / qa {len(train) - nt}) | val {len(val)} ==="
    )
    print(f"    -> {out}")
    print("\n=== 3 SAMPLE TOOL TRACES ===")
    for r in [x for x in train if x["has_tool"]][:3]:
        print("-" * 60)
        print(r["text"].split("<|assistant|>")[0].split("<|user|>")[1].strip(), "=>")
        print(r["text"].split("<|assistant|>")[1].split("<|end|>")[0].strip())


if __name__ == "__main__":
    main()
