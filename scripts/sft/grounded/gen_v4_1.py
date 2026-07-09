#!/usr/bin/env python3
"""H-Grounded v4.1 = v4 + FOUR micro-buckets fixing the v4 gate regressions/gaps
(generic patterns, randomized specifics != gate instances):
  F1) year + attribute -> extract the NON-year number (kills 'Roman 412 -> 2850' confab)
  F2) third-entity car/color -> refuse ; present entity -> answer (kills 'Walter -> blau')
  F3) dense prose with event-month -> extract the month (kills Greifenau-month over-refuse)
  F4) 'haelt/besitzt/zeigt' count: named-uncounted -> refuse ; counted -> answer (verb-generalize)."""

import json
import random

random.seed(46)
ch = random.choice
ri = random.randint
OUT = "/workspace/v2data/data/training/sft_grounded"
REF = [
    "Das steht nicht im Text.",
    "Das geht aus dem Text nicht hervor.",
    "Im Kontext steht das nicht.",
]


def it(c, q, a, ans):
    return {"context": c, "q": q, "a": a, "answerable": ans}


rows = [
    json.loads(l) for l in open(OUT + "/grounded_v4.jsonl", encoding="utf-8")
]  # inherit all of v4

# ---- F1: year + attribute, extract the non-year number ----
WORKS = [
    ("Das Buch", "Seiten"),
    ("Der Roman", "Seiten"),
    ("Der Bericht", "Seiten"),
    ("Die Studie", "Seiten"),
    ("Das Album", "Lieder"),
    ("Der Film", "Minuten"),
    ("Das Heft", "Kapitel"),
    ("Der Katalog", "Seiten"),
    ("Die Broschuere", "Seiten"),
    ("Das Hoerbuch", "Minuten"),
]
for _ in range(60):
    w, unit = ch(WORKS)
    qw = w[0].lower() + w[1:]
    year = ri(1890, 2024)
    n = ri(40, 950)
    ctx = f"{w} erschien {year} und umfasst {n} {unit}."
    if random.random() < 0.6:
        rows.append(it(ctx, f"Wie viele {unit} hat {qw}?", f"{w} hat {n} {unit}.", True))
    else:
        rows.append(it(ctx, f"Wann erschien {qw}?", f"{w} erschien {year}.", True))

# ---- F2: third-entity car/color (refuse) + present entity (answer) ----
NAMES2 = [
    "Weber",
    "Wagner",
    "Walter",
    "Bauer",
    "Berg",
    "Hofer",
    "Keller",
    "Brandt",
    "Schulz",
    "Voss",
    "Reich",
    "Sommer",
    "Fuchs",
    "Vogt",
]
COLORS = [
    "blaues/blau",
    "rotes/rot",
    "gruenes/gruen",
    "weisses/weiss",
    "graues/grau",
    "schwarzes/schwarz",
]
for _ in range(40):  # ask a THIRD driver -> refuse
    a, b, c = random.sample(NAMES2, 3)
    col1, col2 = random.sample(COLORS, 2)
    ctx = f"Herr {a} faehrt ein {col1.split('/')[0]} Auto, Herr {b} ein {col2.split('/')[0]}."
    rows.append(it(ctx, f"Welche Farbe hat das Auto von Herrn {c}?", ch(REF), False))
for _ in range(30):  # ask a PRESENT driver -> answer (guard vs over-refusal)
    a, b = random.sample(NAMES2, 2)
    col1, col2 = random.sample(COLORS, 2)
    c1c, c1a = col1.split("/")
    c2c, c2a = col2.split("/")
    ctx = f"Herr {a} faehrt ein {c1c} Auto, Herr {b} ein {c2c}."
    who, ca = ch([(a, c1a), (b, c2a)])
    rows.append(
        it(
            ctx,
            f"Welche Farbe hat das Auto von Herrn {who}?",
            f"Das Auto von Herrn {who} ist {ca}.",
            True,
        )
    )

# ---- F3: dense prose with event-month -> extract month ----
MONTHS = [
    "Januar",
    "Februar",
    "Maerz",
    "April",
    "Mai",
    "Juni",
    "Juli",
    "August",
    "September",
    "Oktober",
    "November",
    "Dezember",
]
FESTS = [
    "Toepfermesse",
    "Weinfest",
    "Lichterfest",
    "Kraeutermarkt",
    "Erntefest",
    "Musikfest",
    "Fruehlingsfest",
]
TOWNS2 = [
    "Talheim",
    "Moosbach",
    "Rauental",
    "Sonnberg",
    "Kirchsee",
    "Falkenried",
    "Erlbach",
    "Auental",
    "Wieshofen",
    "Hochdorf",
    "Brunnthal",
    "Steinau",
]
for t in TOWNS2:
    year = ri(1100, 1850)
    pop = ri(800, 9000)
    fest = ch(FESTS)
    mon = ch(MONTHS)
    ctx = (
        f"Die Stadt {t} wurde {year} gegruendet. Rund {pop} Menschen leben dort. "
        f"Ihr {fest} findet jedes Jahr im {mon} statt."
    )
    rows.append(it(ctx, f"In welchem Monat findet das {fest} statt?", f"Im {mon}.", True))
    rows.append(
        it(ctx, f"Wie viele Menschen leben in {t}?", f"In {t} leben rund {pop} Menschen.", True)
    )

# ---- F4: 'haelt/besitzt/zeigt' count, named-uncounted -> refuse ; counted -> answer ----
SUBJ4 = [
    ("Bauernhof", "haelt"),
    ("Bauer", "besitzt"),
    ("Zoo", "zeigt"),
    ("Hof", "haelt"),
    ("Reiterhof", "haelt"),
    ("Tierpark", "zeigt"),
]
ANIM = ["Kuehe", "Schafe", "Ziegen", "Pferde", "Huehner", "Esel", "Gaense", "Enten"]
for _ in range(30):  # uncounted -> refuse
    s, v = ch(SUBJ4)
    pick = random.sample(ANIM, 3)
    lst = ", ".join(pick[:-1]) + " und " + pick[-1]
    rows.append(it(f"Der {s} {v} {lst}.", f"Wie viele {ch(pick)} {v} der {s}?", ch(REF), False))
for _ in range(25):  # counted -> answer
    s, v = ch(SUBJ4)
    a, b = random.sample(ANIM, 2)
    na = ri(3, 40)
    rows.append(
        it(
            f"Der {s} {v} {na} {a} und einige {b}.",
            f"Wie viele {a} {v} der {s}?",
            f"{na} {a}.",
            True,
        )
    )

random.shuffle(rows)
seen = set()
uniq = []
for r in rows:
    k = (r["context"][:70] + "|" + r["q"]).lower()
    if k in seen:
        continue
    seen.add(k)
    uniq.append(r)
with open(OUT + "/grounded_v4_1.jsonl", "w", encoding="utf-8") as f:
    for r in uniq:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
na = sum(1 for r in uniq if r["answerable"])
print(
    f"grounded_v4_1: {len(uniq)} | answerable {na} ({100 * na // len(uniq)}%) | refuse {len(uniq) - na}"
)
