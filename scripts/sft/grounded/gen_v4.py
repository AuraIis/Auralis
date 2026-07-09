#!/usr/bin/env python3
"""H-Grounded v4 = v3 + THREE targeted buckets for the v3 gate failures (different specifics
than the gate, so it teaches PATTERNS not gate instances):
  A) dense 5-sentence prose -> extract each fact (kills long-prose over-refusal)
  B) "Wie viele X?" when X is named but NOT counted -> refuse ; when counted -> answer
  C) begin/end & von-bis disambiguation (two times, ask the right one)."""

import json
import random

random.seed(44)
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


def cap(s):
    return s[0].upper() + s[1:]


rows = [
    json.loads(l) for l in open(OUT + "/grounded_v3.jsonl", encoding="utf-8")
]  # inherit all of v3

# ---------- (A) dense multi-sentence prose (Greifenau-style, fictional towns) ----------
TOWNS = [
    "Talheim",
    "Moosbach",
    "Eichfeld",
    "Rauental",
    "Sonnberg",
    "Weidach",
    "Kirchsee",
    "Hagenbruck",
    "Falkenried",
    "Brunnthal",
    "Erlbach",
    "Auental",
    "Lichtenfeld",
    "Wieshofen",
    "Steinau",
    "Reichenau",
    "Hochdorf",
    "Lindenberg",
    "Buchholz",
    "Neubrunn",
]
THINGS = [
    "ihre Toepferkunst",
    "ihren Wochenmarkt",
    "ihre alte Muehle",
    "ihr Weinfest",
    "ihre Glasblaeserei",
    "ihren Kraeutergarten",
    "ihre Holzbruecke",
    "ihr Heimatmuseum",
    "ihre Kaesereien",
    "ihr Lichterfest",
]
LAND = [
    "ein Brunnen aus Stein",
    "eine alte Kirche",
    "ein hoher Uhrturm",
    "ein kleines Rathaus",
    "eine steinerne Bruecke",
]
for t in TOWNS:
    year = ri(1100, 1850)
    pop = ri(800, 9000)
    thing = ch(THINGS)
    land = ch(LAND)
    ctx = (
        f"Die Kleinstadt {t} liegt in einem Tal. Sie wurde {year} gegruendet. "
        f"Rund {pop} Menschen leben dort. Bekannt ist sie fuer {thing}. Im Zentrum steht {land}."
    )
    rows.append(it(ctx, f"Wann wurde {t} gegruendet?", f"{t} wurde {year} gegruendet.", True))
    rows.append(
        it(ctx, f"Wie viele Menschen leben in {t}?", f"In {t} leben rund {pop} Menschen.", True)
    )
    rows.append(it(ctx, f"Wofuer ist {t} bekannt?", f"{t} ist bekannt fuer {thing}.", True))
    rows.append(
        it(
            ctx,
            ch(
                [
                    f"Wie heisst der Buergermeister von {t}?",
                    f"Welche Sprache spricht man in {t}?",
                    f"Wie hoch liegt {t} ueber dem Meer?",
                ]
            ),
            ch(REF),
            False,
        )
    )

# ---------- (B) count: named-but-uncounted -> refuse ; counted -> answer ----------
CONT = [
    ("Korb", "liegen", ["Aepfel", "Birnen", "Bananen", "Kirschen", "Pflaumen"]),
    ("Regal", "stehen", ["Buecher", "Ordner", "Vasen", "Glaeser"]),
    ("Karton", "sind", ["Tassen", "Teller", "Loeffel", "Gabeln"]),
    ("Beet", "wachsen", ["Rosen", "Tulpen", "Nelken", "Lilien"]),
    ("Stall", "stehen", ["Kuehe", "Schafe", "Ziegen", "Pferde"]),
]
for _ in range(55):  # uncounted -> refuse
    c, v, pool = ch(CONT)
    pick = random.sample(pool, 3)
    lst = ", ".join(pick[:-1]) + " und " + pick[-1]
    ctx = f"Im {c} {v} {lst}."
    rows.append(
        it(
            ctx,
            f"Wie viele {ch(pick)} {v.replace('sind', 'sind').replace('liegen', 'liegen')} im {c}?".replace(
                "  ", " "
            ),
            ch(REF),
            False,
        )
    )
for _ in range(45):  # counted -> answer
    c, v, pool = ch(CONT)
    a, b = random.sample(pool, 2)
    na = ri(2, 9)
    nb = ri(2, 9)
    ctx = f"Im {c} {v} {na} {a} und {nb} {b}."
    if random.random() < 0.5:
        rows.append(it(ctx, f"Wie viele {a} {v} im {c}?", f"{na} {a}.", True))
    else:
        rows.append(it(ctx, f"Wie viele {b} {v} im {c}?", f"{nb} {b}.", True))

# ---------- (C) begin/end & von-bis disambiguation (two times) ----------
EVENTS = ["Konzert", "Vorstellung", "Sitzung", "Schulung", "Fuehrung", "Vorlesung", "Probe"]
for _ in range(28):
    e = ch(EVENTS)
    a = ri(8, 14)
    b = ri(a + 2, 21)
    ctx = f"Das {e} beginnt um {a} Uhr und endet um {b} Uhr."
    if random.random() < 0.5:
        rows.append(it(ctx, f"Wann endet das {e}?", f"Um {b} Uhr.", True))
    else:
        rows.append(it(ctx, f"Wann beginnt das {e}?", f"Um {a} Uhr.", True))
PLACES = ["Laden", "Markt", "Schalter", "Kiosk", "Imbiss", "Baeckerei"]
for _ in range(28):
    p = ch(PLACES)
    a = ri(6, 11)
    b = ri(a + 4, 22)
    ctx = f"Der {p} hat von {a} bis {b} Uhr geoeffnet."
    if random.random() < 0.5:
        rows.append(it(ctx, f"Bis wann hat der {p} geoeffnet?", f"Bis {b} Uhr.", True))
    else:
        rows.append(it(ctx, f"Ab wann hat der {p} geoeffnet?", f"Ab {a} Uhr.", True))

random.shuffle(rows)
seen = set()
uniq = []
for r in rows:
    k = (r["context"][:70] + "|" + r["q"]).lower()
    if k in seen:
        continue
    seen.add(k)
    uniq.append(r)
with open(OUT + "/grounded_v4.jsonl", "w", encoding="utf-8") as f:
    for r in uniq:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
na = sum(1 for r in uniq if r["answerable"])
print(
    f"grounded_v4: {len(uniq)} | answerable {na} ({100 * na // len(uniq)}%) | refuse {len(uniq) - na}"
)
