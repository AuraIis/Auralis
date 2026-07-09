#!/usr/bin/env python3
"""H-Grounded v3 — DETERMINISTIC, targets the TWO v2 gaps only:
   (1) structural variety  (2) distractor resolution.
Guaranteed-correct by construction. ~960 items, 60% answerable / 40% traps.
Answerable split ~ 40% numbers-in-varied-prose / 25% similar-entities / 20% time / 15% lists."""

import json
import os
import random

random.seed(41)
rnd = random.random
ch = random.choice
ri = random.randint
OUT = "/workspace/v2data/data/training/sft_grounded"
os.makedirs(OUT, exist_ok=True)
REF = [
    "Das steht nicht im Text.",
    "Das geht aus dem Text nicht hervor.",
    "Im Kontext steht das nicht.",
]


def it(c, q, a, ans):
    return {"context": c, "q": q, "a": a, "answerable": ans}


def cap(s):
    return s[0].upper() + s[1:]


rows = []

# ============ (A) NUMBERS in VARIED prose (40% of answerable) ============
ATTR = [  # frag(n) , question , answer , lo , hi
    ("wiegt {n} Kilogramm", "Wie schwer ist {s}?", "{S} wiegt {n} Kilogramm.", 3, 4000),
    ("ist {n} Meter lang", "Wie lang ist {s}?", "{S} ist {n} Meter lang.", 2, 400),
    ("kostet {n} Euro", "Wie viel kostet {s}?", "{S} kostet {n} Euro.", 5, 5000),
    ("hat {n} Seiten", "Wie viele Seiten hat {s}?", "{S} hat {n} Seiten.", 20, 900),
    ("fasst {n} Liter", "Wie viel fasst {s}?", "{S} fasst {n} Liter.", 1, 1000),
    ("hat {n} Zimmer", "Wie viele Zimmer hat {s}?", "{S} hat {n} Zimmer.", 2, 40),
    ("ist {n} Grad warm", "Wie warm ist {s}?", "{S} ist {n} Grad warm.", 5, 95),
    ("erreicht {n} km/h", "Wie schnell ist {s}?", "{S} erreicht {n} km/h.", 20, 320),
]
SUBJ = [
    "das Lagerhaus",
    "der Lieferwagen",
    "das Aquarium",
    "die Werkhalle",
    "das Hotel",
    "der Tank",
    "die Yacht",
    "das Ferienhaus",
    "der Reisebus",
    "die Kiste",
    "das Gebaeude",
    "der Kessel",
    "die Maschine",
    "das Boot",
]


def num_single():
    s = ch(SUBJ)
    fr, q, an, lo, hi = ch(ATTR)
    n = ri(lo, hi)
    S = cap(s)
    fact = fr.format(n=n)
    wraps = [
        f"{S} {fact}.",
        f"{S} ist neu. {S} {fact}.",
        f"{S} {fact} und steht im Lager.",
        f"Bekannt ist: {S} {fact}.",
        f"{S} liegt am Rand. {S} {fact}.",
    ]
    return it(ch(wraps), q.format(s=s), an.format(S=S, n=n), True)


def num_double():  # two numbers in context, ask one -> OTHER number is a distractor
    s = ch(SUBJ)
    (fr1, q1, an1, l1, h1), (fr2, q2, an2, l2, h2) = random.sample(ATTR, 2)
    n1 = ri(l1, h1)
    n2 = ri(l2, h2)
    S = cap(s)
    ctx = f"{S} {fr1.format(n=n1)} und {fr2.format(n=n2)}."
    return (
        it(ctx, q1.format(s=s), an1.format(S=S, n=n1), True)
        if rnd() < 0.5
        else it(ctx, q2.format(s=s), an2.format(S=S, n=n2), True)
    )


def num_trap():  # 2 attrs present, ask a DIFFERENT absent attr
    s = ch(SUBJ)
    present = random.sample(ATTR, 2)
    absent = ch([a for a in ATTR if a not in present])
    S = cap(s)
    facts = " und ".join(p[0].format(n=ri(p[3], p[4])) for p in present)
    return it(f"{S} {facts}.", absent[1].format(s=s), ch(REF), False)


for _ in range(120):
    rows.append(num_single())
for _ in range(120):
    rows.append(num_double())
for _ in range(150):
    rows.append(num_trap())

# ============ (B) DISTRACTORS: similar entities (25% of answerable) ============
PAIRS = [
    ("Schmidt", "Schmitt"),
    ("Anna", "Anne"),
    ("Meier", "Meyer"),
    ("Bauer", "Baur"),
    ("Kraus", "Krause"),
    ("Peter", "Petra"),
    ("Mueller", "Moeller"),
    ("Hofmann", "Hoffmann"),
    ("Weber", "Wagner"),
    ("Lang", "Lange"),
]
DATTR = [  # frag(v) , question , answer , values (ctx/answer split by '/')
    (
        "wohnt in {v}",
        "Wo wohnt {who}?",
        "{who} wohnt in {v}.",
        ["Koeln", "Bonn", "Mainz", "Trier", "Ulm", "Kassel", "Jena", "Fulda"],
    ),
    (
        "traegt eine {v} Jacke",
        "Welche Farbe hat die Jacke von {who}?",
        "Die Jacke von {who} ist {v}.",
        ["blaue/blau", "rote/rot", "gruene/gruen", "gelbe/gelb", "schwarze/schwarz"],
    ),
    (
        "faehrt ein {v} Auto",
        "Welche Farbe hat das Auto von {who}?",
        "Das Auto von {who} ist {v}.",
        ["blaue/blau", "rote/rot", "weisse/weiss", "graue/grau"],
    ),
    (
        "ist {v} Jahre alt",
        "Wie alt ist {who}?",
        "{who} ist {v} Jahre alt.",
        ["12", "15", "28", "34", "41", "57"],
    ),
    (
        "arbeitet als {v}",
        "Als was arbeitet {who}?",
        "{who} arbeitet als {v}.",
        ["Lehrer", "Arzt", "Tischler", "Pilot", "Koch", "Maler"],
    ),
]


def dv(raw):
    return raw.split("/") if "/" in raw else (raw, raw)


def distractor():
    A, B = ch(PAIRS)
    frag, q, ans, vals = ch(DATTR)
    vA = ch(vals)
    vB = ch([v for v in vals if v != vA])
    ctx = f"{A} {frag.format(v=dv(vA)[0])}. {B} {frag.format(v=dv(vB)[0])}."
    who = ch([A, B])
    v = dv(vA)[1] if who == A else dv(vB)[1]
    return it(ctx, q.format(who=who), ans.format(who=who, v=v), True)


def distractor_trap():  # ask a THIRD name not in context
    A, B = ch(PAIRS)
    C = ch([n for p in PAIRS for n in p if n not in (A, B)])
    frag, q, ans, vals = ch(DATTR)
    vA = ch(vals)
    vB = ch([v for v in vals if v != vA])
    ctx = f"{A} {frag.format(v=dv(vA)[0])}. {B} {frag.format(v=dv(vB)[0])}."
    return it(ctx, q.format(who=C), ch(REF), False)


for _ in range(150):
    rows.append(distractor())
for _ in range(70):
    rows.append(distractor_trap())

# ============ (C) TIME: seit / von-bis / dates (20% of answerable) ============
TS = [
    (
        "Das Museum ist seit {n} Tagen geschlossen.",
        "Seit wann ist das Museum geschlossen?",
        "Seit {n} Tagen.",
        lambda: ri(2, 30),
    ),
    (
        "Der Laden oeffnet von 8 bis {n} Uhr.",
        "Bis wann ist der Laden geoeffnet?",
        "Bis {n} Uhr.",
        lambda: ri(16, 22),
    ),
    (
        "Die Sitzung beginnt um {n} Uhr.",
        "Wann beginnt die Sitzung?",
        "Um {n} Uhr.",
        lambda: ri(8, 17),
    ),
    (
        "Er arbeitet dort seit {n}.",
        "Seit welchem Jahr arbeitet er dort?",
        "Seit {n}.",
        lambda: ri(1995, 2022),
    ),
    (
        "Die Veranstaltung dauert {n} Stunden.",
        "Wie lange dauert die Veranstaltung?",
        "{n} Stunden.",
        lambda: ri(2, 9),
    ),
    (
        "Das Projekt laeuft seit {n} Wochen.",
        "Seit wie vielen Wochen laeuft das Projekt?",
        "Seit {n} Wochen.",
        lambda: ri(2, 40),
    ),
    (
        "Die Lieferung kommt in {n} Tagen.",
        "In wie vielen Tagen kommt die Lieferung?",
        "In {n} Tagen.",
        lambda: ri(2, 14),
    ),
]
for _ in range(120):
    c, q, a, g = ch(TS)
    n = g()
    rows.append(it(c.format(n=n), q, a.format(n=n), True))
for _ in range(40):  # ask a different scenario's question -> not in this context
    i = random.randrange(len(TS))
    j = ch([k for k in range(len(TS)) if k != i])
    c, q, a, g = TS[i]
    n = g()
    rows.append(it(c.format(n=n), TS[j][1], ch(REF), False))

# ============ (D) LISTS / enumerations (15% of answerable) ============
LISTS = [
    ("das Cafe", "bietet", " an", ["Kaffee", "Tee", "Kakao", "Saft", "Limonade", "Suppe"]),
    (
        "der Werkzeugkasten",
        "enthaelt",
        "",
        ["Hammer", "Zange", "Schraubenzieher", "Saege", "Feile", "Bohrer"],
    ),
    ("das Menue", "umfasst", "", ["Suppe", "Salat", "Pasta", "Fisch", "Kuchen", "Eis"]),
    ("der Bauernhof", "haelt", "", ["Kuehe", "Schafe", "Ziegen", "Pferde", "Huehner", "Enten"]),
]
TRAP_Q = [
    "Wie hoch ist der Preis?",
    "Wann ist geoeffnet?",
    "Wo befindet sich das?",
    "Wie viele Stueck gibt es?",
]
for i in range(90):
    subj, verb, suf, pool = ch(LISTS)
    k = ch([3, 3, 4])
    pick = random.sample(pool, k)
    lst = ", ".join(pick[:-1]) + " und " + pick[-1]
    rows.append(
        it(
            cap(f"{subj} {verb} {lst}{suf}."),
            f"Was {verb} {subj}?",
            cap(f"{subj} {verb} {lst}."),
            True,
        )
    )
    if i % 2 == 0:
        rows.append(it(cap(f"{subj} {verb} {lst}{suf}."), ch(TRAP_Q), ch(REF), False))

# ============ (E) world-knowledge traps (keep — these work) ============
WORLD = [
    ("Der Eiffelturm steht in Paris.", "Wie hoch ist der Eiffelturm?"),
    ("Paris ist eine schoene Stadt.", "In welchem Land liegt Paris?"),
    ("Goethe schrieb Faust.", "Wann wurde Goethe geboren?"),
    ("Einstein war ein Physiker.", "Welche Theorie stellte Einstein auf?"),
    ("Mozart komponierte die Zauberfloete.", "Wo wurde Mozart geboren?"),
    ("Der Mount Everest liegt im Himalaya.", "Wie hoch ist der Mount Everest?"),
    ("Berlin ist die deutsche Hauptstadt.", "Wie viele Einwohner hat Berlin?"),
    ("Die Donau ist ein Fluss.", "Durch welche Laender fliesst die Donau?"),
    ("Shakespeare war ein Dramatiker.", "Welches Stueck schrieb Shakespeare ueber Daenemark?"),
    ("Beethoven war Komponist.", "Wann wurde Beethoven geboren?"),
]
for c, q in WORLD:
    for _ in range(6):
        rows.append(it(c, q, ch(REF), False))

random.shuffle(rows)
seen = set()
uniq = []
for r in rows:
    k = (r["context"][:70] + "|" + r["q"]).lower()
    if k in seen:
        continue
    seen.add(k)
    uniq.append(r)
with open(OUT + "/grounded_v3.jsonl", "w", encoding="utf-8") as f:
    for r in uniq:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
na = sum(1 for r in uniq if r["answerable"])
print(
    f"grounded_v3: {len(uniq)} | answerable {na} ({100 * na // len(uniq)}%) | refuse {len(uniq) - na}"
)
