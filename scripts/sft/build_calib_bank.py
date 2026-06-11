#!/usr/bin/env python3
"""Build a GOLD question bank for calibration self-labeling (key-free).

Two kinds of rows:
  - facts:    (question, gold_terms, cat)  -> model SHOULD answer; gold is the truth.
  - invented: (question, None, cat)        -> entity does NOT exist; model SHOULD abstain.

Only facts I am confident about are included (gold must be correct, else we'd train
wrong abstention). Output JSONL is consumed by calib_probe.py --bank."""
import argparse, json, random, pathlib

REPO = pathlib.Path("/workspace/v2data")

CAPITALS = {
    "Deutschland": "Berlin", "Frankreich": "Paris", "Italien": "Rom", "Spanien": "Madrid",
    "Oesterreich": "Wien", "Schweiz": "Bern", "Polen": "Warschau", "Niederlande": "Amsterdam",
    "Belgien": "Bruessel", "Portugal": "Lissabon", "Griechenland": "Athen", "Schweden": "Stockholm",
    "Norwegen": "Oslo", "Daenemark": "Kopenhagen", "Finnland": "Helsinki", "Russland": "Moskau",
    "Japan": "Tokio", "China": "Peking", "USA": "Washington", "Kanada": "Ottawa",
    "Mexiko": "Mexiko-Stadt", "Australien": "Canberra", "Aegypten": "Kairo", "Tuerkei": "Ankara",
    "Grossbritannien": "London", "Irland": "Dublin", "Tschechien": "Prag", "Ungarn": "Budapest",
    "Argentinien": "Buenos Aires", "Suedkorea": "Seoul", "Thailand": "Bangkok", "Kuba": "Havanna",
    "Kenia": "Nairobi", "Brasilien": "Brasilia", "Indien": "Delhi",
}
WORKS = {
    "Faust": "Goethe", "Die Verwandlung": "Kafka", "Der Prozess": "Kafka",
    "Romeo und Julia": "Shakespeare", "Hamlet": "Shakespeare", "Die Zauberfloete": "Mozart",
    "Die Raeuber": "Schiller", "Wilhelm Tell": "Schiller", "Buddenbrooks": "Mann",
    "Der Steppenwolf": "Hesse", "Das Parfum": "Sueskind", "Die Blechtrommel": "Grass",
    "Effi Briest": "Fontane", "Krieg und Frieden": "Tolstoi", "Don Quijote": "Cervantes",
    "Die Odyssee": "Homer",
}
ELEMENTS = {
    "Au": "Gold", "Ag": "Silber", "Fe": "Eisen", "O": "Sauerstoff", "H": "Wasserstoff",
    "C": "Kohlenstoff", "N": "Stickstoff", "Na": "Natrium", "Cl": "Chlor", "He": "Helium",
    "Cu": "Kupfer", "Pb": "Blei", "Sn": "Zinn", "K": "Kalium", "Ca": "Calcium",
    "Hg": "Quecksilber", "Zn": "Zink", "U": "Uran",
}
MISC = [
    ("Wie heisst der hoechste Berg der Welt?", ["everest"], "geo"),
    ("Welcher Ozean ist der groesste?", ["pazifik"], "geo"),
    ("Welcher Planet ist der Sonne am naechsten?", ["merkur"], "science"),
    ("Welcher ist der groesste Planet im Sonnensystem?", ["jupiter"], "science"),
    ("Welcher Planet wird der rote Planet genannt?", ["mars"], "science"),
    ("Wer entdeckte das Penicillin?", ["fleming"], "science"),
    ("Wer entwickelte die Relativitaetstheorie?", ["einstein"], "science"),
    ("Welcher Wissenschaftler stellte die Evolutionstheorie auf?", ["darwin"], "science"),
    ("Wer war der erste Mensch auf dem Mond?", ["armstrong"], "history"),
    ("Wer war der erste Bundeskanzler der Bundesrepublik Deutschland?", ["adenauer"], "history"),
    ("In welchem Jahr fiel die Berliner Mauer?", ["1989"], "history"),
    ("In welchem Jahr begann der Zweite Weltkrieg?", ["1939"], "history"),
    ("Wie viele Kontinente gibt es?", ["sieben", "7"], "geo"),
    ("Wie viele Bundeslaender hat Deutschland?", ["16", "sechzehn"], "geo"),
    ("Was ist die chemische Formel von Wasser?", ["h2o", "h₂o"], "science"),
    ("Welches Gas nehmen Pflanzen fuer die Photosynthese auf?", ["kohlendioxid", "co2", "co₂"], "science"),
    ("Welches Gas atmen Menschen zum Leben ein?", ["sauerstoff"], "science"),
    ("Wie heisst die Waehrung in Japan?", ["yen"], "geo"),
    ("Wie heisst die Waehrung in den USA?", ["dollar"], "geo"),
    ("Welcher Fluss fliesst durch Wien?", ["donau"], "geo"),
    ("Welcher Fluss fliesst durch Paris?", ["seine"], "geo"),
    ("Wie viele Beine hat eine Spinne?", ["acht", "8"], "science"),
    ("Wie viele Beine hat ein Insekt?", ["sechs", "6"], "science"),
    ("Welches Tier ist das groesste der Welt?", ["blauwal", "wal"], "science"),
    ("Welche Farbe entsteht aus Blau und Gelb?", ["gruen"], "science"),
]

SYL = ["bra", "vor", "quel", "zar", "pli", "grom", "fendt", "mox", "tarn", "welk", "drub",
       "skel", "yon", "krav", "pomb", "thal", "wisp", "gorn", "flim", "bos", "nurr", "kel",
       "vant", "rho", "klin", "murr", "dax", "frell", "ston", "glubb"]


def cap(s):
    return s[0].upper() + s[1:]


def make_word(rng, n=2):
    return "".join(rng.choice(SYL) for _ in range(n))


def invented(rng, n):
    out, seen = [], set()
    tmpl = [
        lambda: (f"Wer war {cap(make_word(rng))} {cap(make_word(rng))}?",),
        lambda: (f"Was ist ein {cap(make_word(rng))}?",),
        lambda: (f"Wer schrieb das Buch '{cap(make_word(rng))} von {cap(make_word(rng))}'?",),
        lambda: (f"In welchem Land liegt die Region {cap(make_word(rng))}?",),
        lambda: (f"Was bedeutet das Wort '{cap(make_word(rng, 3))}'?",),
        lambda: (f"Welche Hauptstadt hat das Land {cap(make_word(rng))}?",),
        lambda: (f"Wer erfand die {cap(make_word(rng))}-Maschine?",),
        lambda: (f"Was ist die chemische Formel von {cap(make_word(rng))}?",),
    ]
    tries = 0
    while len(out) < n and tries < n * 40:
        tries += 1
        q = rng.choice(tmpl)()[0]
        if q in seen:
            continue
        seen.add(q); out.append((q, None, "invented"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "data/training/calib/calib_bank.jsonl"))
    ap.add_argument("--n-invented", type=int, default=110)
    ap.add_argument("--seed", type=int, default=20260607)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = []
    for land, hs in CAPITALS.items():
        rows.append((f"Was ist die Hauptstadt von {land}?", [hs.split('-')[0].lower(), hs.lower()], "capital"))
    for work, author in WORKS.items():
        rows.append((f"Wer schrieb '{work}'?", [author.lower()], "work"))
    for sym, name in ELEMENTS.items():
        rows.append((f"Welches Element hat das chemische Symbol {sym}?", [name.lower()], "element"))
    rows.extend(MISC)
    rows.extend(invented(rng, a.n_invented))
    rng.shuffle(rows)
    p = pathlib.Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for q, gold, cat in rows:
            f.write(json.dumps({"q": q, "gold": gold, "cat": cat}, ensure_ascii=False) + "\n")
    nfact = sum(1 for _, g, _ in rows if g is not None)
    ninv = sum(1 for _, g, _ in rows if g is None)
    print(f"=== calib bank: {len(rows)} rows (facts {nfact} / invented {ninv}) -> {a.out} ===")
    from collections import Counter
    print("by cat:", dict(Counter(c for _, _, c in rows)))


if __name__ == "__main__":
    main()
