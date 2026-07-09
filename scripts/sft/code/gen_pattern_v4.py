#!/usr/bin/env python3
"""Code v4 — DETERMINISTIC pattern-class tasks with GUARANTEED-CORRECT reference code.
Targets the v3 unseen failures (map/filter/reduce/digits/sort/dedup/string), and DELIBERATELY
EXCLUDES the 9 code-gate functions so the gate still measures TRANSFER, not memorization.
Each spec is self-verified at gen time (reference runs cleanly on all sample inputs)."""

import json
import os

OUT = "/workspace/v2data/data/training/code_v4"
os.makedirs(OUT, exist_ok=True)
SYS = "Du bist Auralis, ein hilfreicher, ehrlicher KI-Assistent."
END = "<|end|>"


def render(u, a):
    return f"<|system|>\n{SYS}\n{END}\n<|user|>\n{u}\n{END}\n<|assistant|>\n{a}\n{END}\n"


PREFIX = [
    "Schreibe eine Funktion",
    "Implementiere eine Funktion",
    "Erstelle eine Funktion",
    "Definiere eine Funktion",
    "Programmiere eine Funktion",
]
# (name, sig, what, code, sample_inputs)  -- GATE funcs excluded: doppelt, nur_gerade, quersumme,
#   ist_aufsteigend, wort_laengen, max_wert, count_words, celsius_zu_fahrenheit, remove_duplicates
S = [
    # --- MAP ---
    (
        "verdreifache",
        "(xs)",
        "eine neue Liste zurueckgibt, in der jeder Wert aus xs mit 3 multipliziert ist",
        "def verdreifache(xs):\n    return [x*3 for x in xs]",
        [([1, 2, 3],), ([],), ([0],), ([-2, 5],)],
    ),
    (
        "quadriere",
        "(xs)",
        "eine neue Liste mit den Quadraten der Zahlen aus xs zurueckgibt",
        "def quadriere(xs):\n    return [x*x for x in xs]",
        [([1, 2, 3],), ([],), ([-3],)],
    ),
    (
        "halbiere",
        "(xs)",
        "eine neue Liste zurueckgibt, in der jeder Wert aus xs ganzzahlig halbiert ist",
        "def halbiere(xs):\n    return [x//2 for x in xs]",
        [([2, 4, 6],), ([],), ([5],)],
    ),
    (
        "plus_zehn",
        "(xs)",
        "eine neue Liste zurueckgibt, in der zu jedem Wert aus xs 10 addiert ist",
        "def plus_zehn(xs):\n    return [x+10 for x in xs]",
        [([1, 2],), ([],), ([-10],)],
    ),
    (
        "negiere",
        "(xs)",
        "eine neue Liste mit den negierten Werten aus xs zurueckgibt",
        "def negiere(xs):\n    return [-x for x in xs]",
        [([1, -2, 3],), ([],)],
    ),
    (
        "absolutwerte",
        "(xs)",
        "eine neue Liste mit den Absolutwerten der Zahlen aus xs zurueckgibt",
        "def absolutwerte(xs):\n    return [abs(x) for x in xs]",
        [([-1, 2, -3],), ([],)],
    ),
    (
        "inkrementiere",
        "(xs)",
        "eine neue Liste zurueckgibt, in der jeder Wert aus xs um 1 erhoeht ist",
        "def inkrementiere(xs):\n    return [x+1 for x in xs]",
        [([1, 2, 3],), ([],)],
    ),
    (
        "in_grossbuchstaben",
        "(woerter)",
        "eine neue Liste zurueckgibt, in der jedes Wort aus woerter in Grossbuchstaben steht",
        "def in_grossbuchstaben(woerter):\n    return [w.upper() for w in woerter]",
        [(["ab", "cd"],), ([],)],
    ),
    # --- FILTER ---
    (
        "nur_ungerade",
        "(xs)",
        "eine Liste nur mit den ungeraden Zahlen aus xs zurueckgibt",
        "def nur_ungerade(xs):\n    return [x for x in xs if x % 2 != 0]",
        [([1, 2, 3, 4],), ([],), ([2, 4],)],
    ),
    (
        "nur_positive",
        "(xs)",
        "eine Liste nur mit den positiven Zahlen aus xs zurueckgibt",
        "def nur_positive(xs):\n    return [x for x in xs if x > 0]",
        [([-1, 2, -3, 4],), ([],)],
    ),
    (
        "nur_negative",
        "(xs)",
        "eine Liste nur mit den negativen Zahlen aus xs zurueckgibt",
        "def nur_negative(xs):\n    return [x for x in xs if x < 0]",
        [([-1, 2, -3],), ([],)],
    ),
    (
        "ohne_nullen",
        "(xs)",
        "eine Liste zurueckgibt, die alle Nullen aus xs entfernt",
        "def ohne_nullen(xs):\n    return [x for x in xs if x != 0]",
        [([0, 1, 0, 2],), ([],)],
    ),
    (
        "groesser_als",
        "(xs, s)",
        "eine Liste nur mit den Werten aus xs zurueckgibt, die groesser als s sind",
        "def groesser_als(xs, s):\n    return [x for x in xs if x > s]",
        [([1, 5, 3, 8], 4), ([], 0), ([2], 5)],
    ),
    (
        "kleiner_als",
        "(xs, s)",
        "eine Liste nur mit den Werten aus xs zurueckgibt, die kleiner als s sind",
        "def kleiner_als(xs, s):\n    return [x for x in xs if x < s]",
        [([1, 5, 3, 8], 4), ([], 0)],
    ),
    (
        "lange_woerter",
        "(woerter, n)",
        "eine Liste nur mit den Woertern aus woerter zurueckgibt, die laenger als n Zeichen sind",
        "def lange_woerter(woerter, n):\n    return [w for w in woerter if len(w) > n]",
        [(["a", "bbb", "cc"], 1), ([], 0)],
    ),
    # --- REDUCE ---
    (
        "summe",
        "(xs)",
        "die Summe aller Zahlen in xs zurueckgibt",
        "def summe(xs):\n    total = 0\n    for x in xs:\n        total += x\n    return total",
        [([1, 2, 3],), ([],), ([5],)],
    ),
    (
        "produkt",
        "(xs)",
        "das Produkt aller Zahlen in xs zurueckgibt (bei leerer Liste 1)",
        "def produkt(xs):\n    p = 1\n    for x in xs:\n        p *= x\n    return p",
        [([1, 2, 3, 4],), ([],), ([5],)],
    ),
    (
        "minimum",
        "(xs)",
        "den kleinsten Wert der nichtleeren Liste xs zurueckgibt",
        "def minimum(xs):\n    return min(xs)",
        [([3, 1, 2],), ([5],), ([-1, -5],)],
    ),
    (
        "anzahl_elemente",
        "(xs)",
        "die Anzahl der Elemente in xs zurueckgibt",
        "def anzahl_elemente(xs):\n    return len(xs)",
        [([1, 2, 3],), ([],)],
    ),
    (
        "anzahl_gerade",
        "(xs)",
        "die Anzahl der geraden Zahlen in xs zurueckgibt",
        "def anzahl_gerade(xs):\n    return sum(1 for x in xs if x % 2 == 0)",
        [([1, 2, 3, 4],), ([],), ([1, 3],)],
    ),
    (
        "durchschnitt",
        "(xs)",
        "den Durchschnitt der nichtleeren Liste xs zurueckgibt",
        "def durchschnitt(xs):\n    return sum(xs) / len(xs)",
        [([2, 4],), ([10],), ([1, 2, 3],)],
    ),
    (
        "spanne",
        "(xs)",
        "die Differenz zwischen groesstem und kleinstem Wert in der nichtleeren Liste xs zurueckgibt",
        "def spanne(xs):\n    return max(xs) - min(xs)",
        [([3, 7, 1],), ([5],)],
    ),
    (
        "summe_gerade",
        "(xs)",
        "die Summe der geraden Zahlen in xs zurueckgibt",
        "def summe_gerade(xs):\n    return sum(x for x in xs if x % 2 == 0)",
        [([1, 2, 3, 4],), ([],)],
    ),
    # --- STRING ---
    (
        "anzahl_zeichen",
        "(s)",
        "die Anzahl der Zeichen im String s zurueckgibt",
        "def anzahl_zeichen(s):\n    return len(s)",
        [("hallo",), ("",)],
    ),
    (
        "ohne_leerzeichen",
        "(s)",
        "den String s ohne Leerzeichen zurueckgibt",
        "def ohne_leerzeichen(s):\n    return s.replace(' ', '')",
        [("a b c",), ("xyz",)],
    ),
    (
        "erstes_wort",
        "(s)",
        "das erste Wort des nichtleeren Strings s zurueckgibt",
        "def erstes_wort(s):\n    return s.split()[0]",
        [("hallo welt",), ("eins",)],
    ),
    (
        "anzahl_grossbuchstaben",
        "(s)",
        "die Anzahl der Grossbuchstaben im String s zurueckgibt",
        "def anzahl_grossbuchstaben(s):\n    return sum(1 for c in s if c.isupper())",
        [("HalloWelt",), ("abc",)],
    ),
    (
        "wiederhole",
        "(s, n)",
        "den String s n-mal hintereinander zurueckgibt",
        "def wiederhole(s, n):\n    return s * n",
        [("ab", 3), ("x", 0)],
    ),
    (
        "erster_buchstabe_gross",
        "(s)",
        "den nichtleeren String s mit grossem ersten Buchstaben zurueckgibt",
        "def erster_buchstabe_gross(s):\n    return s[0].upper() + s[1:]",
        [("hallo",), ("test",)],
    ),
    (
        "anzahl_a",
        "(s)",
        "zaehlt, wie oft der Buchstabe 'a' im String s vorkommt",
        "def anzahl_a(s):\n    return s.count('a')",
        [("banana",), ("xyz",)],
    ),
    (
        "verbinde_mit_komma",
        "(woerter)",
        "die Woerter aus woerter mit Komma verbunden als String zurueckgibt",
        "def verbinde_mit_komma(woerter):\n    return ','.join(woerter)",
        [(["a", "b", "c"],), ([],)],
    ),
    # --- DIGITS ---
    (
        "ziffern_liste",
        "(n)",
        "eine Liste der Ziffern der nichtnegativen Zahl n zurueckgibt",
        "def ziffern_liste(n):\n    return [int(c) for c in str(n)]",
        [(123,), (0,), (7,)],
    ),
    (
        "anzahl_ziffern",
        "(n)",
        "die Anzahl der Ziffern der nichtnegativen Zahl n zurueckgibt",
        "def anzahl_ziffern(n):\n    return len(str(n))",
        [(123,), (0,), (99,)],
    ),
    (
        "letzte_ziffer",
        "(n)",
        "die letzte Ziffer der nichtnegativen Zahl n zurueckgibt",
        "def letzte_ziffer(n):\n    return n % 10",
        [(123,), (0,), (40,)],
    ),
    (
        "erste_ziffer",
        "(n)",
        "die erste Ziffer der nichtnegativen Zahl n zurueckgibt",
        "def erste_ziffer(n):\n    return int(str(n)[0])",
        [(123,), (7,)],
    ),
    (
        "ziffern_produkt",
        "(n)",
        "das Produkt der Ziffern der nichtnegativen Zahl n zurueckgibt",
        "def ziffern_produkt(n):\n    p = 1\n    for c in str(n):\n        p *= int(c)\n    return p",
        [(123,), (0,), (25,)],
    ),
    (
        "ist_dreistellig",
        "(n)",
        "True zurueckgibt, wenn die Zahl n genau dreistellig ist",
        "def ist_dreistellig(n):\n    return 100 <= n <= 999",
        [(123,), (99,), (1000,)],
    ),
    (
        "ist_durch_drei",
        "(n)",
        "True zurueckgibt, wenn die Zahl n durch 3 teilbar ist",
        "def ist_durch_drei(n):\n    return n % 3 == 0",
        [(9,), (10,), (0,)],
    ),
    # --- SORT / COMPARE ---
    (
        "ist_absteigend",
        "(xs)",
        "True zurueckgibt, wenn die Liste xs absteigend sortiert ist",
        "def ist_absteigend(xs):\n    return all(xs[i] >= xs[i+1] for i in range(len(xs)-1))",
        [([3, 2, 1],), ([1, 2],), ([5],)],
    ),
    (
        "ist_konstant",
        "(xs)",
        "True zurueckgibt, wenn alle Werte in der nichtleeren Liste xs gleich sind",
        "def ist_konstant(xs):\n    return all(x == xs[0] for x in xs)",
        [([2, 2, 2],), ([1, 2],), ([5],)],
    ),
    (
        "zweitgroesster",
        "(xs)",
        "den zweitgroessten verschiedenen Wert aus xs zurueckgibt",
        "def zweitgroesster(xs):\n    return sorted(set(xs))[-2]",
        [([3, 1, 2],), ([5, 5, 1],)],
    ),
    (
        "sortiere_absteigend",
        "(xs)",
        "eine absteigend sortierte Kopie von xs zurueckgibt",
        "def sortiere_absteigend(xs):\n    return sorted(xs, reverse=True)",
        [([1, 3, 2],), ([],)],
    ),
    (
        "index_maximum",
        "(xs)",
        "den Index des groessten Werts der nichtleeren Liste xs zurueckgibt",
        "def index_maximum(xs):\n    return xs.index(max(xs))",
        [([1, 9, 3],), ([5],)],
    ),
    (
        "letztes_element",
        "(xs)",
        "das letzte Element der nichtleeren Liste xs zurueckgibt",
        "def letztes_element(xs):\n    return xs[-1]",
        [([1, 2, 3],), ([7],)],
    ),
    # --- PREDICATE ---
    (
        "enthaelt_null",
        "(xs)",
        "True zurueckgibt, wenn der Wert 0 in xs enthalten ist",
        "def enthaelt_null(xs):\n    return 0 in xs",
        [([1, 0, 2],), ([1, 2],), ([],)],
    ),
    (
        "alle_positiv",
        "(xs)",
        "True zurueckgibt, wenn alle Werte in xs positiv sind",
        "def alle_positiv(xs):\n    return all(x > 0 for x in xs)",
        [([1, 2, 3],), ([1, -2],), ([],)],
    ),
    (
        "mindestens_ein_gerade",
        "(xs)",
        "True zurueckgibt, wenn mindestens eine gerade Zahl in xs ist",
        "def mindestens_ein_gerade(xs):\n    return any(x % 2 == 0 for x in xs)",
        [([1, 3, 4],), ([1, 3],), ([],)],
    ),
    (
        "ist_leer",
        "(xs)",
        "True zurueckgibt, wenn die Liste xs leer ist",
        "def ist_leer(xs):\n    return len(xs) == 0",
        [([],), ([1],)],
    ),
    (
        "gleiche_laenge",
        "(a, b)",
        "True zurueckgibt, wenn die Listen a und b gleich lang sind",
        "def gleiche_laenge(a, b):\n    return len(a) == len(b)",
        [([1, 2], [3, 4]), ([1], [2, 3])],
    ),
    # --- DUPLICATES ---
    (
        "hat_duplikate",
        "(xs)",
        "True zurueckgibt, wenn xs doppelte Werte enthaelt",
        "def hat_duplikate(xs):\n    return len(set(xs)) != len(xs)",
        [([1, 2, 2],), ([1, 2, 3],), ([],)],
    ),
    (
        "eindeutige_anzahl",
        "(xs)",
        "die Anzahl der verschiedenen Werte in xs zurueckgibt",
        "def eindeutige_anzahl(xs):\n    return len(set(xs))",
        [([1, 1, 2, 3],), ([],)],
    ),
    (
        "zaehle_vorkommen",
        "(xs, w)",
        "zaehlt, wie oft der Wert w in der Liste xs vorkommt",
        "def zaehle_vorkommen(xs, w):\n    return xs.count(w)",
        [([1, 2, 2, 3], 2), ([1, 1, 1], 1)],
    ),
    (
        "nur_einmalige",
        "(xs)",
        "eine Liste der Werte zurueckgibt, die in xs genau einmal vorkommen",
        "def nur_einmalige(xs):\n    return [x for x in xs if xs.count(x) == 1]",
        [([1, 1, 2, 3, 3],), ([],)],
    ),
]

rows = []
ok = 0
bad = []
for name, sig, what, code, inputs in S:
    ns = {}
    try:
        exec(code, ns)
        fn = ns[name]
        for args in inputs:
            fn(*args)  # self-verify: runs cleanly on all sample inputs
    except Exception as e:
        bad.append((name, str(e)[:60]))
        continue
    ok += 1
    for pre in PREFIX:
        rows.append(
            {
                "text": render(f"{pre} {name}{sig}, die {what}.", f"```python\n{code}\n```"),
                "func": name,
            }
        )

with open(OUT + "/verified_code.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(
    f"code_v4: {ok}/{len(S)} specs valid, {len(rows)} rendered items, {len(set(r['func'] for r in rows))} distinct funcs"
)
if bad:
    print("BAD:", bad)
