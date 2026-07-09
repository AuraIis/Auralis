#!/usr/bin/env python3
"""Build German Python annealing data as standalone tutorial-style JSONL.

The output is continued-pretraining material, not chat/instruction data. Each
record contains one compact explanation, executable code, tests, expected
stdout, and a beginner-error note.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data/training/python_anneal_de/python_anneal_de_500_v1.jsonl"
SEED = 20260607


@dataclass(frozen=True)
class Item:
    topic: str
    difficulty: str
    tags: list[str]
    explanation: str
    code: str
    tests: str
    typical_error: str
    correction: str


def clean_code(code: str) -> str:
    code = code.strip("\n")
    return "\n".join(line.rstrip() for line in code.splitlines()) + "\n"


def run_blocks(code: str, tests: str) -> str:
    namespace: dict[str, object] = {}
    code = clean_code(code)
    tests = clean_code(tests)
    compile(code, "<code>", "exec")
    compile(tests, "<tests>", "exec")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(code, namespace)
    with contextlib.redirect_stdout(io.StringIO()):
        exec(tests, namespace)
    return buf.getvalue().rstrip("\n")


def render(item: Item, output: str) -> str:
    return (
        f"Titel:\n{item.topic}\n\n"
        f"Erklärung:\n{item.explanation}\n\n"
        f"Code:\n\n```python\n{clean_code(item.code)}```\n\n"
        f"Tests:\n\n```python\n{clean_code(item.tests)}```\n\n"
        f"Erwartete Ausgabe:\n\n```text\n{output}\n```\n\n"
        f"Typischer Fehler:\n{item.typical_error}\n\n"
        f"Korrektur:\n{item.correction}"
    )


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", text.casefold()).strip("_")


def basics() -> list[Item]:
    items: list[Item] = []
    products = [
        ("Apfel", 3, 2.40),
        ("Brot", 2, 3.10),
        ("Heft", 7, 1.25),
        ("Stift", 5, 0.80),
        ("Tasse", 4, 4.50),
        ("Lampe", 2, 12.99),
        ("Karte", 6, 1.10),
        ("Seife", 3, 2.75),
        ("Kabel", 8, 3.20),
        ("Notizbuch", 2, 5.95),
    ]
    for name, amount, price in products:
        var = slug(name)
        total = amount * price
        items.append(
            Item(
                topic=f"Variablen und f-Strings: {name}",
                difficulty="beginner",
                tags=["python", "grundlagen", "variablen", "f-string"],
                explanation=(
                    "Dieses Beispiel zeigt, wie Werte in Variablen gespeichert und später in einem f-String ausgegeben werden. "
                    "Der Gesamtpreis entsteht durch Multiplikation von Menge und Einzelpreis. "
                    "Die Formatangabe :.2f rundet die Ausgabe auf zwei Nachkommastellen."
                ),
                code=f'''
artikel = "{name}"
menge = {amount}
preis_pro_stueck = {price}
gesamtpreis = menge * preis_pro_stueck

print(f"{{menge}} x {{artikel}} kosten {{gesamtpreis:.2f}} Euro.")
''',
                tests=f'''
assert artikel == "{name}"
assert menge == {amount}
assert round(gesamtpreis, 2) == {total:.2f}
assert isinstance(artikel, str)
''',
                typical_error="Ein häufiger Fehler ist, Zahlen als Text zu speichern und dann eine Rechnung zu erwarten.",
                correction="Speichere Preise und Mengen als int oder float, nicht als Zeichenkette.",
            )
        )

    names = ["Mara", "Jonas", "Lea", "Emil", "Noah", "Sofia", "Mina", "Oskar", "Lina", "Timo"]
    for i, name in enumerate(names, 1):
        age = 17 + i
        city = [
            "Bonn",
            "Wien",
            "Hamburg",
            "Zürich",
            "Leipzig",
            "Graz",
            "Köln",
            "Bern",
            "Bremen",
            "Mainz",
        ][i - 1]
        items.append(
            Item(
                topic=f"Strings zusammensetzen: {name}",
                difficulty="beginner",
                tags=["python", "grundlagen", "string", "f-string"],
                explanation=(
                    "Hier werden Text und Zahlen zu einem lesbaren Satz kombiniert. "
                    "Ein f-String kann Variablen direkt in geschweifte Klammern einsetzen. "
                    "So bleibt der Code verständlich und vermeidet umständliche Verkettung mit Pluszeichen."
                ),
                code=f'''
name = "{name}"
alter = {age}
stadt = "{city}"

satz = f"{{name}} ist {{alter}} Jahre alt und wohnt in {{stadt}}."
print(satz)
''',
                tests=f"""
assert name in satz
assert str(alter) in satz
assert satz.endswith("in {city}.")
""",
                typical_error="Ein typischer Fehler ist, eine Zahl direkt mit + an einen String anzuhängen.",
                correction="Nutze f-Strings oder wandle Zahlen mit str() um, wenn du Zeichenketten zusammensetzt.",
            )
        )

    conversions = [
        ("42", 42),
        ("7", 7),
        ("105", 105),
        ("18", 18),
        ("64", 64),
        ("23", 23),
        ("90", 90),
        ("12", 12),
        ("250", 250),
        ("31", 31),
    ]
    for text, value in conversions:
        items.append(
            Item(
                topic=f"Text in Zahl umwandeln: {text}",
                difficulty="beginner",
                tags=["python", "grundlagen", "typumwandlung", "int"],
                explanation=(
                    "Eingaben kommen oft als Text an, auch wenn sie wie Zahlen aussehen. "
                    "Mit int() wird eine Zeichenkette in eine ganze Zahl umgewandelt. "
                    "Erst danach sind mathematische Operationen wie Addition sinnvoll."
                ),
                code=f'''
eingabe = "{text}"
zahl = int(eingabe)
ergebnis = zahl + 8

print(ergebnis)
''',
                tests=f"""
assert zahl == {value}
assert ergebnis == {value + 8}
assert isinstance(zahl, int)
""",
                typical_error="Wer '42' + 8 schreibt, mischt Text und Zahl und bekommt einen TypeError.",
                correction="Wandle Text mit int() oder float() in eine Zahl um, bevor du rechnest.",
            )
        )

    temps = [
        (-3, False),
        (0, True),
        (5, True),
        (12, True),
        (19, True),
        (21, False),
        (28, False),
        (15, True),
        (8, True),
        (-10, False),
    ]
    for temp, ok in temps:
        items.append(
            Item(
                topic=f"Boolean-Ausdruck für Temperatur {temp}",
                difficulty="beginner",
                tags=["python", "grundlagen", "boolean", "vergleich"],
                explanation=(
                    "Dieses Beispiel nutzt Vergleichsoperatoren, um eine Bedingung als Wahrheitswert zu speichern. "
                    "Der Ausdruck 0 <= temperatur <= 20 ist nur wahr, wenn beide Grenzen eingehalten werden. "
                    "Solche booleschen Werte sind die Grundlage für if-Abfragen."
                ),
                code=f"""
temperatur = {temp}
angenehm = 0 <= temperatur <= 20

print(angenehm)
""",
                tests=f"""
assert angenehm is {ok!s}
assert isinstance(angenehm, bool)
""",
                typical_error="Ein häufiger Fehler ist, zwei Vergleiche mit 'and' falsch zu klammern oder als Text zu schreiben.",
                correction="Schreibe Bereichsvergleiche in Python direkt als 0 <= wert <= 20.",
            )
        )

    pairs = [
        (10, 3),
        (17, 5),
        (20, 4),
        (31, 6),
        (48, 7),
        (99, 10),
        (14, 2),
        (27, 8),
        (65, 9),
        (36, 5),
    ]
    for a, b in pairs:
        items.append(
            Item(
                topic=f"Division, Ganzzahldivision und Rest: {a} durch {b}",
                difficulty="beginner",
                tags=["python", "grundlagen", "division", "modulo"],
                explanation=(
                    "Python unterscheidet normale Division, Ganzzahldivision und den Rest einer Division. "
                    "Der Operator / liefert eine Fließkommazahl, // den ganzzahligen Anteil und % den Rest. "
                    "Das ist wichtig für Aufteilungen, Kalenderrechnungen und gerade/ungerade Tests."
                ),
                code=f"""
gesamt = {a}
gruppe = {b}

quotient = gesamt / gruppe
ganze_gruppen = gesamt // gruppe
rest = gesamt % gruppe

print(quotient)
print(ganze_gruppen)
print(rest)
""",
                tests=f"""
assert quotient == {a / b}
assert ganze_gruppen == {a // b}
assert rest == {a % b}
assert ganze_gruppen * gruppe + rest == gesamt
""",
                typical_error="Viele Anfänger erwarten, dass / automatisch eine ganze Zahl liefert.",
                correction="Nutze // für Ganzzahldivision und % für den Rest.",
            )
        )

    rounded = [
        (12.345, 2),
        (7.891, 1),
        (99.995, 2),
        (3.14159, 3),
        (2.71828, 2),
        (45.678, 0),
        (0.3333, 2),
        (123.4567, 1),
        (8.005, 2),
        (19.999, 2),
    ]
    for value, digits in rounded:
        items.append(
            Item(
                topic=f"Runden mit round: {value}",
                difficulty="beginner",
                tags=["python", "grundlagen", "float", "round"],
                explanation=(
                    "Fließkommazahlen haben oft mehr Nachkommastellen als für eine Ausgabe sinnvoll sind. "
                    "round() erzeugt einen gerundeten Wert mit der gewünschten Stellenzahl. "
                    "Das ist nützlich für Messwerte, Geldbeträge und einfache Berichte."
                ),
                code=f"""
messwert = {value}
gerundet = round(messwert, {digits})

print(gerundet)
""",
                tests=f"""
assert gerundet == {round(value, digits)!r}
assert isinstance(gerundet, float)
""",
                typical_error="Ein häufiger Fehler ist zu glauben, round() verändere die ursprüngliche Variable automatisch.",
                correction="Speichere den Rückgabewert von round() in einer neuen Variablen oder überschreibe bewusst die alte.",
            )
        )

    raw_strings = [
        ("  Python  ", "python"),
        (" HALLO ", "hallo"),
        ("\tTest\n", "test"),
        ("  Daten Satz  ", "daten satz"),
        (" Code ", "code"),
        ("  Berlin", "berlin"),
        ("WERT  ", "wert"),
        ("  Ja  ", "ja"),
        ("  Ausgabe ", "ausgabe"),
        ("Name\n", "name"),
    ]
    for raw, expected in raw_strings:
        items.append(
            Item(
                topic=f"Text bereinigen: {expected}",
                difficulty="beginner",
                tags=["python", "grundlagen", "string", "strip"],
                explanation=(
                    "Text enthält häufig überflüssige Leerzeichen oder uneinheitliche Großschreibung. "
                    "strip() entfernt Leerraum am Anfang und Ende. "
                    "lower() wandelt den Text in Kleinbuchstaben um."
                ),
                code=f"""
rohtext = {raw!r}
bereinigt = rohtext.strip().lower()

print(bereinigt)
""",
                tests=f"""
assert bereinigt == {expected!r}
assert bereinigt == bereinigt.lower()
assert not bereinigt.startswith(" ")
""",
                typical_error="Ein typischer Fehler ist, nur lower() zu nutzen und Leerzeichen zu übersehen.",
                correction="Kombiniere strip() und lower(), wenn Eingabetext verglichen werden soll.",
            )
        )

    logic_cases = [
        (16, True),
        (18, True),
        (25, True),
        (70, True),
        (12, False),
        (30, True),
        (65, True),
        (45, True),
        (17, True),
        (22, False),
    ]
    for age, has_ticket in logic_cases:
        allowed = age >= 18 and has_ticket
        items.append(
            Item(
                topic=f"Logische Operatoren: Alter {age}",
                difficulty="beginner",
                tags=["python", "grundlagen", "boolean", "and"],
                explanation=(
                    "Logische Operatoren verbinden mehrere Bedingungen. "
                    "Mit and müssen beide Teilbedingungen wahr sein. "
                    "Das Beispiel prüft, ob eine Person alt genug ist und ein Ticket besitzt."
                ),
                code=f"""
alter = {age}
hat_ticket = {has_ticket!s}

darf_hinein = alter >= 18 and hat_ticket
print(darf_hinein)
""",
                tests=f"""
assert darf_hinein is {allowed!s}
assert isinstance(darf_hinein, bool)
""",
                typical_error="Ein häufiger Fehler ist, and mit dem deutschen Wort 'und' zu schreiben.",
                correction="Nutze in Python die Operatoren and, or und not.",
            )
        )

    index_cases = [
        (["A", "B", "C"], 0),
        (["rot", "blau", "grün"], 1),
        ([10, 20, 30], 2),
        (["Start", "Mitte", "Ziel"], -1),
        (["Mo", "Di", "Mi"], -2),
        ([5, 6, 7, 8], 3),
        (["x", "y"], 0),
        (["links", "rechts"], 1),
        ([100, 200, 300], -1),
        (["klein", "mittel", "gross"], 2),
    ]
    for values, idx in index_cases:
        items.append(
            Item(
                topic=f"Listenindex lesen: Position {idx}",
                difficulty="beginner",
                tags=["python", "grundlagen", "liste", "index"],
                explanation=(
                    "Listen werden in Python ab 0 indiziert. "
                    "Ein negativer Index zählt vom Ende der Liste. "
                    "So kann man gezielt einzelne Werte aus einer Liste lesen."
                ),
                code=f"""
werte = {values!r}
auswahl = werte[{idx}]

print(auswahl)
""",
                tests=f"""
assert auswahl == {values[idx]!r}
assert werte[{idx}] == auswahl
""",
                typical_error="Ein häufiger Fehler ist, das erste Element mit Index 1 lesen zu wollen.",
                correction="Denke daran: Das erste Element hat Index 0, das letzte kann mit -1 gelesen werden.",
            )
        )

    unpack_cases = [
        ("rgb", (255, 128, 0)),
        ("datum", (2026, 6, 7)),
        ("zeit", (8, 30, 15)),
        ("box", (4, 5, 6)),
        ("punkt3d", (1, 2, 3)),
        ("version", (3, 11, 9)),
        ("farbe", (10, 20, 30)),
        ("messung", (7, 8, 9)),
        ("konto", (100, -5, 95)),
        ("vektor", (2, 4, 8)),
    ]
    for name, triple in unpack_cases:
        a, b, c = triple
        items.append(
            Item(
                topic=f"Drei Werte entpacken: {name}",
                difficulty="beginner",
                tags=["python", "grundlagen", "tuple", "unpacking"],
                explanation=(
                    "Mehrere zusammengehörige Werte können in einem Tupel gespeichert werden. "
                    "Beim Entpacken weist Python die Werte von links nach rechts Variablen zu. "
                    "Das macht Code lesbarer als mehrere Indexzugriffe."
                ),
                code=f"""
{name} = ({a}, {b}, {c})
erster, zweiter, dritter = {name}
gesamt = erster + zweiter + dritter

print(gesamt)
""",
                tests=f"""
assert erster == {a}
assert zweiter == {b}
assert dritter == {c}
assert gesamt == {a + b + c}
""",
                typical_error="Ein häufiger Fehler ist, drei Werte auf nur zwei Variablen zu entpacken.",
                correction="Die Anzahl der Variablen links muss zur Anzahl der Werte rechts passen.",
            )
        )

    return items


def functions() -> list[Item]:
    items: list[Item] = []
    greetings = [
        ("begruesse", "Mara"),
        ("willkommen", "Jonas"),
        ("melde_person", "Lea"),
        ("zeige_name", "Emil"),
        ("format_name", "Sofia"),
        ("kurzprofil", "Noah"),
        ("baue_gruss", "Mina"),
        ("sage_hallo", "Oskar"),
        ("person_zeile", "Lina"),
        ("etikett", "Timo"),
    ]
    for fname, name in greetings:
        items.append(
            Item(
                topic=f"Funktion mit Rückgabewert: {fname}",
                difficulty="beginner",
                tags=["python", "funktion", "return", "parameter"],
                explanation=(
                    "Eine Funktion bündelt wiederverwendbaren Code unter einem Namen. "
                    "Der Parameter nimmt den übergebenen Wert entgegen. "
                    "Mit return gibt die Funktion ein Ergebnis zurück, das später weiterverwendet werden kann."
                ),
                code=f'''
def {fname}(name):
    return f"Hallo, {{name}}!"

nachricht = {fname}("{name}")
print(nachricht)
''',
                tests=f'''
assert {fname}("{name}") == "Hallo, {name}!"
assert {fname}("Ada") == "Hallo, Ada!"
assert nachricht.endswith("!")
''',
                typical_error="Ein häufiger Fehler ist, print() statt return zu verwenden, obwohl ein Wert gebraucht wird.",
                correction="Nutze return, wenn die Funktion ein Ergebnis an den aufrufenden Code zurückgeben soll.",
            )
        )

    rectangles = [
        (3, 4),
        (5, 8),
        (10, 2),
        (7, 6),
        (9, 9),
        (12, 3),
        (4, 11),
        (15, 2),
        (6, 13),
        (20, 5),
    ]
    for w, h in rectangles:
        items.append(
            Item(
                topic=f"Fläche berechnen mit Parametern: {w}x{h}",
                difficulty="easy",
                tags=["python", "funktion", "parameter", "mathematik"],
                explanation=(
                    "Diese Funktion berechnet die Fläche eines Rechtecks. "
                    "Breite und Höhe werden als Parameter übergeben. "
                    "Die Funktion ist rein: Sie verändert nichts außerhalb und liefert nur das Ergebnis zurück."
                ),
                code=f"""
def rechteck_flaeche(breite, hoehe):
    return breite * hoehe

flaeche = rechteck_flaeche({w}, {h})
print(flaeche)
""",
                tests=f"""
assert rechteck_flaeche({w}, {h}) == {w * h}
assert rechteck_flaeche(1, 1) == 1
assert rechteck_flaeche(0, 5) == 0
""",
                typical_error="Manchmal werden Breite und Höhe in der Funktion hart codiert.",
                correction="Verwende Parameter, damit dieselbe Funktion für viele Werte funktioniert.",
            )
        )

    discounts = [
        (100, 10),
        (80, 25),
        (250, 20),
        (60, 15),
        (45, 5),
        (120, 30),
        (300, 50),
        (75, 12),
        (90, 40),
        (200, 18),
    ]
    for price, pct in discounts:
        new_price = price - price * pct / 100
        items.append(
            Item(
                topic=f"Funktion mit Prozentrechnung: {pct} Prozent Rabatt",
                difficulty="easy",
                tags=["python", "funktion", "float", "prozent"],
                explanation=(
                    "Dieses Beispiel zeigt eine Funktion mit zwei Parametern für Preis und Rabatt. "
                    "Der Rabattbetrag wird aus dem Preis und dem Prozentsatz berechnet. "
                    "round() macht die Ausgabe für Geldbeträge übersichtlich."
                ),
                code=f"""
def preis_nach_rabatt(preis, rabatt_prozent):
    rabatt = preis * rabatt_prozent / 100
    return round(preis - rabatt, 2)

endpreis = preis_nach_rabatt({price}, {pct})
print(endpreis)
""",
                tests=f"""
assert preis_nach_rabatt({price}, {pct}) == {round(new_price, 2)}
assert preis_nach_rabatt(100, 0) == 100
assert preis_nach_rabatt(100, 50) == 50
""",
                typical_error="Ein häufiger Fehler ist, den Prozentsatz als ganze Zahl abzuziehen, also preis - rabatt_prozent.",
                correction="Berechne zuerst preis * rabatt_prozent / 100 und ziehe diesen Betrag ab.",
            )
        )

    defaults = [
        ("berechne_versand", 29, 4.95, 50),
        ("gesamt_mit_gebuehr", 12, 2.5, 20),
        ("preis_mit_pfand", 8, 0.25, 10),
        ("punkte_mit_bonus", 40, 5, 50),
        ("temperatur_mit_offset", 18, 2, 20),
        ("gewicht_mit_verpackung", 3, 0.5, 5),
        ("zeit_mit_puffer", 45, 10, 60),
        ("laenge_mit_rand", 80, 5, 100),
        ("betrag_mit_aufschlag", 70, 7, 100),
        ("karten_mit_reserve", 33, 2, 40),
    ]
    for fname, base, default, threshold in defaults:
        items.append(
            Item(
                topic=f"Standardparameter verwenden: {fname}",
                difficulty="easy",
                tags=["python", "funktion", "standardparameter", "parameter"],
                explanation=(
                    "Standardparameter geben einer Funktion sinnvolle Vorgabewerte. "
                    "Beim Aufruf kann der Wert weggelassen oder bewusst überschrieben werden. "
                    "Das macht kleine Hilfsfunktionen flexibel, ohne viele Varianten schreiben zu müssen."
                ),
                code=f"""
def {fname}(wert, zuschlag={default}):
    return wert + zuschlag

standard = {fname}({base})
angepasst = {fname}({base}, zuschlag={threshold - base})

print(standard)
print(angepasst)
""",
                tests=f"""
assert {fname}({base}) == {base + default}
assert {fname}({base}, zuschlag={threshold - base}) == {threshold}
assert angepasst == {threshold}
""",
                typical_error="Ein häufiger Fehler ist, beim Aufruf die Reihenfolge der Argumente zu verwechseln.",
                correction="Nutze bei optionalen Werten benannte Argumente wie zuschlag=10.",
            )
        )

    tuples = [
        (8, 3),
        (10, 4),
        (21, 5),
        (42, 8),
        (55, 9),
        (13, 6),
        (99, 12),
        (70, 11),
        (32, 7),
        (18, 5),
    ]
    for a, b in tuples:
        items.append(
            Item(
                topic=f"Mehrere Rückgabewerte: {a} und {b}",
                difficulty="easy",
                tags=["python", "funktion", "tuple", "return"],
                explanation=(
                    "Eine Funktion kann mehrere Werte zurückgeben, indem sie ein Tupel liefert. "
                    "Beim Aufruf können diese Werte direkt auf mehrere Variablen verteilt werden. "
                    "Das ist praktisch, wenn Summe und Differenz gemeinsam gebraucht werden."
                ),
                code=f"""
def summe_und_differenz(a, b):
    return a + b, a - b

summe, differenz = summe_und_differenz({a}, {b})
print(summe)
print(differenz)
""",
                tests=f"""
assert summe_und_differenz({a}, {b}) == ({a + b}, {a - b})
assert summe == {a + b}
assert differenz == {a - b}
""",
                typical_error="Manchmal wird vergessen, beide Rückgabewerte beim Aufruf entgegenzunehmen.",
                correction="Schreibe zum Beispiel summe, differenz = funktion(...), wenn zwei Werte zurückkommen.",
            )
        )

    predicates = [
        ("ist_gerade", "n % 2 == 0", 12, True),
        ("ist_ungerade", "n % 2 != 0", 7, True),
        ("ist_positiv", "n > 0", 5, True),
        ("ist_negativ", "n < 0", -3, True),
        ("ist_null", "n == 0", 0, True),
        ("ist_mehrstellig", "abs(n) >= 10", 42, True),
        ("ist_klein", "n < 100", 99, True),
        ("ist_gross", "n >= 1000", 1200, True),
        ("ist_vielfaches_von_5", "n % 5 == 0", 35, True),
        ("ist_einstellig", "-9 <= n <= 9", 8, True),
    ]
    for fname, expr, sample, expected in predicates:
        items.append(
            Item(
                topic=f"Prädikat-Funktion: {fname}",
                difficulty="easy",
                tags=["python", "funktion", "boolean", "test"],
                explanation=(
                    "Eine Prädikat-Funktion beantwortet eine Frage mit True oder False. "
                    "Der Rückgabewert entsteht direkt aus einem Vergleich oder logischen Ausdruck. "
                    "Solche Funktionen machen Bedingungen im Hauptprogramm lesbarer."
                ),
                code=f"""
def {fname}(n):
    return {expr}

wert = {fname}({sample})
print(wert)
""",
                tests=f"""
assert {fname}({sample}) is {expected!s}
assert isinstance(wert, bool)
""",
                typical_error="Ein häufiger Fehler ist, True und False als Strings zurückzugeben.",
                correction="Gib die booleschen Werte True und False zurück, nicht 'True' oder 'False'.",
            )
        )

    list_funcs = [
        ("summe", "sum(werte)", [1, 2, 3], 6),
        ("anzahl", "len(werte)", [4, 5, 6, 7], 4),
        ("minimum", "min(werte)", [8, 3, 5], 3),
        ("maximum", "max(werte)", [8, 3, 5], 8),
        ("mittelwert", "sum(werte) / len(werte)", [2, 4, 6], 4.0),
        ("erste_zahl", "werte[0]", [9, 1, 2], 9),
        ("letzte_zahl", "werte[-1]", [9, 1, 2], 2),
        ("sortiert", "sorted(werte)", [3, 1, 2], [1, 2, 3]),
        ("umgekehrt", "list(reversed(werte))", [1, 2, 3], [3, 2, 1]),
        ("eindeutig_sortiert", "sorted(set(werte))", [2, 1, 2], [1, 2]),
    ]
    for fname, expr, values, expected in list_funcs:
        items.append(
            Item(
                topic=f"Funktion für Listenwerte: {fname}",
                difficulty="easy",
                tags=["python", "funktion", "liste", "return"],
                explanation=(
                    "Funktionen können auch ganze Listen als Parameter bekommen. "
                    "Das Beispiel berechnet daraus einen einzelnen Wert oder eine neue Liste. "
                    "Der Rückgabewert bleibt unabhängig vom Namen der Eingabeliste."
                ),
                code=f"""
def {fname}(werte):
    return {expr}

ergebnis = {fname}({values!r})
print(ergebnis)
""",
                tests=f"""
assert {fname}({values!r}) == {expected!r}
assert ergebnis == {expected!r}
""",
                typical_error="Ein häufiger Fehler ist, innerhalb der Funktion fest auf eine globale Liste zuzugreifen.",
                correction="Nutze den Parameter werte, damit die Funktion mit jeder passenden Liste arbeitet.",
            )
        )

    converters = [
        ("celsius_zu_fahrenheit", "celsius * 9 / 5 + 32", "celsius", 20, 68.0),
        ("meter_zu_zentimeter", "meter * 100", "meter", 3, 300),
        ("stunden_zu_minuten", "stunden * 60", "stunden", 4, 240),
        ("tage_zu_stunden", "tage * 24", "tage", 2, 48),
        ("euro_zu_cent", "euro * 100", "euro", 7, 700),
        ("kilometer_zu_meter", "kilometer * 1000", "kilometer", 5, 5000),
        ("minuten_zu_sekunden", "minuten * 60", "minuten", 9, 540),
        ("quadratzahl", "n * n", "n", 11, 121),
        ("halbieren", "wert / 2", "wert", 15, 7.5),
        ("verdoppeln", "wert * 2", "wert", 13, 26),
    ]
    for fname, expr, param, sample, expected in converters:
        items.append(
            Item(
                topic=f"Umrechnungsfunktion: {fname}",
                difficulty="easy",
                tags=["python", "funktion", "umrechnung", "parameter"],
                explanation=(
                    "Eine Umrechnungsfunktion kapselt eine feste Formel. "
                    "Dadurch muss die Formel nicht im ganzen Programm wiederholt werden. "
                    "Der Funktionsname beschreibt, welche Umrechnung durchgeführt wird."
                ),
                code=f"""
def {fname}({param}):
    return {expr}

ergebnis = {fname}({sample})
print(ergebnis)
""",
                tests=f"""
assert {fname}({sample}) == {expected!r}
assert ergebnis == {expected!r}
""",
                typical_error="Ein häufiger Fehler ist, die Formel direkt in print() zu schreiben und später zu duplizieren.",
                correction="Lege die Formel in eine Funktion, wenn du sie mehrfach brauchst.",
            )
        )

    formatters = [
        ("formatiere_preis", "f'{betrag:.2f} Euro'", "betrag", 12.5, "12.50 Euro"),
        ("initiale", "name.strip()[0].upper()", "name", " ada", "A"),
        ("dateiname_py", "name.strip().lower() + '.py'", "name", "Test", "test.py"),
        ("rufzeichen", "text.strip() + '!'", "text", "Hallo", "Hallo!"),
        ("klammern", "'(' + text + ')'", "text", "ok", "(ok)"),
        ("tag_label", "'Tag ' + str(n)", "n", 7, "Tag 7"),
        ("kurzer_code", "text.strip().upper()[:3]", "text", "python", "PYT"),
        ("mit_prefix", "'ID-' + str(n)", "n", 42, "ID-42"),
        ("klein_ohne_rand", "text.strip().lower()", "text", " JA ", "ja"),
        ("satzende", "text if text.endswith('.') else text + '.'", "text", "Fertig", "Fertig."),
    ]
    for fname, expr, param, sample, expected in formatters:
        items.append(
            Item(
                topic=f"Formatierungsfunktion: {fname}",
                difficulty="easy",
                tags=["python", "funktion", "string", "formatierung"],
                explanation=(
                    "Formatierungsfunktionen wandeln Werte in eine einheitliche Textform. "
                    "Das ist nützlich für Ausgaben, Labels und einfache Normalisierung. "
                    "Die Funktion trennt die Formatregel vom restlichen Programm."
                ),
                code=f"""
def {fname}({param}):
    return {expr}

ausgabe = {fname}({sample!r})
print(ausgabe)
""",
                tests=f"""
assert {fname}({sample!r}) == {expected!r}
assert ausgabe == {expected!r}
""",
                typical_error="Ein häufiger Fehler ist, Formatlogik an mehreren Stellen leicht unterschiedlich zu schreiben.",
                correction="Nutze eine kleine Funktion, damit dieselbe Regel überall gleich angewendet wird.",
            )
        )

    clamp_cases = [
        (5, 0, 10),
        (-2, 0, 10),
        (15, 0, 10),
        (80, 50, 100),
        (42, 10, 40),
        (3, 3, 9),
        (12, 1, 8),
        (25, 20, 30),
        (0, -5, 5),
        (99, 0, 50),
    ]
    for value, low, high in clamp_cases:
        expected = min(max(value, low), high)
        items.append(
            Item(
                topic=f"Wert begrenzen: {value} in [{low}, {high}]",
                difficulty="easy",
                tags=["python", "funktion", "kontrollfluss", "vergleich"],
                explanation=(
                    "Eine Begrenzungsfunktion hält einen Wert innerhalb eines erlaubten Bereichs. "
                    "Ist der Wert zu klein, wird die untere Grenze zurückgegeben. "
                    "Ist er zu groß, wird die obere Grenze zurückgegeben."
                ),
                code=f"""
def begrenze(wert, minimum, maximum):
    if wert < minimum:
        return minimum
    if wert > maximum:
        return maximum
    return wert

ergebnis = begrenze({value}, {low}, {high})
print(ergebnis)
""",
                tests=f"""
assert begrenze({value}, {low}, {high}) == {expected}
assert begrenze({low - 1}, {low}, {high}) == {low}
assert begrenze({high + 1}, {low}, {high}) == {high}
""",
                typical_error="Ein häufiger Fehler ist, nur den unteren oder nur den oberen Grenzfall zu behandeln.",
                correction="Prüfe beide Grenzen und gib sonst den ursprünglichen Wert zurück.",
            )
        )

    return items


def data_structures() -> list[Item]:
    items: list[Item] = []
    list_sets = [
        ("zahlen", [3, 8, 2, 8, 5]),
        ("punkte", [10, 12, 9, 12, 15]),
        ("preise", [4, 7, 4, 9, 1]),
        ("alter", [21, 19, 21, 30, 19]),
        ("werte", [6, 1, 6, 3, 2]),
        ("noten", [2, 3, 2, 1, 4]),
        ("mengen", [5, 5, 7, 8, 7]),
        ("temperaturen", [18, 20, 18, 21, 19]),
        ("seiten", [12, 8, 12, 5, 8]),
        ("stunden", [4, 6, 4, 2, 6]),
    ]
    for name, values in list_sets:
        items.append(
            Item(
                topic=f"Liste sortieren und Duplikate entfernen: {name}",
                difficulty="easy",
                tags=["python", "liste", "set", "sortieren"],
                explanation=(
                    "Listen behalten Reihenfolge und können doppelte Werte enthalten. "
                    "Ein set entfernt Duplikate, hat aber keine feste Sortierreihenfolge. "
                    "Mit sorted() wird daraus wieder eine sortierte Liste."
                ),
                code=f"""
{name} = {values}
eindeutig_sortiert = sorted(set({name}))

print(eindeutig_sortiert)
""",
                tests=f"""
assert eindeutig_sortiert == {sorted(set(values))}
assert len(eindeutig_sortiert) == len(set({name}))
assert {name}[0] == {values[0]}
""",
                typical_error="Ein häufiger Fehler ist anzunehmen, dass ein set die ursprüngliche Reihenfolge behält.",
                correction="Nutze sorted(set(...)), wenn du eindeutige Werte in stabiler Sortierung brauchst.",
            )
        )

    dicts = [
        ("lager", {"apfel": 5, "banane": 3}, "apfel", 2),
        ("punkte", {"Mara": 12, "Jonas": 9}, "Jonas", 4),
        ("bestand", {"heft": 20, "stift": 15}, "stift", 5),
        ("zaehler", {"rot": 2, "blau": 4}, "rot", 3),
        ("warenkorb", {"brot": 1, "milch": 2}, "milch", 1),
        ("noten", {"Bio": 2, "Mathe": 3}, "Mathe", -1),
        ("stunden", {"Montag": 6, "Dienstag": 5}, "Montag", 2),
        ("profile", {"aktiv": 1, "inaktiv": 0}, "aktiv", 1),
        ("wertung", {"A": 10, "B": 7}, "B", 2),
        ("konto", {"einzahlung": 100, "gebuehr": -5}, "gebuehr", -2),
    ]
    for name, data, key, change in dicts:
        new_val = data[key] + change
        items.append(
            Item(
                topic=f"Dictionary-Wert aktualisieren: {name}",
                difficulty="easy",
                tags=["python", "dictionary", "schluessel", "update"],
                explanation=(
                    "Ein Dictionary ordnet Schlüsseln Werte zu. "
                    "Über einen Schlüssel kann ein vorhandener Wert gelesen und verändert werden. "
                    "Das eignet sich gut für Zähler, Bestände und einfache Zuordnungen."
                ),
                code=f'''
{name} = {data!r}
{name}["{key}"] = {name}["{key}"] + ({change})

print({name}["{key}"])
''',
                tests=f'''
assert {name}["{key}"] == {new_val}
assert "{key}" in {name}
assert isinstance({name}, dict)
''',
                typical_error="Ein häufiger Fehler ist, einen nicht vorhandenen Schlüssel ohne Prüfung zu lesen.",
                correction="Nutze key in dict oder dict.get(key, standardwert), wenn ein Schlüssel fehlen kann.",
            )
        )

    tuple_items = [
        ("punkt", (3, 4)),
        ("koordinate", (10, 2)),
        ("groesse", (1920, 1080)),
        ("bereich", (5, 12)),
        ("position", (7, 9)),
        ("pixel", (255, 128)),
        ("zeit", (14, 30)),
        ("karte", (8, 6)),
        ("fenster", (80, 24)),
        ("ziel", (11, 13)),
    ]
    for name, pair in tuple_items:
        a, b = pair
        items.append(
            Item(
                topic=f"Tupel entpacken: {name}",
                difficulty="beginner",
                tags=["python", "tupel", "unpacking", "datenstruktur"],
                explanation=(
                    "Ein Tupel speichert mehrere zusammengehörige Werte. "
                    "Beim Entpacken werden die einzelnen Positionen direkt Variablen zugewiesen. "
                    "Das ist lesbarer als wiederholt mit Index 0 und 1 zu arbeiten."
                ),
                code=f"""
{name} = ({a}, {b})
x, y = {name}
summe = x + y

print(summe)
""",
                tests=f"""
assert {name} == ({a}, {b})
assert x == {a}
assert y == {b}
assert summe == {a + b}
""",
                typical_error="Ein häufiger Fehler ist, zu viele oder zu wenige Variablen beim Entpacken zu verwenden.",
                correction="Die Anzahl der Variablen links muss zur Anzahl der Werte im Tupel passen.",
            )
        )

    filters = [
        ("namen", ["Ada", "Ben", "Clara", "Dee"], 4),
        ("orte", ["Bonn", "Wien", "Graz", "Hamburg"], 5),
        ("tiere", ["Hund", "Katze", "Maus", "Elefant"], 5),
        ("farben", ["rot", "blau", "gelb", "violett"], 5),
        ("kurse", ["Mathe", "Bio", "Deutsch", "Sport"], 5),
        ("tools", ["git", "python", "pytest", "uv"], 3),
        ("fruechte", ["Apfel", "Birne", "Kiwi", "Banane"], 5),
        ("rollen", ["Admin", "Gast", "Autor", "Leser"], 5),
        ("staedte", ["Ulm", "Kiel", "Berlin", "Essen"], 5),
        ("dateien", ["a.py", "index.html", "README.md", "x"], 5),
    ]
    for name, values, min_len in filters:
        expected = [v for v in values if len(v) >= min_len]
        items.append(
            Item(
                topic=f"List Comprehension filtern: {name}",
                difficulty="easy",
                tags=["python", "liste", "list-comprehension", "filter"],
                explanation=(
                    "Eine List Comprehension erzeugt aus einer bestehenden Liste eine neue Liste. "
                    "Die if-Bedingung entscheidet, welche Elemente übernommen werden. "
                    "So lassen sich einfache Filter kompakt und gut lesbar ausdrücken."
                ),
                code=f"""
{name} = {values!r}
lange_werte = [wert for wert in {name} if len(wert) >= {min_len}]

print(lange_werte)
""",
                tests=f"""
assert lange_werte == {expected!r}
assert all(len(wert) >= {min_len} for wert in lange_werte)
assert {name} != lange_werte
""",
                typical_error="Manchmal wird die ursprüngliche Liste versehentlich überschrieben.",
                correction="Speichere das Ergebnis in einer neuen Variablen, wenn du die Ausgangsdaten behalten willst.",
            )
        )

    char_texts = [
        "banane",
        "mississippi",
        "python",
        "daten",
        "klasse",
        "tutorial",
        "funktion",
        "liste",
        "zeichen",
        "programm",
    ]
    for text in char_texts:
        target = text[0]
        count = text.count(target)
        items.append(
            Item(
                topic=f"Dictionary als Zähler: Buchstabe {target}",
                difficulty="easy",
                tags=["python", "dictionary", "zaehler", "string"],
                explanation=(
                    "Ein Dictionary kann als Zähler verwendet werden. "
                    "Für jedes Zeichen wird der bisherige Wert mit get() gelesen und um eins erhöht. "
                    "So entsteht eine Häufigkeitstabelle."
                ),
                code=f'''
text = "{text}"
haeufigkeit = {{}}

for zeichen in text:
    haeufigkeit[zeichen] = haeufigkeit.get(zeichen, 0) + 1

print(haeufigkeit["{target}"])
''',
                tests=f'''
assert haeufigkeit["{target}"] == {count}
assert sum(haeufigkeit.values()) == len(text)
assert set(haeufigkeit).issubset(set(text))
''',
                typical_error="Ein häufiger Fehler ist, haeufigkeit[zeichen] zu erhöhen, bevor der Schlüssel existiert.",
                correction="Nutze get(zeichen, 0), um beim ersten Auftreten mit 0 zu starten.",
            )
        )

    set_pairs = [
        ([1, 2, 3], [3, 4, 5]),
        (["a", "b"], ["b", "c"]),
        ([10, 20], [20, 30]),
        (["rot", "blau"], ["blau", "grün"]),
        ([2, 4, 6], [1, 2, 3]),
        (["py", "js"], ["go", "py"]),
        ([7, 8], [8, 9]),
        (["Mo", "Di"], ["Di", "Mi"]),
        ([100, 200], [50, 100]),
        (["x", "y"], ["z", "x"]),
    ]
    for left, right in set_pairs:
        inter = sorted(set(left) & set(right))
        union = sorted(set(left) | set(right))
        items.append(
            Item(
                topic=f"Set-Schnittmenge und Vereinigung: {left[0]}",
                difficulty="easy",
                tags=["python", "set", "schnittmenge", "vereinigung"],
                explanation=(
                    "Sets eignen sich für Mengenoperationen. "
                    "Der Operator & liefert gemeinsame Elemente, | liefert die Vereinigung. "
                    "Mit sorted() wird die Ausgabe stabil sortiert."
                ),
                code=f"""
links = {left!r}
rechts = {right!r}

gemeinsam = sorted(set(links) & set(rechts))
alle = sorted(set(links) | set(rechts))

print(gemeinsam)
print(alle)
""",
                tests=f"""
assert gemeinsam == {inter!r}
assert alle == {union!r}
assert set(gemeinsam).issubset(set(alle))
""",
                typical_error="Ein häufiger Fehler ist, Listen direkt mit & zu verknüpfen.",
                correction="Wandle Listen zuerst mit set(...) in Mengen um.",
            )
        )

    nested_lists = [
        [[1, 2], [3]],
        [["a"], ["b", "c"]],
        [[10], [20, 30]],
        [["rot", "blau"], ["grün"]],
        [[5, 6], [7, 8]],
        [["py", "js"], ["go"]],
        [[0], [1, 2, 3]],
        [["Mo"], ["Di", "Mi"]],
        [[100, 200], [300]],
        [["x", "y"], ["z"]],
    ]
    for nested_values in nested_lists:
        flat = [x for part in nested_values for x in part]
        items.append(
            Item(
                topic=f"Verschachtelte Liste abflachen: {len(flat)} Elemente",
                difficulty="medium",
                tags=["python", "liste", "verschachtelt", "schleife"],
                explanation=(
                    "Eine verschachtelte Liste enthält weitere Listen als Elemente. "
                    "Zwei for-Schleifen kopieren jedes innere Element in eine flache Ergebnisliste. "
                    "Das macht die Daten danach einfacher zu verarbeiten."
                ),
                code=f"""
gruppen = {nested_values!r}
flach = []

for gruppe in gruppen:
    for wert in gruppe:
        flach.append(wert)

print(flach)
""",
                tests=f"""
assert flach == {flat!r}
assert len(flach) == {len(flat)}
""",
                typical_error="Ein häufiger Fehler ist, die innere Liste selbst anzuhängen statt ihre Elemente.",
                correction="Hänge in der inneren Schleife den einzelnen wert an, nicht gruppe.",
            )
        )

    comp_nums = [[1, 2, 3], [2, 4, 6], [3, 5, 7], [10, 20], [4, 8, 12]]
    for nums in comp_nums:
        mapping = {n: n * n for n in nums}
        items.append(
            Item(
                topic=f"Dictionary Comprehension: Quadrate ab {nums[0]}",
                difficulty="medium",
                tags=["python", "dictionary", "comprehension", "quadratzahl"],
                explanation=(
                    "Eine Dictionary Comprehension erzeugt Schlüssel und Werte in einem kompakten Ausdruck. "
                    "Hier wird jede Zahl ihrem Quadrat zugeordnet. "
                    "Das Ergebnis ist ein neues Dictionary."
                ),
                code=f"""
zahlen = {nums!r}
quadrate = {{zahl: zahl * zahl for zahl in zahlen}}

print(quadrate)
""",
                tests=f"""
assert quadrate == {mapping!r}
assert set(quadrate.keys()) == set(zahlen)
""",
                typical_error="Ein häufiger Fehler ist, bei einer Dictionary Comprehension den Doppelpunkt zu vergessen.",
                correction="Schreibe {schluessel: wert for ...}, nicht nur {wert for ...}.",
            )
        )

    return items


def control_flow() -> list[Item]:
    items: list[Item] = []
    ranges = [
        (1, 6),
        (2, 8),
        (5, 11),
        (10, 15),
        (3, 13),
        (4, 10),
        (7, 12),
        (1, 4),
        (6, 14),
        (8, 18),
    ]
    for start, stop in ranges:
        total = sum(range(start, stop))
        items.append(
            Item(
                topic=f"for-Schleife mit range: {start} bis {stop - 1}",
                difficulty="beginner",
                tags=["python", "kontrollfluss", "for", "range"],
                explanation=(
                    "Eine for-Schleife wiederholt Code für jedes Element einer Folge. "
                    "range(start, stop) erzeugt Zahlen bis ausschließlich stop. "
                    "Das Beispiel summiert alle erzeugten Zahlen."
                ),
                code=f"""
summe = 0
for zahl in range({start}, {stop}):
    summe += zahl

print(summe)
""",
                tests=f"""
assert summe == {total}
assert list(range({start}, {stop}))[0] == {start}
assert list(range({start}, {stop}))[-1] == {stop - 1}
""",
                typical_error="Ein häufiger Fehler ist zu vergessen, dass range den Endwert nicht einschließt.",
                correction="Wenn der Endwert enthalten sein soll, verwende range(start, endwert + 1).",
            )
        )

    thresholds = [
        (12, 18),
        (20, 18),
        (17, 18),
        (65, 60),
        (59, 60),
        (100, 100),
        (99, 100),
        (5, 10),
        (10, 10),
        (11, 10),
    ]
    for value, threshold in thresholds:
        label = "genug" if value >= threshold else "zu wenig"
        items.append(
            Item(
                topic=f"if-else Entscheidung: {value} gegen {threshold}",
                difficulty="beginner",
                tags=["python", "kontrollfluss", "if", "vergleich"],
                explanation=(
                    "Mit if und else wird Code abhängig von einer Bedingung ausgeführt. "
                    "Der Vergleich wert >= grenze liefert True oder False. "
                    "Die Einrückung bestimmt, welcher Code zum if-Block gehört."
                ),
                code=f"""
wert = {value}
grenze = {threshold}

if wert >= grenze:
    status = "genug"
else:
    status = "zu wenig"

print(status)
""",
                tests=f'''
assert status == "{label}"
assert (wert >= grenze) is {value >= threshold!s}
''',
                typical_error="Ein typischer Fehler ist falsche Einrückung nach if oder else.",
                correction="Rücke alle Zeilen eines Blocks konsequent um vier Leerzeichen ein.",
            )
        )

    words = [
        ("banane", "a"),
        ("tutorial", "t"),
        ("python", "y"),
        ("schleife", "e"),
        ("daten", "z"),
        ("kontrolle", "l"),
        ("variable", "v"),
        ("funktion", "k"),
        ("liste", "s"),
        ("zeichen", "n"),
    ]
    for word, char in words:
        count = word.count(char)
        items.append(
            Item(
                topic=f"while-Schleife zählt Zeichen: {char}",
                difficulty="easy",
                tags=["python", "kontrollfluss", "while", "string"],
                explanation=(
                    "Eine while-Schleife läuft, solange ihre Bedingung wahr ist. "
                    "Hier wird ein Index Schritt für Schritt durch ein Wort bewegt. "
                    "Das Beispiel zeigt bewusst die manuelle Variante, damit der Ablauf klar wird."
                ),
                code=f'''
wort = "{word}"
gesucht = "{char}"
index = 0
anzahl = 0

while index < len(wort):
    if wort[index] == gesucht:
        anzahl += 1
    index += 1

print(anzahl)
''',
                tests=f"""
assert anzahl == {count}
assert index == len(wort)
assert wort.count(gesucht) == anzahl
""",
                typical_error="Bei while-Schleifen wird oft vergessen, den Index zu erhöhen.",
                correction="Sorge dafür, dass sich die Schleifenbedingung im Block irgendwann ändert.",
            )
        )

    nested = [
        (["rot", "blau"], [1, 2]),
        (["A", "B"], [10, 20]),
        (["klein", "gross"], [3, 4]),
        (["x", "y"], [5, 6]),
        (["links", "rechts"], [7, 8]),
        (["oben", "unten"], [9, 10]),
        (["warm", "kalt"], [11, 12]),
        (["tag", "nacht"], [13, 14]),
        (["ja", "nein"], [15, 16]),
        (["start", "ziel"], [17, 18]),
    ]
    for labels, nums in nested:
        expected_len = len(labels) * len(nums)
        items.append(
            Item(
                topic=f"Verschachtelte Schleifen: {labels[0]} und {labels[1]}",
                difficulty="easy",
                tags=["python", "kontrollfluss", "for", "verschachtelt"],
                explanation=(
                    "Verschachtelte Schleifen kombinieren jedes Element der äußeren Schleife mit jedem Element der inneren Schleife. "
                    "Das Ergebnis ist eine Liste von Paaren. "
                    "Solche Muster kommen bei Tabellen, Koordinaten und Kombinationen vor."
                ),
                code=f"""
labels = {labels!r}
nummern = {nums!r}
paare = []

for label in labels:
    for nummer in nummern:
        paare.append((label, nummer))

print(paare)
""",
                tests=f"""
assert len(paare) == {expected_len}
assert paare[0] == ({labels[0]!r}, {nums[0]})
assert paare[-1] == ({labels[-1]!r}, {nums[-1]})
""",
                typical_error="Ein häufiger Fehler ist, append außerhalb der inneren Schleife zu platzieren.",
                correction="Achte auf die Einrückung: Was pro Kombination passieren soll, gehört in die innere Schleife.",
            )
        )

    enum_cases = [
        ["Ada", "Ben", "Clara"],
        ["rot", "blau", "grün"],
        ["Mo", "Di", "Mi"],
        ["A", "B", "C"],
        ["eins", "zwei", "drei"],
        ["py", "js", "go"],
        ["Nord", "Süd"],
        ["Start", "Ziel"],
        ["klein", "mittel", "gross"],
        ["x", "y", "z"],
    ]
    for values in enum_cases:
        first = f"0:{values[0]}"
        items.append(
            Item(
                topic=f"enumerate verwenden: {values[0]}",
                difficulty="easy",
                tags=["python", "kontrollfluss", "enumerate", "liste"],
                explanation=(
                    "enumerate() liefert zu jedem Listenelement auch seinen Index. "
                    "So vermeidet man einen manuell gepflegten Zähler. "
                    "Das Beispiel baut aus Index und Wert kurze Labels."
                ),
                code=f"""
werte = {values!r}
labels = []

for index, wert in enumerate(werte):
    labels.append(f"{{index}}:{{wert}}")

print(labels[0])
""",
                tests=f"""
assert labels[0] == {first!r}
assert len(labels) == {len(values)}
assert labels[-1].endswith({values[-1]!r})
""",
                typical_error="Ein häufiger Fehler ist, einen eigenen Index zu erhöhen und dabei off-by-one-Fehler zu bauen.",
                correction="Nutze enumerate(), wenn du Index und Wert gleichzeitig brauchst.",
            )
        )

    skip_cases = [
        [-2, 1, 3],
        [0, 4, -1],
        [5, -5, 10],
        [-3, -2, 7],
        [8, 0, 2],
        [1, 2, 3],
        [-1, -4, 6],
        [9, -9, 0],
        [10, 11, -1],
        [-8, 4, 5],
    ]
    for values in skip_cases:
        expected = sum(v for v in values if v > 0)
        items.append(
            Item(
                topic=f"continue zum Überspringen: {values}",
                difficulty="easy",
                tags=["python", "kontrollfluss", "continue", "schleife"],
                explanation=(
                    "continue beendet nur den aktuellen Schleifendurchlauf und springt zum nächsten Element. "
                    "Hier werden nicht-positive Zahlen übersprungen. "
                    "Die Summe enthält dadurch nur positive Werte."
                ),
                code=f"""
zahlen = {values!r}
summe = 0

for zahl in zahlen:
    if zahl <= 0:
        continue
    summe += zahl

print(summe)
""",
                tests=f"""
assert summe == {expected}
assert summe == sum(zahl for zahl in zahlen if zahl > 0)
""",
                typical_error="Ein häufiger Fehler ist, continue mit break zu verwechseln.",
                correction="Nutze continue zum Überspringen eines Elements und break zum vollständigen Beenden der Schleife.",
            )
        )

    break_cases = [
        ([2, 4, 9, 3], 8),
        ([1, 2, 3], 5),
        ([10, 5, 2], 7),
        ([3, 6, 12], 10),
        ([4, 4, 4], 3),
        ([0, 1, 2], 1),
        ([8, 9, 10], 9),
        ([5, 15, 1], 10),
        ([7, 3, 2], 6),
        ([11, 12], 20),
    ]
    for values, threshold in break_cases:
        expected = next((v for v in values if v > threshold), None)
        items.append(
            Item(
                topic=f"break bei erstem Treffer: Grenze {threshold}",
                difficulty="easy",
                tags=["python", "kontrollfluss", "break", "suche"],
                explanation=(
                    "break beendet eine Schleife sofort. "
                    "Das ist sinnvoll, wenn nur der erste passende Wert gesucht wird. "
                    "Wird kein Wert gefunden, bleibt die Ergebnisvariable None."
                ),
                code=f"""
zahlen = {values!r}
grenze = {threshold}
treffer = None

for zahl in zahlen:
    if zahl > grenze:
        treffer = zahl
        break

print(treffer)
""",
                tests=f"""
assert treffer == {expected!r}
assert treffer == next((zahl for zahl in zahlen if zahl > grenze), None)
""",
                typical_error="Ein häufiger Fehler ist, break zu vergessen und dadurch den letzten statt den ersten Treffer zu speichern.",
                correction="Setze break direkt nach dem Speichern, wenn nur der erste Treffer gebraucht wird.",
            )
        )

    zip_cases = [
        (["Ada", "Ben"], [10, 12]),
        (["rot", "blau"], [1, 2]),
        (["A", "B", "C"], [3, 4, 5]),
        (["Mo", "Di"], [6, 7]),
        (["x", "y"], [8, 9]),
    ]
    for names, scores in zip_cases:
        expected = [f"{n}:{s}" for n, s in zip(names, scores)]
        items.append(
            Item(
                topic=f"zip für parallele Listen: {names[0]}",
                difficulty="easy",
                tags=["python", "kontrollfluss", "zip", "liste"],
                explanation=(
                    "zip() verbindet mehrere Listen positionsweise. "
                    "So kann man zusammengehörige Namen und Werte gemeinsam durchlaufen. "
                    "Das ist lesbarer als manuelle Indexzugriffe."
                ),
                code=f"""
namen = {names!r}
punkte = {scores!r}
zeilen = []

for name, punktzahl in zip(namen, punkte):
    zeilen.append(f"{{name}}:{{punktzahl}}")

print(zeilen)
""",
                tests=f"""
assert zeilen == {expected!r}
assert len(zeilen) == min(len(namen), len(punkte))
""",
                typical_error="Ein häufiger Fehler ist, anzunehmen, dass zip fehlende Werte auffüllt.",
                correction="zip endet bei der kürzesten Eingabe; prüfe die Längen, wenn das wichtig ist.",
            )
        )

    return items


def errors() -> list[Item]:
    items: list[Item] = []
    raw_values = ["12", "x", "7", "", "42", "3.5", "-8", " 9 ", "abc", "0"]
    for raw in raw_values:
        ok = raw.strip().lstrip("-").isdigit()
        expected = int(raw) if ok else None
        items.append(
            Item(
                topic=f"ValueError vermeiden: {raw!r}",
                difficulty="easy",
                tags=["python", "fehlerbehandlung", "try-except", "valueerror"],
                explanation=(
                    "try und except fangen erwartbare Fehler kontrolliert ab. "
                    "Bei int() kann ein ValueError entstehen, wenn der Text keine ganze Zahl enthält. "
                    "Die Funktion gibt in diesem Fall None zurück, statt das Programm abbrechen zu lassen."
                ),
                code=f"""
def parse_int(text):
    try:
        return int(text)
    except ValueError:
        return None

wert = parse_int({raw!r})
print(wert)
""",
                tests=f"""
assert parse_int({raw!r}) == {expected!r}
assert parse_int("15") == 15
assert parse_int("nicht numerisch") is None
""",
                typical_error="Ein häufiger Fehler ist, alle Exceptions pauschal mit except Exception zu verschlucken.",
                correction="Fange möglichst gezielt den Fehler ab, den du erwartest, hier ValueError.",
            )
        )

    divs = [(10, 2), (10, 0), (9, 3), (5, 0), (12, 4), (7, 0), (100, 10), (8, 2), (6, 0), (15, 5)]
    for a, b in divs:
        expected = None if b == 0 else a / b
        items.append(
            Item(
                topic=f"Division sicher behandeln: {a} durch {b}",
                difficulty="easy",
                tags=["python", "fehlerbehandlung", "zerodivisionerror", "funktion"],
                explanation=(
                    "Division durch null löst in Python einen ZeroDivisionError aus. "
                    "Eine kleine Schutzfunktion kann diesen Fall abfangen und ein klares Ersatzsignal liefern. "
                    "So bleibt der aufrufende Code kontrollierbar."
                ),
                code=f"""
def sicher_teilen(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return None

ergebnis = sicher_teilen({a}, {b})
print(ergebnis)
""",
                tests=f"""
assert sicher_teilen({a}, {b}) == {expected!r}
assert sicher_teilen(8, 2) == 4
assert sicher_teilen(8, 0) is None
""",
                typical_error="Manchmal wird b == 0 nicht geprüft und das Programm bricht ab.",
                correction="Behandle Division durch null bewusst, entweder mit if b == 0 oder mit try/except.",
            )
        )

    dict_cases = [
        ({"name": "Mara"}, "name"),
        ({"alter": 30}, "name"),
        ({"stadt": "Bonn"}, "stadt"),
        ({"punkte": 12}, "rang"),
        ({"kurs": "Python"}, "kurs"),
        ({"aktiv": True}, "aktiv"),
        ({"preis": 9.99}, "preis"),
        ({"tag": "Mo"}, "monat"),
        ({"id": 7}, "id"),
        ({"rolle": "admin"}, "rechte"),
    ]
    for data, key in dict_cases:
        expected = data.get(key, "unbekannt")
        items.append(
            Item(
                topic=f"KeyError vermeiden: Schlüssel {key}",
                difficulty="easy",
                tags=["python", "fehlerbehandlung", "dictionary", "get"],
                explanation=(
                    "Beim Zugriff mit eckigen Klammern entsteht ein KeyError, wenn der Schlüssel fehlt. "
                    "dict.get() erlaubt einen Standardwert. "
                    "Das ist nützlich, wenn Daten unvollständig sein können."
                ),
                code=f'''
datensatz = {data!r}
wert = datensatz.get("{key}", "unbekannt")

print(wert)
''',
                tests=f"""
assert wert == {expected!r}
assert datensatz.get("fehlt", "unbekannt") == "unbekannt"
""",
                typical_error="Ein häufiger Fehler ist datensatz['name'] zu schreiben, obwohl 'name' fehlen kann.",
                correction="Nutze get() mit einem Standardwert, wenn ein Schlüssel optional ist.",
            )
        )

    trace_cases = [
        ("int('x')", "ValueError"),
        ("1 / 0", "ZeroDivisionError"),
        ("{}['x']", "KeyError"),
        ("[1, 2][5]", "IndexError"),
        ("float('abc')", "ValueError"),
        ("10 / 0", "ZeroDivisionError"),
        ("{'a': 1}['b']", "KeyError"),
        ("[][0]", "IndexError"),
        ("int('4.2')", "ValueError"),
        ("5 // 0", "ZeroDivisionError"),
    ]
    for expr, err in trace_cases:
        items.append(
            Item(
                topic=f"Traceback-Typ erkennen: {err}",
                difficulty="medium",
                tags=["python", "fehlerbehandlung", "traceback", "exception"],
                explanation=(
                    "Tracebacks zeigen, welche Exception aufgetreten ist. "
                    "In diesem Beispiel wird der Fehler absichtlich abgefangen und sein Klassenname ausgegeben. "
                    "So lernt man, Fehlermeldungen gezielt zu lesen."
                ),
                code=f"""
try:
    ergebnis = {expr}
except Exception as fehler:
    ergebnis = type(fehler).__name__

print(ergebnis)
""",
                tests=f'''
assert ergebnis == "{err}"
assert isinstance(ergebnis, str)
''',
                typical_error="Ein typischer Anfängerfehler ist, nur die letzte Zeile des Tracebacks zu ignorieren.",
                correction="Lies den Exception-Typ und die Zeile, auf die der Traceback zeigt.",
            )
        )

    index_cases = [
        ([1, 2, 3], 0),
        ([1, 2, 3], 5),
        (["a", "b"], 1),
        (["a", "b"], -1),
        ([], 0),
        ([10], 2),
        ([5, 6, 7], -4),
        (["Start", "Ziel"], 0),
        ([100, 200], 3),
        ([8, 9], -2),
    ]
    for values, idx in index_cases:
        expected = values[idx] if -len(values) <= idx < len(values) and values else None
        items.append(
            Item(
                topic=f"IndexError vermeiden: Index {idx}",
                difficulty="easy",
                tags=["python", "fehlerbehandlung", "indexerror", "liste"],
                explanation=(
                    "Ein IndexError entsteht, wenn ein Listenindex außerhalb des gültigen Bereichs liegt. "
                    "Die Funktion fängt diesen Fehler ab und gibt None zurück. "
                    "Das kann sinnvoll sein, wenn fehlende Werte erlaubt sind."
                ),
                code=f"""
def sicher_lesen(werte, index):
    try:
        return werte[index]
    except IndexError:
        return None

wert = sicher_lesen({values!r}, {idx})
print(wert)
""",
                tests=f"""
assert sicher_lesen({values!r}, {idx}) == {expected!r}
assert sicher_lesen([1, 2], 9) is None
assert sicher_lesen(["x", "y"], 0) == "x"
""",
                typical_error="Ein häufiger Fehler ist, vor dem Zugriff nicht über leere Listen nachzudenken.",
                correction="Fange IndexError ab oder prüfe vorher, ob der Index im gültigen Bereich liegt.",
            )
        )

    return items


def algorithms() -> list[Item]:
    items: list[Item] = []
    sequences = [
        [3, 1, 8, 2],
        [10, 5, 12, 7],
        [-3, -1, -8],
        [4, 4, 2, 9],
        [100, 50, 75],
        [6, 11, 2, 15],
        [1],
        [0, -2, 5],
        [23, 42, 17],
        [8, 16, 4, 32],
    ]
    for values in sequences:
        items.append(
            Item(
                topic=f"Maximum selbst finden: {values[0]}...",
                difficulty="easy",
                tags=["python", "algorithmus", "liste", "schleife"],
                explanation=(
                    "Dieses Beispiel implementiert die Suche nach dem größten Wert ohne max(). "
                    "Der bisher beste Wert wird gespeichert und bei jedem größeren Element ersetzt. "
                    "Das Muster ist grundlegend für viele Suchalgorithmen."
                ),
                code=f"""
zahlen = {values!r}
groesste = zahlen[0]

for zahl in zahlen[1:]:
    if zahl > groesste:
        groesste = zahl

print(groesste)
""",
                tests=f"""
assert groesste == {max(values)}
assert groesste == max(zahlen)
""",
                typical_error="Ein häufiger Fehler ist, groesste mit 0 zu starten; das scheitert bei nur negativen Zahlen.",
                correction="Starte mit dem ersten Listenelement, wenn die Liste nicht leer ist.",
            )
        )

    words = [
        "lagerregal",
        "python",
        "anna",
        "reliefpfeiler",
        "daten",
        "radar",
        "algorithmus",
        "otto",
        "test",
        "level",
    ]
    for word in words:
        is_pal = word == word[::-1]
        items.append(
            Item(
                topic=f"Palindrom prüfen: {word}",
                difficulty="easy",
                tags=["python", "algorithmus", "string", "slicing"],
                explanation=(
                    "Ein Palindrom liest sich vorwärts und rückwärts gleich. "
                    "Mit slicing [::-1] wird eine Zeichenkette umgedreht. "
                    "Der Vergleich mit dem Original liefert einen booleschen Wert."
                ),
                code=f'''
wort = "{word}"
ist_palindrom = wort == wort[::-1]

print(ist_palindrom)
''',
                tests=f'''
assert ist_palindrom is {is_pal!s}
assert "{word}"[::-1] == {word[::-1]!r}
''',
                typical_error="Ein häufiger Fehler ist, Groß-/Kleinschreibung und Leerzeichen bei Sätzen nicht zu normalisieren.",
                correction="Für einfache Wörter reicht der direkte Vergleich; bei Sätzen sollte man vorher bereinigen.",
            )
        )

    fact_nums = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]
    import math

    for n in fact_nums:
        items.append(
            Item(
                topic=f"Fakultät iterativ berechnen: {n}",
                difficulty="medium",
                tags=["python", "algorithmus", "schleife", "fakultaet"],
                explanation=(
                    "Die Fakultät einer Zahl ist das Produkt aller ganzen Zahlen von 1 bis n. "
                    "Die Schleife multipliziert Schritt für Schritt in die Variable ergebnis. "
                    "Für 0 und 1 bleibt das Anfangsergebnis 1 korrekt."
                ),
                code=f"""
def fakultaet(n):
    ergebnis = 1
    for zahl in range(2, n + 1):
        ergebnis *= zahl
    return ergebnis

wert = fakultaet({n})
print(wert)
""",
                tests=f"""
assert fakultaet({n}) == {math.factorial(n)}
assert fakultaet(0) == 1
assert fakultaet(1) == 1
assert fakultaet(5) == 120
""",
                typical_error="Ein häufiger Fehler ist, das Ergebnis mit 0 zu starten.",
                correction="Bei Multiplikation muss der Startwert 1 sein, sonst bleibt das Produkt immer 0.",
            )
        )

    dedup_lists = [
        [1, 2, 1, 3, 2],
        ["a", "b", "a", "c"],
        [5, 5, 5, 6],
        ["rot", "blau", "rot"],
        [3, 1, 3, 2, 1],
        ["x", "y", "x", "z"],
        [10, 20, 10, 30],
        ["Anna", "Ben", "Anna"],
        [0, 0, 1, 2],
        ["py", "js", "py"],
    ]
    for values in dedup_lists:
        expected = []
        for v in values:
            if v not in expected:
                expected.append(v)
        items.append(
            Item(
                topic=f"Duplikate stabil entfernen: {len(values)} Werte",
                difficulty="medium",
                tags=["python", "algorithmus", "liste", "dedup"],
                explanation=(
                    "Ein set entfernt Duplikate, verliert aber die ursprüngliche Reihenfolge. "
                    "Dieses Beispiel sammelt Werte nur dann ein, wenn sie noch nicht gesehen wurden. "
                    "So bleibt die erste Reihenfolge erhalten."
                ),
                code=f"""
werte = {values!r}
ohne_duplikate = []

for wert in werte:
    if wert not in ohne_duplikate:
        ohne_duplikate.append(wert)

print(ohne_duplikate)
""",
                tests=f"""
assert ohne_duplikate == {expected!r}
assert len(ohne_duplikate) == len(set(werte))
assert ohne_duplikate[0] == werte[0]
""",
                typical_error="Ein häufiger Fehler ist set(werte) zu nutzen, obwohl die Reihenfolge wichtig ist.",
                correction="Nutze eine Schleife mit einer Ergebnisliste, wenn die erste Reihenfolge erhalten bleiben soll.",
            )
        )

    evens = [
        [1, 2, 3, 4],
        [10, 11, 12],
        [5, 7, 9],
        [0, 2, 8],
        [-4, -3, -2],
        [13, 14, 15, 16],
        [21, 22],
        [100, 101],
        [6, 6, 7],
        [31, 32, 33],
    ]
    for values in evens:
        expected = [v for v in values if v % 2 == 0]
        items.append(
            Item(
                topic=f"Gerade Zahlen filtern: {values[0]}...",
                difficulty="easy",
                tags=["python", "algorithmus", "modulo", "filter"],
                explanation=(
                    "Der Modulo-Operator % liefert den Rest einer Division. "
                    "Eine Zahl ist gerade, wenn der Rest bei Division durch 2 gleich 0 ist. "
                    "Die Schleife sammelt nur passende Werte in einer neuen Liste."
                ),
                code=f"""
zahlen = {values!r}
gerade = []

for zahl in zahlen:
    if zahl % 2 == 0:
        gerade.append(zahl)

print(gerade)
""",
                tests=f"""
assert gerade == {expected!r}
assert all(zahl % 2 == 0 for zahl in gerade)
""",
                typical_error="Ein häufiger Fehler ist zahl / 2 == 0 zu prüfen.",
                correction="Nutze zahl % 2 == 0, um den Rest der Division zu prüfen.",
            )
        )

    return items


def tests_topic() -> list[Item]:
    items: list[Item] = []
    cases = [
        ("ist_positiv", "n > 0", [(3, True), (0, False), (-2, False)]),
        ("ist_leer", "len(text) == 0", [("", True), ("x", False), ("Hallo", False)]),
        ("hat_mindestlaenge", "len(text) >= 4", [("Test", True), ("abc", False), ("Python", True)]),
        ("ist_erwachsen", "alter >= 18", [(18, True), (17, False), (30, True)]),
        ("ist_kurz", "len(text) <= 5", [("Haus", True), ("Python", False), ("abcde", True)]),
        ("ist_null", "n == 0", [(0, True), (1, False), (-1, False)]),
        ("ist_grossbuchstabe", "zeichen.isupper()", [("A", True), ("a", False), ("Z", True)]),
        ("enthaelt_a", "'a' in text.lower()", [("Anna", True), ("Bonn", False), ("Tag", True)]),
        ("ist_vielfaches_von_3", "n % 3 == 0", [(9, True), (10, False), (0, True)]),
        ("ist_wahr", "wert is True", [(True, True), (False, False), (1, False)]),
    ]
    for fname, expr, values in cases:
        param = (
            "text"
            if "text" in expr
            else "alter"
            if "alter" in expr
            else "zeichen"
            if "zeichen" in expr
            else "wert"
            if "wert" in expr
            else "n"
        )
        sample = values[0][0]
        items.append(
            Item(
                topic=f"assert-Tests für {fname}",
                difficulty="beginner",
                tags=["python", "tests", "assert", "funktion"],
                explanation=(
                    "assert prüft, ob eine Annahme im Code wahr ist. "
                    "Wenn die Bedingung falsch ist, stoppt Python mit einem AssertionError. "
                    "Kleine Tests helfen, Funktionen bei späteren Änderungen abzusichern."
                ),
                code=f"""
def {fname}({param}):
    return {expr}

print({fname}({sample!r}))
""",
                tests="\n".join(f"assert {fname}({v!r}) is {expected!s}" for v, expected in values),
                typical_error="Ein häufiger Fehler ist, nur den erfolgreichen Fall zu testen.",
                correction="Teste auch Grenzfälle und Beispiele, bei denen die Funktion False liefern muss.",
            )
        )

    arithmetic = [
        ("addiere", "a + b", 2, 5),
        ("subtrahiere", "a - b", 9, 4),
        ("multipliziere", "a * b", 6, 7),
        ("teile_ganzzahlig", "a // b", 17, 5),
        ("rest", "a % b", 17, 5),
        ("mittelwert_zwei", "(a + b) / 2", 10, 20),
        ("maximum_zwei", "a if a >= b else b", 8, 12),
        ("minimum_zwei", "a if a <= b else b", 8, 12),
        ("quadratsumme", "a * a + b * b", 3, 4),
        ("differenz_betrag", "abs(a - b)", 3, 10),
    ]
    for fname, expr, a, b in arithmetic:
        expected = eval(expr, {"a": a, "b": b, "abs": abs})
        items.append(
            Item(
                topic=f"Testfunktion für {fname}",
                difficulty="easy",
                tags=["python", "tests", "assert", "mathematik"],
                explanation=(
                    "Bei Rechenfunktionen sind assert-Tests besonders nützlich, weil erwartete Ergebnisse klar sind. "
                    "Die Tests stehen getrennt vom eigentlichen Code und prüfen mehrere Eingaben. "
                    "So fallen Vertauschungen von Operatoren schnell auf."
                ),
                code=f"""
def {fname}(a, b):
    return {expr}

ergebnis = {fname}({a}, {b})
print(ergebnis)
""",
                tests=f"""
assert {fname}({a}, {b}) == {expected!r}
assert {fname}(1, 1) == {eval(expr, {"a": 1, "b": 1, "abs": abs})!r}
assert ergebnis == {expected!r}
""",
                typical_error="Ein häufiger Fehler ist, Tests nur nach Gefühl zu schreiben.",
                correction="Berechne erwartete Werte bewusst und schreibe sie explizit in assert-Anweisungen.",
            )
        )

    validators = [
        (
            "normalisiere_name",
            "' '.join(text.strip().split()).title()",
            "  ada  lovelace ",
            "Ada Lovelace",
        ),
        ("kleinbuchstaben", "text.strip().lower()", "  HALLO ", "hallo"),
        ("ohne_leerzeichen", "text.replace(' ', '')", "a b c", "abc"),
        ("erste_drei", "text[:3]", "Python", "Pyt"),
        ("letzte_drei", "text[-3:]", "Python", "hon"),
        ("ersetze_minus", "text.replace('-', '_')", "a-b-c", "a_b_c"),
        ("zaehle_a", "text.lower().count('a')", "Ananas", 3),
        ("ist_palindrom_text", "text == text[::-1]", "anna", True),
        ("kuerze", "text[:5]", "Datensatz", "Daten"),
        ("endet_mit_punkt", "text.endswith('.')", "Hallo.", True),
    ]
    for fname, expr, value, expected in validators:
        items.append(
            Item(
                topic=f"Stringfunktion testen: {fname}",
                difficulty="easy",
                tags=["python", "tests", "string", "assert"],
                explanation=(
                    "Stringfunktionen verändern oder prüfen Text. "
                    "Mit assert kann man genau festhalten, welches Ergebnis erwartet wird. "
                    "Das ist hilfreich, weil Leerzeichen und Großschreibung leicht übersehen werden."
                ),
                code=f"""
def {fname}(text):
    return {expr}

ergebnis = {fname}({value!r})
print(ergebnis)
""",
                tests=f"""
assert {fname}({value!r}) == {expected!r}
assert ergebnis == {expected!r}
""",
                typical_error="Ein häufiger Fehler ist, führende oder doppelte Leerzeichen in Tests zu vergessen.",
                correction="Nimm genau die Eingabe in den Test auf, die du bereinigen möchtest.",
            )
        )

    collections = [
        ("anzahl_elemente", "len(werte)", [1, 2, 3], 3),
        ("summe_liste", "sum(werte)", [2, 4, 6], 12),
        ("erstes_element", "werte[0]", ["a", "b"], "a"),
        ("letztes_element", "werte[-1]", ["a", "b", "c"], "c"),
        ("sortiere", "sorted(werte)", [3, 1, 2], [1, 2, 3]),
        ("eindeutig", "sorted(set(werte))", [2, 1, 2], [1, 2]),
        ("umdrehen", "list(reversed(werte))", [1, 2, 3], [3, 2, 1]),
        ("positive", "[w for w in werte if w > 0]", [-1, 0, 2, 3], [2, 3]),
        ("verdoppeln", "[w * 2 for w in werte]", [1, 3, 5], [2, 6, 10]),
        ("als_tupel", "tuple(werte)", [4, 5], (4, 5)),
    ]
    for fname, expr, values, expected in collections:
        items.append(
            Item(
                topic=f"Collection-Test: {fname}",
                difficulty="easy",
                tags=["python", "tests", "liste", "datenstruktur"],
                explanation=(
                    "Tests für Listenfunktionen prüfen Inhalt und Reihenfolge. "
                    "Gerade bei Sortierung, Filterung und Umwandlung ist das erwartete Ergebnis gut vorhersehbar. "
                    "assert macht diese Erwartung ausführbar."
                ),
                code=f"""
def {fname}(werte):
    return {expr}

ergebnis = {fname}({values!r})
print(ergebnis)
""",
                tests=f"""
assert {fname}({values!r}) == {expected!r}
assert ergebnis == {expected!r}
""",
                typical_error="Ein häufiger Fehler ist, nur die Länge der Liste zu testen.",
                correction="Teste auch die konkreten Werte und ihre Reihenfolge.",
            )
        )

    runner_cases = [
        ("ist_klein_genug", "wert <= 10", [(5, True), (10, True), (11, False)]),
        (
            "enthaelt_text",
            "teil in text",
            [("Py", "Python", True), ("x", "abc", False), ("a", "Ada", True)],
        ),
        (
            "ist_sortiert",
            "werte == sorted(werte)",
            [([1, 2, 3], True), ([3, 1, 2], False), ([], True)],
        ),
        (
            "hat_genau_drei",
            "len(werte) == 3",
            [([1, 2, 3], True), ([1], False), ([1, 2, 3, 4], False)],
        ),
        (
            "ist_prefix",
            "text.startswith(prefix)",
            [("py", "python", True), ("ja", "nein", False), ("Da", "Daten", True)],
        ),
        (
            "ist_suffix",
            "text.endswith(suffix)",
            [(".py", "test.py", True), (".txt", "test.py", False), ("en", "Daten", True)],
        ),
        (
            "ist_im_bereich",
            "minimum <= wert <= maximum",
            [(1, 1, 3, True), (4, 1, 3, False), (2, 1, 3, True)],
        ),
        (
            "hat_key",
            "key in daten",
            [
                ("name", {"name": "Ada"}, True),
                ("id", {"name": "Ada"}, False),
                ("x", {"x": 1}, True),
            ],
        ),
        (
            "summe_ist_positiv",
            "sum(werte) > 0",
            [([1, 2], True), ([-5, 1], False), ([0, 0], False)],
        ),
        ("alle_wahr", "all(werte)", [([True, True], True), ([True, False], False), ([], True)]),
    ]
    for fname, expr, cases in runner_cases:
        if fname == "enthaelt_text":
            params = "teil, text"
            call_lines = [f"assert {fname}({a!r}, {b!r}) is {e}" for a, b, e in cases]
            sample_call = f"{fname}({cases[0][0]!r}, {cases[0][1]!r})"
        elif fname in {"ist_prefix", "ist_suffix"}:
            params = "prefix, text" if fname == "ist_prefix" else "suffix, text"
            call_lines = [f"assert {fname}({a!r}, {b!r}) is {e}" for a, b, e in cases]
            sample_call = f"{fname}({cases[0][0]!r}, {cases[0][1]!r})"
        elif fname == "ist_im_bereich":
            params = "wert, minimum, maximum"
            call_lines = [f"assert {fname}({a}, {b}, {c}) is {e}" for a, b, c, e in cases]
            sample_call = f"{fname}({cases[0][0]}, {cases[0][1]}, {cases[0][2]})"
        elif fname == "hat_key":
            params = "key, daten"
            call_lines = [f"assert {fname}({a!r}, {b!r}) is {e}" for a, b, e in cases]
            sample_call = f"{fname}({cases[0][0]!r}, {cases[0][1]!r})"
        else:
            params = "werte" if "werte" in expr else "wert"
            call_lines = [f"assert {fname}({a!r}) is {e}" for a, e in cases]
            sample_call = f"{fname}({cases[0][0]!r})"
        items.append(
            Item(
                topic=f"Kleine Testfunktion ausführen: {fname}",
                difficulty="medium",
                tags=["python", "tests", "assert", "testfunktion"],
                explanation=(
                    "Eine kleine Testfunktion kann mehrere assert-Prüfungen bündeln. "
                    "Sie wird am Ende explizit aufgerufen. "
                    "Wenn alle Tests bestehen, läuft das Programm ohne AssertionError weiter."
                ),
                code=f"""
def {fname}({params}):
    return {expr}

print({sample_call})
""",
                tests="\n".join(
                    [
                        "def teste_funktion():",
                        *["    " + line for line in call_lines],
                        "",
                        "teste_funktion()",
                    ]
                ),
                typical_error="Ein häufiger Fehler ist, eine Testfunktion zu definieren, aber nie aufzurufen.",
                correction="Rufe die Testfunktion am Ende auf, damit die assert-Prüfungen wirklich ausgeführt werden.",
            )
        )

    return items


BUILDERS: list[tuple[str, int, Callable[[], list[Item]]]] = [
    ("Python-Grundlagen", 100, basics),
    ("Funktionen und Parameter", 100, functions),
    ("Listen/Dictionaries/Sets", 75, data_structures),
    ("Kontrollfluss und Schleifen", 75, control_flow),
    ("Fehlerbehandlung und Tracebacks", 50, errors),
    ("Kleine Algorithmen", 50, algorithms),
    ("Tests mit assert", 50, tests_topic),
]


def build_records() -> list[dict[str, object]]:
    rng = random.Random(SEED)
    records: list[dict[str, object]] = []
    seen_text: set[str] = set()
    next_id = 1
    for group, target, builder in BUILDERS:
        candidates = builder()
        if len(candidates) < target:
            raise RuntimeError(f"{group}: only {len(candidates)} candidates for target {target}")
        rng.shuffle(candidates)
        picked = 0
        for item in candidates:
            output = run_blocks(item.code, item.tests)
            text = render(item, output)
            if text in seen_text:
                continue
            seen_text.add(text)
            records.append(
                {
                    "id": f"py_anneal_{next_id:06d}",
                    "topic": item.topic,
                    "difficulty": item.difficulty,
                    "tags": item.tags,
                    "text": text,
                }
            )
            next_id += 1
            picked += 1
            if picked == target:
                break
        if picked != target:
            raise RuntimeError(f"{group}: picked {picked}, expected {target}")
    if len(records) != 500:
        raise RuntimeError(f"expected 500 records, got {len(records)}")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = build_records()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(records), "out": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
