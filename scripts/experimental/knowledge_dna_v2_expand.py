"""Generate a larger synthetic-curated Knowledge-DNA v2 seed set.

The first Knowledge-DNA smoke used only a few hand-written concepts. That is
useful for plumbing but too small for a scaling signal. This script creates a
larger deterministic seed set from simple factual templates and writes the same
plain/dna/hybrid corpus layout as ``knowledge_dna_v2.py``.

It deliberately avoids changing the tokenizer and keeps all facts simple enough
to audit by reading the generated entries.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.experimental.knowledge_dna_v2 import (
    KnowledgeDNAEntry,
    Probe,
    build_outputs,
    sample_entries,
)


COUNTRY_CAPITALS = [
    ("Deutschland", "Berlin", "Europa"),
    ("Frankreich", "Paris", "Europa"),
    ("Italien", "Rom", "Europa"),
    ("Spanien", "Madrid", "Europa"),
    ("Portugal", "Lissabon", "Europa"),
    ("Oesterreich", "Wien", "Europa"),
    ("Schweiz", "Bern", "Europa"),
    ("Niederlande", "Amsterdam", "Europa"),
    ("Belgien", "Bruessel", "Europa"),
    ("Polen", "Warschau", "Europa"),
    ("Tschechien", "Prag", "Europa"),
    ("Daenemark", "Kopenhagen", "Europa"),
    ("Schweden", "Stockholm", "Europa"),
    ("Norwegen", "Oslo", "Europa"),
    ("Finnland", "Helsinki", "Europa"),
    ("Griechenland", "Athen", "Europa"),
    ("Tuerkei", "Ankara", "Europa und Asien"),
    ("Japan", "Tokio", "Asien"),
    ("China", "Peking", "Asien"),
    ("Indien", "Neu-Delhi", "Asien"),
    ("Kanada", "Ottawa", "Nordamerika"),
    ("Vereinigte Staaten", "Washington, D.C.", "Nordamerika"),
    ("Mexiko", "Mexiko-Stadt", "Nordamerika"),
    ("Brasilien", "Brasilia", "Suedamerika"),
    ("Argentinien", "Buenos Aires", "Suedamerika"),
    ("Chile", "Santiago de Chile", "Suedamerika"),
    ("Aegypten", "Kairo", "Afrika"),
    ("Suedafrika", "Pretoria", "Afrika"),
    ("Australien", "Canberra", "Australien"),
    ("Neuseeland", "Wellington", "Ozeanien"),
]


GERMAN_CITIES = [
    ("Berlin", "Berlin", "Hauptstadt Deutschlands"),
    ("Hamburg", "Hamburg", "wichtige Hafenstadt"),
    ("Muenchen", "Bayern", "Landeshauptstadt Bayerns"),
    ("Koeln", "Nordrhein-Westfalen", "Stadt am Rhein"),
    ("Frankfurt am Main", "Hessen", "Finanzzentrum in Hessen"),
    ("Stuttgart", "Baden-Wuerttemberg", "Landeshauptstadt Baden-Wuerttembergs"),
    ("Duesseldorf", "Nordrhein-Westfalen", "Landeshauptstadt Nordrhein-Westfalens"),
    ("Dortmund", "Nordrhein-Westfalen", "Stadt im Ruhrgebiet"),
    ("Essen", "Nordrhein-Westfalen", "Stadt im Ruhrgebiet"),
    ("Leipzig", "Sachsen", "Stadt in Sachsen"),
    ("Dresden", "Sachsen", "Landeshauptstadt Sachsens"),
    ("Hannover", "Niedersachsen", "Landeshauptstadt Niedersachsens"),
    ("Nuernberg", "Bayern", "Stadt in Franken"),
    ("Bremen", "Bremen", "Hansestadt"),
    ("Bonn", "Nordrhein-Westfalen", "fruehere Bundeshauptstadt"),
]


SCIENCE_TERMS = [
    ("Atom", "Ein Atom ist eine kleine Einheit chemischer Elemente.", "Atomkern", "Elektronen"),
    ("Molekuel", "Ein Molekuel besteht aus zwei oder mehr verbundenen Atomen.", "Atome", "chemische Bindung"),
    ("Zelle", "Eine Zelle ist eine grundlegende Einheit lebender Organismen.", "Zellkern", "Membran"),
    ("DNA", "DNA traegt genetische Information in Lebewesen.", "Gene", "Erbinformation"),
    ("Protein", "Ein Protein ist ein biologisches Molekuel aus Aminosaeuren.", "Aminosaeure", "Enzym"),
    ("Evolution", "Evolution beschreibt die Veraenderung vererbbarer Merkmale ueber Generationen.", "Mutation", "Selektion"),
    ("Gravitation", "Gravitation ist die Anziehung zwischen Massen.", "Masse", "Schwerkraft"),
    ("Energie", "Energie ist die Faehigkeit, Arbeit zu verrichten oder Waerme abzugeben.", "Arbeit", "Waerme"),
    ("Magnetismus", "Magnetismus ist eine physikalische Wirkung bewegter elektrischer Ladungen.", "Magnetfeld", "Ladung"),
    ("Elektron", "Ein Elektron ist ein negativ geladenes Elementarteilchen.", "Ladung", "Atom"),
    ("Oekosystem", "Ein Oekosystem besteht aus Lebewesen und ihrer Umwelt.", "Lebensraum", "Wechselwirkung"),
    ("Klima", "Klima beschreibt typische Wetterbedingungen ueber lange Zeitraeume.", "Temperatur", "Niederschlag"),
]


CODE_TERMS = [
    ("Variable", "Eine Variable speichert einen Wert unter einem Namen.", "name = 42"),
    ("Schleife", "Eine Schleife wiederholt Anweisungen.", "for x in items:"),
    ("Liste", "Eine Liste speichert mehrere Werte in einer Reihenfolge.", "items = [1, 2, 3]"),
    ("Dictionary", "Ein Dictionary speichert Werte unter Schluesseln.", "data = {'name': 'Ada'}"),
    ("Bedingung", "Eine Bedingung fuehrt Code nur bei erfuellter Voraussetzung aus.", "if x > 0:"),
    ("Klasse", "Eine Klasse beschreibt Bauplan und Verhalten von Objekten.", "class User:"),
    ("Methode", "Eine Methode ist eine Funktion, die zu einem Objekt oder einer Klasse gehoert.", "user.save()"),
    ("Import", "Ein Import bindet Code aus einem Modul ein.", "import math"),
    ("Exception", "Eine Exception beschreibt einen Fehlerzustand im Programmablauf.", "try:"),
    ("JSON", "JSON ist ein textbasiertes Format fuer strukturierte Daten.", '{"ok": true}'),
]


def country_entries() -> list[KnowledgeDNAEntry]:
    out = []
    for country, capital, continent in COUNTRY_CAPITALS:
        out.append(
            KnowledgeDNAEntry(
                term=f"Hauptstadt von {country}",
                definition=f"Die Hauptstadt von {country} ist {capital}.",
                facts=[
                    f"{country} liegt in {continent}.",
                    f"{capital} ist der Regierungssitz oder eine zentrale Hauptstadt von {country}.",
                ],
                examples=[f"Frage: Was ist die Hauptstadt von {country}? Antwort: {capital}."],
                related=[country, capital, continent],
                counterfacts=[f"Die Hauptstadt von {country} ist nicht Berlin, ausser bei Deutschland."],
                source="generated_country_capitals_seed",
                probes=[
                    Probe(f"Was ist die Hauptstadt von {country}?", f"{capital}.", "fact", [capital]),
                    Probe(f"Liegt {country} in {continent}?", "Ja.", "fact", ["Ja"]),
                ],
            )
        )
    return out


def city_entries() -> list[KnowledgeDNAEntry]:
    out = []
    for city, state, role in GERMAN_CITIES:
        out.append(
            KnowledgeDNAEntry(
                term=city,
                definition=f"{city} ist eine deutsche Stadt in {state}.",
                facts=[f"{city} ist {role}.", f"{city} liegt in Deutschland."],
                examples=[f"{city} kann in deutschen Geografiefragen als Stadt erkannt werden."],
                related=[state, "Deutschland"],
                counterfacts=[f"{city} ist nicht dasselbe wie Berlin, ausser wenn der Begriff Berlin selbst ist."],
                source="generated_german_cities_seed",
                probes=[
                    Probe(f"In welchem Land liegt {city}?", "Deutschland.", "fact", ["Deutschland"]),
                    Probe(f"Ist {city} eine deutsche Stadt?", "Ja.", "fact", ["Ja"]),
                ],
            )
        )
    return out


def arithmetic_entries(limit: int) -> list[KnowledgeDNAEntry]:
    out = []
    for a in range(limit + 1):
        for b in range(limit + 1):
            s = a + b
            out.append(
                KnowledgeDNAEntry(
                    term=f"Addition {a} plus {b}",
                    definition=f"Die Addition {a} + {b} ergibt {s}.",
                    facts=[f"{a} + {b} = {s}.", "Addition zaehlt Zahlen zusammen."],
                    examples=[f"Wenn man {a} und {b} addiert, erhaelt man {s}."],
                    related=["Addition", "Summe", "Mathematik"],
                    counterfacts=[f"{a} + {b} ergibt nicht {s + 1}."],
                    source="generated_arithmetic_seed",
                    probes=[
                        Probe(f"Rechne {a} + {b}.", f"{s}.", "fact", [str(s)], [str(s + 1)]),
                        Probe(f"Ist {a} + {b} gleich {s + 1}?", f"Nein. {a} + {b} ergibt {s}.", "counterfact", ["Nein", str(s)], ["Ja."]),
                    ],
                )
            )
    for a in range(1, min(limit, 12) + 1):
        for b in range(1, min(limit, 12) + 1):
            p = a * b
            out.append(
                KnowledgeDNAEntry(
                    term=f"Multiplikation {a} mal {b}",
                    definition=f"Die Multiplikation {a} mal {b} ergibt {p}.",
                    facts=[f"{a} * {b} = {p}.", "Multiplikation kann als wiederholte Addition verstanden werden."],
                    examples=[f"{a} Gruppen mit jeweils {b} Elementen enthalten zusammen {p} Elemente."],
                    related=["Multiplikation", "Produkt", "Mathematik"],
                    counterfacts=[f"{a} * {b} ergibt nicht {p + 1}."],
                    source="generated_arithmetic_seed",
                    probes=[
                        Probe(f"Rechne {a} mal {b}.", f"{p}.", "fact", [str(p)], [str(p + 1)]),
                    ],
                )
            )
    return out


def science_entries() -> list[KnowledgeDNAEntry]:
    out = []
    for term, definition, rel_a, rel_b in SCIENCE_TERMS:
        out.append(
            KnowledgeDNAEntry(
                term=term,
                definition=definition,
                facts=[f"{term} ist ein Begriff aus Naturwissenschaft oder Technik.", f"{term} steht in Beziehung zu {rel_a} und {rel_b}."],
                examples=[f"Der Begriff {term} kann in einer kurzen Definition erklaert werden."],
                related=[rel_a, rel_b],
                counterfacts=[f"{term} ist kein beliebiger Ortsname."],
                source="generated_science_seed",
                probes=[
                    Probe(f"Was ist {term}?", definition, "definition", [term.split()[0]]),
                ],
            )
        )
    return out


def code_entries() -> list[KnowledgeDNAEntry]:
    out = []
    for term, definition, example in CODE_TERMS:
        out.append(
            KnowledgeDNAEntry(
                term=f"Python {term}",
                definition=definition,
                facts=[f"{term} ist ein Programmierbegriff.", f"Ein Beispiel ist: {example}"],
                examples=[example],
                related=["Python", "Programmierung", "Code"],
                counterfacts=[f"{term} ist keine deutsche Stadt."],
                source="generated_code_seed",
                probes=[
                    Probe(f"Was ist eine {term} in der Programmierung?", definition, "definition", [term]),
                ],
            )
        )
    return out


def build_expanded_entries(min_entries: int) -> list[KnowledgeDNAEntry]:
    entries = []
    entries.extend(sample_entries())
    entries.extend(country_entries())
    entries.extend(city_entries())
    entries.extend(science_entries())
    entries.extend(code_entries())
    limit = 10
    while True:
        arithmetic = arithmetic_entries(limit)
        if len(entries) + len(arithmetic) >= min_entries or limit >= 20:
            entries.extend(arithmetic)
            break
        limit += 2
    return entries[:min_entries]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer/helix_v2_tokenizer.model"))
    parser.add_argument("--entries", type=int, default=750)
    parser.add_argument("--seed", type=int, default=20260514)
    args = parser.parse_args()
    build_outputs(build_expanded_entries(args.entries), args.output_dir, args.tokenizer, args.seed)


if __name__ == "__main__":
    main()
