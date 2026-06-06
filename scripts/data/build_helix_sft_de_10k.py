#!/usr/bin/env python3
"""Build 10k German SFT examples for Helix response behavior.

The dataset is intentionally deterministic and template-backed. It is meant as
a first clean SFT behavior set: short answers, compact explanations, explicit
uncertainty, and corrections of common misunderstandings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "data/training/helix_sft_de_10k"
SEED = 20260605


CAPITALS = [
    ("Deutschland", "Berlin"),
    ("Österreich", "Wien"),
    ("Schweiz", "Bern"),
    ("Frankreich", "Paris"),
    ("Italien", "Rom"),
    ("Spanien", "Madrid"),
    ("Portugal", "Lissabon"),
    ("Niederlande", "Amsterdam"),
    ("Belgien", "Brüssel"),
    ("Dänemark", "Kopenhagen"),
    ("Norwegen", "Oslo"),
    ("Schweden", "Stockholm"),
    ("Finnland", "Helsinki"),
    ("Polen", "Warschau"),
    ("Tschechien", "Prag"),
    ("Ungarn", "Budapest"),
    ("Griechenland", "Athen"),
    ("Irland", "Dublin"),
    ("Island", "Reykjavik"),
    ("Kanada", "Ottawa"),
    ("Japan", "Tokio"),
    ("Südkorea", "Seoul"),
    ("China", "Peking"),
    ("Indien", "Neu-Delhi"),
    ("Australien", "Canberra"),
    ("Neuseeland", "Wellington"),
    ("Brasilien", "Brasília"),
    ("Argentinien", "Buenos Aires"),
    ("Chile", "Santiago de Chile"),
    ("Ägypten", "Kairo"),
]

FACTS = [
    ("Wie viele Minuten hat eine Stunde?", "Eine Stunde hat 60 Minuten."),
    ("Wie viele Sekunden hat eine Minute?", "Eine Minute hat 60 Sekunden."),
    ("Wie viele Monate hat ein Jahr?", "Ein Jahr hat 12 Monate."),
    ("Wie viele Tage hat eine Woche?", "Eine Woche hat 7 Tage."),
    ("Wie viele Seiten hat ein Dreieck?", "Ein Dreieck hat 3 Seiten."),
    ("Wie viele Seiten hat ein Quadrat?", "Ein Quadrat hat 4 Seiten."),
    ("Wie viele Grad hat ein rechter Winkel?", "Ein rechter Winkel hat 90 Grad."),
    ("Was ist H2O?", "H2O ist die chemische Formel für Wasser."),
    ("Welcher Planet ist der Sonne am nächsten?", "Merkur ist der Sonne am nächsten."),
    ("Wie heißt der natürliche Satellit der Erde?", "Der natürliche Satellit der Erde ist der Mond."),
    ("Welches Gas atmen Menschen hauptsächlich aus?", "Menschen atmen vor allem Stickstoff, Sauerstoff, Kohlendioxid und Wasserdampf aus; der Anteil an Kohlendioxid ist dabei höher als in der eingeatmeten Luft."),
    ("Was ist die kleinste Primzahl?", "Die kleinste Primzahl ist 2."),
    ("Ist 1 eine Primzahl?", "Nein. 1 ist keine Primzahl."),
    ("Wie viele Kontinente werden meist unterschieden?", "Meist werden 7 Kontinente unterschieden."),
    ("Was misst ein Thermometer?", "Ein Thermometer misst die Temperatur."),
    ("Was misst ein Barometer?", "Ein Barometer misst den Luftdruck."),
    ("Was ist die chemische Formel von Kohlendioxid?", "Die chemische Formel von Kohlendioxid ist CO2."),
    ("Wie heißt die größte Wüste der Erde?", "Die Antarktis ist die größte Wüste der Erde, wenn man Kältewüsten mitzählt."),
    ("Wie viele Bundesländer hat Deutschland?", "Deutschland hat 16 Bundesländer."),
    ("Wie viele Zähne hat ein erwachsener Mensch normalerweise?", "Ein erwachsener Mensch hat normalerweise 32 Zähne, einschließlich Weisheitszähnen."),
    ("Welche Einheit wird für elektrische Spannung verwendet?", "Elektrische Spannung wird in Volt angegeben."),
    ("Welche Einheit wird für elektrische Stromstärke verwendet?", "Elektrische Stromstärke wird in Ampere angegeben."),
    ("Welche Einheit wird für Leistung verwendet?", "Leistung wird in Watt angegeben."),
    ("Was ist die SI-Einheit der Masse?", "Die SI-Einheit der Masse ist das Kilogramm."),
    ("Was ist die SI-Einheit der Länge?", "Die SI-Einheit der Länge ist der Meter."),
    ("Wie viele Byte hat ein Kilobyte im Dezimalsystem?", "Ein Kilobyte hat im Dezimalsystem 1.000 Byte."),
    ("Wie viele Bit hat ein Byte?", "Ein Byte hat 8 Bit."),
    ("Was ist ein Synonym?", "Ein Synonym ist ein Wort mit gleicher oder sehr ähnlicher Bedeutung."),
    ("Was ist ein Antonym?", "Ein Antonym ist ein Wort mit gegenteiliger Bedeutung."),
    ("Was ist ein Verb?", "Ein Verb ist ein Tätigkeits- oder Zustandswort."),
]

CONCEPTS = [
    ("Was ist Photosynthese?", "Photosynthese ist ein Prozess, bei dem Pflanzen, Algen und einige Bakterien Lichtenergie nutzen, um aus Wasser und Kohlendioxid energiereiche Stoffe aufzubauen. Dabei entsteht oft Sauerstoff als Nebenprodukt."),
    ("Was ist ein Vulkan?", "Ein Vulkan ist eine Öffnung in der Erdkruste, durch die Magma, Gase und Asche an die Oberfläche gelangen können. Bei einem Ausbruch spricht man von einer vulkanischen Eruption."),
    ("Was ist Inflation?", "Inflation bedeutet, dass das allgemeine Preisniveau steigt. Dadurch verliert Geld an Kaufkraft, weil man für denselben Betrag weniger kaufen kann."),
    ("Was ist Demokratie?", "Demokratie ist eine Staatsform, in der politische Macht vom Volk ausgeht. Bürgerinnen und Bürger entscheiden direkt oder wählen Vertreter."),
    ("Was ist ein Algorithmus?", "Ein Algorithmus ist eine eindeutige Schrittfolge zur Lösung eines Problems. Er beschreibt, was in welcher Reihenfolge getan werden soll."),
    ("Was ist ein Ökosystem?", "Ein Ökosystem besteht aus Lebewesen und ihrer unbelebten Umwelt. Beide beeinflussen sich gegenseitig, zum Beispiel durch Nahrungsketten, Wasser und Nährstoffe."),
    ("Was ist ein Atom?", "Ein Atom ist ein sehr kleiner Baustein der Materie. Es besteht aus einem Atomkern und einer Hülle aus Elektronen."),
    ("Was ist Gravitation?", "Gravitation ist die Anziehung zwischen Massen. Sie sorgt zum Beispiel dafür, dass Dinge auf der Erde nach unten fallen."),
    ("Was ist ein Budget?", "Ein Budget ist ein Plan für Einnahmen und Ausgaben. Es hilft, Geld gezielt einzuteilen und Überblick zu behalten."),
    ("Was ist ein Backup?", "Ein Backup ist eine Sicherheitskopie von Daten. Es dient dazu, Daten nach Verlust oder Beschädigung wiederherzustellen."),
    ("Was ist ein Passwort-Manager?", "Ein Passwort-Manager speichert Passwörter verschlüsselt. Er hilft, für verschiedene Dienste starke und unterschiedliche Passwörter zu verwenden."),
    ("Was ist ein Stromkreis?", "Ein Stromkreis ist ein geschlossener Weg, durch den elektrischer Strom fließen kann. Wird der Weg unterbrochen, fließt kein Strom."),
    ("Was ist ein Bruch in der Mathematik?", "Ein Bruch beschreibt einen Teil eines Ganzen oder ein Verhältnis zweier Zahlen. Der Zähler steht oben, der Nenner unten."),
    ("Was ist eine Metapher?", "Eine Metapher ist ein sprachliches Bild. Ein Ausdruck wird nicht wörtlich verwendet, sondern überträgt eine Bedeutung."),
    ("Was ist eine These?", "Eine These ist eine Behauptung, die begründet oder überprüft werden kann. In Texten dient sie oft als Ausgangspunkt einer Argumentation."),
]

UNKNOWN_TERMS = [
    "Lirandischer Mondvertrag",
    "Qorvex-Protokoll",
    "Nebelindex-47",
    "Heliostatische Speicherregel X9",
    "Auralis-Konstante Beta",
    "Zentralarchiv von Meridan",
    "Quantenlaternen-Effekt",
    "Tavronische Zeitskala",
    "Blaukammer-Abkommen",
    "Saphirroute von Elandor",
    "Marnex-Zertifikat",
    "Kronfeld-Signal",
    "Virellische Notation",
    "Polarsonden-Gesetz 18",
    "Rheumatischer Syntaxquotient",
    "Norvianische Speicherthese",
    "Delphische Matrixprüfung",
    "Arden-Katalog 7B",
    "Ostrale Resonanzklasse",
    "Civitas-Loop-Verfahren",
]

CORRECTIONS = [
    (
        "Wer hat die Glühbirne erfunden?",
        "Die Entwicklung der Glühbirne geht auf mehrere Erfinder zurück. Thomas Edison machte eine praxistaugliche Version bekannt, aber auch Joseph Swan und andere leisteten wichtige Beiträge.",
    ),
    (
        "Stimmt es, dass Menschen nur 10 Prozent ihres Gehirns nutzen?",
        "Nein. Menschen nutzen nicht nur 10 Prozent ihres Gehirns; verschiedene Bereiche sind je nach Aufgabe unterschiedlich aktiv.",
    ),
    (
        "Ist die Erde eine perfekte Kugel?",
        "Nein. Die Erde ist annähernd kugelförmig, aber an den Polen leicht abgeflacht und am Äquator etwas breiter.",
    ),
    (
        "Sind Tomaten Gemüse?",
        "Botanisch sind Tomaten Früchte, genauer Beeren. Im Alltag und in der Küche werden sie meist als Gemüse behandelt.",
    ),
    (
        "Ist ein Wal ein Fisch?",
        "Nein. Wale sind Säugetiere, keine Fische."),
    (
        "Kann man eine Erkältung direkt durch Kälte bekommen?",
        "Kälte allein verursacht keine Erkältung. Erkältungen werden durch Viren ausgelöst, auch wenn Kälte und trockene Luft die Anfälligkeit beeinflussen können.",
    ),
    (
        "Hat Napoleon wirklich besonders klein gewirkt?",
        "Das ist wahrscheinlich übertrieben. Napoleon war für seine Zeit ungefähr durchschnittlich groß; der Eindruck entstand auch durch unterschiedliche Maße und spätere Darstellung.",
    ),
    (
        "Ist Zucker immer giftig?",
        "Nein. Zucker ist nicht grundsätzlich giftig, aber dauerhaft zu viel Zucker kann der Gesundheit schaden.",
    ),
    (
        "Sind alle Bakterien schädlich?",
        "Nein. Viele Bakterien sind harmlos oder nützlich, zum Beispiel in der Verdauung oder bei der Herstellung von Lebensmitteln.",
    ),
    (
        "Ist der höchste Berg vom Meeresspiegel aus der Mauna Kea?",
        "Nein. Vom Meeresspiegel aus ist der Mount Everest der höchste Berg. Mauna Kea ist höher, wenn man von seinem Fuß am Meeresboden misst.",
    ),
]


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def render(user: str, assistant: str) -> str:
    return f"<|user|>\n{clean(user)}\n\n<|assistant|>\n{clean(assistant)}\n\n<|end|>\n"


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def key(user: str, assistant: str) -> str:
    return hashlib.blake2b(f"{norm(user)}\n{norm(assistant)}".encode("utf-8"), digest_size=16).hexdigest()


def add(
    rows: list[dict[str, str]],
    seen: set[str],
    counts: Counter[str],
    user: str,
    assistant: str,
    category: str,
    family: str,
) -> bool:
    k = key(user, assistant)
    if k in seen:
        return False
    seen.add(k)
    rows.append(
        {
            "messages": [
                {"role": "user", "content": clean(user)},
                {"role": "assistant", "content": clean(assistant)},
            ],
            "text": render(user, assistant),
            "category": category,
            "family": family,
            "source": "synthetic_helix_sft_de_10k_v1",
        }
    )
    counts[category] += 1
    return True


def build_short_direct(rows: list[dict[str, str]], seen: set[str], counts: Counter[str], target: int) -> None:
    rng = random.Random(SEED + 1)
    family_counts: Counter[str] = Counter()
    quotas = {
        "addition": 1300,
        "subtraction": 1000,
        "multiplication": 850,
        "division": 550,
        "comparison": 700,
        "unit_conversion": 1200,
        "capital": 450,
        "basic_fact": 950,
    }
    variants = [
        "Was ist {a} plus {b}?",
        "Rechne {a} + {b}.",
        "Wie viel ergibt {a} + {b}?",
        "Berechne {a} plus {b}.",
    ]
    for a in range(2, 250):
        for b in range(2, 250):
            if family_counts["addition"] >= quotas["addition"]:
                break
            if (a + b) % 3 == 0:
                q = rng.choice(variants).format(a=a, b=b)
                if add(rows, seen, counts, q, f"{a} plus {b} ergibt {a + b}.", "short_direct", "addition"):
                    family_counts["addition"] += 1
        if family_counts["addition"] >= quotas["addition"]:
            break

    for a in range(30, 500):
        for b in range(2, 29):
            if family_counts["subtraction"] >= quotas["subtraction"]:
                break
            q = rng.choice(["Was ist {a} minus {b}?", "Rechne {a} - {b}.", "Wie viel ergibt {a} - {b}?"]).format(a=a, b=b)
            if add(rows, seen, counts, q, f"{a} minus {b} ergibt {a - b}.", "short_direct", "subtraction"):
                family_counts["subtraction"] += 1
        if family_counts["subtraction"] >= quotas["subtraction"]:
            break

    for a in range(2, 31):
        for b in range(2, 31):
            if family_counts["multiplication"] >= quotas["multiplication"]:
                break
            q = rng.choice(["Was ist {a} mal {b}?", "Rechne {a} * {b}.", "Wie viel ergibt {a} mal {b}?"]).format(a=a, b=b)
            if add(rows, seen, counts, q, f"{a} mal {b} ergibt {a * b}.", "short_direct", "multiplication"):
                family_counts["multiplication"] += 1
        if family_counts["multiplication"] >= quotas["multiplication"]:
            break

    for result in range(2, 140):
        for divisor in range(2, 20):
            if family_counts["division"] >= quotas["division"]:
                break
            dividend = result * divisor
            q = rng.choice(["Was ist {a} geteilt durch {b}?", "Rechne {a} / {b}.", "Wie viel ergibt {a} durch {b}?"]).format(a=dividend, b=divisor)
            if add(rows, seen, counts, q, f"{dividend} geteilt durch {divisor} ergibt {result}.", "short_direct", "division"):
                family_counts["division"] += 1
        if family_counts["division"] >= quotas["division"]:
            break

    for a in range(3, 400):
        for b in range(2, 399):
            if family_counts["comparison"] >= quotas["comparison"]:
                break
            if a == b or (a + b) % 11 != 0:
                continue
            bigger = max(a, b)
            q = rng.choice(["Welche Zahl ist größer: {a} oder {b}?", "Was ist größer, {a} oder {b}?"]).format(a=a, b=b)
            if add(rows, seen, counts, q, f"{bigger} ist größer.", "short_direct", "comparison"):
                family_counts["comparison"] += 1
        if family_counts["comparison"] >= quotas["comparison"]:
            break

    units = [
        ("Kilometer", "Meter", 1000, "km", "m"),
        ("Meter", "Zentimeter", 100, "m", "cm"),
        ("Kilogramm", "Gramm", 1000, "kg", "g"),
        ("Stunden", "Minuten", 60, "Stunden", "Minuten"),
        ("Tage", "Stunden", 24, "Tage", "Stunden"),
        ("Wochen", "Tage", 7, "Wochen", "Tage"),
    ]
    for src, dst, factor, src_short, dst_short in units:
        for n in range(2, 500):
            if family_counts["unit_conversion"] >= quotas["unit_conversion"]:
                break
            q = rng.choice([f"Wie viele {dst} sind {n} {src}?", f"Wandle {n} {src_short} in {dst_short} um."])
            if add(rows, seen, counts, q, f"{n} {src} sind {n * factor} {dst}.", "short_direct", "unit_conversion"):
                family_counts["unit_conversion"] += 1
        if family_counts["unit_conversion"] >= quotas["unit_conversion"]:
            break

    for country, capital in CAPITALS:
        for q in [
            f"Was ist die Hauptstadt von {country}?",
            f"Wie heißt die Hauptstadt von {country}?",
            f"Nenne die Hauptstadt von {country}.",
            f"Welche Stadt ist die Hauptstadt von {country}?",
            f"Wie lautet die Hauptstadt von {country}?",
            f"Hauptstadt von {country}?",
            f"Welche Hauptstadt hat {country}?",
            f"Sag mir die Hauptstadt von {country}.",
            f"Gib die Hauptstadt von {country} an.",
            f"Wie nennt man die Hauptstadt von {country}?",
            f"Welche Stadt ist politisches Zentrum von {country}?",
            f"Was ist {country}s Hauptstadt?",
            f"Direkt gefragt: Hauptstadt von {country}?",
            f"Kurze Antwort: Hauptstadt von {country}?",
            f"Welche Hauptstadt gehört zu {country}?",
        ]:
            if family_counts["capital"] >= quotas["capital"]:
                break
            if add(rows, seen, counts, q, f"Die Hauptstadt von {country} ist {capital}.", "short_direct", "capital"):
                family_counts["capital"] += 1
        if family_counts["capital"] >= quotas["capital"]:
            break

    for q, a in FACTS:
        for suffix in [
            "",
            " Antworte kurz.",
            " Bitte direkt antworten.",
            " Nur die Antwort.",
            " Ohne Erklärung.",
            " Kurz und sachlich.",
            " In einem Satz.",
            " Was ist die direkte Antwort?",
            " Sag es knapp.",
            " Bitte ohne Abschweifung.",
            " Antworte in natürlichem Deutsch.",
            " Gib nur die relevante Information.",
            " Keine Zusatzinfos.",
            " Direkt beantworten.",
            " Kurzfassung bitte.",
            " Präzise Antwort.",
            " Einfache Antwort.",
            " Ohne Einleitung.",
            " Bitte sachlich.",
            " Knapp formuliert.",
            " Bitte nicht ausführen.",
            " Nur das Wesentliche.",
            " In kurzer Form.",
            " Antwort in einem kurzen Satz.",
            " Faktenantwort.",
            " Kein Kontext nötig.",
            " Was stimmt?",
            " Sag die Lösung.",
            " Bitte kompakt.",
            " Maximal zwei Sätze.",
            " Antwort ohne Beispiele.",
            " Direkte Faktenantwort.",
        ]:
            if family_counts["basic_fact"] >= quotas["basic_fact"]:
                break
            if add(rows, seen, counts, q + suffix, a, "short_direct", "basic_fact"):
                family_counts["basic_fact"] += 1
        if family_counts["basic_fact"] >= quotas["basic_fact"]:
            break

    for percent in range(5, 55, 5):
        for base in range(20, 1000, 20):
            if counts["short_direct"] >= target:
                break
            value = base * percent // 100
            if base * percent % 100 != 0:
                continue
            q = rng.choice([
                "Was sind {p} Prozent von {b}?",
                "Berechne {p}% von {b}.",
                "Wie viel sind {p}% von {b}?",
            ]).format(p=percent, b=base)
            if add(rows, seen, counts, q, f"{percent} Prozent von {base} sind {value}.", "short_direct", "percentage"):
                family_counts["percentage"] += 1
        if counts["short_direct"] >= target:
            break

    if counts["short_direct"] != target:
        raise ValueError(f"short_direct target not reached: {counts['short_direct']} != {target}")


def build_explanations(rows: list[dict[str, str]], seen: set[str], counts: Counter[str], target: int) -> None:
    starters = [
        'Erkläre den Begriff "{concept}".',
        'Was bedeutet der Begriff "{concept}"?',
        'Was ist mit "{concept}" gemeint?',
        'Gib eine kurze Erklärung zu "{concept}".',
        'Erkläre "{concept}" in einfachen Worten.',
        'Fasse "{concept}" knapp zusammen.',
        'Wie lässt sich "{concept}" verständlich erklären?',
        'Was sollte man über "{concept}" wissen?',
        'Beschreibe "{concept}" sachlich.',
        "Erkläre kurz, was {concept} ist.",
    ]
    constraints = [
        "",
        " Bitte ohne lange Einleitung.",
        " Antworte in zwei bis vier Sätzen.",
        " Bleib sachlich.",
        " Nutze natürliches Deutsch.",
        " Bitte nicht abschweifen.",
        " Erkläre es kompakt.",
        " Gib nur die wichtigsten Punkte.",
        " Ohne Rollenspiel.",
        " So, dass es gut verständlich ist.",
    ]
    while counts["explanation"] < target:
        made_progress = False
        for q, a in CONCEPTS:
            concept = q.removeprefix("Was ist ").rstrip("?")
            for starter in starters:
                for constraint in constraints:
                    if counts["explanation"] >= target:
                        return
                    question = starter.format(concept=concept) + constraint
                    made_progress |= add(rows, seen, counts, question, a, "explanation", "concept")
        if not made_progress:
            raise ValueError("could not create enough explanation rows")


def build_uncertainty(rows: list[dict[str, str]], seen: set[str], counts: Counter[str], target: int) -> None:
    rng = random.Random(SEED + 3)
    templates = [
        "Was bedeutet {term} genau?",
        "Kannst du {term} zuverlässig erklären?",
        "Ist {term} ein bekannter Fachbegriff?",
        "Wofür steht {term}?",
        "Gibt es verlässliche Informationen zu {term}?",
        "Was ist über {term} bekannt?",
        "Ordne {term} bitte sachlich ein.",
        "Ist {term} historisch belegt?",
        "Kannst du kurz erklären, was {term} ist?",
        "Wie sicher ist die Bedeutung von {term}?",
    ]
    answers = [
        'Mir ist "{term}" nicht als gesicherter Begriff bekannt. Ohne weitere Informationen kann ich die Frage nicht zuverlässig beantworten.',
        'Zu "{term}" habe ich keine verlässliche Information. Ich würde hier nicht spekulieren.',
        'Ich kann "{term}" nicht eindeutig zuordnen. Bitte gib mehr Kontext, wenn du einen bestimmten Begriff meinst.',
    ]
    qualifiers = [
        "A-17",
        "B-42",
        "C-09",
        "Nord",
        "Süd",
        "Ost",
        "West",
        "Alpha",
        "Beta",
        "Gamma",
        "Delta",
        "im Archivkontext",
        "aus der Miranor-Liste",
        "nach der Kantos-Tabelle",
        "des Jahres 1842",
        "des Jahres 1907",
        "des Jahres 1973",
        "der dritten Fassung",
        "im technischen Sinn",
        "in der historischen Lesart",
        "für Speicherklassen",
        "für Navigationsräume",
        "im Meridan-Modell",
        "aus Protokollgruppe N",
        "aus Protokollgruppe R",
        "mit Index 12",
        "mit Index 28",
        "mit Index 64",
        "nach Anlage K",
        "nach Anlage T",
        "in Stufe 3",
        "in Stufe 5",
        "im Kurzregister",
        "im Langregister",
        "für Prüfpfad Rot",
        "für Prüfpfad Blau",
        "im Standardentwurf",
        "in der Sonderfassung",
        "nach Abschnitt 4",
        "nach Abschnitt 9",
        "im alten Katalog",
        "im neuen Katalog",
        "mit Zusatzcode Q",
        "mit Zusatzcode Z",
        "im Quellenband",
        "im Feldbericht",
        "nach Lesart A",
        "nach Lesart B",
        "im Referenzblatt",
        "im Datenraum",
    ]
    while counts["uncertainty"] < target:
        made_progress = False
        for base in UNKNOWN_TERMS:
            for qualifier in qualifiers:
                if counts["uncertainty"] >= target:
                    return
                term = f"{base} {qualifier}"
                q = rng.choice(templates).format(term=term)
                a = rng.choice(answers).format(term=term)
                made_progress |= add(rows, seen, counts, q, a, "uncertainty", "unknown_term")
        if not made_progress:
            raise ValueError("could not create enough uncertainty rows")


def build_corrections(rows: list[dict[str, str]], seen: set[str], counts: Counter[str], target: int) -> None:
    prefixes = [
        "",
        "Korrigiere knapp: ",
        "Stimmt das? ",
        "Bitte sachlich korrigieren: ",
        "Was ist daran falsch? ",
        "Antworte kurz: ",
        "Korrigiere die Annahme: ",
        "Gib die korrekte Einordnung: ",
        "Prüfe diese Aussage: ",
        "Direkt antworten: ",
    ]
    suffixes = [
        "",
        " Bitte ohne Abschweifung.",
        " Antworte in wenigen Sätzen.",
        " Bleib sachlich.",
        " Gib nur die Korrektur.",
    ]
    while counts["correction"] < target:
        made_progress = False
        for q, a in CORRECTIONS:
            for prefix in prefixes:
                for suffix in suffixes:
                    if counts["correction"] >= target:
                        return
                    made_progress |= add(rows, seen, counts, prefix + q + suffix, a, "correction", "misconception")
        if not made_progress:
            raise ValueError("could not create enough correction rows")


def validate(rows: list[dict[str, str]], expected: int) -> dict[str, object]:
    if len(rows) != expected:
        raise ValueError(f"expected {expected} rows, got {len(rows)}")
    text_keys = [key(r["messages"][0]["content"], r["messages"][1]["content"]) for r in rows]
    if len(text_keys) != len(set(text_keys)):
        raise ValueError("duplicate user/assistant pairs found")
    for idx, row in enumerate(rows, 1):
        text = row["text"]
        if text.count("<|user|>") != 1 or text.count("<|assistant|>") != 1 or text.count("<|end|>") != 1:
            raise ValueError(f"bad marker count in row {idx}")
        if not row["messages"][0]["content"] or not row["messages"][1]["content"]:
            raise ValueError(f"empty message in row {idx}")
    return {
        "rows": len(rows),
        "category_counts": dict(Counter(r["category"] for r in rows)),
        "family_counts": dict(Counter(r["family"] for r in rows)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    targets = {
        "short_direct": 7000,
        "explanation": 1500,
        "uncertainty": 1000,
        "correction": 500,
    }

    build_short_direct(rows, seen, counts, targets["short_direct"])
    build_explanations(rows, seen, counts, targets["explanation"])
    build_uncertainty(rows, seen, counts, targets["uncertainty"])
    build_corrections(rows, seen, counts, targets["correction"])

    random.Random(args.seed).shuffle(rows)
    manifest = validate(rows, 10_000)

    jsonl_path = out_dir / "helix_sft_de_10k_v1.jsonl"
    txt_path = out_dir / "helix_sft_de_10k_v1.txt"
    manifest_path = out_dir / "helix_sft_de_10k_v1.manifest.json"

    with jsonl_path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with txt_path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(row["text"])
            fh.write("\n")

    manifest.update(
        {
            "dataset": "helix_sft_de_10k_v1",
            "seed": args.seed,
            "format": "<|user|> ... <|assistant|> ... <|end|>",
            "jsonl": str(jsonl_path),
            "txt": str(txt_path),
            "notes": [
                "Deterministic synthetic German SFT behavior dataset.",
                "Weighted toward concise direct answers to reduce over-answering.",
                "Uncertainty rows use invented or ambiguous terms and explicitly avoid speculation.",
            ],
        }
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
