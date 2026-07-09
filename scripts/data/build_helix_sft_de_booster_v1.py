#!/usr/bin/env python3
"""Build a small curated German SFT booster from the 10k template set.

This is intentionally not a primary SFT dataset. It keeps the useful parts of
the deterministic set while avoiding answer-side repetition and arithmetic
dominance. Additional stable real-term examples are added by hand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "data/training/helix_sft_de_10k/helix_sft_de_10k_v1.jsonl"
DEFAULT_OUT = REPO / "data/training/helix_sft_de_booster_v1"
SEED = 20260605


REAL_TERMS = [
    (
        "Was ist ein Palimpsest?",
        "Ein Palimpsest ist ein wiederverwendetes Schriftmaterial, bei dem ältere Schrift entfernt oder überschrieben wurde. Oft sind Spuren des ursprünglichen Textes noch erkennbar.",
    ),
    (
        "Was bedeutet Heliotropismus?",
        "Heliotropismus ist die Ausrichtung von Pflanzenteilen zum Licht, besonders zur Sonne. Das Verhalten hilft Pflanzen, Licht besser zu nutzen.",
    ),
    (
        "Was ist die Endosymbiontentheorie?",
        "Die Endosymbiontentheorie erklärt, dass Mitochondrien und Chloroplasten aus ehemals eigenständigen Bakterien hervorgegangen sind, die in frühen Zellen lebten.",
    ),
    (
        "Was ist Homöostase?",
        "Homöostase ist die Fähigkeit eines Organismus, innere Bedingungen wie Temperatur oder pH-Wert in einem stabilen Bereich zu halten.",
    ),
    (
        "Was ist ein Antonym?",
        "Ein Antonym ist ein Wort mit gegenteiliger Bedeutung, zum Beispiel 'hell' und 'dunkel'.",
    ),
    (
        "Was ist ein Oxymoron?",
        "Ein Oxymoron verbindet zwei scheinbar widersprüchliche Begriffe, etwa 'beredtes Schweigen'.",
    ),
    (
        "Was ist Isostasie?",
        "Isostasie beschreibt das Gleichgewicht zwischen Teilen der Erdkruste und dem darunterliegenden Erdmantel. Leichtere Krustenbereiche können dabei höher aufragen.",
    ),
    (
        "Was ist der Antikythera-Mechanismus?",
        "Der Antikythera-Mechanismus ist ein antikes griechisches Gerät zur Berechnung astronomischer Zyklen. Er gilt als frühes Beispiel komplexer mechanischer Rechentechnik.",
    ),
    (
        "Was ist Occams Rasiermesser?",
        "Occams Rasiermesser ist ein heuristisches Prinzip: Wenn mehrere Erklärungen passen, bevorzugt man oft die einfachere mit weniger Zusatzannahmen.",
    ),
    (
        "Was ist ein Peer-Review?",
        "Peer-Review ist ein Begutachtungsverfahren, bei dem Fachleute eine wissenschaftliche Arbeit prüfen, bevor sie veröffentlicht wird.",
    ),
    (
        "Was ist ein Isotop?",
        "Ein Isotop ist eine Variante eines chemischen Elements mit gleicher Protonenzahl, aber unterschiedlicher Neutronenzahl.",
    ),
    (
        "Was ist ein Exoplanet?",
        "Ein Exoplanet ist ein Planet außerhalb unseres Sonnensystems, der einen anderen Stern oder ein anderes Objekt umkreist.",
    ),
    (
        "Was ist ein Neologismus?",
        "Ein Neologismus ist ein neu gebildetes Wort oder eine neue Bedeutung eines bestehenden Wortes.",
    ),
    (
        "Was ist ein Pleonasmus?",
        "Ein Pleonasmus ist eine überflüssige Wiederholung ähnlicher Bedeutungen, zum Beispiel 'nasser Regen'.",
    ),
    (
        "Was ist ein Katalysator?",
        "Ein Katalysator beschleunigt eine chemische Reaktion, ohne dabei dauerhaft verbraucht zu werden.",
    ),
    (
        "Was ist Osmose?",
        "Osmose ist die Bewegung von Wasser durch eine halbdurchlässige Membran hin zu einer höheren Konzentration gelöster Stoffe.",
    ),
    (
        "Was ist ein Habitat?",
        "Ein Habitat ist der Lebensraum einer Art, also die Umgebung, in der sie natürlicherweise lebt.",
    ),
    (
        "Was ist eine Fabel?",
        "Eine Fabel ist eine kurze Erzählung, oft mit Tieren als Figuren, die eine moralische Lehre vermittelt.",
    ),
    (
        "Was ist ein Sonett?",
        "Ein Sonett ist eine Gedichtform mit 14 Versen und festen formalen Traditionen.",
    ),
    (
        "Was ist ein Axiom?",
        "Ein Axiom ist eine grundlegende Annahme, die innerhalb eines Systems nicht bewiesen, sondern vorausgesetzt wird.",
    ),
    (
        "Was ist ein Paradoxon?",
        "Ein Paradoxon ist eine Aussage oder Situation, die widersprüchlich wirkt, aber auf ein tieferes Problem oder eine überraschende Wahrheit hinweisen kann.",
    ),
    (
        "Was ist ein Morphem?",
        "Ein Morphem ist die kleinste bedeutungstragende Einheit einer Sprache.",
    ),
    (
        "Was ist ein Phonem?",
        "Ein Phonem ist die kleinste lautliche Einheit, die in einer Sprache Bedeutungen unterscheiden kann.",
    ),
    (
        "Was ist ein Meridian?",
        "Ein Meridian ist eine gedachte Linie auf der Erde, die vom Nordpol zum Südpol verläuft.",
    ),
    (
        "Was ist ein Äquinoktium?",
        "Ein Äquinoktium ist ein Zeitpunkt, an dem Tag und Nacht annähernd gleich lang sind.",
    ),
    (
        "Was ist ein Solstitium?",
        "Ein Solstitium ist eine Sonnenwende. Dabei erreicht die Sonne ihren höchsten oder niedrigsten Stand im Jahreslauf.",
    ),
    (
        "Was ist ein Fossil?",
        "Ein Fossil ist ein erhaltener Rest oder Abdruck eines Lebewesens aus früheren Erdzeitaltern.",
    ),
    ("Was ist ein Archipel?", "Ein Archipel ist eine Inselgruppe."),
    (
        "Was ist ein Delta in der Geografie?",
        "Ein Delta ist eine Flussmündung, bei der sich Ablagerungen verzweigen und Landflächen bilden können.",
    ),
    (
        "Was ist ein Manuskript?",
        "Ein Manuskript ist eine handschriftliche oder noch nicht endgültig veröffentlichte Textfassung.",
    ),
    (
        "Was ist ein Korpus in der Sprachwissenschaft?",
        "Ein Korpus ist eine Sammlung von Texten oder Sprachdaten, die systematisch untersucht wird.",
    ),
    (
        "Was ist ein Register in der Sprache?",
        "Ein Register ist eine sprachliche Stilebene, die von Situation, Publikum oder Zweck abhängt.",
    ),
    (
        "Was ist ein Vektor in der Mathematik?",
        "Ein Vektor ist eine Größe mit Richtung und Betrag. Er wird oft durch Pfeile oder Zahlenlisten dargestellt.",
    ),
    (
        "Was ist ein Skalar?",
        "Ein Skalar ist eine Größe, die nur durch einen Zahlenwert beschrieben wird, zum Beispiel Temperatur oder Masse.",
    ),
    ("Was ist ein Byte?", "Ein Byte ist eine Dateneinheit, die aus 8 Bit besteht."),
    (
        "Was ist ein Hashwert?",
        "Ein Hashwert ist eine kurze Prüfsumme, die aus Daten berechnet wird. Er wird zum Beispiel für Vergleiche und Integritätsprüfungen genutzt.",
    ),
    (
        "Was ist ein Cache?",
        "Ein Cache ist ein Zwischenspeicher, der häufig benötigte Daten schneller verfügbar macht.",
    ),
    (
        "Was ist eine API?",
        "Eine API ist eine Programmierschnittstelle, über die Softwarekomponenten miteinander kommunizieren können.",
    ),
    (
        "Was ist ein Token in der Sprachmodellierung?",
        "Ein Token ist eine Verarbeitungseinheit eines Sprachmodells. Es kann ein Wort, Wortteil, Zeichen oder Satzzeichen sein.",
    ),
    (
        "Was ist Overfitting?",
        "Overfitting bedeutet, dass ein Modell Trainingsdaten zu stark auswendig lernt und dadurch auf neuen Daten schlechter funktioniert.",
    ),
]


AMBIGUOUS_REAL = [
    (
        "Was ist Python?",
        "Python kann eine Programmiersprache oder eine Schlange bezeichnen. Ohne Kontext ist nicht eindeutig, welche Bedeutung gemeint ist.",
    ),
    (
        "Was bedeutet Java?",
        "Java kann eine Programmiersprache, eine Insel oder Kaffee meinen. Ohne Kontext sollte man die Bedeutung nicht festlegen.",
    ),
    (
        "Was ist Rust?",
        "Rust kann die Programmiersprache oder auf Englisch Rost bedeuten. Der Kontext entscheidet.",
    ),
    (
        "Was ist Go?",
        "Go kann ein Brettspiel oder eine Programmiersprache sein. Ohne Kontext ist die Frage mehrdeutig.",
    ),
    (
        "Was bedeutet Mercury?",
        "Mercury kann Merkur, Quecksilber oder einen Namen bezeichnen. Ohne Kontext ist keine eindeutige Antwort möglich.",
    ),
    (
        "Was ist Corona?",
        "Corona kann eine Viruskrankheit, eine Sonnenkorona oder einen Markennamen meinen. Ohne Kontext ist die Frage mehrdeutig.",
    ),
    (
        "Was ist ein Terminal?",
        "Terminal kann eine Kommandozeile, einen Flughafenbereich oder ein Endgerät meinen. Ohne Kontext ist die Bedeutung offen.",
    ),
    (
        "Was ist ein Kernel?",
        "Kernel kann den Betriebssystemkern, einen mathematischen Kern oder einen Saatkern meinen. Der Kontext ist entscheidend.",
    ),
    (
        "Was bedeutet Matrix?",
        "Matrix kann eine mathematische Tabelle, ein Film, eine Struktur oder eine Umgebung meinen. Ohne Kontext ist die Bedeutung nicht eindeutig.",
    ),
    (
        "Was ist ein Modell?",
        "Modell kann ein vereinfachtes Abbild, ein KI-System, eine Person in der Mode oder ein Produktmodell bedeuten. Es braucht Kontext.",
    ),
]


UNKNOWN_SMALL = [
    (
        "Was bedeutet der Begriff Lirandischer Mondvertrag?",
        'Mir ist "Lirandischer Mondvertrag" nicht als gesicherter Begriff bekannt. Ohne weitere Informationen kann ich die Frage nicht zuverlässig beantworten.',
    ),
    (
        "Kannst du Qorvex-Protokoll zuverlässig erklären?",
        'Zu "Qorvex-Protokoll" habe ich keine verlässliche Information. Ich würde hier nicht spekulieren.',
    ),
    (
        "Was ist die Auralis-Konstante Beta?",
        'Mir ist "Auralis-Konstante Beta" nicht als etablierter Fachbegriff bekannt. Ohne Kontext kann ich das nicht zuverlässig beantworten.',
    ),
    (
        "Wofür steht Nebelindex-47?",
        'Ich kann "Nebelindex-47" nicht eindeutig zuordnen. Bitte gib mehr Kontext, wenn du einen bestimmten Begriff meinst.',
    ),
    (
        "Ist das Blaukammer-Abkommen historisch belegt?",
        'Mir ist kein historisch gesichertes Abkommen mit dem Namen "Blaukammer-Abkommen" bekannt. Ich würde daraus keine Fakten ableiten.',
    ),
]

PRACTICAL = [
    (
        "Wie bewahre ich Brot am besten auf?",
        "Brot bleibt meist länger frisch, wenn es trocken und luftdurchlässig gelagert wird, zum Beispiel in einem Brotkasten. Im Kühlschrank wird es oft schneller altbacken.",
    ),
    (
        "Was hilft, wenn ein Reißverschluss klemmt?",
        "Ziehe nicht mit Gewalt. Prüfe zuerst, ob Stoff eingeklemmt ist, und bewege den Schieber vorsichtig zurück.",
    ),
    (
        "Wie entferne ich Kalk aus einem Wasserkocher?",
        "Fülle Wasser mit etwas Essig oder Zitronensäure ein, lasse es einwirken und spüle danach gründlich mit klarem Wasser nach.",
    ),
    (
        "Wie lange sollte man Hände waschen?",
        "Gründliches Händewaschen dauert etwa 20 bis 30 Sekunden.",
    ),
    (
        "Was sollte ich vor einem Backup prüfen?",
        "Prüfe, ob die wichtigen Dateien enthalten sind und ob sich das Backup wiederherstellen lässt.",
    ),
    (
        "Wie erkenne ich eine seriöse E-Mail eher?",
        "Achte auf Absender, Sprache, Links, unerwartete Anhänge und ungewöhnlichen Druck. Bei Unsicherheit öffne keine Links und frage direkt beim Anbieter nach.",
    ),
    (
        "Was mache ich, wenn mein WLAN langsam ist?",
        "Starte Router und Gerät neu, prüfe die Signalstärke und teste, ob andere Geräte das Problem auch haben.",
    ),
    (
        "Wie kann ich ein starkes Passwort erstellen?",
        "Nutze ein langes, einzigartiges Passwort oder eine Passphrase. Verwende es nicht mehrfach und speichere es am besten in einem Passwort-Manager.",
    ),
    (
        "Was gehört in eine kurze Einkaufsliste?",
        "Schreibe nur die benötigten Artikel auf und gruppiere sie nach Bereichen wie Obst, Kühlung oder Vorräte.",
    ),
    (
        "Wie plane ich einen Arbeitstag sinnvoll?",
        "Notiere die wichtigsten Aufgaben, priorisiere wenige davon und plane Puffer für Unterbrechungen ein.",
    ),
    (
        "Was tun, wenn ein Termin unklar ist?",
        "Frage nach Datum, Uhrzeit, Ort und Zweck des Termins. So vermeidest du Missverständnisse.",
    ),
    (
        "Wie formuliere ich eine höfliche Absage?",
        "Sag klar ab, bedanke dich für die Einladung und gib bei Bedarf einen kurzen Grund an.",
    ),
    (
        "Was ist beim Online-Kauf wichtig?",
        "Prüfe Anbieter, Preis, Versandkosten, Rückgaberecht und Zahlungsmethode, bevor du bestellst.",
    ),
    (
        "Wie kann ich mich besser konzentrieren?",
        "Reduziere Ablenkungen, arbeite in überschaubaren Zeitblöcken und mache kurze Pausen.",
    ),
    (
        "Was mache ich, wenn ein Dokument nicht gespeichert wurde?",
        "Prüfe automatische Wiederherstellung, temporäre Dateien und den zuletzt verwendeten Speicherort.",
    ),
    (
        "Wie vergleiche ich zwei Angebote?",
        "Vergleiche nicht nur den Preis, sondern auch Leistung, Laufzeit, Zusatzkosten und Bedingungen.",
    ),
    (
        "Wie schreibe ich eine kurze Entschuldigung?",
        "Benenne den Fehler, übernimm Verantwortung und sage knapp, wie du es künftig vermeiden willst.",
    ),
    (
        "Was sollte auf eine To-do-Liste?",
        "Schreibe konkrete Aufgaben auf, nicht nur große Ziele. Jede Aufgabe sollte klar erledigbar sein.",
    ),
    (
        "Wie kann ich Dateien sinnvoll benennen?",
        "Nutze klare Namen mit Datum, Thema und Version, zum Beispiel '2026-06-05_protokoll_v1'.",
    ),
    (
        "Wie prüfe ich eine Rechnung?",
        "Kontrolliere Absender, Betrag, Leistungsbeschreibung, Datum, Zahlungsziel und Kontodaten.",
    ),
    (
        "Was hilft gegen Chaos auf dem Schreibtisch?",
        "Sortiere zuerst Müll, lose Notizen und aktive Aufgaben. Lege danach feste Plätze für häufig genutzte Dinge fest.",
    ),
    (
        "Wie bereite ich ein kurzes Gespräch vor?",
        "Notiere Ziel, wichtigste Punkte und offene Fragen. Das reicht oft für ein strukturiertes Gespräch.",
    ),
    (
        "Was tun, wenn ich eine Aufgabe nicht verstehe?",
        "Frage nach Ziel, gewünschtem Ergebnis, Frist und Beispielen. Das klärt meistens die entscheidenden Punkte.",
    ),
    (
        "Wie kann ich eine Entscheidung dokumentieren?",
        "Halte fest, was entschieden wurde, warum, von wem und bis wann die nächsten Schritte erfolgen.",
    ),
    (
        "Wie erkenne ich eine schlechte Quelle?",
        "Warnzeichen sind fehlende Autorenangaben, extreme Behauptungen, keine Belege und viele reißerische Formulierungen.",
    ),
    (
        "Wie sichere ich ein Smartphone vor Verlust ab?",
        "Aktiviere Bildschirmsperre, Gerätesuche und Backups. Speichere wichtige Zugangsdaten nicht ungeschützt auf dem Gerät.",
    ),
    (
        "Was sollte ich vor einer Reise prüfen?",
        "Prüfe Ausweis, Tickets, Unterkunft, Zahlungsmittel, Wetter und wichtige Fristen.",
    ),
    (
        "Wie kann ich eine Datei kleiner machen?",
        "Komprimiere Bilder, entferne unnötige Inhalte oder speichere sie in einem geeigneten Format.",
    ),
    (
        "Was mache ich bei einem vergessenen Passwort?",
        "Nutze die offizielle Passwort-zurücksetzen-Funktion des Dienstes. Verwende keine Links aus verdächtigen E-Mails.",
    ),
    (
        "Wie schreibe ich eine klare Betreffzeile?",
        "Nenne Thema und Anlass kurz, zum Beispiel 'Rückfrage zum Angebot vom 5. Juni'.",
    ),
    (
        "Was ist bei Medikamenten wichtig?",
        "Nimm Medikamente nur wie angegeben ein und frage bei Unsicherheit eine Ärztin, einen Arzt oder eine Apotheke.",
    ),
    (
        "Wie kann ich Strom im Alltag sparen?",
        "Schalte ungenutzte Geräte aus, nutze effiziente Beleuchtung und vermeide unnötigen Stand-by-Betrieb.",
    ),
    (
        "Wie räume ich einen vollen Posteingang auf?",
        "Lösche Offensichtliches, archiviere Erledigtes und markiere nur echte Aufgaben als offen.",
    ),
    (
        "Wie formuliere ich Feedback sachlich?",
        "Beschreibe konkrete Beobachtungen, die Wirkung und einen möglichen Wunsch. Vermeide pauschale Vorwürfe.",
    ),
    (
        "Was tun, wenn ein Paket nicht ankommt?",
        "Prüfe Sendungsverfolgung, Ablageort und Benachrichtigung. Danach kontaktiere Händler oder Versanddienst.",
    ),
    (
        "Wie kann ich Notizen übersichtlicher machen?",
        "Nutze kurze Überschriften, Stichpunkte und klare Datumsangaben. Trenne Aufgaben von Hintergrundinformationen.",
    ),
    (
        "Was ist bei einem Mietvertrag wichtig?",
        "Prüfe Miete, Nebenkosten, Kündigungsfristen, Kaution und besondere Vereinbarungen. Bei Unsicherheit lohnt rechtliche Beratung.",
    ),
    (
        "Wie verhindere ich doppelte Dateien?",
        "Nutze klare Ordner, einheitliche Dateinamen und suche vor dem Speichern nach bestehenden Versionen.",
    ),
    (
        "Wie kann ich eine Präsentation kürzen?",
        "Streiche Nebendetails, fasse ähnliche Punkte zusammen und behalte nur Aussagen, die das Ziel unterstützen.",
    ),
    (
        "Wie gehe ich mit einer falschen Überweisung um?",
        "Kontaktiere schnell deine Bank und den Empfänger, falls bekannt. Je früher du reagierst, desto besser sind die Chancen auf Klärung.",
    ),
]

LOGIC = [
    (
        "Alle Quadrate sind Rechtecke. Ist jedes Rechteck ein Quadrat?",
        "Nein. Jedes Quadrat ist ein Rechteck, aber nicht jedes Rechteck ist ein Quadrat.",
    ),
    (
        "Wenn alle Rosen Blumen sind und diese Pflanze eine Rose ist, ist sie dann eine Blume?",
        "Ja. Wenn die Aussagen stimmen, ist die Pflanze eine Blume.",
    ),
    ("Anna ist älter als Ben. Ben ist älter als Cem. Wer ist am jüngsten?", "Cem ist am jüngsten."),
    (
        "Ein Zug fährt vor dem Bus ab. Der Bus fährt vor dem Taxi ab. Was fährt zuerst ab?",
        "Der Zug fährt zuerst ab.",
    ),
    (
        "Wenn es regnet, ist die Straße nass. Die Straße ist nass. Muss es geregnet haben?",
        "Nein. Die Straße kann auch aus anderen Gründen nass sein.",
    ),
    (
        "Wenn ein Passwort falsch ist, schlägt die Anmeldung fehl. Die Anmeldung ist fehlgeschlagen. War das Passwort sicher falsch?",
        "Nicht zwingend. Es kann auch andere Ursachen geben, zum Beispiel ein gesperrtes Konto.",
    ),
    (
        "Lisa hat mehr Punkte als Tom. Tom hat mehr Punkte als Mira. Wer hat die meisten Punkte?",
        "Lisa hat die meisten Punkte.",
    ),
    ("Ein Rechteck hat vier Ecken. Hat ein Quadrat vier Ecken?", "Ja. Ein Quadrat hat vier Ecken."),
    ("Wenn heute Montag ist, welcher Tag ist übermorgen?", "Übermorgen ist Mittwoch."),
    (
        "Wenn ein Glas halb voll ist und nichts verschüttet wird, ist es auch halb leer?",
        "Ja. Beide Aussagen beschreiben denselben Füllstand aus unterschiedlicher Perspektive.",
    ),
    ("Max hat 3 Äpfel und gibt 1 ab. Wie viele bleiben übrig?", "Es bleiben 2 Äpfel übrig."),
    (
        "Eine Aussage kann nicht gleichzeitig vollständig wahr und vollständig falsch sein. Wie nennt man so etwas allgemein?",
        "Das beschreibt den Grundgedanken des Satzes vom Widerspruch.",
    ),
    (
        "Wenn A vor B kommt und B vor C kommt, kommt A vor C?",
        "Ja. Aus den beiden Reihenfolge-Aussagen folgt, dass A vor C kommt.",
    ),
    (
        "Drei Lampen sind an. Eine wird ausgeschaltet. Wie viele sind noch an?",
        "Es sind noch 2 Lampen an.",
    ),
    (
        "Wenn kein Vogel im Käfig ist, kann dann ein Kanarienvogel im Käfig sein?",
        "Nein. Ein Kanarienvogel wäre ein Vogel.",
    ),
    (
        "Alle Teilnehmenden haben ein Ticket. Jana nimmt teil. Hat Jana ein Ticket?",
        "Ja. Wenn beide Aussagen stimmen, hat Jana ein Ticket.",
    ),
    (
        "Ein Raum hat zwei Türen. Eine Tür wird geschlossen. Wie viele Türen hat der Raum danach?",
        "Der Raum hat weiterhin zwei Türen; nur eine davon ist geschlossen.",
    ),
    (
        "Wenn etwas notwendig ist, ist es dann automatisch ausreichend?",
        "Nein. Eine notwendige Bedingung muss erfüllt sein, reicht aber allein nicht immer aus.",
    ),
    (
        "Wenn etwas ausreichend ist, ist es dann automatisch notwendig?",
        "Nein. Eine ausreichende Bedingung genügt, kann aber eine von mehreren Möglichkeiten sein.",
    ),
    (
        "Ein Buch liegt links vom Heft. Der Stift liegt links vom Buch. Was liegt am weitesten links?",
        "Der Stift liegt am weitesten links.",
    ),
]


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def norm(text: str) -> str:
    return clean(text).casefold()


def pair_key(user: str, assistant: str) -> str:
    return hashlib.blake2b(f"{norm(user)}\n{norm(assistant)}".encode(), digest_size=16).hexdigest()


def render(user: str, assistant: str) -> str:
    return f"<|user|>\n{clean(user)}\n\n<|assistant|>\n{clean(assistant)}\n\n<|end|>\n"


def add(
    rows: list[dict],
    seen_pairs: set[str],
    user: str,
    assistant: str,
    category: str,
    family: str,
    source: str,
) -> bool:
    key = pair_key(user, assistant)
    if key in seen_pairs:
        return False
    seen_pairs.add(key)
    rows.append(
        {
            "messages": [
                {"role": "user", "content": clean(user)},
                {"role": "assistant", "content": clean(assistant)},
            ],
            "text": render(user, assistant),
            "category": category,
            "family": family,
            "source": source,
        }
    )
    return True


def load_source(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def select_from_source(source_rows: list[dict], rows: list[dict], seen_pairs: set[str]) -> None:
    rng = random.Random(SEED)
    by_family: dict[str, list[dict]] = defaultdict(list)
    for row in source_rows:
        by_family[row["family"]].append(row)
    for values in by_family.values():
        rng.shuffle(values)

    quotas = {
        "misconception": 10,
        "concept": 15,
        "capital": 90,
        "basic_fact": 30,
        "addition": 40,
        "subtraction": 35,
        "multiplication": 35,
        "division": 30,
        "comparison": 30,
        "unit_conversion": 60,
        "percentage": 5,
        "unknown_term": 5,
    }
    answer_seen_by_family: dict[str, set[str]] = defaultdict(set)
    for family, quota in quotas.items():
        picked = 0
        for row in by_family.get(family, []):
            assistant = row["messages"][1]["content"]
            answer_key = norm(assistant)
            if (
                family in {"concept", "misconception", "basic_fact", "unknown_term"}
                and answer_key in answer_seen_by_family[family]
            ):
                continue
            user = row["messages"][0]["content"]
            if add(
                rows,
                seen_pairs,
                user,
                assistant,
                row["category"],
                family,
                "synthetic_helix_sft_de_booster_v1/from_10k",
            ):
                answer_seen_by_family[family].add(answer_key)
                picked += 1
            if picked >= quota:
                break
        if picked != quota:
            raise ValueError(f"quota not reached for {family}: {picked}/{quota}")


def add_real_terms(rows: list[dict], seen_pairs: set[str]) -> None:
    for user, assistant in REAL_TERMS:
        add(
            rows,
            seen_pairs,
            user,
            assistant,
            "real_term_explanation",
            "real_term",
            "synthetic_helix_sft_de_booster_v1/hand_curated",
        )
    for user, assistant in AMBIGUOUS_REAL:
        add(
            rows,
            seen_pairs,
            user,
            assistant,
            "uncertainty",
            "ambiguous_real_term",
            "synthetic_helix_sft_de_booster_v1/hand_curated",
        )
    for user, assistant in UNKNOWN_SMALL:
        add(
            rows,
            seen_pairs,
            user,
            assistant,
            "uncertainty",
            "unknown_term_small",
            "synthetic_helix_sft_de_booster_v1/hand_curated",
        )
    for user, assistant in PRACTICAL:
        add(
            rows,
            seen_pairs,
            user,
            assistant,
            "practical",
            "everyday",
            "synthetic_helix_sft_de_booster_v1/hand_curated",
        )
    for user, assistant in LOGIC:
        add(
            rows,
            seen_pairs,
            user,
            assistant,
            "logic",
            "reasoning",
            "synthetic_helix_sft_de_booster_v1/hand_curated",
        )


def validate(rows: list[dict]) -> dict:
    if len(rows) != 500:
        raise ValueError(f"expected 500 rows, got {len(rows)}")
    pairs = [(norm(r["messages"][0]["content"]), norm(r["messages"][1]["content"])) for r in rows]
    if len(pairs) != len(set(pairs)):
        raise ValueError("duplicate prompt/answer pairs")
    text = "".join(r["text"] for r in rows)
    if (
        text.count("<|user|>") != len(rows)
        or text.count("<|assistant|>") != len(rows)
        or text.count("<|end|>") != len(rows)
    ):
        raise ValueError("bad chat markers")
    if "\ufffd" in text:
        raise ValueError("replacement character found")
    return {
        "rows": len(rows),
        "category_counts": dict(Counter(r["category"] for r in rows)),
        "family_counts": dict(Counter(r["family"] for r in rows)),
        "unique_pairs": len(set(pairs)),
        "unique_answers": len({norm(r["messages"][1]["content"]) for r in rows}),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict] = []
    seen_pairs: set[str] = set()
    select_from_source(load_source(args.source), rows, seen_pairs)
    add_real_terms(rows, seen_pairs)
    random.Random(SEED).shuffle(rows)
    manifest = validate(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.out_dir / "helix_sft_de_booster_v1.jsonl"
    txt_path = args.out_dir / "helix_sft_de_booster_v1.txt"
    manifest_path = args.out_dir / "helix_sft_de_booster_v1.manifest.json"

    with jsonl_path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with txt_path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(row["text"] + "\n")

    manifest.update(
        {
            "dataset": "helix_sft_de_booster_v1",
            "seed": SEED,
            "source_10k": str(args.source),
            "jsonl": str(jsonl_path),
            "txt": str(txt_path),
            "notes": [
                "Small booster only; not intended as a primary SFT foundation.",
                "Answer-side dedup is enforced for repeated explanation, correction, basic_fact, and unknown-term families.",
                "Arithmetic and unit examples are capped deliberately.",
                "Includes hand-curated stable real-term explanations and ambiguous real-term uncertainty cases.",
            ],
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
