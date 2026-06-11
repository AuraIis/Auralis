import argparse
import ast
import hashlib
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


OLLAMA_URL = "http://localhost:11434"


DATASET_TYPES = {
    "assistant_profile": {
        "description": "Antworten ueber Faehigkeiten, Wissen, Grenzen und Arbeitsweise eines KI-Assistenten",
        "topics": [
            "Welche Wissensbereiche deckst du ab?",
            "Was kannst du im Bereich Wissenschaft und Technik?",
            "Was kannst du beim Programmieren und Debuggen?",
            "Welche Arten von Texten kannst du schreiben?",
            "Wie gehst du mit unsicheren Informationen um?",
            "Welche Grenzen hast du ohne Internetzugriff?",
            "Wie kannst du beim Lernen helfen?",
            "Wie kannst du bei Projektplanung helfen?",
            "Was kannst du in Mathematik und Datenanalyse?",
            "Was kannst du in Geschichte, Philosophie und Kultur?",
            "Wie erklaerst du komplexe Themen einfach?",
            "Wie arbeitest du sicherheitsbewusst?",
            "Was kannst du nicht garantieren?",
            "Wie unterscheidest du Wissen, Vermutung und Meinung?",
            "Wie hilfst du beim Brainstorming?",
            "Wie kannst du Code reviewen?",
            "Wie hilfst du beim Schreiben und Uebersetzen?",
            "Wie gehst du mit medizinischen oder rechtlichen Fragen um?",
        ],
        "formats": [
            "strukturierte Antwort mit kurzen Abschnitten",
            "freundliche Erklaerung mit Bulletpoints",
            "knappe aber vollstaendige Antwort",
            "ausfuehrliche Antwort mit wichtigen Hinweisen",
            "Antwort mit Beispielen fuer Anwendungsfaelle",
        ],
    },
    "general_instruction": {
        "description": "Allgemeine deutschsprachige Instruction-Following-Daten",
        "topics": [
            "erklaere ein technisches Konzept einfach",
            "fasse einen Text zusammen",
            "erstelle einen Lernplan",
            "vergleiche zwei Optionen",
            "formuliere eine professionelle Nachricht",
            "erstelle eine Checkliste",
            "analysiere Vor- und Nachteile",
            "wandle Stichpunkte in einen Text um",
            "erstelle eine Schritt-fuer-Schritt-Anleitung",
            "beantworte eine Wissensfrage mit Unsicherheiten",
        ],
        "formats": [
            "kurze Antwort",
            "mittellange Antwort",
            "Antwort mit nummerierten Schritten",
            "Antwort mit klaren Bulletpoints",
            "Antwort mit kurzer Zusammenfassung am Ende",
        ],
    },
    "code_debug": {
        "description": "Deutschsprachige Programmier- und Debugging-Daten",
        "topics": [
            "Python TypeError",
            "Python IndexError",
            "Python KeyError",
            "Python ValueError",
            "JavaScript Fehler",
            "SQL Query korrigieren",
            "C++ Syntaxfehler",
            "Code refactoring",
            "Fehlererklaerung fuer Anfaenger",
            "Unit-Test fuer einfachen Code schreiben",
        ],
        "formats": [
            "Fehler erklaeren und korrigierten Code geben",
            "kurze Diagnose plus Fix",
            "ausfuehrliche Erklaerung fuer Anfaenger",
            "Code Review mit konkretem Bug",
            "Testfall plus Korrektur",
        ],
    },
    "mixed": {
        "description": "Mix aus Assistant-Profil, Allgemeinwissen, Produktivitaet und Code",
        "topics": [],
        "formats": [],
    },
}

DATASET_TYPES["mixed"]["topics"] = (
    DATASET_TYPES["assistant_profile"]["topics"]
    + DATASET_TYPES["general_instruction"]["topics"]
    + DATASET_TYPES["code_debug"]["topics"]
)
DATASET_TYPES["mixed"]["formats"] = (
    DATASET_TYPES["assistant_profile"]["formats"]
    + DATASET_TYPES["general_instruction"]["formats"]
    + DATASET_TYPES["code_debug"]["formats"]
)

DIFFICULTIES = ["einfach", "mittel", "anspruchsvoll"]
TONES = ["freundlich", "praezise", "natuerlich", "hilfsbereit", "klar und sachlich"]


def post_json(path, payload, timeout=240):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(path, timeout=15):
    with urllib.request.urlopen(OLLAMA_URL + path, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def list_ollama_models():
    try:
        data = get_json("/api/tags")
    except urllib.error.URLError as exc:
        raise SystemExit(
            "Konnte Ollama nicht erreichen. Starte Ollama zuerst. "
            f"Erwartete URL: {OLLAMA_URL}. Fehler: {exc}"
        )
    return [model["name"] for model in data.get("models", [])]


def choose_model(models):
    if not models:
        raise SystemExit("Keine Ollama-Modelle gefunden. Beispiel: ollama pull qwen2.5:7b")

    print("\nVerfuegbare Ollama-Modelle:\n")
    for i, model in enumerate(models, start=1):
        print(f"{i:2d}. {model}")

    while True:
        choice = input("\nWelches Modell soll Daten erzeugen? Nummer eingeben: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            return models[int(choice) - 1]
        print("Ungueltige Auswahl.")


def build_prompt(dataset_type, batch_size, assistant_name):
    cfg = DATASET_TYPES[dataset_type]
    topic = random.choice(cfg["topics"])
    answer_format = random.choice(cfg["formats"])
    difficulty = random.choice(DIFFICULTIES)
    tone = random.choice(TONES)
    seed = random.randint(100000, 999999)

    assistant_rule = (
        f"Wenn die Frage nach Identitaet oder Faehigkeiten fragt, darf der Assistent natuerlich sagen: Ich bin {assistant_name}. Nutze den Namen aber nicht kuenstlich in jeder Antwort."
        if assistant_name
        else "Der Assistent soll keine konkrete Markenidentitaet behaupten."
    )

    if batch_size == 8:
        return f"""
Erzeuge ein deutsches Finetuning-Beispiel als gueltiges JSON-Array mit genau einem Objekt.

Thema: {topic}
Stil: {answer_format}
Schwierigkeit: {difficulty}
Ton: {tone}
Seed: {seed}

Schema:
[{{"instruction":"Nutzerfrage","input":"","ideal_output":"gute hilfreiche Antwort"}}]

Regeln:
- Nur JSON ausgeben, kein Markdown, kein Text davor.
- Die Antwort muss natuerlich, konkret und hilfreich sein.
- Keine kuenstliche Selbstnennung.
- Bei Grenzen oder unsicheren Themen transparent bleiben.
""".strip()

    return f"""
Du erzeugst hochwertige synthetische Finetuning-Daten auf Deutsch.

Erzeuge genau {batch_size} unterschiedliche Trainingsbeispiele.

Datentyp: {dataset_type}
Beschreibung: {cfg["description"]}
Thema: {topic}
Antwortformat: {answer_format}
Schwierigkeit: {difficulty}
Ton: {tone}
Zufallsseed: {seed}

Regeln:
- Jedes Beispiel muss deutlich anders sein.
- Keine fast gleichen Fragen, keine Wiederholungen, keine generischen Duplikate.
- Die Antwort soll natuerlich klingen, nicht wie eine starre Vorlage.
- Keine holprigen Saetze wie "Ich bin als Modell bestrebt" oder Selbstnennung in der dritten Person.
- Schreibe wie ein hilfreicher Assistent direkt zum Nutzer.
- Schreibe auf Deutsch.
- {assistant_rule}
- Bei Grenzen, Unsicherheit, Recht, Medizin oder Finanzen: vorsichtig und transparent antworten.
- Kein Markdown-Codeblock fuer das JSON selbst.
- Antworte ausschliesslich als gueltiges JSON-Array. Kein Text davor oder danach.
- Escape Zeilenumbrueche in JSON-Strings korrekt als \\n.

Schema pro Element:
{{
  "instruction": "Nutzerfrage oder Aufgabe",
  "input": "optionaler Kontext, sonst leerer String",
  "ideal_output": "hochwertige Antwort des Assistenten"
}}

Wichtig:
- ideal_output muss konkret, hilfreich und sauber formuliert sein.
- Wenn es um Faehigkeiten/Wissen geht, nenne auch Grenzen.
- Wenn es um Code geht, gib eine konkrete Korrektur oder Vorgehensweise.
""".strip()


def generate_batch(model, dataset_type, batch_size, assistant_name):
    prompt = build_prompt(dataset_type, batch_size, assistant_name)
    result = post_json(
        "/api/generate",
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.9,
                "top_p": 0.95,
                "repeat_penalty": 1.12,
                "num_ctx": 8192,
                "num_predict": 1600,
            },
        },
        timeout=240,
    )
    response = result.get("response", "")
    if response.strip():
        return response

    chat_result = post_json(
        "/api/chat",
        {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "Du antwortest ausschliesslich mit gueltigem JSON. Kein Markdown, kein Text davor oder danach.",
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {
                "temperature": 0.8,
                "top_p": 0.92,
                "repeat_penalty": 1.12,
                "num_ctx": 8192,
                "num_predict": 1600,
            },
        },
        timeout=240,
    )
    return chat_result.get("message", {}).get("content", "")


def extract_json_array(text):
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and all(key in parsed for key in ("instruction", "input", "ideal_output")):
            return [parsed]
        if isinstance(parsed, dict):
            for key in ("samples", "data", "examples", "items"):
                if isinstance(parsed.get(key), list):
                    return parsed[key]
    except Exception:
        pass

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Keine JSON-Liste gefunden")
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(candidate)
        if isinstance(parsed, list):
            return parsed
        raise


def normalize(value):
    value = value.lower()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^\wäöüß ]+", "", value)
    return value.strip()


def sample_hash(sample):
    key = normalize(sample["instruction"] + " " + sample.get("input", ""))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def too_similar(sample, seen_texts):
    current_words = set(normalize(sample["instruction"]).split())
    if len(current_words) < 4:
        return True
    for old_words in seen_texts[-500:]:
        overlap = len(current_words & old_words) / max(len(current_words | old_words), 1)
        if overlap >= 0.82:
            return True
    seen_texts.append(current_words)
    return False


def validate_sample(sample):
    if not isinstance(sample, dict):
        return False, "sample ist kein Objekt"
    for key in ["instruction", "input", "ideal_output"]:
        if key not in sample or not isinstance(sample[key], str):
            return False, f"ungueltiges Feld: {key}"
    if len(sample["instruction"].strip()) < 12:
        return False, "instruction zu kurz"
    if len(sample["ideal_output"].strip()) < 120:
        return False, "ideal_output zu kurz"
    if "Als KI" in sample["ideal_output"][:80] and len(sample["ideal_output"]) < 300:
        return False, "zu generische KI-Antwort"
    awkward_phrases = [
        "ich bin als",
        "empfehle qwen",
        "qwen hilft dir",
        "in ihrem namen oder deinem kontext",
        "trainingsspeicherwissen",
    ]
    lowered = sample["ideal_output"].lower()
    if any(phrase in lowered for phrase in awkward_phrases):
        return False, "holprige Selbstbeschreibung/Formulierung"
    return True, "ok"


def append_jsonl(path, samples):
    with path.open("a", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def load_existing(path):
    seen_hashes = set()
    seen_texts = []
    count = 0
    if not path.exists():
        return seen_hashes, seen_texts, count

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            sample = json.loads(line)
            seen_hashes.add(sample_hash(sample))
            seen_texts.append(set(normalize(sample["instruction"]).split()))
            count += 1
        except Exception:
            pass
    return seen_hashes, seen_texts, count


def save_raw_debug(out_path, attempt, raw):
    debug_dir = out_path.with_suffix("").parent / (out_path.stem + "_debug_raw")
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_file = debug_dir / f"rejected_batch_{attempt:04d}.txt"
    debug_file.write_text(raw, encoding="utf-8", errors="replace")
    return debug_file


def main():
    parser = argparse.ArgumentParser(description="Erzeugt viele saubere JSONL-Finetuning-Daten mit Ollama.")
    parser.add_argument("--count", type=int, default=1000, help="Zielanzahl guter Samples")
    parser.add_argument("--batch-size", type=int, default=1, help="Samples pro Modellaufruf")
    parser.add_argument("--out", default="ollama_mass_dataset.jsonl", help="Ausgabedatei")
    parser.add_argument("--model", default=None, help="Ollama-Modell direkt angeben")
    parser.add_argument("--max-attempts", type=int, default=0, help="Maximale Batch-Versuche, 0 = automatisch")
    parser.add_argument(
        "--type",
        default="assistant_profile",
        choices=sorted(DATASET_TYPES.keys()),
        help="Art der Daten",
    )
    parser.add_argument(
        "--assistant-name",
        default="",
        help="Optionaler Name, den der Assistent in Profilantworten verwenden darf, z.B. Qwen",
    )
    args = parser.parse_args()

    models = list_ollama_models()
    model = args.model if args.model else choose_model(models)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path
    out_path = out_path.resolve()
    seen_hashes, seen_texts, accepted = load_existing(out_path)
    rejected = 0
    attempts = 0
    max_attempts = args.max_attempts or max(args.count * 5, 20)

    print(f"\nModell: {model}", flush=True)
    print(f"Datentyp: {args.type}", flush=True)
    print(f"Ziel: {args.count}", flush=True)
    print(f"Ausgabe: {out_path}", flush=True)
    if str(out_path).lower().startswith(r"c:\windows\system32"):
        print("Warnung: Ausgabe liegt in C:\\Windows\\System32. Nutze besser --out mit einem normalen Ordner.", flush=True)
    if accepted:
        print(f"Bestehende Samples geladen: {accepted}", flush=True)
    print(f"Max. Versuche: {max_attempts}\n", flush=True)

    while accepted < args.count and attempts < max_attempts:
        attempts += 1
        print(f"[{accepted}/{args.count}] Batch {attempts} startet...", flush=True)
        try:
            raw = generate_batch(model, args.type, args.batch_size, args.assistant_name.strip())
            print(f"  Antwort erhalten: {len(raw)} Zeichen", flush=True)
            batch = extract_json_array(raw)
        except Exception as exc:
            rejected += args.batch_size
            try:
                debug_file = save_raw_debug(out_path, attempts, raw if "raw" in locals() else "")
                print(f"  Batch verworfen: {exc}. Rohantwort: {debug_file}", flush=True)
            except Exception:
                print(f"  Batch verworfen: {exc}", flush=True)
            time.sleep(1)
            continue

        good = []
        for sample in batch:
            ok, reason = validate_sample(sample)
            if not ok:
                rejected += 1
                print(f"  Verworfen: {reason}", flush=True)
                continue
            h = sample_hash(sample)
            if h in seen_hashes or too_similar(sample, seen_texts):
                rejected += 1
                print("  Verworfen: Duplikat/aehnlich", flush=True)
                continue
            seen_hashes.add(h)
            good.append(
                {
                    "instruction": sample["instruction"].strip(),
                    "input": sample["input"].strip(),
                    "ideal_output": sample["ideal_output"].strip(),
                }
            )

        if good:
            good = good[: args.count - accepted]
            append_jsonl(out_path, good)
            accepted += len(good)
            print(f"  Akzeptiert: {len(good)}", flush=True)
        else:
            print("  Keine guten Samples.", flush=True)

    if accepted < args.count:
        print("\nAbgebrochen, weil zu viele Versuche ohne genug gute Samples noetig waren.", flush=True)
        print("Tipp: anderes Modell nutzen, --type wechseln oder --max-attempts erhoehen.", flush=True)
    else:
        print("\nFertig.", flush=True)
    print(f"Gespeichert: {out_path}", flush=True)
    print(f"Akzeptiert: {accepted}", flush=True)
    print(f"Verworfen: {rejected}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAbgebrochen.")
        sys.exit(130)
