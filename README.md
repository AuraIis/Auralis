# Auralis v2 — Helix

**Ein von Grund auf selbst trainiertes, deutsch-primäres Hybrid-LLM (~0,9B).**
Kein Fine-Tune eines fremden Modells — eigene Architektur, eigener Tokenizer, from scratch auf **einer einzigen GPU** trainiert.

> **Status:** Läuft & wird iteriert. Das Modell ist live, beantwortet Fragen, nutzt Werkzeuge und Wissensquellen — und ist **ehrlich darüber, was es nicht weiß**. Jede Fähigkeit ist mit Tests („Gates") vermessen; auch die Grenzen sind dokumentiert, nicht versteckt.

---

## Was es ist

- **~0,9B Parameter**, Hybrid-Architektur: **6× Mamba-2 + 16× Gated Linear Attention (GLA) + 6× Sparse Attention** = 28 Layer, `d_model 1280`, RoPE.
- **Eigener 200k-SentencePiece-Tokenizer** (auf deutsche + Code-Fertilität optimiert), getrennt gemessen gegen o200k / Llama-3.
- Trainiert & betrieben auf **einer RTX PRO 5000 Blackwell (48 GB)**.
- Wird über eine **OpenAI-/Ollama-kompatible API** ausgeliefert und ist in einem Chat-Hub auswählbar.

## Was es **kann** (getestet & funktionsfähig)

| Fähigkeit | Verhalten |
|---|---|
| **Deutsche Fakten-Fragen** | Hauptstädte, Autoren, Grundwissen — schnell und auf häufigen Fakten korrekt |
| **Ehrliches Abstain** (Signatur) | Sagt „Ich weiß nicht" bei erfundenen/unbekannten Begriffen statt zu halluzinieren |
| **Mathe über Werkzeuge** | Rechnet **nie im Kopf** — ruft ein Python-Tool auf, führt es aus, gibt das verifizierte Ergebnis zurück |
| **RAG / gegroundetes Wissen** | Lokale deutsche Wikipedia (2,84 Mio. Artikel, 6,69 Mio. Volltext-Chunks) + Live-Websuche; liest den Kontext und antwortet belegt — oder abstaint |
| **Code** | Erzeugt einfache, lauffähige Funktionen; stoppt sauber |
| **Auto-Router** | Ein Einstieg wählt automatisch den richtigen Modus (Mathe / Code / RAG / Web / Chat) |
| **Robust gegen „dreckige" Eingaben** | Ein Normalisierer putzt Tippfehler/Slang/Klein-/Umlaute *vor* dem Modell: `was is die haupstadt von italien` → `Was ist die Hauptstadt von Italien?` |
| **Single-Turn-Kontext** | Beantwortet die aktuelle Frage, ohne an Themen aus vorherigen Turns „kleben" zu bleiben |

## Was es **(noch) nicht gut kann** — ehrlich gemessen

Es ist ein **0,9B-Modell — klein**. Es **memoriert mehr, als es versteht.**

| Grenze | Was passiert |
|---|---|
| **Dünnes Weltwissen** | Konfabuliert Fakten, die es nicht trainiert hat (z. B. „Hund" → Tiger-Beschreibung), und kann sogar einen *korrekt vorliegenden* Artikel falsch lesen → **RAG mildert das, der echte Fix ist ein größeres Modell** |
| **Tiefe/offene Erklärungen** | Liefert die Form, aber nicht immer den korrekten Inhalt |
| **Code-Logik / Generalisierung** | Scheitert jenseits einfacher Funktionen; keine Selbst-Reparatur |
| **Semantische Umschreibungen** | „das Energy-Getränk mit dem Stier" findet nicht zuverlässig „Red Bull" (Embeddings getestet → verworfen) |
| **Mehrturnige Gespräche** | Schwach (deshalb Single-Turn im Betrieb) |
| **🇬🇧 Englische Antworten** | **Deutlich schwächer als deutsche** — siehe unten |

## 🇩🇪 vs 🇬🇧 — warum Englisch schlechter ist

Helix **versteht** Englisch (der Pretrain-Korpus ist zweisprachig), aber er wurde **nur auf Deutsch instruktions-trainiert (SFT)**. Ergebnis:

- Englische Fragen werden korrekt **verstanden**, aber die **Antwort** kippt: mehr Konfabulation, manchmal Sprach-Mischung (englische Frage → deutscher Satz).
- Beispiel: „Wer schrieb Faust?" → **„Goethe"** ✅, aber „Who wrote the play Faust?" → **„John Williams"** ❌.

Das ist **kein Defekt, sondern Design**: Helix ist ein **deutsch-primärer** Assistent. Englisch-Verstehen ist ein Nebenprodukt des Pretrainings; englisch-**Antworten** wurde bewusst (noch) nicht trainiert. **Für beste Ergebnisse: auf Deutsch fragen.**

## Wie es gebaut ist

- **From-scratch-Pretrain** mit eigenen Kernels (Mamba / GLA / FlashAttention) auf einem deutsch-primären, bereinigten Korpus.
- **LoRA/DoRA/MoRA-Adapter** auf eingefrorenem Basismodell, pro Aufgabe (corrective / grounded / code) — zur Laufzeit vom Router gewechselt.
- **Serving-Stack:** Normalisierer → Router → Tool-Ausführung → RAG (FTS5/BM25 + Titel/Alias-Auflösung) → gegroundeter Reader.

## Methodik — Gates statt Bauchgefühl

Jede Fähigkeit hat ein **Test-Gate**. Entscheidungen fallen über Gates, **nicht über Val-Loss**. **Negativergebnisse werden dokumentiert und geparkt** statt geschönt ausgeliefert — z. B. Embeddings-Retrieval, Dirty-Data-SFT und ein offener „Erklären"-Archetyp wurden getestet, als unzureichend gemessen und **bewusst nicht eingebaut**. Leitsatz: **„Kurz und richtig schlägt lang und falsch."**

## Was als Nächstes kommt

Die gemessene Decke ist überall dieselbe: **Modellgröße.** Der nächste Schritt ist ein **größeres Modell** (Upcycle auf ~2B oder ein 3B from scratch) für zuverlässiges Wissen, besseres Lesen, Englisch und Logik. Der **gesamte Serving-Stack** (Tokenizer, Router, Tools, RAG, Normalisierer, Gates) wird dabei **direkt übernommen** — das 0,9B hat das Gerüst gebaut, das Größere fällt als besserer *Reader* hinein.

---

## Projekt-Struktur

```
/src/auralis/     Python-Paket (tokenizer · model · training · inference · lora)
/scripts/         CLI-Scripts pro Phase
/configs/         YAML-Hyperparameter (Modell / Training / LoRA)
/docs/            Architektur, Doktrin, Daten-Pipeline, Postmortems
/eval/            Baseline-Fragen + Gate-Ergebnisse
/data /checkpoints   Datensätze & Gewichte — NICHT im Git
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows
pip install -e ".[dev]"
pytest
```

## Grundregeln

1. Alles modular — keine hardcoded Werte, alles über Configs
2. Type Hints (Python 3.11+) & Docstrings überall
3. **Ein** Prompt-Builder für Training = Inference = Eval = API
4. Pro Experiment eine MANIFEST/Config (Git-Hash + Daten-Hash + Metrics)
5. **Gates entscheiden, nicht Val-Loss** — und Grenzen werden ehrlich dokumentiert

---

*Auralis/Helix ist ein Forschungs- & Lernprojekt: ein kleines, ehrliches, deutsch-primäres Modell, an dem Architektur, Daten-Pipeline, Tools, Gates — und die realen Grenzen kleiner Modelle — sichtbar gemacht werden.*
