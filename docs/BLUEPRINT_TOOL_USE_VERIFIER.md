# Blueprint — Tool-Use & Selbst-Verifikation (Helix v2)

> **Status:** Design / beschlossen (Michael + GPT + Claude, Juni 2026). Noch nicht implementiert.
> **Phase:** kommt NACH dem aktuellen Reasoning-SFT (sft_v2). Siehe `ZUKUNFT_BACKLOG.md` → Phase 3–4.
> **Leitsatz:** *Ein kleines Modell soll nicht alles wissen oder können. Es soll lernen, wann es prüfen muss.*
> Nicht raten — verifizieren. Konsequente Fortsetzung des Anti-Halluzinations-USP.

---

## 1) Warum (Evidenz, nicht Hoffnung)
Helix ist ~0,9B und untertrainiert. Benchmarks (Juni 2026) zeigen: das Modell rechnet
mehrstufige Mathe **nicht** zuverlässig im Kopf — das ist bei der Größe physikalisch normal,
kein Bug. Mehr Parameter „ins Hirn prügeln" skaliert schlecht.

**Der Hebel ist nicht Rechnen-Können, sondern Prüfen-Lernen.** Ein kleines Modell kann sehr
gut lernen, ein Muster zu erkennen („das muss geprüft werden") und ein Werkzeug zu rufen.
Die Korrektheit kommt dann von **außen** (Rechner / Code-Runner), nicht aus dem Modell.

```
347 × 892 im Kopf      → 0.9B rät, meist falsch
print(347*892) rufen   → Modell muss nur WISSEN, dass es rufen soll → lernbar
```

## 2) Ehrliche Decke (was wir NICHT versprechen)
- Du bekommst zuverlässig das **Verhalten**: testen-vor-akzeptieren, bei Fehler reparieren,
  bei Unsicherheit ein Tool rufen.
- Du bekommst **nicht garantiert** gute Code-Qualität oder autonomes Debugging. Die
  Selbst-Reparatur-Schleife setzt voraus, dass das Modell einen Traceback liest und einen
  *korrekten* Fix produziert — genau die Fähigkeit, die bei 0,9B wackelt.
- Reparatur-Traces lehren das **Muster** („Fehler = Feedback, nochmal"), nicht garantiert
  die Lösung. Trotzdem ein Gewinn und voll auf der USP-Linie.

## 3) Die 80/20-Wahrheit
> **Die SFT-Traces sind die einfachen 20 %. Das Inferenz-Harness ist die echten 80 %.**

Trainingsdaten mit *vorab ausgerechnetem* `<result>` zu erzeugen ist trivial. Der schwere
Teil ist der Laufzeit-Loop, der das Tool **wirklich ausführt** und das Ergebnis live einspeist.
Ohne diesen Loop *halluziniert* das Modell die Zahl nur verkleidet = exakt das Problem,
das wir lösen wollen.

---

## 4) Knallharte Reihenfolge (nicht überspringen)
| Stufe | Inhalt | Voraussetzung |
|---|---|---|
| 0 | Reasoning-SFT fertig (sft_v2) — prüfen, ob Antworten strukturierter werden | läuft |
| **1** | **Mathe-Tool-Harness ALLEIN** (simpelster Fall — beweist das Harness) | Stufe 0 |
| 2 | Tool-Use-SFT: Rechnen / Einheiten / kleine Zahlenlogik | Stufe 1 grün |
| 3 | Code-Annealing (Python-Edu liegt in `anneal_candidates/`) → Code latent im Base | Stufe 2 |
| 4 | Code + eigene Tests + Selbst-Reparatur + Hidden-Test-Daten-Gate | Stufe 3 |
| 5 | *eventuell* Code-DoRA auf annealtem Base | Stufe 4 |

**Begründung Stufe 1 zuerst:** Mathe-Tool prüft nur 5 Dinge, alle isoliert testbar:
1. Erkennt Helix „ich brauche ein Tool"? 2. Schreibt es den Call korrekt? 3. Stoppt die
Generierung am Call? 4. Führt das Harness Python aus? 5. Baut Helix das Ergebnis korrekt ein?
Code bringt sofort 7+ Probleme gleichzeitig (Qualität, Tests, Tracebacks, Reparatur, Spec,
Hidden Tests, Sandbox) — zu viel auf einmal.

**Begründung Code-DoRA zuletzt:** Ein Adapter **verstärkt latente Fähigkeit — er installiert
keine neue.** Helix hat 0 % echten Code im Pretraining gesehen (nur Code *als Prosa*). Code-DoRA
auf diesem Base = auf Sand bauen. Erst Code-Annealing (latente Fähigkeit), dann Adapter.

---

## 5) Harness-Architektur (der eigentliche Bau)

### 5.1 Inferenz-Loop (State-Machine)
```
1. Modell generiert Text
2. Stop-Sequenz </tool> erreicht?  → Generierung anhalten, Kontrolle abgeben
3. Tool-Call parsen (Sprache + Body zwischen <tool:python> … </tool>)
4. Sandbox führt aus → stdout/stderr/exit
5. <result> … </result> in den Kontext injizieren
6. Modell generiert weiter (zurück zu 1) bis <|end|>
7. Guard: max. N Tool-Calls pro Antwort (Endlosschleifen-Schutz)
```

### 5.2 Format (eine Konvention, byte-exakt, train == inference)
```
<|user|>
Was ist 47 mal 83?
<|end|>
<|assistant|>
<tool:python>
print(47 * 83)
</tool>
<result>
3901
</result>
47 mal 83 ergibt 3901.
<|end|>
```
- `</tool>` ist **Stop-Sequenz** der Generierung (zentrale Design-Entscheidung).
- `<result>…</result>` schreibt **das Harness**, nie das Modell (im Training vorab ausgerechnet,
  zur Laufzeit live).
- Tags als Plain-Text lernen (keine zwingenden Spezial-Tokens), aber **konsistent** — sonst
  bricht der byte-exakte Train/Inference-Match (vgl. L-001: Prompt-Mismatch).

### 5.3 Sandbox (nicht verhandelbar)
Modell-generierten Code **niemals** ungeschützt laufen lassen.
```
Docker-Container · KEIN Netzwerk · Timeout (z.B. 5 s) · RAM-Limit · CPU-Limit
nur tmpdir · keine Systempfade · read-only Root · non-root user
```
Wir sitzen ohnehin in Docker → Substrat vorhanden. Minimal: ephemerer Sub-Container/`nsjail`
pro Call.

## 6) Trainingsdaten

### 6.1 Mathe-Tool-MVP (Stufe 2)
Drei Aufgabentypen, klein und messbar:
- **Rechnen** (`print(347*892)`)
- **Einheiten** (`print(3*60+25)` → Stunden→Minuten)
- **kleine Zahlenlogik** (Durchschnitt, Prozent, Verhältnis)

Generierung: Teacher erzeugt Frage + korrekten Tool-Call; das **Harness** rechnet `<result>`
deterministisch aus (nicht der Teacher → keine Teacher-Mathe-Fehler). Format byte-exakt wie 5.2.

### 6.2 Code-Phase (Stufe 4) — Datentyp-Mix
```
40 %  einfache Python-Aufgaben mit Tests
20 %  Fehler → Traceback → Reparatur → erneut testen
15 %  Randfälle / Edge Cases
15 %  Code erklären
10 %  "Spec unklar, ich brauche Details"
```
Aufgaben klein halten (Liste sortieren, Duplikate, Primzahl, Palindrom, CSV/JSON, kleine
Klassen) — **keine großen Projekte**.

### 6.3 Hidden Tests = Daten-/Eval-Werkzeug, NICHT Inferenz
> Das ist exakt unser gpt-4o-Verify-Muster, nur für Code (= externe, unbestechliche Prüfung).
- **Daten-Bauen:** Modell schreibt Lösung **+ eigene Tests** → Sandbox führt aus → *unabhängige
  Hidden Tests* prüfen gegen → nur Traces, die **beide** bestehen, kommen ins SFT-Set.
  (Verhindert „besteht eigene Tests, ist aber Mist" wie `def addiere(a,b): return 5`.)
- **Echter User:** es gibt **keine** Hidden Tests (der User *ist* die Spec). Da bleibt nur
  „eigene Tests schreiben + ausführen". Nicht verwechseln.

## 6b) STAND: Mathe-Harness MVP gebaut (Juni 2026)
`scripts/sft/tool_harness.py` — gebaut + getestet:
- **Sicherer Rechner** (AST-Whitelist, kein RCE): Selftest 14/14, lehnt `import`,
  `__import__/system`, `open`, Compute-Bombe `9**9**9`, Zuweisung, /0 ab.
- **Loop bewiesen** (`--selftest-only`, gescriptetes Fake-Modell): Stop@`</tool>`
  → Executor → `<result>`-Injektion → Resume. Transcript korrekt, Result vom
  Executor (nicht Modell).
- **Offen:** Modell *emittiert* noch keine Tool-Calls → braucht Tool-SFT.
- **Self-generating-Daten:** Mathe-Tool-Traces brauchen KEINEN Teacher/Key — der
  Rechner IST die Ground Truth (Problem → kanonischer Call → Executor-Result →
  Antwort-Template). Damit ist Tool-SFT-Daten auch ohne OpenRouter-Key baubar.

### Hartes Tool-Gate (vor Promotion eines tool-SFT-Checkpoints)
```
✗ Modell-Tokenstrom enthält selbst "<result>"  → Stop-Sequenz versagt → FAIL
✗ kein parsebarer <tool:python>…</tool>          → FAIL
✗ finale Antwort-Zahl ≠ Executor-Zahl            → Result nicht übernommen → FAIL
✓ stoppt @</tool> · Executor rechnet · Antwort == Executor-Zahl
```
(Im Harness kann das Modell `<result>` gar nicht schreiben — die Generierung stoppt
bei `</tool>`. „Fake-result" = Stop-Sequenz hat versagt. Genau das prüft das Gate.)

**Stand:** Phase-1 (call_only) bestanden — best step_400: tool_rate 100%, false_tool 0%,
fake_result 0%, parse 97%, correct **68%**. Phase 1.1 (enrichte Übersetzungs-Traces) zielt
auf correct ≥80% bevor Phase 2. Trainer-`<result>`-Masking gebaut + token-genau verifiziert.

### Phase-2-End-to-End-Gate (zusätzlich — misst Result-NUTZUNG)
Phase 2 kann auf 3 Arten scheitern → 3 Metriken:
```
result_usage_rate    : nutzt die finale Antwort die Executor-Zahl ueberhaupt?  (Fail 1: ignoriert <result>)
answer_numeric_match : steht EXAKT die Executor-Zahl in der Antwort?           (Fail 2: Zahl falsch abgeschrieben)
fake_result_rate     : hat das Modell trotzdem selbst <result> geschrieben?    (Fail 3: halluziniert Block) -> MUSS 0
```
Plus weiterhin: false_tool 0 · parse >95% · correct (gedeckelt durch Phase 1.1). Promotion nur wenn alle grün.

## 7) Erfolgskriterien (messbar, nicht „sieht gut aus")
- **Stufe 1/2:** auf einem Mathe-Probe-Set (n≥200): Tool-Call-Trefferquote ↑, End-Antwort-
  Korrektheit deutlich > Base-ohne-Tool (Ziel: Korrektheit folgt dem Rechner, nicht dem Raten).
- Stop-Sequenz greift in ~100 % (kein „Modell schreibt `<result>` selbst").
- **Stufe 4:** Anteil Lösungen, die unabhängige Hidden Tests bestehen, ↑ ggü. Code-SFT-ohne-Loop.
- Negativ-Guard: Tool-Use darf die allgemeine SFT-Qualität (Benchmarks de/en) **nicht**
  verschlechtern (separat gegenmessen).

## 8) Bausteine, die schon liegen
- **Verify-Muster** (gpt-4o-Pass) = Hidden-Test-Mechanik, bereits gebaut & validiert.
- **Python-Edu** (`data/raw/anneal_candidates/`) = Code-Annealing-Daten, schon geladen.
- **Docker** = Sandbox-Substrat, vorhanden.
- **Byte-exakter Prompt-Builder** (L-001-Lektion) = Grundlage für train==inference Format.

## 9) Offene Implementierungs-Fragen (vor Stufe 1 klären)
- Stop-Sequenz im Inferenz-Pfad sauber verdrahten (Sampler muss bei `</tool>` anhalten).
- Sandbox-Aufruf-Latenz pro Call (Container-Spawn vs. persistenter Worker-Pool).
- Tokenizer: Tags effizient kodiert? (`<tool:python>` etc. nicht in 10 Tokens zerfallen lassen.)
- DoRA-Targeting auf Hybrid-Arch (Mamba `in_proj/out_proj`, GLA `q/k/v/g` = linear → adaptierbar,
  muss aber im Trainer verdrahtet sein) — erst relevant ab Stufe 5.

---
*Dieses Dokument hält eine dreifach trianguliert beschlossene Richtung fest. Reihenfolge ist
gegated. Kein Vorziehen von Stufe 4/5, bevor Stufe 1–3 grün sind (Erinnerung 500M-Sackgasse:
Schicht vor Fundament = Müll).*
