# Zukunft-Backlog — geparkte Ideen & Ressourcen (nach Phase)

> **Zweck:** Michaels Ideen-Maschine produziert viele gute, aber oft *weit vorausliegende*
> Richtungen. Hier werden sie geparkt — mit ehrlichem Phasen-Tag, damit nichts verloren
> geht und nichts vorgezogen wird, bevor der Base/die Größe es trägt.
>
> **Grundregel:** Jede Stufe ist gegated auf die vorige + auf Modellgröße.
> Reihenfolge nicht überspringen. (Erinnerung an die 500M-Sackgasse: Schicht vor Fundament = Müll.)
>
> **Leitsatz (Tool-Use-Philosophie):** *Ein kleines Modell soll nicht alles wissen oder können.
> Es soll lernen, wann es prüfen muss.* — nicht raten, sondern verifizieren.

---

## Wo wir stehen
- ✅ **Pretraining** (1B, de55/en45) — abgeschlossen, Wissen + Sprache nachgewiesen.
- ⚗ **Phase 1 — Basis-SFT** (gerade aktiv): aus dem Base einen *antwortenden* Assistenten machen.
  Smoke bestanden („Berlin"), Pilot (282 Beispiele, DeepSeek + Premium-Verify) läuft.

## Phase 2 — Kalibrierungs-/Ehrlichkeits-SFT  *(als Anteil im SFT-Mix)*
- Verhalten: Antwort zuerst, stabile Fakten direkt; **Marker nur bei Risiko** (🔧 aktuelle Quelle,
  ⚠️ Vorsicht, ❗ definitionsabhängig); falsche Prämisse korrigieren; kein Fake-Live-Check.
- Status: **Generator + Verify-Pass gebaut & validiert.** Kommt als Slice in die Haupt-SFT.

### Helix-R-Tuning v1 (Self-Labeling, KEY-FREI) — Quelle: R-Tuning (arXiv 2311.09677)
> Befund des Papers = exakt unser Problem: klassisches Instruction-Tuning **zwingt** zu einer
> Antwort → das Modell rät bei Unbekanntem. Lösung: Fragen in „im Modellwissen / nicht sicher"
> trennen und Unsicherheit bewusst trainieren. Refusal ist eine **generalisierende Meta-Skill**.
>
> **WICHTIG — Self-Labeling heißt NICHT „Helix beurteilt sich selbst".** Es heißt: Helix beantwortet
> Fragen mit BEKANNTER Lösung, ein **Script** vergleicht gegen Gold/MC/Regex/Executor:
> ```
> richtig beantwortet  -> confident-answer (BEHALTEN, Retention-Anker)
> falsch beantwortet   -> uncertainty/refusal trainieren ("kann ich nicht zuverlässig beantworten")
> Rechnung             -> tool-needed (Tool rufen, nicht raten)
> ```
> **Key-frei machbar** bei verifizierbaren Antworten: MC (MMLU-DE/ARC-DE), Faktenfragen mit
> eindeutiger Antwort, Mathe (Executor), Übersetzung mit klarem Ziel. **Nicht** key-frei bei
> offenen Antworten ("Erkläre einen Vulkan") → später Teacher/Rule-Checks/lokaler Judge.
>
> **Rezept Helix-R-Tuning v1:** (1) Gold-Label-Fragenbank (MMLU-DE/ARC-DE + eigene Faktenbatterie
> + kontrastive Fälle Bonn/Berlin, Pluto). (2) Helix antworten lassen. (3) Auto-labeln (s.o.).
> (4) SFT: bekannt→korrekt antworten, unbekannt→nicht erfinden, Rechnung→Tool.
>
> **R-Tuning-R (Replacement)** = bei Unsicherem die richtige Antwort NICHT zeigen, sondern echtes
> „weiß ich nicht" trainieren → höchste Refusal-Rate, aber **Über-Verweigerungs-Gefahr**.
> → **PFLICHT-Retention-Gate** (Wien/Madrid/Berlin/Pluto müssen beantwortet bleiben; erfundene
> Entitäten dürfen NICHT ausgeschmückt werden; `12+15`→Tool). Eine Retention-Regression = nicht
> promotable. Abstention moderat dosieren (RLVR-Humility-Befund: zu viel = Verweigerungs-Bot).

### GEMESSEN — Calib v1 (Juni 2026): Fähigkeit bewiesen, Rezept zu grob
- Probe (key-frei): step_600 halluziniert **100 %** der erfundenen Entitäten (60/60), kennt
  Hauptstädte ~74 %, Werke nur 6 %. → klarer Kalibrierungsbedarf.
- Calib-SFT v1 (714 Bsp: 155 Abstention / 38 confident / 600 Anker, ~20 % Abstention).
- Dual-Gate (Held-out): Honesty **0→93 % Abstention** auf NEUEN Erfundenen, Hauptstädte
  gehalten (89 %). **ABER Demo deckte auf, was das Aggregat-Gate maskierte:** Über-Verweigerung
  leakt auf bekannte Fakten (Einstein→„weiß ich nicht") UND bricht Math-Dispatch (`12+15`→
  Abstention statt Tool; `15 %` ging noch). → **step_50 NICHT promotable.**
- **Lektion:** Gate maß nur Hauptstädte → Über-Verweigerung versteckte sich im ungemessenen Raum.
  Greedy-Demo fing es, gesampeltes Aggregat nicht. („Prüfe, ob die Zahl misst, was du glaubst.")
- **Fix Calib v2:** (a) **Tool-SFT-Traces beimischen** (Dispatch nicht vergessen), (b) confident-
  Anker BREITER (Personen/Fakten, nicht nur Hauptstädte) + viele kurze Known-Facts (gegen
  „kurz→abstain"), (c) Abstention-Anteil < 20 %. **Gate v2:** Retention breiter messen
  (Hauptstädte + Personen + Math-Dispatch), nicht nur Capitals.

## Leitprinzip — besserer Entwicklungs-Loop, NICHT rekursive Selbstverbesserung
> Quelle: Anthropic, "Recursive Self-Improvement". Helix wird **nicht durch „mehr Modell"** besser,
> sondern durch einen **besseren Loop um das Modell**: Modell + Tools + Tests + Mensch.
- **Mensch = Direction-Setter** (was testen? was ist Erfolg/Fail? welche Richtung?). Modell/Tools
  führen aus (Daten erzeugen, Tests laufen, Fehler suchen, Reports) — aber **kein Auto-Promote**.
- **Neuer Engpass = Review/Verifikation**, nicht Generierung. Darum harte Gates überall:
  ```
  kein Gate     -> kein Promote
  kein Verify   -> kein Datensatz
  kein Benchmark-> keine Behauptung
  kein Sandbox  -> kein Tool-Use
  ```
- **Kein** „Helix trainiert sich selbst und wir lassen laufen" (bei 0.9B technisch Unsinn + gefährlich).
- Später optional: Auto-Experiment-Loop (kleiner Lauf → Gate → Report → promote/reject/retry),
  weiterhin mit Mensch an den Zielen.

## Phase 3–4 — Tool-Nutzung / Verifier (NACH solidem SFT)  ⭐ BESCHLOSSEN (Michael + GPT + Claude, Juni 2026)
> **Dreifach trianguliert.** Tool-Use ist der nächste *große* Schritt nach dem SFT — DoRA kommt
> erst später. Begründung: ein 0.9B rechnet `347×892` zuverlässig falsch, kann aber lernen,
> *wann* es einen Rechner ruft. Verifikation **außerhalb** des Modells statt Raten im Kopf.
> Das ist die konsequente Fortsetzung des Anti-Halluzinations-USP: **nicht raten, prüfen.**

**Knallharte Reihenfolge (nicht überspringen):**
1. Reasoning-SFT fertig + prüfen, ob Antworten strukturierter werden.
2. **Mathe-Tool-Harness ZUERST** (allein, simpelster Fall) — beweist das Harness.
3. Tool-Use-SFT für einfache Rechnungen / Einheiten / kleine Zahlenlogik.
4. **Erst danach** Code-Annealing (Python-Edu liegt bereit).
5. Dann Code + eigene Tests + Selbst-Reparatur.
6. Dann *eventuell* Code-DoRA.

**Harness ist die eigentliche Arbeit (80 %), die SFT-Traces sind die einfachen 20 %:**
- **Stop-Sequenz `</tool>`** → Generierung stoppt, Modell gibt Kontrolle ab.
- **Sandbox führt aus** (Docker, **kein Netz**, Timeout, RAM/CPU-Limit, nur tmpdir, keine Systempfade).
- **Loop injiziert `<result>…</result>`** → Modell generiert weiter bis `<|end|>`.
- Ohne diesen Loop *halluziniert* das Modell die Zahl nur verkleidet = genau das zu lösende Problem.
- **Hidden Tests = Daten-/Eval-Werkzeug, NICHT Inferenz** (= unser gpt-4o-Verify-Muster für Code):
  beim Daten-Bauen prüft eine *externe* Instanz unabhängig; beim echten User gibt es keine
  Hidden Tests (der User *ist* die Spec), da bleibt nur „eigene Tests schreiben + ausführen".
- **Ehrliche Decke bei 0.9B:** du bekommst zuverlässig das *Verhalten* (testen-vor-akzeptieren,
  bei Fehler reparieren), **nicht garantiert** gute Code-Qualität/autonomes Debugging.
  Reparatur-Traces lehren das *Muster*, nicht garantiert den korrekten Fix. Trotzdem ein Gewinn.
- **Bausteine liegen schon:** Verify-Muster (= Hidden Tests) · Python-Edu (= Code-Annealing) · Docker (= Sandbox).

## Phase 5+ — fortgeschritten (braucht Code-Können + Größe 3B–7B+)
- **Modell-*gebaute* Tools** (schreibt + testet Tools in Sandbox). Riskant, erst mit Code-
  Pretraining (`code.bin` ~677M Tokens, noch nicht im Mix) + Code-SFT.
- **Reasoning-RL / RLVR:**
  - 📌 **MiniMaxAI/SynLogic** — https://huggingface.co/datasets/MiniMaxAI/SynLogic
    - ~49k Beispiele, **35 Logik-Tasks** (Sudoku, Cipher, Game-of-24, Cryptarithm…).
    - Format: **RL-mit-verifizierbaren-Belohnungen** (`<think>/<answer>` + Verifier) — *nicht* SFT.
    - Sprache: **EN + ZH** (kein Deutsch). Lizenz: **MIT** ✅.
    - Zielgröße laut Autoren: **7B / 32B**. Für 0.9B unbrauchbar.
    - **Verdikt:** sehr gut, aber falsche Phase. Braucht: größeres Modell **+** komplette RL-Pipeline
      (GRPO/Verifier). Erst Reasoning-/Skalierungs-Phase.

## Skalierungs-Kontext (warum die Reihenfolge zählt)
- Vokab bleibt fix 200k → Anteil sinkt mit Größe (27 %@1B → ~13 %@3B → ~10 %@7B+).
- Reasoning-RL, autonome Tools, harte Logik: lohnen erst ab 3B–7B+, auf solider SFT-Basis.

---

## Gelernt von vergleichbaren Projekten (Zamba, SmolLM2, TinyLlama, Karpathy)
> Quelle: Berichte gelesen (arXiv 2405.16712, 2502.02737, 2401.02385, llm.c). Befund:
> Helix hat unabhängig dieselben Problem-Klassen getroffen → echtes Model-Engineering.
> Diese Punkte sind **NEU** oder **Verfeinerungen**, anzuwenden NACH dem aktuellen SFT-Lauf.

### ⭐ v-next Schritt 1 — Annealing-Phase (Zambas größter Hebel: MMLU 50,8→57,7)
- Kurze Continued-Pretrain-Phase (~5% Tokens) mit **nur Top-Daten** (sauberste DE +
  Mathe + **Code aus `code.bin`**) + **LR fast auf null**. Poliert den Base, bevor SFT draufkommt.
- Bild: letzte Lern-Woche vor der Prüfung nur aus dem besten Lehrbuch.
- Aufwand: mittel (Anneal-Mix + kurzer Lauf). Bei 0.9B evtl. kleiner als +7. Gated auf SFT-Ergebnis.

### 💸 Billige Verbesserungen (sofort/günstig, nach dem Lauf)
- **Substring-Dekontamination** (SmolLM2): nicht nur exakte Eval-Probes filtern, auch
  *umformulierte* (substring/fuzzy). ~20 Zeilen → wasserdichte Eval-Ehrlichkeit.
- **EOS-Audit** (TinyLlama verlor 2,3 Bio. Tokens an EOS-Bug): einmal verifizieren, dass jedes
  SFT-Beispiel mit genau einem `<|end|>` an richtiger Stelle endet + Loss-Maske greift.
- **Datenmix iterativ nach Eval** (SmolLM2): per-Kategorie-Eval → Mix gezielt nachjustieren
  statt schätzen. = unsere Wissensprofil-Strategie, nur systematischer.

### 📊 Benchmarks — NÄCHSTER konkreter Schritt (direkt nach dem SFT-Lauf)
> Experten-Konsens: ohne Standard-Zahlen ist Helix nicht *bewertbar*. Benchmarks sind eine
> Messung, kein Reifegrad — Base ist technisch schon benchmarkbar.
- **Blocker = `lm-eval-harness`-Wrapper:** Helix ist eigene Architektur, braucht einen kleinen
  Wrapper (Loglikelihood-Interface). Das ist die eigentliche „Readiness"-Arbeit, sonst nichts.
- **Messen:** `Base (50k)` vs `Base+SFT` vs Baselines (**Qwen2.5-0.5B, SmolLM2-360M/1.7B, TinyLlama**)
  auf demselben Harness → fairer Vergleich + zeigt den SFT-Lift.
- **Benchmarks:** MMLU/HellaSwag/ARC (Multiple-Choice, Likelihood — kein Generieren nötig)
  + **deutsche Varianten** (Heimstärke, fairer) + IFEval/MT-Bench (Instruction, braucht SFT).
- **Ehrliche Erwartung:** erste Zahlen bescheiden (0.9B, untertrainiert → MMLU evtl. ~Zufall).
  Niedrig ≠ Versagen — es ist der erste *ehrliche* Standort. DE-Benchmarks zuerst.
- *Öffentlich-wirksam* erst nach dem **Annealing** (da ist Helix am stärksten).

### 🏗️ Architektur-Notiz (v3 / Skalierung)
- Zamba: **EINE geteilte Attention** statt mehrerer separater → param-effizient. Für Helix v-next
  prüfen, ob Attention-Layer geteilt werden können → mehr Budget fürs Mamba.

### ✅ Bestätigt (machen wir schon richtig)
- Grad-Clipping 1.0 · deterministische Eval · Qualität>Menge · Code braucht GitHub nicht Web.

---

### Geparkte „großartig-aber-weit-voraus"-Ideen (Chronik)
1. Autonomer Tool-bauender Agent (decompose→build→test→run) → **Phase 5+**.
2. Verifier-Agent (Claim-Zerlegung, Konfidenz, Prüfbedarf) → **Phase 2 sichtbar / 3–4 mit Tools**.
3. SynLogic / Reasoning-RL für 7B–32B → **Phase 5+**.

*Alle drei richtig. Alle drei brauchen erst den fertigen kleinen Assistenten + mehr Größe.*
