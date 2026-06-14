# Milestone — Code-Skill (Helix 0.9B)

**Status: Tür geöffnet, aber Code als 0.9B-Grenze sauber bewiesen. v3 ist das Code-Adapter.
Kommt erst bei 3B/7B wieder auf den Tisch.** Ehrlicher Forschungsstand inkl. zweier Negativbefunde.

## Kernergebnis
Helix 0.9B schreibt nach dem Code-SFT **syntaktisch perfekten, lauffähigen Code, der sauber stoppt
und bekannte Muster abruft** — aber es **generalisiert neue Logik nicht** und **kann sich nicht aus
Test-Feedback korrigieren**. Das ist eine Repräsentations-Kapazitäts-Grenze, parallel zur
Grounded-Decke ([[grounded-archetype-h-ceiling]]).

## Der Bogen (3 Stufen, je mit Executor-Gate verifiziert)

### 1. v3 — Tür auf ✅
- Base: `step_60000` (final pretrain) + **Narrow-Embedding-EOS-Fix** (Rows 4–17).
- Daten: `code_curated_v1` (1189, executor-verifiziert) + `code_verified_v1` (173) + corrective + abstain.
- Gate (Executor, 18 Tasks): **syntax 1.0, eos 1.0, pass 11/18, unseen 2/9 (22%)**.
- **Wurzel des alten „0/5":** die früheren `code_lora_v1/v2` trainierten auf dem falschen Base
  (`sft_smoke_step_2000`) **ohne** den Embedding-Fix → konnten `<|end|>` nicht emittieren → Kauderwelsch.
  Mit korrektem Base + Fix: kohärenter, lauffähiger Code.

### 2. v4 — mehr Musterdaten ✗ (Negativbefund)
- +53 deterministische Musterklassen-Funktionen (map/filter/reduce/digits/sort/dedup/string),
  ×3 gewichtet, Gate-Funktionen bewusst ausgeschlossen (Transfer-Test).
- Gate (24 Tasks): **unseen 3/14 (21%) — flach**; auf den *gleichen* 18 Tasks **9/18 vs v3 11/18 ↓**.
- **Interferenz** statt Generalisierung: `nur_gerade → filtert ungerade`, `dritte_potenz → ×3`,
  `doppelt → +2`; brach sogar `remove_duplicates` + `zaehle_vokale`, die v3 konnte.
- **Lehre:** dichte ähnliche Code-SFT-Daten erzeugen bei 0.9B Interferenz, keine Generalisierung.

### 3. Repair-Loop — Feedback nutzen ✗ (Negativbefund)
- Inference-Loop auf v3: schreiben → Hidden-Tests → **kurzes standardisiertes** Fehlerfeedback
  (`TEST FAILED / function / input / expected / got / Fix the function only.`) → 1 Repair.
- **pass@1 12/24, repair@1 12/24, repair_gain +0** (unseen ebenfalls +0).
- **Lehre:** Das Modell produziert bei gezeigtem Fehler dieselbe falsche Logik erneut —
  0.9B kann Test-Feedback nicht zur Selbstkorrektur nutzen.

## Schlussfolgerung
| Fähigkeit | 0.9B |
|---|---|
| Syntax / kompiliert | ✅ 100% |
| Sauberer Stop (`<\|end\|>`) | ✅ 100% |
| Bekannte Muster abrufen | ✅ |
| **Neue Logik generalisieren** | ❌ (~22%, mehr Daten = schlechter) |
| **Aus Feedback reparieren** | ❌ (gain +0) |

Code ist für 0.9B **als Grenze bewiesen**, nicht als „fast geschafft". Der nächste sinnvolle Hebel
ist **kein** weiteres SFT/Repair, sondern ein **größeres Basismodell (3B/7B)** oder deutlich mehr
Code-*Pretraining*. Bis dahin: **v3 als Code-Adapter behalten** (kohärent, sicher, stoppt sauber).

## Methode / Reproduktion
- Deterministisch generierte, **garantiert-korrekte** verifizierte Tasks (Referenz-Code self-checked).
- **Executor-Gate** (`scripts/sft/code/executor_gate.py`): generiert Code → `compile` → führt Asserts aus;
  Metriken syntax_rate / pass_rate / eos_rate / **unseen_pass** (Hauptmetrik = Transfer, nicht Memorieren).
- **Repair-Gate** (`scripts/sft/code/repair_gate.py`): pass@1 / repair@1 / repair_gain, kurzes Feedback.
- Trainer = derselbe Narrow-Embedding-Fix-Trainer wie Grounded ([[sft-eos-embedding-fix]]).

## Artefakte
- Adapter (NICHT in git): `checkpoints/sft_code_v3/adapter_best.pt` (Server). v4 verworfen.
- Daten: `code_curated_v1` (1189 verifiziert), `code_verified_v1` (173), `code_v4` (Muster, Negativbefund).
- Scripts: `scripts/sft/code/`.
