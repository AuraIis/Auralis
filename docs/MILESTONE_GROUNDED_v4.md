# Milestone — Archetyp H (Grounded) v4

**Status: bester aktueller Grounded-Stand bei 0.9B — NICHT final bestanden (80%-Ziel knapp verfehlt).**

Ehrlicher Forschungsstand, kein Schönreden. v4 ist das Adapter, das wir behalten;
v4.1 wurde gebaut, getestet und **bewusst nicht promotet** (siehe Plateau-Befund unten).

## Ergebnis (Grounded-Stress-Gate, 54 Fälle)

| Metrik | Wert | Bar | |
|---|---|---|---|
| `answer_ok` | **23/30 (77%)** | >80% | ❌ (1 Fall zu wenig) |
| `refuse_ok` | **22/24 (92%)** | >90% | ✅ |
| `world_leaks` | **0** | =0 | ✅ |
| `stop_rate` | **1.0 (100%)** | >95% | ✅ |

Das **Sicherheitskritische ist voll erfüllt**: 0 Welt-Wissen-Leaks über 11 Trap-Fälle,
sauberer Stop auf allen 54 Fällen. Die fehlenden 3 Prozentpunkte sind reine
Hard-Distraktor-Edge-Cases — kein Korrektheits- oder Halluzinations-Risiko im Kern.

## Iterationsverlauf — und warum wir bei v4 stoppen

| Version | answer_ok | refuse_ok | leaks | stop | Befund |
|---|---|---|---|---|---|
| v2 (Zahlen-Fokus) | 8/17 (47%)* | 18/19 | 0 | 1.0 | Distraktoren + variierte Prosa schwach |
| **v3** (Strukturvielfalt + Distraktoren) | 19/30 (63%) | 21/24 (87.5%) | 0 | 1.0 | Distraktoren deutlich besser, Zwei-Zahlen-Prosa funktioniert |
| **v4** (dichte Prosa + Zähl-Absage + Beginn/Ende) | **23/30 (77%)** | 22/24 (92%) | 0 | 1.0 | **bester Stand**; +Schmidt/Schmitt gelöst |
| v4.1 (Regressions-Fixes) | 22/30 (73%) | 23/24 (96%) | 0 | 1.0 | lateral — siehe Plateau |

\*v2-Zahlen auf dem alten 36-Fälle-Gate; ab v3 das erweiterte 54-Fälle-Gate.

### Plateau-Befund (Whack-a-Mole)
v4.1 hat **alle 4 gezielten Fixes nachweislich erreicht** (Roman→„412 Seiten" statt
2850-Halluzination; Greifenau-Monat→„Mai"; Dritt-Entität→Absage; „Wie viele Kühe
hält…?"→Absage). **Aber** die zusätzliche Absage-Pflicht hat die Prior Richtung
„ablehnen" verschoben und **3 vorher korrekte Fälle gebrochen** (Lindau-Einwohner,
Bauernhof-Tiere-Liste, Berger-Zuordnung). Netto: **+4 gefixt, −3 kaputt → 77%→73%**.

**Schlussfolgerung:** LoRA-SFT-Daten-Iteration tauscht bei 0.9B Fehler ~1:1 statt sie
zu summieren. Die Methode hat ihr Plateau bei **~77%** erreicht. Weitere Gewinne
brauchen einen **stärkeren Hebel**, nicht mehr SFT-Daten-Runden.

## Was funktioniert (robust)
- **Welt-Wissen-Traps:** Kontext nennt Entität, Frage zielt auf bekanntes Faktum
  außerhalb des Texts → lehnt ab, **ergänzt nie aus dem Kopf** (0 Leaks).
- **Zahlen-Extraktion:** einfache + Zwei-Zahlen-Prosa (Distraktor-Zahl korrekt ignoriert).
- **Dichte 5-Satz-Prosa:** Gründungsjahr / Einwohner / „bekannt für" / Event-Monat.
- **Zeit:** Beginn/Ende, von-bis, ab/bis, seit-Jahr.
- **Dritt-Entität:** zwei genannte Personen, Frage nach einer dritten → Absage.
- **Mehrere ähnliche Entitäten (teilweise):** Schmidt/Schmitt, Anna/Anne, Kraus/Krause ✅.

## Die Decke (Restfehler bei 0.9B)
- **Harte Near-Duplicate-Namen:** Tom/Tim, Jonas/Jonah (Über-Ablehnung).
- **Positionale Zuordnung:** „linkes/rechtes Haus" (falsche Wahl).
- **Kompositionale Zeit:** „montags bis freitags … freitags?" (Über-Ablehnung).
- **Tail-Truncation:** „läuft noch fünf Wochen" → schneidet „fünf Wochen" ab.

Diese sind repräsentations-/decoding-limitiert, nicht daten-limitiert.

## Methode (reproduzierbar)
- **Deterministische Generierung** → garantiert-korrekte Labels (kein LLM-Judge).
- **Narrow-Embedding-EOS-Fix:** nur Spezial-Token-Rows 4–17 erhalten Gradient
  (LR 3e-5), Rest via LoRA r=64 — sonst kann das Modell `<|end|>` nicht emittieren.
- **54-Fälle-Stress-Gate, unverändert über v3/v4/v4.1** + 18 Generalisierungs-Fälle
  mit **anderen Specifics als das Training** → testet Generalisierung, nicht Auswendiglernen.
  Das Gate wurde **nie aufgeweicht**, um eine Version gut aussehen zu lassen.

## Artefakte
- Adapter (NICHT in git): `checkpoints/sft_grounded_v4/adapter_best.pt` (auf dem Server).
- Generatoren/Assembler/Gate: `scripts/sft/grounded/`.
- Basis: `step_60000` (Pretraining final), Mix grounded ~61% / corrective 33% / tool 6% / abstain 3%.

## Weg nach vorn (stärkerer Hebel, nicht mehr SFT-Runden)
1. **Grounded-/QA-Pretraining** statt nur LoRA-SFT (Extraktion früher verankern).
2. **Größeres Basismodell** für die Near-Duplicate-Distraktoren.
3. **Decoding-Hilfe** (constrained extraction / span-copy) gegen Truncation + Zuordnung.
4. H Grounded später mit (1)–(3) neu anfassen.
