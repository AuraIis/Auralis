# Blueprint — Domänen-Adapter (DoRA: Mathe / Logik / Code) (Helix v2)

> **Status:** Design / beschlossen-mit-Vorbehalt (Michael + GPT + Claude, Juni 2026). Nicht implementiert.
> **Phase:** **NACH** Tool-Use **und** NACH Code-Annealing. Siehe `BLUEPRINT_TOOL_USE_VERIFIER.md`
> (Stufe 5) und `ZUKUNFT_BACKLOG.md` Phase 3–5.
> **Kernprinzip (nicht verhandelbar):**
> *Ein Adapter verstärkt latente Fähigkeit — er installiert keine neue.*

---

## 1) Idee
Hauptmodell **einfrieren**, pro Domäne einen kleinen Adapter trainieren, der die *Fähigkeit*
(nicht das Wissen) verstärkt:
- `dora:math` — Rechenweg-Struktur, Schritt-für-Schritt
- `dora:logic` — Schlussfolgern, Fallunterscheidung
- `dora:code` — Code-Struktur, Tests, Reparatur-Muster

Adapter werden separat trainiert und je nach Aufgabe geladen/getauscht (oder geroutet).

## 2) Warum DoRA — und nicht LoRA oder MoRA
Aus den v1-Lektionen (**L-002**): *LoRA lernte Muster, keine Fakten.* Daraus die saubere Trennung:
| Technik | wofür | Helix-Einsatz |
|---|---|---|
| **DoRA** (weight-decomposed) | **Muster / Skills** | Mathe-Rechenweg, Code-Struktur, Logik → **hier richtig** |
| **MoRA** (high-rank) | **Fakten / Wissen** | Wissens-Injektion (separater Pfad, nicht dieses Dokument) |

Mathe-/Logik-/Code-Können ist ein **Skill-Muster** → DoRA passt. Fakten würden MoRA brauchen.

## 3) Der Haken — Reihenfolge ist Pflicht, nicht Stil
Ein Adapter dreht an Gewichten, die **schon da** sind. Fehlt die Fähigkeit im Base *latent*,
hat der Adapter nichts zum Verstärken.

| Adapter | latente Basis im aktuellen Base? | Verdikt |
|---|---|---|
| `dora:code` | **0 % echter Code im Pretraining** (nur Code-als-Prosa) | **gesperrt** bis Code-Annealing — sonst auf Sand gebaut |
| `dora:math` | dünn vorhanden (war im Mix) | moderat verstärkbar, lohnt eher nach Annealing |
| `dora:logic` | dünn vorhanden | moderat verstärkbar |

→ **Code-Annealing (Python-Edu liegt bereit) ist die Voraussetzung für `dora:code`.**
Reihenfolge umgedreht = verbrannte Zeit.

## 4) Verhältnis zu Tool-Use (wichtig — nicht konkurrierend)
DoRA **ersetzt den Verifier nicht.** Tool-Use macht Antworten *verifizierbar* (Korrektheit von
außen); DoRA macht das Modell *flüssiger/strukturierter* in der Domäne. Reihenfolge:
1. **Tool-Use zuerst** (höchster Hebel bei 0,9B, Korrektheit von außen).
2. **Annealing** (latente Fähigkeit in den Base).
3. **DoRA danach** (verstärkt das Latente, inkl. „ruft Tool sauber").

Ein DoRA-Mathe-Adapter sollte idealerweise das **Tool-Use-Verhalten mitlernen**, nicht
Kopfrechnen ersetzen wollen.

## 5) Targeting auf der Hybrid-Architektur (Implementierungs-Kern)
Helix ist kein reines Transformer-Stack. DoRA hängt an **Linear-Projektionen** — die existieren
in allen drei Layer-Typen, müssen aber im Trainer verdrahtet werden:
| Layer-Typ (Anzahl) | adaptierbare Linears |
|---|---|
| Mamba-2 (6×) | `in_proj`, `out_proj` (ggf. `x_proj`/`dt_proj`) |
| GLA (16×) | `q_proj`, `k_proj`, `v_proj`, `g_proj`, `out_proj` |
| Sparse-Attn (6×) | `q/k/v/o_proj` |
| MLP (SwiGLU, alle) | `gate/up/down_proj` |

**Offene Design-Entscheidung:** alle Projektionen targeten (max. Kapazität, mehr Params) vs.
nur Attention/GLA-Projektionen (üblich, sparsamer). **Erst Ablation, dann Default.** Embedding
(tied 200k) bleibt eingefroren.

## 6) Trainingsrezept (Startwerte, zu kalibrieren)
```
base:        eingefroren (kein grad)
adapter:     DoRA, rank r ∈ {8,16,32} (ablatieren), alpha = 2r
target:      siehe §5 (Default-Entscheidung via Ablation)
lr:          ~1e-4 (höher als Full-SFT, da nur Adapter lernt)
data/adapter: domänenrein (math-only / logic-only / code-only), je 5–20k
              — math/logic aus unserem Reasoning-Slice ableitbar
eval:        disjunkte Val-Splits pro Domäne (L-002: Memorization-Falle vermeiden)
             + Negativ-Guard: allgemeine Benchmarks dürfen NICHT fallen (Adapter aus = Baseline)
```

## 7) Multi-Adapter zur Laufzeit
- **Variante A (einfach, zuerst):** expliziter Modus — User/Caller wählt Adapter
  (`--adapter math`). Kein Router-Risiko.
- **Variante B (später):** kleiner Router/Klassifikator wählt Adapter aus der Frage.
  Eigene Fehlerquelle (falscher Adapter) → erst wenn A steht und sich lohnt.
- Adapter sind klein → mehrere im RAM, schnelles Tauschen; **nicht** in den Base mergen
  (sonst Modularität weg).

## 8) Ehrliche Decke bei 0,9B
- DoRA hebt v.a. **Form/Flüssigkeit** in der Domäne — kein Sprung auf „starkes Reasoning".
- Effekt **gated auf Base-Qualität**: vor Annealing klein, danach größer.
- Kein Ersatz für Skalierung. DoRA ist Feinschliff, nicht Fundament.
- Realistische Erwartung: messbarer, *moderater* Lift auf Domänen-Benchmarks bei gehaltenem
  Allgemein-Niveau — nicht mehr, nicht weniger. „Erst messen, dann entscheiden."

## 9) Erfolgskriterien
- Domänen-Benchmark (z.B. mmlu_de Mathe-Slice / GSM8K-de / Logik-Probe) **mit Adapter > ohne**.
- Allgemein-Benchmarks (de/en MMLU/ARC/HellaSwag) mit Adapter **≥** ohne (kein Allgemein-Verlust).
- Adapter-Größe ≪ Base; Laden/Tauschen < 1 s.

## 10) Voraussetzungen (Gates) — vor Start abhaken
- [ ] Tool-Use Stufe 1–2 grün (Mathe-Tool beweist Harness)
- [ ] Code-Annealing gelaufen (für `dora:code`) — `math`/`logic` ggf. früher
- [ ] DoRA-Targeting im v2-Trainer verdrahtet (§5) + Ablation rank/target
- [ ] disjunkte Domänen-Val-Splits gebaut (L-002)
- [ ] Negativ-Guard-Eval (Allgemein-Benchmarks) als Pflicht-Gate eingerichtet

---
*Reihenfolge gegated. `dora:code` ohne Code-Annealing = §1-Kernprinzip verletzt = auf Sand bauen.
Erinnerung 500M-Sackgasse: Schicht vor Fundament = Müll.*
