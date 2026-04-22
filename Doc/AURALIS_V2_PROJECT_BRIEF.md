# Auralis v2 — Projekt-Einweisung für Claude Code

**Projekt:** Auralis (das System/der Assistent)
**Modell-Name:** Helix v2 (das LLM darunter)
**Maintainer:** Michael Speckels
**Start-Datum:** April 2026
**Vorgänger:** Auralis v1 / Helix v1 (v33/v35) — siehe PROJEKT_HISTORIE.md

---

## 1. Kontext & Motivation

Auralis v1 war der Prototyp der bewiesen hat dass das modulare Konzept
(kleines Basismodell + LoRAs + Tools) funktioniert. In 4 Wochen wurde
ein funktionierender 1.2B deutscher Chat-Assistent gebaut.

**Was aus v1 gelernt wurde (kritische Erkenntnisse):**

1. **Prompt-Format-Konsistenz ist kritisch** — ein einziger Bug (`<|user|>`
   statt `User:\n`) hat wochenlang die Inference-Qualität verschleiert.
   Lösung in v2: EIN Prompt-Builder für Training + Inference + Eval + API.

2. **LoRA kann Patterns, nicht immer Fakten** — Blutdruck-Adapter v1 hat
   bei 212 Samples Loss 0.0099 erreicht (Memorization). Neue Fragen
   scheiterten teilweise. Lösung: MoRA für Fakten, DoRA für Patterns,
   Val-Split mit disjunkten Fakten, Early-Stopping bei Val ~0.2-0.3.

3. **Tokenizer matters** — GPT-2 Tokenizer war ineffizient für Deutsch
   (~50% mehr Tokens als nötig). Lösung: eigener 200k Multilingual
   Tokenizer (EN/DE/Code).

4. **Daten-Mischung vor dem Training prüfen** — german-commons Cultural
   Subset dominierte → historischer Deutsch-Bias. Lösung: bewusste Mix-
   Ratios, Stichproben-Reviews vor Training.

5. **Baseline-Tests ab Tag 1** — ohne feste Eval-Questions keine ehrliche
   Progress-Messung. Lösung: 50 Baseline-Fragen ins Repo committed,
   automatisierte Ausführung bei jedem Checkpoint.

6. **Optimizer-State bewusst behandeln** — `--reset-optimizer` ist Standard,
   nicht Ausnahme. Drei Versionen (v20, v28, v30) wurden durch vergessene
   Resets verloren.

---

## 2. Architektur-Philosophie

**Leitprinzip:** Kleines, aber gutes Basismodell. Alles Spezifische kommt
über LoRAs und Tools. Das Gegenteil zu GPT-4's "alles in einem riesigen
Modell" — eher nach dem Vorbild des menschlichen Gehirns:

- **Thalamus** = Router-LoRA (entscheidet: einfach oder komplex?)
- **Broca/Wernicke** = Basismodell (Sprache)
- **Präfrontaler Kortex** = Denk-LoRA (Reasoning)
- **Logik-Zentrum** = Logik-LoRA (Selbstprüfung)
- **Hippocampus** = Memory-LoRA + JSON-Dict
- **Temporal-Lappen** = Topic-LoRAs (on-demand Fachwissen)
- **Cerebellum** = Autopilot-Patterns im Basismodell
- **Tools** = erweiterter Körper (Python, Web, Code)

**Fundamentaler Unterschied zu GPT-4:**

GPT-4 nutzt für "Hallo" die gleiche Rechenleistung wie für eine
komplexe Analyse. Helix v2 skaliert Compute mit Komplexität:

- Level 0 (Autopilot): <100ms, kein Denken
- Level 1 (einfach): <500ms, direkt antworten
- Level 2 (Spezialwissen): <2s, Topic-LoRA laden
- Level 3 (Reasoning): <5s, Denk-LoRA + Topic-LoRA
- Level 4 (Tools nötig): <10s, Python-Calls
- Level 5 (kritisch): <15s, alles + Self-Verification

---

## 3. Technische Spezifikation

### 3.1 Basismodell

```
Name:           Helix v2
Größe:          2-3B Parameter (dense, NICHT MoE für v2)
Architektur:    Heterogener Hybrid-Stack

Layer-Aufbau (28 Layers total):
  Layer 0-5:    Mamba-2 / SSM (lokaler Kontext, günstig)
  Layer 6-21:   GLA (Gated Linear Attention, Hauptkörper)
  Layer 22-27:  Sparse Attention (Needle-in-Haystack)

Dimensionen:
  d_model:      2048
  n_heads:      16
  d_head:       128
  d_ffn:        5632 (2.75x d_model)
  vocab_size:   200000

Modularität (von Anfang an eingeplant):
  → FFN-Blöcke als austauschbare Module
  → MoE-Router-Skelett vorbereitet (default: aus)
  → Multi-Token Prediction Heads optional (1-3)
  → Quantisierung-ready (AWQ-kompatibel)
```

### 3.2 Tokenizer

```
Name:           helix_v2_tokenizer
Typ:            SentencePiece Unigram
Vocab:          200000 (bilingual + code optimiert)

Daten-Mix:
  Englisch:     35 GB (60%)    — FineWeb-Edu
  Deutsch:      18 GB (30%)    — german-commons (gefiltert)
  Code:         7 GB (10%)     — The Stack v2

Aufteilung (ziel):
  80k:          Englisch-optimiert
  60k:          Deutsch-optimiert
  40k:          Code-Tokens
  20k:          Symbole, Zahlen, Unicode, Special Tokens

Special Tokens (von Anfang an):
  Chat:         <|system|>, <|user|>, <|assistant|>, <|end|>
  Reasoning:    <think>, </think>, <reflection>, </reflection>
  LoRA:         <lora>, </lora>, <route>, </route>
  Tools:        <tool>, </tool>, <tool_result>, </tool_result>
  Memory:       <memory>, </memory>, <recall>, </recall>
  Code:         <code>, </code>, <|python|>, <|javascript|>, <|rust|>
  MTP:          <|mtp_1|>, <|mtp_2|>, <|mtp_3|>

Ziel-Effizienz:
  Englisch:     ~130 tokens/100 words (state-of-art)
  Deutsch:      ~145 tokens/100 words (kompetitiv)
  Code:         ~160 tokens/100 words
```

### 3.3 Trainings-Strategie

**Phase 0: Tokenizer-Training (2-3 Tage)**

Voraussetzung für alles andere. Korpus zusammenstellen, SentencePiece
trainieren, Effizienz-Tests. Einmal richtig machen — Tokenizer-Wechsel
später = neues Pretraining.

**Phase 1: Englisch-Heavy Pretraining (3-4 Wochen, H200)**

```
Ziel:         Starke englische Basis (Weltwissen, Reasoning)
Mix:          75% EN + 20% DE + 5% Code
Tokens:       ~30-50B
LR:           3e-4 (AdamW oder Muon Optimizer)
Batch:        effective 256-512
Grund:        Mehr/bessere englische Daten verfügbar,
              Konzepte cross-lingual lernbar

Benchmarks alle 1000 Steps:
  English:    HellaSwag-subset, MMLU-subset
  German:     Belebele-DE, Okapi-DE
  Code:       HumanEval-subset
```

**Phase 2: Bilingual Continued Pretraining (1-2 Wochen)**

```
Ziel:         Deutsch-Skills ohne Englisch zu verlieren
Mix:          30% EN + 60% DE + 10% Code
Tokens:       ~10-15B
LR:           3e-5 (10x niedriger als Phase 1)

KRITISCH: KL-Distillation aktiv
  Teacher:    Phase-1-Checkpoint (frozen)
  Student:    Weitertrainierendes Modell
  Lambda:     0.5 (adaptive, siehe Eval-Hooks)
  Temperature: 3.0

Zusätzlich Replay Buffer:
  5-10% goldene Phase-1-Samples
  Verhindert dass Englisch vergessen wird

Monitoring:
  Wenn English-Retention < 90%: Lambda erhöhen
  Wenn Deutsch stagniert: Lambda senken
```

**Phase 3: SFT (1 Woche)**

```
Mix:          50% EN (Tülu 3) + 45% DE (eigene) + 5% Code
Samples:      100k-200k
Optimizer:    GaLore (volles Finetuning in LoRA-VRAM)
LR:           2e-5
Epochs:       2-3 (early stopping)

Daten-Quellen:
  Englisch:   Tülu 3, UltraChat 200k
  Deutsch:    DeepSeek V3 generated (wie Auralis v1),
              neu + qualitätsgefiltert
  Code:       Tülu-Code, MagiCoder
  Smalltalk:  Explizit generiert (Lücke aus v1!)
  Unsicherheit: Explizit generiert ("weiß ich nicht")
```

**Phase 4: Alignment via ORPO (3-5 Tage)**

```
Ziel:         Präferenz-Alignment ohne komplexes RLHF
Daten:        50k-100k Preference Pairs
  → Gleiche Frage, zwei Antworten
  → Gut (DeepSeek V3) vs. Schlechter (Llama-2 oder frühes Helix)
  → Automatisch generierbar
Methode:      ORPO (SFT + Alignment in einem Schritt)
LR:           1e-5
```

**Phase 5: LoRA-System aufbauen (2 Wochen)**

```
Meta-LoRAs (permanent aktiv):
  1. Router-LoRA — Komplexitäts-Level 0-5 Entscheidung
  2. Denk-LoRA — Chain-of-Thought, Problem-Zerlegung
  3. Logik-LoRA — Widerspruch-Erkennung, Selbstprüfung

Topic-LoRAs (on-demand):
  Medizin, Recht, Technik, Kochen, etc.
  Pro Topic: ~1000 Samples, disjunkter Val-Split,
  Early-Stopping bei Val ~0.2-0.3

LoRA-Methoden:
  → MoRA für Topic-LoRAs (Fakten-Lernen)
  → DoRA für Meta-LoRAs (Pattern-Lernen)

Nicht gleiche Lektion wie v1 wiederholen:
  ✗ Training ohne Val-Set
  ✗ Loss 0.0099 als Erfolg werten
  ✗ Nur gleiche-Formulierung Paraphrasen als Val
  ✓ Val mit disjunkten Fakten
  ✓ Early-Stopping bei Val-Loss-Plateau
  ✓ ~800-1500 Samples pro Topic minimum
```

### 3.4 Anti-Forgetting Strategie (KL-Distillation)

**Das Kernproblem:** Wenn man nach Englisch auf Deutsch continued
pretrained, vergisst das Modell Englisch (catastrophic forgetting).

**Die Lösung:** KL-Divergence Loss hält Student nah am Teacher.

```python
# Kern-Formel:
total_loss = task_loss + lambda * kl_loss(student_logits, teacher_logits)

# lambda = 0.5: balanciert
# Student lernt Neues (task_loss), bleibt aber dem Teacher ähnlich (kl_loss)
```

**Implementation (kurz):**

```python
class KLDistillationTrainer:
    def __init__(self, student, teacher_checkpoint_path, lambda_kd=0.5):
        self.student = student
        self.teacher = load_frozen(teacher_checkpoint_path)
        self.lambda_kd = lambda_kd
        self.temperature = 3.0
    
    def compute_loss(self, batch):
        # Student forward
        student_out = self.student(batch)
        task_loss = student_out.loss
        
        # Teacher forward (no gradients)
        with torch.no_grad():
            teacher_out = self.teacher(batch)
        
        # KL-Divergence mit Temperature
        kl_loss = F.kl_div(
            F.log_softmax(student_out.logits / self.temperature, dim=-1),
            F.softmax(teacher_out.logits / self.temperature, dim=-1),
            reduction='batchmean',
        ) * (self.temperature ** 2)
        
        return task_loss + self.lambda_kd * kl_loss
```

**Adaptive Lambda während Training:**

```
Alle 1000 Steps evaluieren:
  retention = english_score / teacher_english_score * 100
  
  if retention < 90%:
      lambda *= 1.5  # Bewahrung verstärken
  elif retention > 98%:
      lambda *= 0.7  # Mehr Lernen zulassen
```

---

## 4. Datenquellen

### 4.1 Pretraining-Daten

```
Englisch (Priorität):
  ✓ FineWeb-Edu (15T tokens, beste Qualität)
  ✓ The Stack v2 (900B tokens Code)
  ✓ Proof-Pile-2 (Mathe/Reasoning)
  ✓ Wikipedia EN

Deutsch:
  ✓ german-commons (18B, aber modern gefiltert: Perplexity < 500)
  ✓ Wikipedia DE (aktuell dump)
  ✓ FineWeb-Edu-DE (wenn verfügbar)
  ✗ NICHT: Gutenberg historisch (max 5%)

Code:
  ✓ The Stack v2 filtered (Top-10 Sprachen)
  ✓ GitHub Code Search API results
  ✓ Codeparrot

Filter-Regeln:
  → Duplikate: aggressiv (MinHash)
  → Perplexity: < 500 für Pretraining
  → Länge: 200-50000 Zeichen
  → Sprache: fastText-Identification
  → Toxicity: Standard-Filter
```

### 4.2 SFT-Daten

```
Englisch:
  → Tülu 3 (1M+ SFT pairs, curated)
  → UltraChat 200k
  → OpenMathInstruct
  → HelpSteer2

Deutsch:
  → Eigene Generation via DeepSeek V3
    (aus Auralis v1 bewährt: 150K+ Samples in 4 Wochen)
  → Gemini 2.0 Flash für breite Themen
  → Tülu 3 Übersetzungen
  → Mayflower (ABER: stark filtern, war v1-Problem)

Code:
  → Tülu-Code
  → MagiCoder
  → OpenCodeInterpreter

Spezial-Kategorien (Lücken aus v1 schließen):
  → Smalltalk (~2000 pairs)
  → Unsicherheit / "Weiß ich nicht" (~1000 pairs)
  → Multi-turn Dialoge (~5000 pairs)
  → Fakten-Q&A (~20000 pairs)
```

### 4.3 LoRA-Training-Daten

```
Pro Topic-Adapter:
  Kern-Fakten:  100 atomare Wissenseinheiten (YAML-Spec)
    → Quellenangabe PFLICHT (Leitlinien, Fachinfo, etc.)
    → source_quote Feld für Verifikation
    → confidence: high/medium/unsure
  
  Generation:   80 Train-Fakten × 3 Paraphrasen = 240 Train-Samples
                20 Val-Fakten × 1 Formulierung = 20 Val-Samples
                (Val-Fakten DISJUNKT zu Train!)
  
  Erweiterung: +500-750 kontextuelle/Fall-Beispiele → ~1000 total
  
  Kategorien (Beispiel Medizin):
    A. Definitionen (~150 samples)
    B. Zahlen-Interpretation (~250)
    C. Medikamente (~200)
    D. Symptome & Ursachen (~150)
    E. Lifestyle/Ernährung (~100)
    F. Fallbeispiele (~100)
    G. Grenzen/Unsicherheit (~50)
```

---

## 5. Infrastruktur-Anforderungen

### 5.1 Hardware

```
Pretraining:
  GPU:        H200 (143GB) oder H100 (80GB)
  Dauer:      4-6 Wochen
  Kosten:     ~$500-800 auf RunPod

SFT + LoRA:
  GPU:        RTX Pro 5000 (48GB) oder H200
  Dauer:      2-3 Wochen total
  Kosten:     ~$100-200

Local Development:
  GPU:        RTX 3090 (24GB)
  Nutzung:    LoRA-Training, Inference-Tests, kleine Experimente
```

### 5.2 Software-Stack

```
Training Framework:
  → PyTorch 2.5+ (Basis)
  → Unsloth (2-5x schnelleres Finetuning)
  → TRL (für ORPO, DPO, KTO)
  → Flash Attention 3
  → DeepSpeed ZeRO-3 (falls Multi-GPU)

Tokenizer:
  → SentencePiece (Training)
  → tiktoken-kompatibler Wrapper (Inference)

Inference:
  → vLLM (Production, PagedAttention)
  → llama.cpp + GGUF (lokale Deployment)
  → AWQ 4-bit Quantisierung

Monitoring:
  → WandB oder eigenes Dashboard
  → Custom Baseline-Eval Script
  → Prometheus + Grafana (optional)

LoRA/Adapter:
  → MoRA implementation (eigene oder community)
  → DoRA implementation (wie Auralis v1)
  → GaLore (für Basis-Finetuning)
```

### 5.3 Datenmanagement

```
Struktur:
  /data/
    raw/              # Originaldaten, NICHT ändern
    cleaned/          # Validiert, dedupliziert
    training/         # Final, versioniert
    eval/             # Benchmark-Daten, stabil
    
  /checkpoints/
    phase1_pretrain/
    phase2_continued/
    phase3_sft/
    phase4_aligned/
    lora_adapters/
      meta/           # Router, Denk, Logik
      topics/         # Medizin, Recht, etc.
  
  /configs/
    *.yaml            # Alle Hyperparameter hier
  
  /scripts/
    tokenizer/
    pretrain/
    sft/
    lora/
    eval/

Wichtig:
  → Jeder Run produziert eine MANIFEST.yaml
  → Config + Git-Hash + Daten-Hash + Metrics
  → Reproduzierbarkeit garantiert
```

---

## 6. Qualitäts-Sicherung

### 6.1 Baseline-Test (Tag 1!)

50 feste Testfragen committen, die bei JEDEM Checkpoint durchlaufen:

```yaml
# eval/baseline_questions.yaml
questions:
  - id: geo_001
    category: geography
    question: "Was ist die Hauptstadt von Frankreich?"
    expected_keywords: ["Paris"]
    language: de
  
  - id: math_001
    category: math
    question: "Was ist 15 × 7?"
    expected: "105"
    tool_required: true
  
  # ... 48 weitere
```

```python
# scripts/eval/run_baseline.py
def evaluate_checkpoint(checkpoint_path):
    model = load_model(checkpoint_path)
    results = []
    for q in load_baseline_questions():
        answer = generate(model, q.question)
        score = score_answer(answer, q.expected_keywords)
        results.append({"id": q.id, "score": score})
    
    save_results(f"eval/results/{checkpoint_name}.json", results)
    return aggregate_score(results)
```

**Nach JEDEM Training-Run: Baseline laufen lassen, Regression prüfen.**

### 6.2 Prompt-Builder Test (kritisch!)

```python
# tests/test_prompt_builder.py
def test_training_inference_identical():
    """
    Verhindert den v1-Bug: Training + Inference müssen
    byte-weise identischen Input bauen.
    """
    messages = [
        {"role": "system", "content": "Du bist Helix."},
        {"role": "user", "content": "Hallo"},
    ]
    
    training_prompt = build_training_prompt(messages)
    inference_prompt = build_inference_prompt(messages)
    
    assert training_prompt == inference_prompt, (
        f"Training != Inference!\n"
        f"Training:  {repr(training_prompt)}\n"
        f"Inference: {repr(inference_prompt)}"
    )
    
    # Tokenisierung muss auch identisch sein
    tokens_t = tokenizer.encode(training_prompt)
    tokens_i = tokenizer.encode(inference_prompt)
    assert tokens_t == tokens_i
```

### 6.3 Preflight-Checklist

Vor jedem Training-Run durchgehen:

```
□ Config-File existiert und ist valid YAML
□ Checkpoint-Pfad ist korrekt
□ --reset-optimizer gesetzt (bei SFT!)
□ Val-Files existieren und sind DISJUNKT zu Train
□ Variant stimmt mit Pretrain überein
□ Daten-Pfade absolut (keine relativen)
□ Guthaben auf RunPod > geschätzte Kosten × 1.5
□ Baseline-Fragen neu durchgelaufen auf Start-Checkpoint
□ Git-Commit von Config + Scripts
□ MANIFEST.yaml mit Run-Details angelegt
□ Monitoring-Dashboard erreichbar
□ Auto-Refill auf RunPod aktiv
```

### 6.4 Monitoring während Training

```
Alle 100 Steps:
  → train_loss
  → grad_norm
  → learning_rate
  → tokens/second

Alle 1000 Steps:
  → val_loss
  → Baseline-Fragen Score (subset, 10 Fragen schnell)
  → VRAM-Usage
  → Disk-Usage

Alle 5000 Steps:
  → Vollständiger Baseline-Test (50 Fragen)
  → Cross-Lingual Retention (EN + DE + Code)
  → Checkpoint speichern

Alarme:
  → Val-Loss steigt 3 Evals in Folge
  → Baseline-Score fällt um > 5%
  → VRAM > 90%
  → Disk > 80%
```

---

## 7. Projekt-Phasen & Timeline

```
Monat 1: Vorbereitung
  Woche 1:    Tokenizer-Korpus vorbereiten
  Woche 2:    Tokenizer trainieren + testen
  Woche 3:    Baseline-Questions erstellen + Infrastruktur
  Woche 4:    Modell-Architektur implementieren + Tests

Monat 2: Pretraining Phase 1 (Englisch-Heavy)
  Woche 1-3:  Continuous pretrain auf H200
  Woche 4:    Evaluation + Baseline

Monat 3: Continued Pretrain + SFT
  Woche 1-2:  Phase 2 (Bilingual) mit KL-Distillation
  Woche 3:    Phase 3 (SFT) mit GaLore
  Woche 4:    Phase 4 (ORPO Alignment)

Monat 4: LoRA-System + Deployment
  Woche 1-2:  Meta-LoRAs (Router, Denk, Logik)
  Woche 3:    Erste Topic-LoRAs (Medizin, Recht)
  Woche 4:    vLLM Deployment + API + Open WebUI

Monat 5: Optimierung + Erweiterung
  → AWQ 4-bit Quantisierung
  → Triton Kernels für Speed
  → Weitere Topic-LoRAs
  → User-Testing + Feedback-Loop

Total: 5 Monate, ~$1000-1500 Kosten
```

---

## 8. Arbeits-Anweisungen für Claude Code

### 8.1 Grundregeln

1. **Alles modular** — keine Hardcoded-Werte, alles über Config-Files
2. **Type Hints überall** — Python 3.11+ Syntax
3. **Docstrings für jede Funktion** — Args, Returns, Examples
4. **Tests für kritische Komponenten** — besonders Prompt-Builder, Tokenizer
5. **Git-Commit nach jeder funktionalen Einheit** — atomare Commits
6. **MANIFEST.yaml pro Experiment** — Reproduzierbarkeit
7. **Kein "schnell mal was" machen** — immer sauber, immer dokumentiert

### 8.2 Reihenfolge der Implementation

```
1. Projekt-Struktur aufsetzen
   → /data, /scripts, /configs, /tests, /docs
   → pyproject.toml mit allen Dependencies
   → Git-Repo initialisieren
   → .gitignore für Checkpoints, Daten

2. Baseline-Questions YAML definieren (50 Fragen)
   → Kategorien: Geo, Math, Science, History, Language, Code
   → Je Sprache: EN + DE
   → Check-Logik (Keywords, exakte Matches, etc.)

3. Prompt-Builder + Tokenizer-Tests schreiben
   → Unit-Tests die v1-Bug verhindern
   → Training = Inference = Eval = API

4. Tokenizer-Training implementieren
   → Daten-Download Scripts
   → SentencePiece Training Script
   → Qualitäts-Test Script

5. Modell-Architektur implementieren
   → Modulare Config (YAML)
   → Layer-Typen (Mamba, GLA, Sparse Attention)
   → FFN-Module (Dense + MoE-Ready)
   → Tests auf kleinem Modell (100M)

6. Pretraining-Pipeline
   → Daten-Mixing (streaming, deterministic)
   → Training Loop mit Checkpoints
   → Metriken & Monitoring
   → Preflight-Checker

7. KL-Distillation implementieren
   → Teacher Loading
   → KL-Loss mit Temperature
   → Adaptive Lambda
   → Retention-Monitoring

8. SFT-Pipeline (mit GaLore)
9. ORPO Pipeline
10. LoRA-System (MoRA + DoRA + GaLore-Hybrid)
11. Inference Pipeline (vLLM)
12. FastAPI + Open WebUI Integration
```

### 8.3 Code-Style

```python
# GUT: Modular, typed, documented
from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class TrainingConfig:
    """Konfiguration für Pretraining-Run.
    
    Attributes:
        model_size: Parameter-Anzahl (z.B. 3e9 für 3B)
        batch_size: Effektive Batch-Größe nach grad_accum
        learning_rate: Peak LR für Cosine Schedule
    """
    model_size: float
    batch_size: int
    learning_rate: float
    # ... mit Defaults wo sinnvoll


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    config: TrainingConfig,
) -> dict[str, float]:
    """Trainiert eine Epoche und returnt Metrics.
    
    Args:
        model: Das zu trainierende Modell
        dataloader: Training-Daten
        config: Hyperparameter
    
    Returns:
        Dict mit 'loss', 'grad_norm', 'tokens_per_second'
    """
    ...


# SCHLECHT: Globale Variablen, keine Types, keine Docs
model_path = "some/path"

def train(data):  # was ist data?
    # ... 200 Zeilen ...
    pass
```

### 8.4 Git-Commit-Format

```
<type>(<scope>): <subject>

<body>

<footer>

Beispiele:
  feat(tokenizer): implement SentencePiece training pipeline
  fix(prompt): unify training/inference prompt builders
  refactor(training): extract KL-distillation to separate module
  test(prompt): add byte-wise comparison test for builders
  docs(readme): document phase 1 pretrain configuration
```

### 8.5 Wichtige Anti-Patterns (aus v1 gelernt)

```
✗ NICHT: Adapter trainieren ohne Val-Split
✗ NICHT: Loss unter 0.1 als Erfolg feiern
✗ NICHT: Val aus Train-Subset splitten (muss disjunkt sein)
✗ NICHT: Prompt-Format in mehreren Files duplizieren
✗ NICHT: Hyperparameter hardcoden (alles in Config)
✗ NICHT: "schnell mal ohne Baseline-Test" deployen
✗ NICHT: Optimizer-State beim SFT laden (immer --reset-optimizer)
✗ NICHT: Alte Daten im neuen Run wiederverwenden ohne Re-Validierung
✗ NICHT: Viele Markdown-Dokumente mit Überlappung erstellen

✓ STATT: Val mit disjunkten Fakten, Early-Stopping bei Plateau
✓ STATT: Loss 0.2-0.3 ist Ziel (echtes Lernen, keine Memorization)
✓ STATT: Val + 50 Baseline-Fragen + Retention-Check kombiniert
✓ STATT: EIN Prompt-Builder, used everywhere
✓ STATT: Alle Hyperparams in configs/*.yaml
✓ STATT: Baseline IMMER, egal wie "kleines" Experiment
✓ STATT: --reset-optimizer in jedem SFT-Script
✓ STATT: Daten-Hash im MANIFEST, Re-Validierung bei Reuse
✓ STATT: EIN Master-STATUS.md, archivierte Details in /docs/archive/
```

---

## 9. Erfolgskriterien

**Helix v2 ist "fertig" wenn:**

```
Pretraining:
  ✓ English MMLU > 45 (gut für 3B)
  ✓ German eval > 55 (sehr gut für deutsches Modell)
  ✓ HumanEval > 20 (basic Coding)

SFT + Alignment:
  ✓ 50 Baseline-Fragen: > 80% korrekt
  ✓ Photosynthese-Test (v1-Baseline): korrekt
  ✓ Smalltalk: natürlich, nicht halluzinierend
  ✓ Unsicherheit: sagt "weiß ich nicht" statt zu raten

LoRA-System:
  ✓ Router-LoRA: korrektes Level 0-5 Routing in > 90%
  ✓ Topic-LoRA Blutdruck v2: > 70% auf disjunkten Val-Fakten
  ✓ Hot-Swap funktioniert, keine Crashes
  ✓ Intent-basiertes Auto-Routing funktioniert

Deployment:
  ✓ vLLM Inference läuft stabil
  ✓ OpenAI-kompatible API
  ✓ 4-bit Quantisierung funktioniert
  ✓ Läuft auf RTX 3090 (Consumer-GPU)
  ✓ Open WebUI Integration
```

---

## 10. Besonderheiten & Konstraints

### 10.1 Sprachen-Regeln

- **Code/Kommentare im Repo:** Englisch
- **Dokumentation:** Deutsch (außer API-Docs)
- **Git-Commits:** Englisch
- **Variablen/Funktionen:** Englisch
- **Konfig-Keys:** Englisch
- **User-facing Output:** je nach User-Sprache (deutsch-first)

### 10.2 Datenschutz

- Keine Training-Daten mit PII (Personally Identifiable Information)
- DSGVO-konform: keine Namen, Adressen, Telefonnummern in Daten
- Gesundheitsdaten: nur anonymisierte/synthetische
- User-Conversations: optional, explizites Opt-in

### 10.3 Ethik

- Keine Erzeugung von Malware, Exploits, Waffen-Anleitungen
- Keine Deepfakes von real existierenden Personen
- Medizinische Beratung: immer Hinweis "kein Arzt"
- Rechtliche Beratung: immer Hinweis "kein Anwalt"
- Transparenz: Modell soll wissen was es ist (KI-Assistent)

### 10.4 Lizenzierung

- Training-Code: MIT oder Apache 2.0 (open)
- Modell-Gewichte: zu entscheiden (open weights vs. proprietary)
- Trainings-Daten: alle License-kompatibel verwenden
- Dependencies: License-Check pro Library

---

## 11. Kommunikation

**Bei Fragen/Unsicherheit:**
- Immer Rückfrage stellen statt raten
- Alternativen mit Trade-offs präsentieren
- Bei kritischen Entscheidungen: User-Approval einholen

**Bei Fehlern:**
- Nicht verstecken, direkt ansprechen
- Root-Cause analysieren, nicht nur Symptom-Fix
- Lesson learned in LESSONS.md dokumentieren

**Bei Erfolgen:**
- Kurz feiern, dann nächster Schritt
- Benchmark ins Repo committen
- Progress in STATUS.md aktualisieren

**Dokumentations-Regel:**
- EIN aktives STATUS.md (der aktuelle Stand)
- EIN LESSONS.md (Append-Only, beste Erkenntnisse)
- EIN HISTORY.md (chronologisch, Milestones)
- Alle anderen Docs: /docs/archive/

---

## 12. Ressourcen & Referenzen

**Forschungspapers die die v2-Architektur inspirieren:**
- Mamba-2: State Space Duality
- Gated Linear Attention (GLA)
- Mixture of Experts (Mixtral)
- DoRA: Weight-Decomposed LoRA
- MoRA: Matrix-Rank Adaptation
- GaLore: Gradient Low-Rank Projection
- ORPO: Monolithic Preference Optimization
- KL-Distillation for Continual Learning

**Open Source Referenzen:**
- Llama 3 (Meta) — Architektur-Referenz
- Gemma 3 (Google) — Multilingual Tokenizer
- DeepSeek V3 — Moderne Training-Tricks
- Mistral 7B — Effiziente Inference

**Auralis v1 Dokumentation:**
- PROJEKT_HISTORIE.md — Was lief gut/schlecht
- 24 dokumentierte Bugs — was NICHT wiederholen
- helix_api.py — API-Referenz (funktioniert!)

---

## 13. Start-Checkliste

**Bevor du anfängst zu programmieren:**

```
□ Diese Einweisung vollständig gelesen
□ Auralis v1 Historie gelesen (PROJEKT_HISTORIE.md)
□ Auralis v1 Architecture Spec gelesen (AURALIS_Architecture_Spec.md)
□ Mit Michael kurz abgestimmt welche Phase zuerst
□ Git-Repo initialisiert
□ Python 3.11+ venv aktiv
□ CUDA 12+ verfügbar (oder Pod gemietet)
□ WandB Account (optional) oder eigenes Monitoring aufgesetzt
□ Erste todo-Liste erstellt
□ MANIFEST.yaml Template angelegt
```

**Los geht's!**

---

*Dieses Dokument ist die Master-Einweisung für Auralis v2. Alle Details,
Code-Beispiele, und spezifische Implementationen werden in begleitenden
Dokumenten (SPEC_*.md, IMPLEMENTATION_*.md) ausgearbeitet.*

*Version: 1.0 — April 2026*
*Autor: Auralis Team (Michael + Claude)*
