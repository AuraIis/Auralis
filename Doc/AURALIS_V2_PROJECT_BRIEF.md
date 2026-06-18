# Auralis v2 — Project Briefing for Claude Code

> Current note (2026-05-17): This is the original master brief and still the
> best source for the Auralis/Helix idea. Some concrete numbers and schedules
> are historical. For the current run state, use `STATUS.md`.

**Project:** Auralis (the system / the assistant)
**Model name:** Helix v2 (the LLM underneath)
**Maintainer:** Michael Speckels
**Start date:** April 2026
**Predecessor:** Auralis v1 / Helix v1 (v33/v35) — see PROJEKT_HISTORIE.md

---

## 1. Context & Motivation

Auralis v1 was the prototype that proved the modular concept
(small base model + LoRAs + tools) works. In 4 weeks a working
1.2B German chat assistant was built.

**What was learned from v1 (critical insights):**

1. **Prompt-format consistency is critical** — a single bug (`<|user|>`
   instead of `User:\n`) obscured inference quality for weeks.
   Solution in v2: ONE prompt builder for training + inference + eval + API.

2. **LoRA can learn patterns, not always facts** — the blood-pressure adapter v1
   reached loss 0.0099 on 212 samples (memorization). New questions
   partly failed. Solution: MoRA for facts, DoRA for patterns,
   val split with disjoint facts, early stopping at val ~0.2-0.3.

3. **Tokenizer matters** — the GPT-2 tokenizer was inefficient for German
   (~50% more tokens than necessary). Solution: own 200k multilingual
   tokenizer (EN/DE/Code).

4. **Check the data mix before training** — the german-commons cultural
   subset dominated → historical German bias. Solution: deliberate mix
   ratios, sample reviews before training.

5. **Baseline tests from day 1** — without fixed eval questions there is no honest
   progress measurement. Solution: 50 baseline questions committed into the repo,
   automated execution at every checkpoint.

6. **Handle optimizer state deliberately** — `--reset-optimizer` is the standard,
   not the exception. Three versions (v20, v28, v30) were lost due to forgotten
   resets.

---

## 2. Architecture Philosophy

**Guiding principle:** small but good base model. Everything specific comes
via LoRAs and tools. The opposite of GPT-4's "everything in one giant
model" — rather modeled on the human brain:

- **Thalamus** = router LoRA (decides: simple or complex?)
- **Broca/Wernicke** = base model (language)
- **Prefrontal cortex** = thinking LoRA (reasoning)
- **Logic center** = logic LoRA (self-checking)
- **Hippocampus** = memory LoRA + JSON dict
- **Temporal lobe** = topic LoRAs (on-demand domain knowledge)
- **Cerebellum** = autopilot patterns in the base model
- **Tools** = extended body (Python, Web, Code)

**Fundamental difference from GPT-4:**

GPT-4 uses the same compute for "Hello" as for a
complex analysis. Helix v2 scales compute with complexity:

- Level 0 (autopilot): <100ms, no thinking
- Level 1 (simple): <500ms, answer directly
- Level 2 (specialized knowledge): <2s, load topic LoRA
- Level 3 (reasoning): <5s, thinking LoRA + topic LoRA
- Level 4 (tools needed): <10s, Python calls
- Level 5 (critical): <15s, everything + self-verification

---

## 3. Technical Specification

### 3.1 Base model

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

### 3.3 Training strategy

**Phase 0: Tokenizer training (2-3 days)**

A prerequisite for everything else. Assemble the corpus, train SentencePiece,
efficiency tests. Do it right once — a later tokenizer swap = new pretraining.

**Phase 1: English-heavy pretraining (3-4 weeks, H200)**

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

**Phase 2: Bilingual continued pretraining (1-2 weeks)**

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

**Phase 3: SFT (1 week)**

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

**Phase 4: Alignment via ORPO (3-5 days)**

```
Ziel:         Präferenz-Alignment ohne komplexes RLHF
Daten:        50k-100k Preference Pairs
  → Gleiche Frage, zwei Antworten
  → Gut (DeepSeek V3) vs. Schlechter (Llama-2 oder frühes Helix)
  → Automatisch generierbar
Methode:      ORPO (SFT + Alignment in einem Schritt)
LR:           1e-5
```

**Phase 5: Build up the LoRA system (2 weeks)**

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

### 3.4 Anti-forgetting strategy (KL distillation)

**The core problem:** when you continued-pretrain on German after English,
the model forgets English (catastrophic forgetting).

**The solution:** KL-divergence loss keeps the student close to the teacher.

```python
# Kern-Formel:
total_loss = task_loss + lambda * kl_loss(student_logits, teacher_logits)

# lambda = 0.5: balanciert
# Student lernt Neues (task_loss), bleibt aber dem Teacher ähnlich (kl_loss)
```

**Implementation (brief):**

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

**Adaptive lambda during training:**

```
Alle 1000 Steps evaluieren:
  retention = english_score / teacher_english_score * 100
  
  if retention < 90%:
      lambda *= 1.5  # Bewahrung verstärken
  elif retention > 98%:
      lambda *= 0.7  # Mehr Lernen zulassen
```

---

## 4. Data Sources

### 4.1 Pretraining data

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

### 4.2 SFT data

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

### 4.3 LoRA training data

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

## 5. Infrastructure Requirements

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

### 5.2 Software stack

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

### 5.3 Data management

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

## 6. Quality Assurance

### 6.1 Baseline test (day 1!)

Commit 50 fixed test questions that run at EVERY checkpoint:

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

**After EVERY training run: run the baseline, check for regression.**

### 6.2 Prompt-builder test (critical!)

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

### 6.3 Preflight checklist

Go through this before every training run:

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

### 6.4 Monitoring during training

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

## 7. Project Phases & Timeline

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

## 8. Working Instructions for Claude Code

### 8.1 Ground rules

1. **Everything modular** — no hardcoded values, everything via config files
2. **Type hints everywhere** — Python 3.11+ syntax
3. **Docstrings for every function** — args, returns, examples
4. **Tests for critical components** — especially prompt builder, tokenizer
5. **Git commit after every functional unit** — atomic commits
6. **MANIFEST.yaml per experiment** — reproducibility
7. **No "quick hacks"** — always clean, always documented

### 8.2 Order of implementation

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

### 8.3 Code style

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

### 8.4 Git commit format

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

### 8.5 Important anti-patterns (learned from v1)

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

## 9. Success Criteria

**Helix v2 is "done" when:**

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

## 10. Specifics & Constraints

### 10.1 Language rules

- **Code/comments in the repo:** English
- **Documentation:** German (except API docs)
- **Git commits:** English
- **Variables/functions:** English
- **Config keys:** English
- **User-facing output:** depending on the user's language (German-first)

### 10.2 Data protection

- No training data with PII (Personally Identifiable Information)
- GDPR-compliant: no names, addresses, phone numbers in the data
- Health data: only anonymized/synthetic
- User conversations: optional, explicit opt-in

### 10.3 Ethics

- No generation of malware, exploits, weapons instructions
- No deepfakes of real existing persons
- Medical advice: always include the note "not a doctor"
- Legal advice: always include the note "not a lawyer"
- Transparency: the model should know what it is (AI assistant)

### 10.4 Licensing

- Training code: MIT or Apache 2.0 (open)
- Model weights: to be decided (open weights vs. proprietary)
- Training data: use all license-compatible
- Dependencies: license check per library

---

## 11. Communication

**On questions/uncertainty:**
- Always ask back instead of guessing
- Present alternatives with trade-offs
- For critical decisions: obtain user approval

**On errors:**
- Don't hide, address directly
- Analyze root cause, not just a symptom fix
- Document the lesson learned in LESSONS.md

**On successes:**
- Celebrate briefly, then the next step
- Commit the benchmark into the repo
- Update progress in STATUS.md

**Documentation rule:**
- ONE active STATUS.md (the current state)
- ONE LESSONS.md (append-only, best insights)
- ONE HISTORY.md (chronological, milestones)
- All other docs: /docs/archive/

---

## 12. Resources & References

**Research papers inspiring the v2 architecture:**
- Mamba-2: State Space Duality
- Gated Linear Attention (GLA)
- Mixture of Experts (Mixtral)
- DoRA: Weight-Decomposed LoRA
- MoRA: Matrix-Rank Adaptation
- GaLore: Gradient Low-Rank Projection
- ORPO: Monolithic Preference Optimization
- KL-Distillation for Continual Learning

**Open source references:**
- Llama 3 (Meta) — architecture reference
- Gemma 3 (Google) — multilingual tokenizer
- DeepSeek V3 — modern training tricks
- Mistral 7B — efficient inference

**Auralis v1 documentation:**
- PROJEKT_HISTORIE.md — what went well/badly
- 24 documented bugs — what NOT to repeat
- helix_api.py — API reference (works!)

---

## 13. Start Checklist

**Before you start programming:**

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

**Let's go!**

---

*This document is the master briefing for Auralis v2. All details,
code examples, and specific implementations are elaborated in accompanying
documents (SPEC_*.md, IMPLEMENTATION_*.md).*

*Version: 1.0 — April 2026*
*Author: Auralis Team (Michael + Claude)*
