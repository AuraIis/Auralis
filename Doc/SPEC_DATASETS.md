# Auralis v2 — Dataset Specification

**Projekt:** Auralis v2 / Helix v2 (1B Model)
**Strategie:** EN-first (75/20/5) → Continued DE
**Token Budget Phase 1:** 25B Tokens
**Daten-Quelle:** Maximum Open-Source (HuggingFace)
**Letzter Stand:** April 2026

---

## 1. Gesamt-Übersicht

```
Phase 1 Pretraining:      25B Tokens  (EN-heavy)
Phase 2 Continued:        15B Tokens  (DE-heavy + KL)
Phase 3 SFT:              ~200k Samples
Phase 4 ORPO:             ~60k Preference Pairs
Phase 5 LoRA:             per Adapter (~1-5k Samples)

Total Disk (tokenized):   ~200 GB
Total Download (raw):     ~120 GB
```

---

## 2. Phase 1 Pretraining Daten (25B Tokens)

### 2.1 Englisch (75% = 18.75B Tokens)

**Hauptquelle: FineWeb-Edu**

```yaml
source:
  name: "FineWeb-Edu"
  huggingface: "HuggingFaceFW/fineweb-edu"
  config: "sample-10BT"  # oder sample-100BT für mehr
  license: "ODC-BY"
  quality: "höchste Englisch-Web-Qualität"
  
what_it_is: |
  Gefiltert aus Common Crawl mit Education-Classifier.
  Nur Dokumente mit hoher akademischer/lehrreicher Qualität.
  
expected_tokens: 10B (sample-10BT)
target_tokens: 10B (alles nehmen)
disk_size: ~25 GB

filters:
  min_score: 2.5  # Education Score
  min_length: 200
  max_length: 100000
  language: "en"
```

**Zweitquelle: Wikipedia Englisch**

```yaml
source:
  name: "Wikipedia English"
  huggingface: "wikimedia/wikipedia"
  config: "20231101.en"
  license: "CC BY-SA 3.0"
  quality: "Hoch (kuriert)"

what_it_is: |
  Aktuelle Wikipedia Snapshots.
  Clean, faktual, breite Themen.
  
expected_tokens: ~3-4B (nach Tokenisierung)
target_tokens: 3B
disk_size: ~10 GB

filters:
  min_length: 500
  exclude_disambiguation: true
  exclude_list_articles: true
```

**Drittquelle: Diverse Text (SlimPajama subset)**

```yaml
source:
  name: "SlimPajama (subset)"
  huggingface: "cerebras/SlimPajama-627B"
  license: "mixed (mostly permissive)"
  quality: "kuratiert, dedupliziert"

what_it_is: |
  Gereinigte Version von RedPajama.
  Mix aus: C4, ArXiv, Books, StackExchange, GitHub, Wikipedia, CommonCrawl
  Bereits stark dedupliziert.

expected_tokens: 627B total (wir nehmen subset)
target_tokens: 3B
disk_size: ~8 GB

filters:
  # Schon vor-gefiltert
  subset_selection:
    - "arxiv"       # Wissenschaft
    - "stackexchange"  # Q&A Format
    - "book"        # Längere Texte
    - "wikipedia"   # Faktual
  skip:
    - "commoncrawl" # haben wir schon via FineWeb-Edu
    - "c4"          # redundant
    - "github"      # haben wir eigene Code-Quelle
```

**Viertquelle: OpenMath / Reasoning**

```yaml
source:
  name: "OpenMathInstruct-2"
  huggingface: "nvidia/OpenMathInstruct-2"
  license: "CC BY 4.0"
  quality: "sehr hoch"

what_it_is: |
  Mathe-Problemlösungen mit Reasoning.
  Stärkt Modell-Fähigkeit in step-by-step Denken.
  Auch für Pretraining nutzbar (nicht nur SFT).
  
expected_tokens: ~2-3B
target_tokens: 2B (mit etwas Downsample)
disk_size: ~6 GB

format: "pretraining-ready als Text-Konkatenation"
```

**Englisch Total:**

```
FineWeb-Edu:              10.0 B tokens   (53%)
Wikipedia EN:              3.0 B tokens   (16%)
SlimPajama subset:         3.0 B tokens   (16%)
OpenMathInstruct:          2.0 B tokens   (11%)
Reserve/Diversity:         0.75 B tokens  (4%)
─────────────────────────────────────────────
Total Englisch:           18.75 B tokens  (75% of 25B)
```

### 2.2 Deutsch (20% = 5B Tokens)

**Hauptquelle: german-commons**

```yaml
source:
  name: "german-commons"
  huggingface: "coral-nlp/german-commons"
  license: "ODC-BY"
  quality: "variabel, braucht Filtering"

what_it_is: |
  154B Tokens deutsche Texte.
  Mix aus Web, News, Bücher, Wikipedia, kulturellen Texten.
  Das ist die größte verfügbare deutsche Text-Sammlung.

LESSONS AUS V1:
  Cultural Subset ist historisch (Gutenberg etc.) → Bias!
  Max 5-10% historisch nehmen.

expected_tokens: 154B (riesig)
target_tokens: 3B (gefiltert)
disk_size: ~9 GB nach Filtering

filters:
  max_perplexity: 500  # Modernes Deutsch
  
  subset_ratios:
    news: 0.40          # Hauptanteil modernes DE
    web_modern: 0.30    # OSCAR-ähnlich
    wikipedia: 0.15     # Faktual
    books_modern: 0.10  # ab 1990
    cultural: 0.05      # nur wenig historisch!
  
  exclude:
    - "gutenberg_pre_1900"  # zu alt
    - "dta_historical"       # Deutsches Textarchiv
  
  min_length: 300
  language_check: "fastText must be 'de'"
```

**Zweitquelle: Wikipedia Deutsch**

```yaml
source:
  name: "Wikipedia Deutsch"
  huggingface: "wikimedia/wikipedia"
  config: "20231101.de"
  license: "CC BY-SA 3.0"

expected_tokens: ~1B
target_tokens: 1B
disk_size: ~3 GB

filters:
  min_length: 500
  exclude_disambiguation: true
```

**Drittquelle: OSCAR-2301 Deutsch (modern filtered)**

```yaml
source:
  name: "OSCAR-2301 German"
  huggingface: "oscar-corpus/OSCAR-2301"
  config: "de"
  license: "CC0"
  quality: "variable, strong filtering needed"

what_it_is: |
  Common Crawl multilingual subset, deutsche Teile.
  Mehr Web-Diversität als german-commons.

expected_tokens: ~40-50B (nur DE subset)
target_tokens: 1B (stark gefiltert)
disk_size: ~3 GB

filters:
  max_perplexity: 400  # strenger als german-commons
  # KwargsSettings von OSCAR:
  quality_warnings:
    - "header"
    - "footer"
    - "noisy"
    - "tiny"
  exclude_quality_warnings: true
  min_length: 400
```

**Deutsch Total:**

```
german-commons (filtered): 3.0 B tokens   (60%)
Wikipedia DE:              1.0 B tokens   (20%)
OSCAR-2301 DE:             1.0 B tokens   (20%)
─────────────────────────────────────────────
Total Deutsch:             5.0 B tokens   (20% of 25B)
```

### 2.3 Code (5% = 1.25B Tokens)

**Hauptquelle: StarCoderData**

```yaml
source:
  name: "StarCoderData"
  huggingface: "bigcode/starcoderdata"
  license: "mixed (permissive)"
  quality: "sehr gut, bereits kuratiert"

what_it_is: |
  Kuratierte Code-Daten von GitHub, 86 Sprachen.
  Schon dedupliziert und gefiltert.
  Besser als roher The Stack für Pretraining.

expected_tokens: 250B
target_tokens: 1.0B
disk_size: ~3 GB

filters:
  languages:
    python: 0.30
    javascript: 0.20
    typescript: 0.10
    rust: 0.10
    cpp: 0.10
    go: 0.08
    java: 0.07
    shell: 0.03
    sql: 0.02
  
  min_stars: 5
  max_length: 30000
  exclude_auto_generated: true
```

**Zweitquelle: ProofPile-2 (Reasoning/Math)**

```yaml
source:
  name: "Proof-Pile-2"
  huggingface: "EleutherAI/proof-pile-2"
  license: "mixed"

what_it_is: |
  Math proofs, formal reasoning, ArXiv Math Papers.
  Schärft logisches Reasoning, nicht nur Code.

target_tokens: 250M
disk_size: ~1 GB

filters:
  subsets:
    - "open-web-math"  # math-focused web content
    - "arxiv"
    # skip: "algebraic-stack" (zu speziell)
```

**Code Total:**

```
StarCoderData subset:      1.00 B tokens   (80%)
Proof-Pile-2:              0.25 B tokens   (20%)
─────────────────────────────────────────────
Total Code:                1.25 B tokens   (5% of 25B)
```

### 2.4 Phase 1 Gesamt

```
Englisch:        18.75 B tokens  (75%)   ~49 GB disk
Deutsch:          5.00 B tokens  (20%)   ~15 GB disk
Code:             1.25 B tokens  ( 5%)   ~4 GB disk
─────────────────────────────────────────────────
Total Phase 1:   25.00 B tokens          ~68 GB raw
Tokenized:                              ~100 GB (uint32)
```

---

## 3. Phase 2 Continued Pretraining Daten (15B Tokens)

### 3.1 Strategie

```
Phase 2 dreht das Verhältnis um:
  Phase 1: 75% EN, 20% DE, 5% Code
  Phase 2: 30% EN, 60% DE, 10% Code
  
Ziel: Deutsch aufbauen ohne Englisch zu verlieren.
      KL-Distillation bewahrt Phase-1-Wissen.
```

### 3.2 Englisch Replay (30% = 4.5B)

```yaml
source_1:
  name: "Phase-1 Golden Replay"
  what: "Top 5% quality samples from Phase 1 english data"
  rationale: |
    Wenn du Best-Samples von Phase 1 wieder zeigst,
    hilft das Retention massiv.
  target_tokens: 2.5B
  
source_2:
  name: "FineWeb-Edu (fresh)"
  huggingface: "HuggingFaceFW/fineweb-edu"
  target_tokens: 1.5B
  
source_3:
  name: "OpenMathInstruct-2"
  target_tokens: 500M
  note: "Reasoning-Schutz"
```

### 3.3 Deutsch Stark (60% = 9B)

Hier kommt jetzt Deutsch richtig rein:

```yaml
source_1:
  name: "german-commons (aggressiv gefiltert)"
  target_tokens: 4B
  filters:
    max_perplexity: 300   # noch strenger als Phase 1
    subset_ratios:
      news: 0.50          # mehr News
      web_modern: 0.25
      wikipedia: 0.15
      books: 0.10
      cultural: 0.0       # gar nichts historisches mehr

source_2:
  name: "Wikipedia DE (full)"
  target_tokens: 1.5B
  config: "20231101.de"

source_3:
  name: "OSCAR-2301 DE"
  target_tokens: 2B
  filters:
    max_perplexity: 300
    year_filter: ">=2020"  # Nur modern

source_4:
  name: "FineWeb-2 German"
  huggingface: "HuggingFaceFW/fineweb-2"
  config: "deu_Latn"
  target_tokens: 1.5B
  note: |
    Falls verfügbar (2026): FineWeb-Edu-Qualität auf Deutsch.
    Falls nicht: durch OSCAR ersetzen.
```

### 3.4 Code Verstärkung (10% = 1.5B)

```yaml
source_1:
  name: "StarCoderData"
  target_tokens: 1.25B
  same_as: "Phase 1 Code, aber mehr"

source_2:
  name: "Proof-Pile-2"
  target_tokens: 250M
```

### 3.5 Phase 2 Gesamt

```
Englisch Replay:   4.50 B tokens  (30%)   ~12 GB
Deutsch:           9.00 B tokens  (60%)   ~27 GB
Code:              1.50 B tokens  (10%)   ~4 GB
─────────────────────────────────────────────────
Total Phase 2:    15.00 B tokens          ~43 GB raw
Tokenized:                              ~60 GB
```

---

## 4. Phase 3 SFT Datasets (~200k Samples)

### 4.1 Englisch SFT (50% = 100k Samples)

**Tülu 3 SFT Mixture**

```yaml
source:
  name: "Tülu 3 SFT"
  huggingface: "allenai/tulu-3-sft-mixture"
  license: "ODC-BY"
  quality: "state-of-the-art SFT data"

what_it_is: |
  Kuratierte SFT Data von AllenAI.
  ~1M samples total, wir nehmen subset.
  Diverse Formate: Instructions, Reasoning, Code, Math.

target_samples: 80k
disk_size: ~500 MB

subsets_ratios:
  instructions: 0.40      # allgemeine Tasks
  reasoning: 0.20         # Math, Logic
  coding: 0.10            # Code tasks
  creative_writing: 0.10  # Stories, Essays
  knowledge: 0.10         # Facts, Explanations
  safety: 0.10            # Refusals, Harm
```

**UltraChat 200k**

```yaml
source:
  name: "UltraChat 200k"
  huggingface: "HuggingFaceH4/ultrachat_200k"
  license: "MIT"
  quality: "sehr gut, multi-turn"

what_it_is: |
  Multi-turn Dialoge, GPT-4 generated.
  Gut für Chat-Format Lernen.

target_samples: 15k (filtered)
disk_size: ~100 MB

filters:
  min_turns: 2
  max_turns: 8
  min_response_length: 50
  exclude_short_responses: true
```

**OpenOrca (Reasoning-Heavy)**

```yaml
source:
  name: "OpenOrca subset"
  huggingface: "Open-Orca/OpenOrca"
  license: "MIT"

target_samples: 5k
focus: "FLAN-style reasoning prompts"
```

### 4.2 Deutsch SFT (45% = 90k Samples)

**Hier ist Deutsch-Lage schlechter. Die beste verfügbaren Open-Source DE SFT Datasets:**

**LAION-GPT4-prompts Deutsch**

```yaml
source:
  name: "German GPT-4 Prompts"
  huggingface: "LAION-LLM/GPT4-DE-Instructions"
  license: "check license"
  quality: "gut für DE"

target_samples: 30k
```

**Sauerkraut / Hermes DPO übersetzt**

```yaml
source:
  name: "VAGOsolutions German Instructions"
  huggingface: "VAGOsolutions/SauerkrautLM-Mixtral-7B-Instruct-data"
  license: "Apache 2.0"

target_samples: 20k
note: "Mixtral-quality translated instructions"
```

**OpenAssistant Deutsch**

```yaml
source:
  name: "OpenAssistant OASST2 German"
  huggingface: "OpenAssistant/oasst2"
  license: "Apache 2.0"
  filter: "lang == 'de'"

target_samples: 10k
quality: "human-curated, sehr hochwertig"
```

**Alpaca-DE (VORSICHT!)**

```yaml
source:
  name: "German Alpaca"
  note: |
    WARNUNG: Bei Helix v1 als "Gift" identifiziert!
    Machine-Translation-Artefakte, oft unnatürlich.
    
    Wenn nutzen: MAX 5k samples UND stark filtern.
    Oder komplett weglassen.

target_samples: 0  # besser komplett weglassen
```

**Deutsch Smalltalk (generieren)**

```yaml
source:
  name: "Self-Generated Smalltalk"
  how_to_generate: |
    Phase 3a: Script das Smalltalk-Paare generiert
    Prompts wie:
      - "Hallo"
      - "Wie geht's dir?"
      - "Was machst du gerade?"
      - "Danke!"
      - "Schönen Tag noch"
    
    Mit verschiedenen Antwort-Stilen:
      - kurz, freundlich
      - mit Gegenfrage
      - mit Interesse an User
    
    Automatisch generieren ist ok (einfache Patterns).

target_samples: 5k
rationale: |
  Lektion aus v1: Smalltalk war schlecht,
  weil keine expliziten Smalltalk-Samples im Training.
```

**Deutsch Unsicherheit (generieren)**

```yaml
source:
  name: "Self-Generated Uncertainty"
  how_to_generate: |
    Samples wo das Modell "weiß ich nicht" lernen soll:
      - "Was sagt die Firma XYZ zu ABC?" (nicht bekannt)
      - "Wann genau passiert X?" (spezifische Zahlen unsicher)
      - "Was denkt Person Y?" (nicht antwortbar)
    
    Antworten-Pattern:
      - "Das weiß ich nicht sicher, aber..."
      - "Darüber habe ich keine verlässlichen Informationen"
      - "Ich kann das nicht mit Gewissheit sagen"

target_samples: 3k
rationale: |
  v1 hat halluziniert statt "weiß nicht" zu sagen.
  Explizites Training nötig.
```

**Deutsch Reasoning**

```yaml
source:
  name: "MMLU German (translated)"
  huggingface: "openbmb/UltraInteract_sft"
  filter: "contains German"

target_samples: 10k
focus: "Multi-step reasoning in Deutsch"
```

**Multi-turn Deutsch**

```yaml
source:
  name: "WildChat German subset"
  huggingface: "allenai/WildChat-1M"
  filter: "language == 'German'"
  license: "AI2 ImpACT-Low-Risk"

target_samples: 12k
note: "Real user conversations mit GPT-3.5/4"
```

### 4.3 Code SFT (5% = 10k Samples)

```yaml
source_1:
  name: "Magicoder-Evol-Instruct"
  huggingface: "ise-uiuc/Magicoder-Evol-Instruct-110K"
  license: "MIT"
  target_samples: 7k

source_2:
  name: "CodeAlpaca"
  huggingface: "sahil2801/CodeAlpaca-20k"
  target_samples: 3k
```

### 4.4 Phase 3 SFT Gesamt

```
Englisch:    100k samples   (50%)   ~250 MB
Deutsch:      90k samples   (45%)   ~200 MB
Code:         10k samples   ( 5%)   ~30 MB
─────────────────────────────────────────
Total:       200k samples           ~480 MB
Tokenized:                        ~3 GB
```

---

## 5. Phase 4 ORPO Preference Pairs (~60k)

### 5.1 Englisch Preferences (30k Pairs)

**UltraFeedback (Gold Standard)**

```yaml
source:
  name: "UltraFeedback Binarized"
  huggingface: "HuggingFaceH4/ultrafeedback_binarized"
  license: "MIT"
  quality: "höchste verfügbare Qualität"

what_it_is: |
  60k+ Preference Pairs, GPT-4 evaluated.
  Chosen vs Rejected gut unterscheidbar.

target_pairs: 25k
disk_size: ~150 MB
```

**HH-RLHF**

```yaml
source:
  name: "Anthropic HH-RLHF"
  huggingface: "Anthropic/hh-rlhf"
  license: "MIT"
  quality: "fokus auf Helpfulness + Harmlessness"

target_pairs: 5k
filter: "helpful-base only (skip harmful)"
```

### 5.2 Deutsch Preferences (27k Pairs)

**Hier wird's knapp — es gibt wenig DE Preference Data:**

**Orca DPO Deutsch (übersetzt)**

```yaml
source:
  name: "VAGOsolutions German DPO"
  huggingface: "VAGOsolutions/SauerkrautLM-DPO-data"
  license: "Apache 2.0"

target_pairs: 15k
```

**Selbst generieren (Hauptstrategie)**

```yaml
approach: |
  Weil DE Preference Data knapp ist, 
  generiere selbst aus SFT-Prompts:
  
  1. Nimm 15k Prompts aus Phase 3 SFT
  2. Generiere 2 Antworten:
     CHOSEN:   DeepSeek V3 (klar, direkt, hilfreich)
     REJECTED: Füge Floskeln hinzu oder kürze drastisch
  3. Auch: Helix v1 als "natürlich schlechter" Rejected

target_pairs: 12k (self-generated)
time_needed: ~2 Tage Scripting + Generation
cost: "nur DeepSeek API ~10€"
```

### 5.3 Code Preferences (3k Pairs)

```yaml
source:
  name: "CodeUltraFeedback"
  huggingface: "coseal/CodeUltraFeedback"
  target_pairs: 3k
```

### 5.4 Phase 4 ORPO Gesamt

```
Englisch:  30k pairs   (50%)   ~200 MB
Deutsch:   27k pairs   (45%)   ~180 MB
Code:       3k pairs   ( 5%)    ~20 MB
─────────────────────────────────────
Total:     60k pairs           ~400 MB
```

---

## 6. Phase 5 LoRA Datasets

### 6.1 Router-LoRA (~8k Samples)

```yaml
approach: "Self-generated + handcurated"

categories:
  level_0_smalltalk: 1000
    examples:
      - "Hallo" → {level: 0, topics: [], tools: false}
      - "Danke" → ...
      - "Gute Nacht" → ...
  
  level_1_simple_qa: 1500
    examples:
      - "Was ist die Hauptstadt von Frankreich?"
      - "Wie alt ist die Erde?"
  
  level_2_tools_needed: 1500
    examples:
      - "Was ist 1234 mal 567?"
      - "Wie viel Uhr ist es in Tokyo?"
  
  level_3_topic_knowledge: 1500
    examples:
      - "Wie wirkt Ramipril?" → topics: [medizin]
      - "Was ist Paragraph 573 BGB?" → topics: [recht]
  
  level_4_reasoning: 1500
    examples:
      - Complex multi-step problems
  
  level_5_critical: 1000
    examples:
      - Medical diagnosis questions
      - Legal advice critical

total: 8000 samples
storage: ~5 MB
```

### 6.2 Denk-LoRA (~8k Samples)

```yaml
sources:
  - name: "OpenMathInstruct-2 (CoT format)"
    samples: 3000
    format: "<think>...</think>\n\n{answer}"
  
  - name: "MetaMathQA"
    huggingface: "meta-math/MetaMathQA"
    samples: 2000
  
  - name: "Self-generated German CoT"
    samples: 3000
    approach: "DeepSeek V3 generiert CoT für deutsche Fragen"

total: 8000 samples
```

### 6.3 Logik-LoRA (~4k Samples)

```yaml
approach: "Self-generated"

categories:
  fact_checking:
    samples: 1500
    example: |
      Input: "[DRAFT] Die Erde ist 4.5 Millionen Jahre alt."
      Output: "<reflection>Fehler: Die Erde ist 4.5 MILLIARDEN Jahre alt, nicht Millionen.</reflection>\n[CORRECTED] Die Erde ist etwa 4.5 Milliarden Jahre alt."
  
  contradiction_detection:
    samples: 1000
  
  incomplete_answer_detection:
    samples: 1000
  
  confidence_calibration:
    samples: 500

total: 4000 samples
```

### 6.4 Topic-LoRAs (each ~1-2k Samples)

**Medizin-LoRA (Proof of Concept):**

```yaml
approach: "YAML-Fact-Spec + Generation (wie in Phase 5 Spec)"

facts_file: "data/lora/topics/medizin/facts.yaml"
structure:
  train_facts: 80    # 80 atomare Fakten
  val_facts: 20      # disjunkt
  
generation:
  per_fact: 3 paraphrases × 3 formats = 9 samples
  kontextuell: 500-750 zusätzlich
  total: ~1000 samples

categories:
  definitionen: 150
  zahlen_interpretation: 250
  medikamente: 200
  symptome_ursachen: 150
  lifestyle: 100
  fallbeispiele: 100
  grenzen: 50
```

---

## 7. Storage-Planung

### 7.1 Disk Space Requirements

```
Raw Downloads (während Preparation):
  Phase 1 EN:        ~50 GB
  Phase 1 DE:        ~15 GB
  Phase 1 Code:      ~4 GB
  Phase 2 DE:        ~30 GB  (viele DE Quellen)
  Phase 2 rest:      ~15 GB
  Phase 3 SFT:       ~500 MB
  Phase 4 ORPO:      ~400 MB
  Phase 5 LoRA:      ~10 MB
  ────────────────────────
  Total Raw:         ~115 GB

Tokenized (uint32, 4 bytes/token):
  Phase 1:           ~100 GB
  Phase 2:           ~60 GB
  Phase 3+4+5:       ~3 GB
  ────────────────────────
  Total Tokenized:   ~163 GB

Working Space Buffer:    ~50 GB
─────────────────────────────
Empfohlener Platz:       250-300 GB
```

### 7.2 Storage-Strategie für Unraid

```
/mnt/user/auralis_v2/
├── raw/                    # Original Downloads
│   ├── english/            # FineWeb-Edu, Wikipedia, etc.
│   ├── german/             # german-commons, OSCAR, etc.
│   └── code/               # StarCoderData, ProofPile
│
├── cleaned/                # Nach Filtering
│   ├── english.txt         # Ein Doc pro Line
│   ├── german.txt
│   └── code.txt
│
├── tokenized/              # Binary format (.bin + .idx)
│   ├── phase1/
│   │   ├── english.bin     # ~45 GB
│   │   ├── english.idx
│   │   ├── german.bin      # ~12 GB
│   │   ├── german.idx
│   │   ├── code.bin        # ~3 GB
│   │   └── code.idx
│   └── phase2/
│       └── ...
│
├── sft/                    # Phase 3 JSONL
│   ├── train.jsonl
│   └── val.jsonl
│
├── preferences/            # Phase 4 JSONL
│   └── ...
│
└── lora/                   # Phase 5 per Adapter
    ├── router/
    ├── denk/
    ├── logik/
    └── topics/
        └── medizin/

Empfehlung: auf SSD Pool (schneller I/O während Training)
```

---

## 8. Downloaden & Vorbereiten

### 8.1 HuggingFace Token Setup

```bash
# Einmalig:
pip install huggingface_hub
huggingface-cli login
# Paste dein HF Token (https://huggingface.co/settings/tokens)

# Test:
huggingface-cli whoami
```

### 8.2 Download-Reihenfolge (nach Priorität)

```bash
# Download-Skript
cd /mnt/user/auralis_v2

# Phase 1 - Englisch (kritisch, zuerst)
python scripts/data/download_english.py
  → FineWeb-Edu sample-10BT    (~25 GB)
  → Wikipedia EN                (~10 GB)
  → SlimPajama subset           (~8 GB)
  → OpenMathInstruct            (~6 GB)
  
# Phase 1 - Deutsch  
python scripts/data/download_german.py
  → german-commons              (~9 GB nach Filter)
  → Wikipedia DE                (~3 GB)
  → OSCAR-2301 DE               (~3 GB)

# Phase 1 - Code
python scripts/data/download_code.py
  → StarCoderData subset        (~3 GB)
  → ProofPile-2                 (~1 GB)

# Phase 3 SFT (klein, schnell)
python scripts/data/download_sft.py

# Phase 4 ORPO
python scripts/data/download_orpo.py
```

### 8.3 Download-Geschwindigkeit Realistisch

```
Mit 500 Mbit/s Connection:
  FineWeb-Edu 25 GB:    ~7 Minuten
  Wikipedia EN 10 GB:    ~3 Minuten
  Alle Phase 1 Daten:    ~20-30 Minuten
  
Mit 100 Mbit/s:
  Alle Phase 1 Daten:    ~2-3 Stunden

Bottleneck oft HuggingFace API limits,
nicht deine Bandbreite.
```

---

## 9. Download Scripts

### 9.1 Englisch Download

**Datei:** `scripts/data/download_english.py`

```python
"""
Lädt englische Pretraining-Daten von HuggingFace.
Speichert als gefilterte .txt Files (ein Doc pro Zeile).
"""

from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import random


OUTPUT_DIR = Path("/mnt/user/auralis_v2/raw/english")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_fineweb_edu(target_tokens: int = 10_000_000_000):
    """FineWeb-Edu: Hauptquelle Englisch."""
    print(f"Downloading FineWeb-Edu (target: {target_tokens/1e9:.1f}B tokens)")
    
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    
    output = OUTPUT_DIR / "fineweb_edu.txt"
    n_docs = 0
    total_bytes = 0
    target_bytes = int(target_tokens * 4)  # grobe Schätzung 4 chars/token
    
    with open(output, 'w', encoding='utf-8') as f:
        for example in tqdm(dataset, desc="FineWeb-Edu"):
            text = example.get('text', '')
            score = example.get('score', 0)
            
            # Filter
            if len(text) < 200 or len(text) > 100000:
                continue
            if score < 2.5:
                continue
            
            # Clean
            clean = text.replace('\n', ' ').replace('\r', ' ')
            clean = ' '.join(clean.split())
            
            f.write(clean + '\n')
            total_bytes += len(clean.encode('utf-8'))
            n_docs += 1
            
            if total_bytes >= target_bytes:
                break
    
    print(f"  ✓ {n_docs:,} docs, {total_bytes/1e9:.2f} GB")


def download_wikipedia_en(target_tokens: int = 3_000_000_000):
    """Wikipedia Englisch."""
    print(f"Downloading Wikipedia EN (target: {target_tokens/1e9:.1f}B tokens)")
    
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )
    
    output = OUTPUT_DIR / "wikipedia_en.txt"
    n_docs = 0
    total_bytes = 0
    target_bytes = int(target_tokens * 4)
    
    with open(output, 'w', encoding='utf-8') as f:
        for example in tqdm(dataset, desc="Wiki EN"):
            text = example.get('text', '')
            title = example.get('title', '')
            
            # Filter: skip disambiguation pages
            if 'disambiguation' in title.lower():
                continue
            if len(text) < 500:
                continue
            
            # Clean
            clean = text.replace('\n\n', ' ').replace('\n', ' ')
            clean = ' '.join(clean.split())
            
            f.write(clean + '\n')
            total_bytes += len(clean.encode('utf-8'))
            n_docs += 1
            
            if total_bytes >= target_bytes:
                break
    
    print(f"  ✓ {n_docs:,} docs")


def download_slimpajama_subset(target_tokens: int = 3_000_000_000):
    """SlimPajama: arxiv, stackexchange, books, wikipedia."""
    print(f"Downloading SlimPajama subset")
    
    # SlimPajama hat verschiedene sources
    dataset = load_dataset(
        "cerebras/SlimPajama-627B",
        split="train",
        streaming=True,
    )
    
    output = OUTPUT_DIR / "slimpajama.txt"
    wanted_sources = {'arxiv', 'stackexchange', 'book', 'wikipedia'}
    
    n_docs = 0
    total_bytes = 0
    target_bytes = int(target_tokens * 4)
    
    with open(output, 'w', encoding='utf-8') as f:
        for example in tqdm(dataset, desc="SlimPajama"):
            source = example.get('meta', {}).get('redpajama_set_name', '').lower()
            
            # Nur gewünschte Sources
            if not any(s in source for s in wanted_sources):
                continue
            
            text = example.get('text', '')
            if len(text) < 200 or len(text) > 100000:
                continue
            
            clean = ' '.join(text.split())
            
            f.write(clean + '\n')
            total_bytes += len(clean.encode('utf-8'))
            n_docs += 1
            
            if total_bytes >= target_bytes:
                break
    
    print(f"  ✓ {n_docs:,} docs")


def download_openmath(target_tokens: int = 2_000_000_000):
    """OpenMathInstruct: Reasoning-heavy."""
    print(f"Downloading OpenMathInstruct")
    
    dataset = load_dataset(
        "nvidia/OpenMathInstruct-2",
        split="train",
        streaming=True,
    )
    
    output = OUTPUT_DIR / "openmath.txt"
    n_docs = 0
    total_bytes = 0
    target_bytes = int(target_tokens * 4)
    
    with open(output, 'w', encoding='utf-8') as f:
        for example in tqdm(dataset, desc="OpenMath"):
            problem = example.get('problem', '')
            solution = example.get('generated_solution', '')
            
            if not problem or not solution:
                continue
            
            # Format als Pretraining-Text
            combined = f"Problem: {problem}\n\nSolution: {solution}"
            clean = ' '.join(combined.split())
            
            f.write(clean + '\n')
            total_bytes += len(clean.encode('utf-8'))
            n_docs += 1
            
            if total_bytes >= target_bytes:
                break
    
    print(f"  ✓ {n_docs:,} docs")


if __name__ == "__main__":
    print("=" * 60)
    print("ENGLISH PRETRAINING DATA DOWNLOAD")
    print("=" * 60)
    
    download_fineweb_edu(target_tokens=10_000_000_000)
    download_wikipedia_en(target_tokens=3_000_000_000)
    download_slimpajama_subset(target_tokens=3_000_000_000)
    download_openmath(target_tokens=2_000_000_000)
    
    # Stats
    print("\n=== Summary ===")
    for f in OUTPUT_DIR.glob("*.txt"):
        size_gb = f.stat().st_size / 1024**3
        print(f"  {f.name}: {size_gb:.2f} GB")
```

### 9.2 Deutsch Download

**Datei:** `scripts/data/download_german.py`

```python
"""
Lädt deutsche Pretraining-Daten.
Mit aggressivem Filtering (Lessons aus v1!).
"""

from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import random


OUTPUT_DIR = Path("/mnt/user/auralis_v2/raw/german")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_german_commons(target_tokens: int = 3_000_000_000):
    """german-commons mit Quality-Filter."""
    print(f"Downloading german-commons (filtered)")
    
    dataset = load_dataset(
        "coral-nlp/german-commons",
        split="train",
        streaming=True,
    )
    
    output = OUTPUT_DIR / "german_commons.txt"
    n_docs = 0
    n_filtered_cultural = 0
    n_filtered_perplexity = 0
    total_bytes = 0
    target_bytes = int(target_tokens * 5)  # DE hat mehr chars/token
    
    random.seed(42)
    
    with open(output, 'w', encoding='utf-8') as f:
        for example in tqdm(dataset, desc="german-commons"):
            text = example.get('text', '')
            subset = example.get('subset', '')
            perplexity = example.get('perplexity', 1000)
            
            # FILTER 1: Perplexity (modern German only)
            if perplexity > 500:
                n_filtered_perplexity += 1
                continue
            
            # FILTER 2: Cultural/Historical subsample (AUS V1!)
            if subset in ('cultural', 'gutenberg', 'dta'):
                # Nur 5% behalten (aus v1 gelernt)
                if random.random() > 0.05:
                    n_filtered_cultural += 1
                    continue
            
            # FILTER 3: Length
            if len(text) < 300:
                continue
            if len(text) > 100000:
                text = text[:100000]
            
            clean = text.replace('\n', ' ').replace('\r', ' ')
            clean = ' '.join(clean.split())
            
            f.write(clean + '\n')
            total_bytes += len(clean.encode('utf-8'))
            n_docs += 1
            
            if total_bytes >= target_bytes:
                break
    
    print(f"  ✓ {n_docs:,} docs, {total_bytes/1e9:.2f} GB")
    print(f"  ✗ {n_filtered_cultural:,} cultural filtered")
    print(f"  ✗ {n_filtered_perplexity:,} high-PPL filtered")


def download_wikipedia_de(target_tokens: int = 1_000_000_000):
    """Wikipedia Deutsch."""
    print(f"Downloading Wikipedia DE")
    
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.de",
        split="train",
        streaming=True,
    )
    
    output = OUTPUT_DIR / "wikipedia_de.txt"
    n_docs = 0
    total_bytes = 0
    target_bytes = int(target_tokens * 5)
    
    with open(output, 'w', encoding='utf-8') as f:
        for example in tqdm(dataset, desc="Wiki DE"):
            text = example.get('text', '')
            title = example.get('title', '')
            
            if 'Begriffsklärung' in title:
                continue
            if len(text) < 500:
                continue
            
            clean = ' '.join(text.split())
            
            f.write(clean + '\n')
            total_bytes += len(clean.encode('utf-8'))
            n_docs += 1
            
            if total_bytes >= target_bytes:
                break
    
    print(f"  ✓ {n_docs:,} docs")


def download_oscar_de(target_tokens: int = 1_000_000_000):
    """OSCAR-2301 German (modernes Web)."""
    print(f"Downloading OSCAR-2301 DE")
    
    dataset = load_dataset(
        "oscar-corpus/OSCAR-2301",
        "de",
        split="train",
        streaming=True,
    )
    
    output = OUTPUT_DIR / "oscar_de.txt"
    n_docs = 0
    total_bytes = 0
    target_bytes = int(target_tokens * 5)
    
    with open(output, 'w', encoding='utf-8') as f:
        for example in tqdm(dataset, desc="OSCAR DE"):
            text = example.get('text', '')
            meta = example.get('meta', {})
            
            # OSCAR quality warnings
            warnings = meta.get('quality_warnings', [])
            if warnings:
                continue
            
            if len(text) < 400:
                continue
            if len(text) > 100000:
                text = text[:100000]
            
            clean = ' '.join(text.split())
            
            f.write(clean + '\n')
            total_bytes += len(clean.encode('utf-8'))
            n_docs += 1
            
            if total_bytes >= target_bytes:
                break
    
    print(f"  ✓ {n_docs:,} docs")


if __name__ == "__main__":
    print("=" * 60)
    print("GERMAN PRETRAINING DATA DOWNLOAD")
    print("=" * 60)
    
    download_german_commons(target_tokens=3_000_000_000)
    download_wikipedia_de(target_tokens=1_000_000_000)
    download_oscar_de(target_tokens=1_000_000_000)
    
    # Stats
    print("\n=== Summary ===")
    for f in OUTPUT_DIR.glob("*.txt"):
        size_gb = f.stat().st_size / 1024**3
        print(f"  {f.name}: {size_gb:.2f} GB")
```

### 9.3 SFT Download

**Datei:** `scripts/data/download_sft.py`

```python
"""
Lädt SFT Datasets für Phase 3.
"""

from pathlib import Path
from datasets import load_dataset
import json
from tqdm import tqdm


OUTPUT_DIR = Path("/mnt/user/auralis_v2/raw/sft")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_tulu3(target: int = 80_000):
    """Tülu 3 SFT - best English SFT data."""
    print(f"Downloading Tülu 3 SFT (target: {target})")
    
    dataset = load_dataset(
        "allenai/tulu-3-sft-mixture",
        split="train",
    )
    
    # Shuffle and sample
    dataset = dataset.shuffle(seed=42).select(range(target))
    
    output = OUTPUT_DIR / "tulu3_en.jsonl"
    with open(output, 'w') as f:
        for ex in tqdm(dataset, desc="Tülu 3"):
            sample = {
                "messages": ex['messages'],
                "source": "tulu3",
                "language": "en",
            }
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def download_ultrachat(target: int = 15_000):
    """UltraChat 200k - multi-turn."""
    print(f"Downloading UltraChat 200k")
    
    dataset = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split="train_sft",
    )
    
    dataset = dataset.shuffle(seed=42).select(range(target))
    
    output = OUTPUT_DIR / "ultrachat_en.jsonl"
    with open(output, 'w') as f:
        for ex in tqdm(dataset, desc="UltraChat"):
            # Filter: min 2 turns
            messages = ex['messages']
            if len(messages) < 2:
                continue
            
            sample = {
                "messages": messages,
                "source": "ultrachat",
                "language": "en",
            }
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def download_oasst2_de(target: int = 10_000):
    """OpenAssistant OASST2 - deutsche Samples."""
    print(f"Downloading OASST2 German")
    
    dataset = load_dataset(
        "OpenAssistant/oasst2",
        split="train",
    )
    
    # Filter German only
    de_samples = dataset.filter(lambda ex: ex.get('lang') == 'de')
    print(f"  Found {len(de_samples)} German samples")
    
    if len(de_samples) > target:
        de_samples = de_samples.shuffle(seed=42).select(range(target))
    
    # Convert tree format to messages
    # (OASST is tree-structured, need to linearize)
    output = OUTPUT_DIR / "oasst2_de.jsonl"
    with open(output, 'w') as f:
        for ex in tqdm(de_samples, desc="OASST2"):
            # Simple: just use the conversation path
            if ex.get('role') != 'assistant':
                continue
            
            # Get parent message (user question)
            parent_id = ex.get('parent_id')
            if not parent_id:
                continue
            
            # Skip complex tree handling for now
            # Production: build conversation chains
            sample = {
                "messages": [
                    {"role": "user", "content": ex.get('parent_text', '')},
                    {"role": "assistant", "content": ex['text']},
                ],
                "source": "oasst2_de",
                "language": "de",
            }
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


def download_magicoder(target: int = 7_000):
    """Magicoder-Evol-Instruct Code Data."""
    print(f"Downloading Magicoder")
    
    dataset = load_dataset(
        "ise-uiuc/Magicoder-Evol-Instruct-110K",
        split="train",
    )
    
    dataset = dataset.shuffle(seed=42).select(range(target))
    
    output = OUTPUT_DIR / "magicoder_code.jsonl"
    with open(output, 'w') as f:
        for ex in tqdm(dataset, desc="Magicoder"):
            sample = {
                "messages": [
                    {"role": "user", "content": ex['instruction']},
                    {"role": "assistant", "content": ex['response']},
                ],
                "source": "magicoder",
                "language": "code",
            }
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    print("=" * 60)
    print("SFT DATA DOWNLOAD")
    print("=" * 60)
    
    download_tulu3(80_000)
    download_ultrachat(15_000)
    download_oasst2_de(10_000)
    download_magicoder(7_000)
    
    print("\n=== Summary ===")
    for f in OUTPUT_DIR.glob("*.jsonl"):
        with open(f) as fh:
            n = sum(1 for _ in fh)
        size_mb = f.stat().st_size / 1024**2
        print(f"  {f.name}: {n:,} samples, {size_mb:.1f} MB")
```

### 9.4 ORPO Download

```python
# scripts/data/download_orpo.py

"""
Lädt Preference Data für Phase 4 ORPO.
"""

from pathlib import Path
from datasets import load_dataset
import json
from tqdm import tqdm


OUTPUT_DIR = Path("/mnt/user/auralis_v2/raw/orpo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_ultrafeedback(target: int = 25_000):
    """UltraFeedback: Gold Standard."""
    print("Downloading UltraFeedback")
    
    dataset = load_dataset(
        "HuggingFaceH4/ultrafeedback_binarized",
        split="train_prefs",
    )
    
    dataset = dataset.shuffle(seed=42).select(range(min(target, len(dataset))))
    
    output = OUTPUT_DIR / "ultrafeedback.jsonl"
    with open(output, 'w') as f:
        for ex in tqdm(dataset, desc="UltraFeedback"):
            pair = {
                "prompt": ex['chosen'][0]['content'],  # first user message
                "chosen": ex['chosen'][-1]['content'],  # last assistant
                "rejected": ex['rejected'][-1]['content'],
                "source": "ultrafeedback",
                "language": "en",
            }
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')


def download_hh_rlhf(target: int = 5_000):
    """Anthropic HH-RLHF."""
    print("Downloading HH-RLHF helpful-base")
    
    dataset = load_dataset(
        "Anthropic/hh-rlhf",
        data_dir="helpful-base",
        split="train",
    )
    
    dataset = dataset.shuffle(seed=42).select(range(target))
    
    output = OUTPUT_DIR / "hh_rlhf.jsonl"
    with open(output, 'w') as f:
        for ex in tqdm(dataset, desc="HH-RLHF"):
            # Parse conversation format
            chosen = ex['chosen']
            rejected = ex['rejected']
            
            # Extract last "Human:" and first "Assistant:" response
            # (HH-RLHF has specific format)
            pair = {
                "prompt": "extract prompt from conversation",
                "chosen": chosen,
                "rejected": rejected,
                "source": "hh_rlhf",
                "language": "en",
            }
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    download_ultrafeedback(25_000)
    download_hh_rlhf(5_000)
```

---

## 10. Tokenization Pipeline

### 10.1 Tokenization Script

Nach dem Download: Tokenisieren in Binary für Training.

**Datei:** `scripts/data/tokenize_for_pretraining.py`

```python
"""
Tokenisiert cleaned text files in Binary .bin Format.
Optimiert für memmap-based Streaming Training.
"""

from pathlib import Path
import numpy as np
from tqdm import tqdm
from auralis.tokenizer import HelixTokenizer


def tokenize_file(
    input_file: str,
    output_file: str,
    tokenizer: HelixTokenizer,
    add_eos: bool = True,
):
    """Tokenisiert ein Text-File zu Binary."""
    input_path = Path(input_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Tokenizing {input_path.name}")
    
    total_tokens = 0
    
    with open(output_path, 'wb') as fbin:
        with open(input_path, 'r', encoding='utf-8') as fin:
            for line in tqdm(fin, desc="Lines"):
                text = line.strip()
                if not text:
                    continue
                
                tokens = tokenizer.encode(text, add_eos=add_eos)
                arr = np.array(tokens, dtype=np.uint32)
                arr.tofile(fbin)
                
                total_tokens += len(tokens)
    
    size_gb = output_path.stat().st_size / 1024**3
    print(f"  ✓ {total_tokens:,} tokens, {size_gb:.2f} GB")
    
    return total_tokens


def main():
    tokenizer = HelixTokenizer()
    
    base_raw = Path("/mnt/user/auralis_v2/raw")
    base_out = Path("/mnt/user/auralis_v2/tokenized/phase1")
    base_out.mkdir(parents=True, exist_ok=True)
    
    # Englisch
    en_files = list((base_raw / "english").glob("*.txt"))
    en_total = 0
    with open(base_out / "english.bin", 'wb') as combined:
        for f in en_files:
            tokens = tokenize_file(
                f, 
                base_out / "english.bin",
                tokenizer,
            )
            en_total += tokens
    
    # Deutsch
    de_files = list((base_raw / "german").glob("*.txt"))
    de_total = 0
    for f in de_files:
        tokens = tokenize_file(
            f,
            base_out / "german.bin",
            tokenizer,
        )
        de_total += tokens
    
    # Code
    code_files = list((base_raw / "code").glob("*.txt"))
    code_total = 0
    for f in code_files:
        tokens = tokenize_file(
            f,
            base_out / "code.bin",
            tokenizer,
        )
        code_total += tokens
    
    # Summary
    print("\n" + "=" * 60)
    print("PHASE 1 TOKENIZATION SUMMARY")
    print("=" * 60)
    print(f"Englisch: {en_total/1e9:.2f}B tokens")
    print(f"Deutsch:  {de_total/1e9:.2f}B tokens")
    print(f"Code:     {code_total/1e9:.2f}B tokens")
    print(f"Total:    {(en_total+de_total+code_total)/1e9:.2f}B tokens")


if __name__ == "__main__":
    main()
```

---

## 11. Quality-Checks (vor Training!)

### 11.1 Sanity-Check Script

Bevor du das 12-Tage Training startest: prüfe ob die Daten ok sind.

**Datei:** `scripts/data/sanity_check.py`

```python
"""
Sanity-Checks vor dem Pretraining.
Verhindert böse Überraschungen nach Tag 3.
"""

from pathlib import Path
import numpy as np
from auralis.tokenizer import HelixTokenizer


def check_tokenized_file(path: Path, tokenizer: HelixTokenizer):
    """Checks ein tokenized .bin file."""
    print(f"\n=== {path.name} ===")
    
    # Size
    size_gb = path.stat().st_size / 1024**3
    print(f"Size: {size_gb:.2f} GB")
    
    # Load sample (nicht komplett laden!)
    arr = np.memmap(path, dtype=np.uint32, mode='r')
    n_tokens = len(arr)
    print(f"Tokens: {n_tokens:,} ({n_tokens/1e9:.2f}B)")
    
    # Check vocab range
    # Sample 1M random tokens
    n_sample = min(1_000_000, n_tokens)
    idx = np.random.choice(n_tokens, n_sample, replace=False)
    sample = arr[idx]
    
    max_token = sample.max()
    min_token = sample.min()
    n_unique = len(np.unique(sample))
    
    print(f"Token range: {min_token} - {max_token}")
    print(f"Unique in sample: {n_unique:,}")
    print(f"Vocab utilization: {100*n_unique/tokenizer.vocab_size:.1f}%")
    
    # Warnings
    if max_token >= tokenizer.vocab_size:
        print("❌ ERROR: Token ID out of vocab range!")
    if n_unique / tokenizer.vocab_size < 0.1:
        print("⚠️  WARNING: Low vocab utilization (< 10%)")
    
    # Decode random samples
    print("\nSample decodings:")
    for i in range(3):
        start = np.random.randint(0, n_tokens - 100)
        chunk = arr[start:start+100].tolist()
        decoded = tokenizer.decode(chunk)
        print(f"  [{i+1}] {decoded[:150]}...")


def main():
    tokenizer = HelixTokenizer()
    
    base = Path("/mnt/user/auralis_v2/tokenized/phase1")
    
    for bin_file in sorted(base.glob("*.bin")):
        check_tokenized_file(bin_file, tokenizer)
    
    # Mix-Ratios prüfen
    print("\n=== Mix Ratios ===")
    total = 0
    counts = {}
    for bin_file in base.glob("*.bin"):
        arr = np.memmap(bin_file, dtype=np.uint32, mode='r')
        counts[bin_file.stem] = len(arr)
        total += len(arr)
    
    print(f"Total: {total/1e9:.2f}B tokens")
    for name, count in counts.items():
        pct = 100 * count / total
        print(f"  {name}: {count/1e9:.2f}B ({pct:.1f}%)")
    
    # Expected ratios für Phase 1:
    expected = {"english": 75, "german": 20, "code": 5}
    print("\nExpected vs Actual:")
    for name, target_pct in expected.items():
        actual_pct = 100 * counts.get(name, 0) / total
        diff = actual_pct - target_pct
        status = "✓" if abs(diff) < 3 else "⚠️"
        print(f"  {name}: {actual_pct:.1f}% (target {target_pct}%, diff {diff:+.1f}) {status}")


if __name__ == "__main__":
    main()
```

---

## 12. Fehlerbehebung

### 12.1 Häufige Download-Probleme

```
Problem: "Rate limited" bei HuggingFace
Lösung: 
  - HF_TOKEN setzen (authenticated hat höhere Limits)
  - Sleep zwischen Requests
  - Mit Streaming arbeiten (weniger Load)

Problem: Disk full während Download
Lösung:
  - Nicht alle Files gleichzeitig laden
  - Streaming + Filter + Write (Memory nicht auslasten)
  - Raw Downloads nach Tokenization löschen

Problem: Encoding Errors (UTF-8)
Lösung:
  - errors='ignore' bei Read
  - Aber: Log welche Zeilen übersprungen
```

### 12.2 Datasets Deprecated / Moved

```
Falls ein Dataset auf HF nicht mehr verfügbar:

Alternativen für:
  FineWeb-Edu → RedPajama-V2 Edu subset
  SlimPajama → Dolma (AllenAI)
  UltraFeedback → CompMix
  OSCAR-2301 → mc4 Deutsch

Check: https://huggingface.co/datasets/... vor Download
```

---

## 13. Next Steps

Nach dieser Dataset-Spec:

```
1. Download-Scripts implementieren
   → scripts/data/download_english.py
   → scripts/data/download_german.py
   → scripts/data/download_code.py
   → scripts/data/download_sft.py
   → scripts/data/download_orpo.py

2. Tokenizer zuerst trainieren (Phase 0)
   → braucht ~30GB Text von den Downloads
   → nutzt gleiche Daten die auch Pretraining nutzt

3. Alle Daten downloaden
   → ~2-3 Stunden bei guter Bandbreite
   → Parallel: Modell-Architektur implementieren

4. Tokenize alle Files
   → ~1-2 Stunden auf Pro 5000

5. Sanity-Check
   → PFLICHT vor dem Training
   → Mix-Ratios prüfen
   → Sample-Decodings anschauen

6. Phase 1 Pretraining starten
```

---

## 14. Zusammenfassung

```
Phase 1 (Pretraining 25B):     ~68 GB raw, 100 GB tokenized
Phase 2 (Continued 15B):       ~43 GB raw, 60 GB tokenized
Phase 3 (SFT 200k):            ~500 MB
Phase 4 (ORPO 60k):            ~400 MB
Phase 5 (LoRA):                ~50 MB

Total Download:                ~120 GB
Total Storage after prep:      ~165 GB
Empfohlener Disk-Platz:        300 GB
```

Die Datasets sind pragmatisch gewählt:
- Hauptsächlich aus HuggingFace (wie gewünscht)
- Qualitäts-gefiltert (aus v1-Lessons)
- Mix-Ratios realistisch (75/20/5 → 30/60/10)
- Alles reproduzierbar (Seeds gesetzt)
- Alles lizenz-konform (ODC-BY, Apache, MIT, CC)

---

*Dataset Specification Version 1.0 — April 2026*
*Für Auralis v2 / Helix v2 1B Model*
