# Phase 0: Tokenizer Training

**Project:** Auralis v2 / Helix v2
**Phase:** 0 (prerequisite for everything else)
**Duration:** 2-3 days
**Goal:** Production-ready 200k bilingual + code tokenizer

---

## 1. Why this phase first?

The tokenizer is the only component that CANNOT be swapped out
afterwards. Every bit of training work is tied to the vocabulary.
Tokenizer change = complete re-pretraining.

Therefore: do it right ONCE, then never touch it again.

---

## 2. Deliverables

At the end of this phase the following exist:

```
/tokenizer/
    helix_v2_tokenizer.model    # SentencePiece model file
    helix_v2_tokenizer.vocab    # Vocab as text (for inspection)
    training_manifest.yaml      # Which data, which config
    quality_report.md           # Efficiency per language/domain

/scripts/tokenizer/
    prepare_corpus.py           # Data download + filtering
    train_tokenizer.py          # SentencePiece training
    test_tokenizer.py           # Quality tests
    
/src/auralis/tokenizer/
    __init__.py
    helix_tokenizer.py          # Python wrapper for training + inference
    chat_template.py            # Chat format handling

/tests/tokenizer/
    test_helix_tokenizer.py     # Unit tests
    test_chat_template.py       # Chat format tests
    test_roundtrip.py           # Encode→Decode tests
```

---

## 3. Work Steps

### 3.1 Set up project structure (30 min)

```bash
# Create repo
mkdir -p auralis-v2
cd auralis-v2
git init

# Structure
mkdir -p {data,scripts,src,tests,configs,docs,checkpoints}
mkdir -p data/{raw,cleaned,training,eval}
mkdir -p scripts/{tokenizer,pretrain,sft,lora,eval,utils}
mkdir -p src/auralis/{tokenizer,model,training,inference,lora}
mkdir -p tests/{tokenizer,model,training}

# pyproject.toml
cat > pyproject.toml << 'EOF'
[project]
name = "auralis-v2"
version = "0.1.0"
description = "Auralis v2 - Modular AI Assistant (Helix v2 Model)"
requires-python = ">=3.11"

dependencies = [
    "sentencepiece>=0.2.0",
    "torch>=2.5.0",
    "transformers>=4.45.0",
    "datasets>=3.0.0",
    "huggingface_hub>=0.25.0",
    "tqdm>=4.66.0",
    "pyyaml>=6.0",
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "black>=24.0",
    "ruff>=0.5",
    "mypy>=1.10",
]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
EOF

# .gitignore
cat > .gitignore << 'EOF'
# Data (never commit)
data/raw/*
data/cleaned/*
data/training/*
!data/**/.gitkeep

# Checkpoints
checkpoints/
*.pt
*.bin
*.safetensors

# Tokenizer binary
*.model

# Python
__pycache__/
*.py[cod]
.venv/
venv/
*.egg-info/

# IDE
.vscode/
.idea/
*.swp

# Logs
*.log
logs/

# Experiments
wandb/
runs/
EOF

# First placeholders
touch data/raw/.gitkeep
touch checkpoints/.gitkeep

git add .
git commit -m "chore: initial project structure"
```

### 3.2 Prepare training corpus (4-6 hours)

**Concept:**

```
Target sizes:
  English:  35 GB  (60% — most data, best quality)
  German:   18 GB  (30% — filtered, modern)
  Code:      7 GB  (10% — multi-language)
  
Total:     ~60 GB of text for tokenizer training
```

**Script:** `scripts/tokenizer/prepare_corpus.py`

```python
"""
Bereitet den Trainings-Korpus für den Tokenizer vor.
Lädt Daten aus HuggingFace, filtert nach Qualität,
schreibt in Text-Dateien für SentencePiece.
"""

from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import random
import argparse


OUTPUT_DIR = Path("data/raw/tokenizer_corpus")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def prepare_english(target_gb: float = 35.0) -> None:
    """FineWeb-Edu als englische Hauptquelle.
    
    Höchste Qualität englischer Web-Daten.
    Vorgefiltert nach Education-Score.
    """
    print(f"Lade Englisch (Ziel: {target_gb} GB)...")
    
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",  # 10B Token Sample
        split="train",
        streaming=True,
    )
    
    output = OUTPUT_DIR / "english.txt"
    total_bytes = 0
    target_bytes = int(target_gb * 1024**3)
    n_docs = 0
    
    with open(output, 'w', encoding='utf-8') as f:
        for example in tqdm(dataset, desc="Englisch"):
            text = example.get('text', '')
            
            # Filter
            if len(text) < 200:
                continue
            if len(text) > 100000:
                text = text[:100000]
            
            # Score-Filter (FineWeb-Edu hat Score)
            if example.get('score', 5.0) < 2.5:
                continue
            
            # Schreiben (ein Dokument pro Zeile)
            clean_text = text.replace('\n', ' ').replace('\r', ' ')
            clean_text = ' '.join(clean_text.split())  # Multi-Whitespace zu single
            
            f.write(clean_text + '\n')
            total_bytes += len(clean_text.encode('utf-8'))
            n_docs += 1
            
            if total_bytes >= target_bytes:
                break
    
    actual_gb = total_bytes / 1024**3
    print(f"  ✓ {n_docs} Dokumente, {actual_gb:.2f} GB")


def prepare_german(target_gb: float = 18.0) -> None:
    """german-commons, aggressive Qualitäts-Filterung.
    
    Lessons aus v1:
      - Cultural Subset (historisch) max 10%
      - Perplexity < 500
      - Modernes Deutsch bevorzugen
    """
    print(f"Lade Deutsch (Ziel: {target_gb} GB)...")
    
    dataset = load_dataset(
        "coral-nlp/german-commons",
        split="train",
        streaming=True,
    )
    
    output = OUTPUT_DIR / "german.txt"
    total_bytes = 0
    target_bytes = int(target_gb * 1024**3)
    n_docs = 0
    n_filtered_cultural = 0
    n_filtered_perplexity = 0
    
    random.seed(42)  # Reproduzierbar
    
    with open(output, 'w', encoding='utf-8') as f:
        for example in tqdm(dataset, desc="Deutsch"):
            text = example.get('text', '')
            subset = example.get('subset', '')
            perplexity = example.get('perplexity', 0)
            
            # Filter 1: Perplexity (modernes Deutsch)
            if perplexity > 500:
                n_filtered_perplexity += 1
                continue
            
            # Filter 2: Cultural Subset reduzieren
            if subset in ('cultural', 'gutenberg', 'dta'):
                # Nur 10% beibehalten
                if random.random() > 0.10:
                    n_filtered_cultural += 1
                    continue
            
            # Filter 3: Länge
            if len(text) < 200:
                continue
            if len(text) > 100000:
                text = text[:100000]
            
            # Schreiben
            clean_text = text.replace('\n', ' ').replace('\r', ' ')
            clean_text = ' '.join(clean_text.split())
            
            f.write(clean_text + '\n')
            total_bytes += len(clean_text.encode('utf-8'))
            n_docs += 1
            
            if total_bytes >= target_bytes:
                break
    
    actual_gb = total_bytes / 1024**3
    print(f"  ✓ {n_docs} Dokumente, {actual_gb:.2f} GB")
    print(f"  ✗ {n_filtered_cultural} cultural-Docs gefiltert")
    print(f"  ✗ {n_filtered_perplexity} High-PPL-Docs gefiltert")


def prepare_code(target_gb: float = 7.0) -> None:
    """The Stack v2 mit gewichteten Top-Sprachen.
    
    Nicht 100 Sprachen wild mischen, sondern fokussiert
    auf die 10 meist-genutzten.
    """
    print(f"Lade Code (Ziel: {target_gb} GB)...")
    
    # Gewichte pro Sprache
    target_langs = {
        "Python":     0.25,
        "JavaScript": 0.20,
        "TypeScript": 0.10,
        "Rust":       0.10,
        "C++":        0.10,
        "Go":         0.08,
        "Java":       0.07,
        "C":          0.05,
        "Shell":      0.03,
        "SQL":        0.02,
    }
    
    dataset = load_dataset(
        "bigcode/the-stack-v2",
        split="train",
        streaming=True,
    )
    
    output = OUTPUT_DIR / "code.txt"
    total_bytes = 0
    target_bytes = int(target_gb * 1024**3)
    n_docs = 0
    lang_counts = {lang: 0 for lang in target_langs}
    
    with open(output, 'w', encoding='utf-8') as f:
        for example in tqdm(dataset, desc="Code"):
            lang = example.get('language', '')
            
            if lang not in target_langs:
                continue
            
            # Gewichtung: eingabe-wahrscheinlichkeit
            if random.random() > target_langs[lang] * 5:
                continue
            
            text = example.get('content', '')
            if len(text) < 100:
                continue
            if len(text) > 30000:
                text = text[:30000]
            
            # Code behält Struktur (Zeilenumbrüche wichtig)
            # Aber mit Separator zwischen Dateien
            f.write(text + '\n<|endcode|>\n')
            
            total_bytes += len(text.encode('utf-8'))
            n_docs += 1
            lang_counts[lang] += 1
            
            if total_bytes >= target_bytes:
                break
    
    actual_gb = total_bytes / 1024**3
    print(f"  ✓ {n_docs} Dokumente, {actual_gb:.2f} GB")
    print(f"  Verteilung:")
    for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
        print(f"    {lang}: {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--english-gb', type=float, default=35.0)
    parser.add_argument('--german-gb', type=float, default=18.0)
    parser.add_argument('--code-gb', type=float, default=7.0)
    parser.add_argument('--skip-english', action='store_true')
    parser.add_argument('--skip-german', action='store_true')
    parser.add_argument('--skip-code', action='store_true')
    args = parser.parse_args()
    
    if not args.skip_english:
        prepare_english(args.english_gb)
    if not args.skip_german:
        prepare_german(args.german_gb)
    if not args.skip_code:
        prepare_code(args.code_gb)
    
    # Summary
    print("\n=== Korpus fertig ===")
    for path in OUTPUT_DIR.glob("*.txt"):
        size_gb = path.stat().st_size / 1024**3
        print(f"  {path.name}: {size_gb:.2f} GB")


if __name__ == "__main__":
    main()
```

**Execution:**

```bash
cd auralis-v2
pip install -e ".[dev]"

# Run once, takes time + bandwidth
python scripts/tokenizer/prepare_corpus.py

# Test mode with smaller data first:
python scripts/tokenizer/prepare_corpus.py \
    --english-gb 1.0 \
    --german-gb 0.5 \
    --code-gb 0.2
```

### 3.3 Train tokenizer (4-8 hours)

**Script:** `scripts/tokenizer/train_tokenizer.py`

```python
"""
SentencePiece Unigram Training für Helix v2.
200k Vocab, bilingual + code optimiert.
"""

import sentencepiece as spm
from pathlib import Path
import yaml
from datetime import datetime


def train_helix_tokenizer(
    corpus_dir: str = "data/raw/tokenizer_corpus",
    output_dir: str = "tokenizer",
    vocab_size: int = 200000,
    num_threads: int = 16,
):
    """Trainiert den Helix v2 Tokenizer.
    
    Args:
        corpus_dir: Verzeichnis mit Text-Dateien (english.txt, german.txt, code.txt)
        output_dir: Wohin die Modell-Datei geschrieben wird
        vocab_size: Target Vocabulary Size
        num_threads: CPU-Threads für Training
    """
    
    corpus_path = Path(corpus_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Input-Files
    input_files = [
        str(corpus_path / "english.txt"),
        str(corpus_path / "german.txt"),
        str(corpus_path / "code.txt"),
    ]
    
    # Pre-check
    for f in input_files:
        if not Path(f).exists():
            raise FileNotFoundError(f"Fehlt: {f}")
        size_gb = Path(f).stat().st_size / 1024**3
        print(f"✓ {Path(f).name}: {size_gb:.2f} GB")
    
    # Manifest für Reproduzierbarkeit
    manifest = {
        "version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "vocab_size": vocab_size,
        "model_type": "unigram",
        "input_files": input_files,
        "file_sizes_gb": {
            Path(f).name: Path(f).stat().st_size / 1024**3
            for f in input_files
        },
    }
    
    print(f"\nStarte Tokenizer-Training:")
    print(f"  Vocab Size: {vocab_size}")
    print(f"  Model Type: unigram")
    print(f"  Threads:    {num_threads}")
    print(f"  Erwartete Dauer: 4-8 Stunden\n")
    
    # SentencePiece Training
    spm.SentencePieceTrainer.train(
        # === INPUT ===
        input=",".join(input_files),
        input_format="text",
        
        # === OUTPUT ===
        model_prefix=str(output_path / "helix_v2_tokenizer"),
        model_type="unigram",
        vocab_size=vocab_size,
        
        # === SUB-WORD REGELN ===
        character_coverage=0.9999,
        max_sentence_length=32768,
        shuffle_input_sentence=True,
        
        # === SPECIAL TOKENS ===
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        
        # === USER-DEFINED TOKENS ===
        user_defined_symbols=[
            # Chat-Rollen
            "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
            
            # Reasoning
            "<think>", "</think>",
            "<reflection>", "</reflection>",
            
            # LoRA-Routing
            "<lora>", "</lora>",
            "<route>", "</route>",
            
            # Tools
            "<tool>", "</tool>",
            "<tool_result>", "</tool_result>",
            
            # Memory
            "<memory>", "</memory>",
            "<recall>", "</recall>",
            
            # Code
            "<code>", "</code>",
            "<|python|>", "<|javascript|>", "<|typescript|>",
            "<|rust|>", "<|cpp|>", "<|go|>",
            "<|java|>", "<|shell|>", "<|sql|>",
            
            # Multi-Token Prediction
            "<|mtp_1|>", "<|mtp_2|>", "<|mtp_3|>",
            
            # Structure
            "<|endcode|>",
        ],
        
        # === NORMALISIERUNG ===
        normalization_rule_name="nmt_nfkc",
        
        # === CODE-FREUNDLICH ===
        remove_extra_whitespaces=False,
        
        # === ZAHLEN ===
        split_digits=True,
        
        # === BYTE FALLBACK ===
        byte_fallback=True,
        
        # === PERFORMANCE ===
        num_threads=num_threads,
        train_extremely_large_corpus=True,
        
        # === PRE-TOKENIZATION ===
        add_dummy_prefix=True,
    )
    
    # Manifest speichern
    with open(output_path / "training_manifest.yaml", 'w') as f:
        yaml.dump(manifest, f, default_flow_style=False)
    
    print("\n✓ Tokenizer fertig!")
    print(f"  Datei: {output_path / 'helix_v2_tokenizer.model'}")
    print(f"  Vocab: {output_path / 'helix_v2_tokenizer.vocab'}")
    print(f"  Manifest: {output_path / 'training_manifest.yaml'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus-dir', default='data/raw/tokenizer_corpus')
    parser.add_argument('--output-dir', default='tokenizer')
    parser.add_argument('--vocab-size', type=int, default=200000)
    parser.add_argument('--threads', type=int, default=16)
    args = parser.parse_args()
    
    train_helix_tokenizer(
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        num_threads=args.threads,
    )
```

### 3.4 Python wrapper for model integration

**Script:** `src/auralis/tokenizer/helix_tokenizer.py`

```python
"""
PyTorch-freundlicher Wrapper um SentencePiece.
Stellt einheitliche API für Training + Inference bereit.

WICHTIG: Einziger Ort wo Tokenizer-Logik lebt.
         Nie duplizieren — nur hier hinein commiten.
"""

from __future__ import annotations

import sentencepiece as spm
import torch
from pathlib import Path
from typing import Union


class HelixTokenizer:
    """Helix v2 Tokenizer — Production API."""
    
    def __init__(
        self,
        model_path: str = "tokenizer/helix_v2_tokenizer.model",
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Tokenizer-Modell fehlt: {model_path}\n"
                f"Trainiere mit: python scripts/tokenizer/train_tokenizer.py"
            )
        
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(self.model_path))
        
        # Standard IDs
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        
        # Chat-Rollen (IDs dynamisch holen)
        self.system_token_id = self.sp.piece_to_id("<|system|>")
        self.user_token_id = self.sp.piece_to_id("<|user|>")
        self.assistant_token_id = self.sp.piece_to_id("<|assistant|>")
        self.end_token_id = self.sp.piece_to_id("<|end|>")
        
        # Validation: Special Tokens müssen erkannt werden
        for name, token_id in [
            ("<|system|>", self.system_token_id),
            ("<|user|>", self.user_token_id),
            ("<|assistant|>", self.assistant_token_id),
            ("<|end|>", self.end_token_id),
        ]:
            if token_id == self.unk_token_id:
                raise ValueError(
                    f"Special Token {name} wurde nicht als eigenes "
                    f"Token erkannt! Tokenizer-Training fehlerhaft."
                )
        
        self._vocab_size = self.sp.vocab_size()
    
    @property
    def vocab_size(self) -> int:
        return self._vocab_size
    
    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        return_tensor: bool = False,
    ) -> Union[list[int], torch.Tensor]:
        """Encodiert Text zu Token-IDs.
        
        Args:
            text: Input-Text
            add_bos: BOS-Token vorne anfügen
            add_eos: EOS-Token hinten anfügen
            return_tensor: Als torch.Tensor statt Liste
        
        Returns:
            Token-IDs als Liste oder Tensor (1D, long)
        """
        ids = self.sp.encode(text)
        
        if add_bos:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]
        
        if return_tensor:
            return torch.tensor(ids, dtype=torch.long)
        return ids
    
    def decode(
        self,
        token_ids: Union[list[int], torch.Tensor],
        skip_special: bool = True,
    ) -> str:
        """Dekodiert Token-IDs zu Text.
        
        Args:
            token_ids: Liste oder 1D-Tensor
            skip_special: Special Tokens entfernen
        
        Returns:
            Dekodierter Text
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        if skip_special:
            specials = {
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id,
            }
            token_ids = [t for t in token_ids if t not in specials]
        
        return self.sp.decode(token_ids)
    
    def apply_chat_template(
        self,
        messages: list[dict],
        add_generation_prompt: bool = False,
    ) -> str:
        """Wandelt Messages in Helix-Chat-Format um.
        
        KRITISCH: Diese Funktion ist die EINZIGE Quelle für
        Chat-Format. Training + Inference + Eval + API müssen
        diese Funktion aufrufen.
        
        Format:
            <|system|>
            {content}
            <|end|>
            <|user|>
            {content}
            <|end|>
            <|assistant|>
            {content}
            <|end|>
        
        Args:
            messages: Liste von {role, content} dicts
            add_generation_prompt: Am Ende "<|assistant|>\n" anfügen
                                   (für Inference)
        
        Returns:
            Formatierter Prompt-String
        """
        parts = []
        
        for msg in messages:
            role = msg['role']
            content = msg['content'].strip()
            
            if role == 'system':
                parts.append(f"<|system|>\n{content}\n<|end|>")
            elif role == 'user':
                parts.append(f"<|user|>\n{content}\n<|end|>")
            elif role == 'assistant':
                parts.append(f"<|assistant|>\n{content}\n<|end|>")
            else:
                raise ValueError(f"Unknown role: {role}")
        
        prompt = "\n".join(parts)
        
        if add_generation_prompt:
            prompt = prompt + "\n<|assistant|>\n"
        
        return prompt
    
    def __len__(self) -> int:
        return self.vocab_size
    
    def __repr__(self) -> str:
        return (
            f"HelixTokenizer("
            f"vocab_size={self.vocab_size}, "
            f"model={self.model_path.name})"
        )
```

### 3.5 Tests (CRITICAL — prevent v1 bugs)

**Script:** `tests/tokenizer/test_helix_tokenizer.py`

```python
"""
Tests die den v1-Prompt-Bug verhindern.
Training, Inference, Eval, API MÜSSEN identische Prompts bauen.
"""

import pytest
import torch
from src.auralis.tokenizer import HelixTokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return HelixTokenizer()


class TestBasicOperations:
    """Grundlegende Encode/Decode Operationen."""
    
    def test_roundtrip_english(self, tokenizer):
        text = "The capital of France is Paris."
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert decoded.strip() == text.strip()
    
    def test_roundtrip_german(self, tokenizer):
        text = "Die Hauptstadt von Österreich ist Wien."
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert decoded.strip() == text.strip()
    
    def test_roundtrip_code(self, tokenizer):
        text = "def hello(name):\n    return f'Hello, {name}!'"
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert decoded.strip() == text.strip()
    
    def test_tensor_output(self, tokenizer):
        ids = tokenizer.encode("Test", return_tensor=True)
        assert isinstance(ids, torch.Tensor)
        assert ids.dtype == torch.long
        assert ids.dim() == 1
    
    def test_bos_eos(self, tokenizer):
        ids = tokenizer.encode("Hallo", add_bos=True, add_eos=True)
        assert ids[0] == tokenizer.bos_token_id
        assert ids[-1] == tokenizer.eos_token_id


class TestSpecialTokens:
    """Special Tokens müssen einzelne Tokens sein."""
    
    def test_system_token(self, tokenizer):
        ids = tokenizer.encode("<|system|>")
        # Sollte als 1 Token erkannt werden (plus evtl. dummy prefix)
        assert tokenizer.system_token_id in ids
    
    def test_chat_tokens_distinct(self, tokenizer):
        """Alle Chat-Tokens haben unterschiedliche IDs."""
        ids = {
            tokenizer.system_token_id,
            tokenizer.user_token_id,
            tokenizer.assistant_token_id,
            tokenizer.end_token_id,
        }
        assert len(ids) == 4, "Chat-Tokens müssen distinct sein!"
    
    def test_no_unknown_special_tokens(self, tokenizer):
        """Keiner der Special Tokens darf UNK sein."""
        for name, token_id in [
            ("system", tokenizer.system_token_id),
            ("user", tokenizer.user_token_id),
            ("assistant", tokenizer.assistant_token_id),
        ]:
            assert token_id != tokenizer.unk_token_id, f"{name} ist UNK!"


class TestChatTemplate:
    """
    Die kritischen Tests die den v1-Bug verhindern.
    """
    
    def test_simple_conversation(self, tokenizer):
        messages = [
            {"role": "system", "content": "Du bist Helix."},
            {"role": "user", "content": "Hallo"},
            {"role": "assistant", "content": "Hi!"},
        ]
        prompt = tokenizer.apply_chat_template(messages)
        
        # Format muss stimmen
        assert "<|system|>" in prompt
        assert "<|user|>" in prompt
        assert "<|assistant|>" in prompt
        assert "<|end|>" in prompt
        assert "Du bist Helix." in prompt
        assert "Hallo" in prompt
        assert "Hi!" in prompt
    
    def test_generation_prompt(self, tokenizer):
        """Inference-Modus fügt <|assistant|>\n am Ende hinzu."""
        messages = [
            {"role": "user", "content": "Test"},
        ]
        
        # Ohne generation prompt
        prompt1 = tokenizer.apply_chat_template(messages)
        assert not prompt1.endswith("<|assistant|>\n")
        
        # Mit generation prompt
        prompt2 = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        assert prompt2.endswith("<|assistant|>\n")
    
    def test_training_inference_consistency(self, tokenizer):
        """
        DER TEST DER v1 VERHINDERT HÄTTE.
        
        Training (ohne generation prompt) und Inference (mit)
        müssen für die gleichen Messages IDENTISCHE Prefixes
        produzieren.
        """
        messages_training = [
            {"role": "user", "content": "Was ist Photosynthese?"},
            {"role": "assistant", "content": "Ein biologischer Prozess."},
        ]
        
        messages_inference = [
            {"role": "user", "content": "Was ist Photosynthese?"},
        ]
        
        training_prompt = tokenizer.apply_chat_template(messages_training)
        inference_prompt = tokenizer.apply_chat_template(
            messages_inference,
            add_generation_prompt=True,
        )
        
        # Inference-Prompt muss Prefix von Training-Prompt sein
        # (bis zum Assistant-Content)
        user_part_training = training_prompt.split("<|assistant|>")[0]
        user_part_inference = inference_prompt.split("<|assistant|>")[0]
        
        assert user_part_training == user_part_inference, (
            f"Training und Inference bauen DIFFERENT Prompts!\n"
            f"Training:\n{repr(user_part_training)}\n\n"
            f"Inference:\n{repr(user_part_inference)}"
        )
    
    def test_byte_identical_tokenization(self, tokenizer):
        """Training und Inference tokenisieren IDENTISCH."""
        messages = [
            {"role": "user", "content": "Hallo"},
        ]
        
        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        
        # Zwei Mal tokenisieren — muss gleich sein
        tokens_1 = tokenizer.encode(prompt)
        tokens_2 = tokenizer.encode(prompt)
        
        assert tokens_1 == tokens_2


class TestEfficiency:
    """Effizienz-Ziele müssen erreicht werden."""
    
    def test_english_efficiency(self, tokenizer):
        """~130 tokens / 100 words für Englisch."""
        text = (
            "The capital of France is Paris. "
            "Photosynthesis is the process by which plants "
            "convert light energy into chemical energy. "
            "I went to the store yesterday and bought some groceries."
        )
        words = len(text.split())
        tokens = len(tokenizer.encode(text))
        
        tokens_per_100_words = (tokens / words) * 100
        assert tokens_per_100_words < 150, (
            f"English tokenization ineffizient: "
            f"{tokens_per_100_words:.1f} tokens/100 words"
        )
    
    def test_german_efficiency(self, tokenizer):
        """~145 tokens / 100 words für Deutsch."""
        text = (
            "Die Hauptstadt von Österreich ist Wien. "
            "Photosynthese ist der Prozess, bei dem Pflanzen "
            "Lichtenergie in chemische Energie umwandeln. "
            "Ich bin gestern einkaufen gegangen."
        )
        words = len(text.split())
        tokens = len(tokenizer.encode(text))
        
        tokens_per_100_words = (tokens / words) * 100
        assert tokens_per_100_words < 170, (
            f"German tokenization ineffizient: "
            f"{tokens_per_100_words:.1f} tokens/100 words"
        )
    
    def test_long_german_words(self, tokenizer):
        """Deutsche Komposita sollten effizient tokenisiert werden."""
        words = [
            "Donaudampfschifffahrtsgesellschaftskapitän",
            "Rindfleischetikettierungsüberwachungsgesetz",
        ]
        
        for word in words:
            tokens = len(tokenizer.encode(word))
            # Faustregel: max 10 Tokens für extrem lange deutsche Wörter
            assert tokens < 12, (
                f"'{word}' → {tokens} tokens (zu ineffizient)"
            )


class TestVocabSize:
    def test_vocab_size(self, tokenizer):
        assert tokenizer.vocab_size == 200000
```

**Execution:**

```bash
pytest tests/tokenizer/ -v
```

**All tests must be GREEN before Phase 0 is complete.**

### 3.6 Generate quality report

**Script:** `scripts/tokenizer/test_tokenizer.py`

```python
"""
Umfassender Qualitäts-Report für den trainierten Tokenizer.
Ausgabe: tokenizer/quality_report.md
"""

from src.auralis.tokenizer import HelixTokenizer
from pathlib import Path
from datetime import datetime


def generate_quality_report(output_path: str = "tokenizer/quality_report.md"):
    tok = HelixTokenizer()
    
    report = []
    report.append(f"# Helix v2 Tokenizer Quality Report\n")
    report.append(f"Generated: {datetime.now().isoformat()}\n")
    report.append(f"Vocab Size: {tok.vocab_size}\n\n")
    
    # === Effizienz ===
    report.append("## Token-Effizienz\n")
    report.append("| Domäne | Tokens/100 Wörter | Status |\n")
    report.append("|--------|-------------------|--------|\n")
    
    test_sets = {
        "Englisch": [
            "The capital of France is Paris.",
            "Photosynthesis converts light to energy.",
        ],
        "Deutsch": [
            "Die Hauptstadt von Österreich ist Wien.",
            "Photosynthese wandelt Lichtenergie um.",
        ],
        "Code Python": [
            "def hello():\n    return 'world'",
        ],
        "Code JavaScript": [
            "const greet = (name) => `Hello ${name}`;",
        ],
    }
    
    for domain, texts in test_sets.items():
        total_tokens = sum(len(tok.encode(t)) for t in texts)
        total_words = sum(len(t.split()) for t in texts)
        ratio = (total_tokens / total_words) * 100
        
        # Ziel-Benchmarks
        targets = {
            "Englisch": 130,
            "Deutsch": 145,
            "Code Python": 160,
            "Code JavaScript": 160,
        }
        target = targets.get(domain, 200)
        status = "✓" if ratio <= target * 1.15 else "✗"
        
        report.append(f"| {domain} | {ratio:.1f} | {status} (Ziel: <{target}) |\n")
    
    # === Beispiele ===
    report.append("\n## Tokenization-Beispiele\n")
    
    examples = [
        ("Einfach DE", "Die Hauptstadt von Österreich ist Wien."),
        ("Komposita DE", "Donaudampfschifffahrtsgesellschaftskapitän"),
        ("Chat", "<|user|>\nHallo\n<|end|>"),
        ("Code", "import numpy as np\narr = np.array([1, 2, 3])"),
    ]
    
    for label, text in examples:
        tokens = tok.encode(text)
        pieces = [tok.sp.id_to_piece(t) for t in tokens]
        report.append(f"\n### {label}\n")
        report.append(f"**Text:** `{text}`\n\n")
        report.append(f"**Tokens ({len(tokens)}):** `{pieces[:15]}"
                      f"{'...' if len(pieces) > 15 else ''}`\n")
    
    # Schreiben
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.writelines(report)
    
    print(f"✓ Report: {output_path}")


if __name__ == "__main__":
    generate_quality_report()
```

---

## 4. Acceptance Criteria

**Phase 0 is complete when:**

```
Code:
  □ All scripts exist and run through
  □ helix_tokenizer.py production-ready
  □ All unit tests green (pytest tests/tokenizer/ passes)
  □ Prompt-builder consistency test green (critical!)

Tokenizer:
  □ helix_v2_tokenizer.model exists
  □ Vocab size = 200000 exactly
  □ All special tokens correctly recognized
  □ quality_report.md generated

Efficiency:
  □ English: < 150 tokens/100 words
  □ German: < 170 tokens/100 words
  □ Code: < 180 tokens/100 words
  □ Long German words: < 12 tokens

Documentation:
  □ training_manifest.yaml committed
  □ quality_report.md committed
  □ README.md in /tokenizer/ with usage example

Git:
  □ All changes committed
  □ Tag: "v0.1.0-tokenizer"
```

---

## 5. Troubleshooting

**Problem: "Special token recognized as UNK"**

```
Cause:    user_defined_symbols not passed correctly
Solution: Check that the list is a Python list, not a string
```

**Problem: "German efficiency poor (>180)"**

```
Cause:    Too little German data in the corpus
Solution: Run prepare_german(target_gb=25.0) again
          Or: add more diverse sources
```

**Problem: "Tokenizer training crashed with OOM"**

```
Cause:    RAM too small for corpus
Solution: Reduce num_threads (less parallelism)
          Or: reduce corpus size
```

**Problem: "Roundtrip changes whitespace"**

```
Cause:    add_dummy_prefix=True + normalization
Solution: Expected! `text.strip()` when comparing
```

---

## 6. Next Steps after Phase 0

After successful completion:

```
1. Set git tag: git tag -a v0.1.0-tokenizer
2. Push to remote: git push origin v0.1.0-tokenizer
3. Start model architecture (Phase 0.5)
   → Embeddings size = vocab_size (200k × d_model)
   → See: SPEC_PHASE_0.5_MODEL_ARCHITECTURE.md
4. In parallel: prepare pretraining data (Phase 1)
   → See: SPEC_PHASE_1_PRETRAINING.md
```

---

## 7. Schedule

```
Day 1 (6-8 hours):
  09:00-10:00  Project structure + pyproject.toml
  10:00-14:00  Corpus preparation (english + german)
  14:00-15:00  Corpus preparation (code)
  15:00-16:00  Write tests
  16:00-17:00  Documentation + commits

Day 2 (8-10 hours):
  09:00         Start tokenizer training
  09:00-17:00   Training runs (in parallel: plan model arch)
  17:00-18:00   Quality test + report

Day 3 (4-6 hours):
  09:00-12:00  Run tests, debug
  12:00-14:00  Final tuning if needed
  14:00-15:00  Documentation + git tag
  15:00-16:00  Kick-off Phase 1 planning
```

---

*Phase 0 Spec Version 1.0 — April 2026*
