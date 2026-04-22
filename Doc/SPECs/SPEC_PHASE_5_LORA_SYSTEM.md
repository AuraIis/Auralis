# Phase 5: LoRA-System

**Projekt:** Auralis v2 / Helix v2
**Phase:** 5 (Modulares Erweiterungssystem)
**Dauer:** 2-3 Wochen
**Ziel:** Brain-inspiriertes LoRA-System mit Meta-LoRAs + Topic-LoRAs
**Voraussetzung:** Phase 4 abgeschlossen (aligniertes Basismodell)
**Hardware:** RTX 3090 für LoRAs, H200 nur bei Meta-LoRAs
**Budget:** ~$50-100

---

## 1. Vision: Brain-Inspired Architecture

```
Das menschliche Gehirn hat verschiedene Module:
  → Thalamus:          Routing (was ist wichtig?)
  → Broca/Wernicke:    Sprache (Basismodell)
  → Präfrontaler Kx:   Reasoning, Planning
  → Logik-Zentrum:     Widerspruch-Erkennung
  → Hippocampus:       Gedächtnis
  → Temporal-Lappen:   Fachwissen
  → Cerebellum:        Automatische Patterns
  → Körper:            Tools (Hände, Augen, etc.)

Helix v2 bildet das ab:
  → Router-LoRA:       Komplexitäts-Entscheidung
  → Basismodell:       Sprache + Basis-Weltwissen
  → Denk-LoRA:         Chain-of-Thought, Reasoning
  → Logik-LoRA:        Selbstprüfung
  → Memory-LoRA:       Persistentes Wissen
  → Topic-LoRAs:       Fachgebiete on-demand
  → Autopilot:         Im Basismodell (schnelle Antworten)
  → Tool-System:       Python, Web, Code-Execution

Vorteil: Compute skaliert mit Komplexität
  Level 0: ~100ms (Basismodell only)
  Level 5: ~15s (Alles + Self-Verification)
```

---

## 2. Struktur des LoRA-Systems

### 2.1 Hierarchie

```
Immer aktiv (Meta-LoRAs):
  ┌─ Router-LoRA (immer zuerst)
  ├─ Denk-LoRA (bei Level 2+)
  └─ Logik-LoRA (bei Level 4+)

On-Demand (Topic-LoRAs):
  ┌─ Medizin
  ├─ Recht
  ├─ Technik
  ├─ Kochen
  ├─ Reisen
  ├─ Finanzen
  └─ ...

Tools (extern):
  ┌─ Python-Sandbox
  ├─ Web-Search
  ├─ Code-Execution
  ├─ Calendar
  └─ ...
```

### 2.2 Inference-Pipeline

```
User: "Kann ich Ramipril mit Bisoprolol kombinieren?"

Step 1: Router-LoRA
  Output: {level: 4, topics: ["medizin"], tools: false}
  → Medizinische Frage, erfordert Fachwissen + Vorsicht

Step 2: Topic-LoRA laden (Medizin)
  → Hot-swap in Runtime

Step 3: Denk-LoRA aktiv
  <think>
  Ramipril: ACE-Hemmer (Blutdruck)
  Bisoprolol: Beta-Blocker (Blutdruck, Herz)
  Kombinationstherapie häufig bei Hypertonie
  Aber: Orthostase-Risiko, Kalium-Monitoring
  </think>

Step 4: Generate Answer
  "Die Kombination von ACE-Hemmern und Beta-Blockern ist eine 
  gängige Therapie bei Bluthochdruck..."

Step 5: Logik-LoRA (Selbstprüfung)
  <reflection>
  Habe ich Warnungen erwähnt? Ja.
  Rechtliche Abgrenzung? Sollte ich ergänzen.
  </reflection>

Step 6: Final Answer (mit Disclaimer)
```

---

## 3. Meta-LoRAs

### 3.1 Router-LoRA

**Zweck:** Entscheidet Komplexitäts-Level und welche Topics/Tools nötig.

**Datenformat:**

```jsonl
{
  "input": "Hallo",
  "output": "{\"level\": 0, \"topics\": [], \"tools\": false, \"reason\": \"smalltalk\"}"
}
{
  "input": "Was ist 25 mal 47?",
  "output": "{\"level\": 2, \"topics\": [], \"tools\": true, \"reason\": \"math_calculation\"}"
}
{
  "input": "Wie wirkt Ramipril?",
  "output": "{\"level\": 3, \"topics\": [\"medizin\"], \"tools\": false, \"reason\": \"medical_knowledge\"}"
}
{
  "input": "Kann ich meinen Vermieter wegen Schimmel verklagen?",
  "output": "{\"level\": 4, \"topics\": [\"recht\", \"medizin\"], \"tools\": false, \"reason\": \"legal_complex\"}"
}
```

**Dataset:**
- 5000-8000 Samples
- Alle 6 Level vertreten
- Breite Topic-Abdeckung
- Handcurated Gold Standard

**Config:**

```yaml
# configs/lora/router_lora.yaml

name: "router_lora"
base_model: "checkpoints/phase4_orpo/best.pt"

lora:
  method: "dora"  # DoRA für Pattern-Learning
  r: 32
  alpha: 64
  dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

training:
  data: "data/lora/router/train.jsonl"
  val: "data/lora/router/val.jsonl"
  lr: 3.0e-4
  batch_size: 16
  epochs: 3
  
  early_stopping:
    patience: 3
    metric: "val_accuracy"  # JSON-Parseability + Level-Match
```

### 3.2 Denk-LoRA (Reasoning Patterns)

**Zweck:** Chain-of-Thought, Problem-Zerlegung in `<think>` Blöcke.

**Datenformat:**

```jsonl
{
  "prompt": "Wenn ein Zug um 14:00 in Berlin abfährt und 6 Stunden nach Hamburg braucht, wann kommt er an?",
  "response": "<think>\nAbfahrt: 14:00\nFahrtzeit: 6 Stunden\nAnkunft = Abfahrt + Fahrtzeit\n14:00 + 6:00 = 20:00\n</think>\n\nDer Zug kommt um 20:00 in Hamburg an."
}
{
  "prompt": "Was ist wichtiger beim Kauf eines Laptops: RAM oder CPU?",
  "response": "<think>\nEs kommt auf den Use-Case an:\n- Multitasking/Browser: RAM wichtiger\n- Video-Editing/Games: CPU wichtiger\n- Büro/Office: Beides ausreichend\nIch sollte nach Use-Case fragen oder beide Szenarien aufzeigen.\n</think>\n\nDas hängt davon ab, wofür du den Laptop nutzen willst:\n- Für viele Browser-Tabs und Office-Arbeit: RAM ist wichtiger\n- Für Video-Editing oder Gaming: CPU ist wichtiger"
}
```

**Dataset:**
- 5000-10000 Samples
- Verschiedene Reasoning-Arten:
  - Mathematisch
  - Logisch (Syllogismen)
  - Ursache-Wirkung
  - Vergleiche
  - Pro/Contra
  - Schrittweise Anleitung
- `<think>` immer vor finaler Antwort

**Config:**

```yaml
# configs/lora/denk_lora.yaml

name: "denk_lora"
base_model: "checkpoints/phase4_orpo/best.pt"

lora:
  method: "dora"
  r: 64
  alpha: 128
  dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"]

training:
  data: "data/lora/denk/train.jsonl"
  val: "data/lora/denk/val.jsonl"
  lr: 2.0e-4
  batch_size: 8
  epochs: 2
```

### 3.3 Logik-LoRA (Self-Verification)

**Zweck:** Prüft eigene Antworten in `<reflection>` auf Widersprüche, Lücken, Fehler.

**Datenformat:**

```jsonl
{
  "prompt": "[DRAFT] Wasser kocht bei 90°C.",
  "response": "<reflection>\nMoment - Wasser kocht bei 100°C auf Meereshöhe, nicht 90°C. Das ist ein Fehler im Draft.\n</reflection>\n\n[CORRECTED] Wasser kocht bei 100°C (auf Meereshöhe, bei Normaldruck)."
}
{
  "prompt": "[DRAFT] Paris ist die Hauptstadt von Italien.",
  "response": "<reflection>\nFehler: Paris ist Hauptstadt von Frankreich, nicht Italien. Italiens Hauptstadt ist Rom.\n</reflection>\n\n[CORRECTED] Paris ist die Hauptstadt von Frankreich. Die Hauptstadt von Italien ist Rom."
}
```

**Dataset:**
- 3000-5000 Samples
- Universelle Logik-Regeln
- Bekannte Fakten-Checks
- Widersprüche-Erkennung
- Abschätzungen prüfen

---

## 4. Topic-LoRAs

### 4.1 Beispiel: Medizin-LoRA

**Fakten-Spec (YAML):**

```yaml
# data/lora/topics/medizin/facts.yaml

topic: "medizin"
version: "1.0"

# Train-Fakten (80% der totalen)
train_facts:
  - id: "med_001"
    fact: "ACE-Hemmer wie Ramipril senken den Blutdruck durch Hemmung der Angiotensin-Konversion"
    source_quote: "S3-Leitlinie Hypertonie 2023"
    confidence: "high"
  
  - id: "med_002"
    fact: "Beta-Blocker wie Bisoprolol blockieren Adrenalin-Wirkung am Herzen"
    source_quote: "ESC Guidelines 2021"
    confidence: "high"
  
  # ... 80+ weitere

# Val-Fakten (DISJUNKT zu Train!)
val_facts:
  - id: "med_val_001"
    fact: "Kalziumantagonisten entspannen die Gefäßmuskulatur"
    source_quote: "Leitlinie"
    confidence: "high"
  
  # ... 20 weitere
```

**Sample Generation:**

```python
# scripts/lora/generate_topic_samples.py

def generate_samples_for_topic(facts_yaml, output_path):
    """
    Generiert aus Facts-YAML vielfältige Training-Samples.
    
    Pro Fact:
      - 3 verschiedene Formulierungen der Frage
      - Naturliche Antworten mit Fakten-Inhalt
      - Kontextuelle Variationen
    """
    # 80 train_facts × 3 paraphrases = 240 core samples
    # + 250 medikamente (Dosierung, Wechselwirkung)
    # + 150 symptome
    # + 100 lifestyle
    # + 100 fallbeispiele
    # + 50 grenzen/unsicherheit
    # = ~1000 total
    
    # Multiple Paraphrasen pro Fact:
    variations = generate_variations(fact)  # via LLM
    
    # Multiple Question-Types:
    q_types = [
        "Was ist X?",
        "Wie wirkt X?",
        "Wofür nutzt man X?",
        "Welche Nebenwirkungen hat X?",
    ]
```

**Config:**

```yaml
# configs/lora/topics/medizin.yaml

name: "medizin_lora"
base_model: "checkpoints/phase4_orpo/best.pt"

lora:
  method: "mora"  # MoRA für Fakten-Lernen (> DoRA)
  r: 128
  alpha: 256
  dropout: 0.05
  target_modules: ["q_proj", "v_proj", "up_proj", "down_proj"]

training:
  data: "data/lora/topics/medizin/train.jsonl"
  val: "data/lora/topics/medizin/val.jsonl"
  lr: 1.0e-4
  batch_size: 8
  
  # KRITISCH: Lessons aus v1!
  epochs: 3
  early_stopping:
    enabled: true
    patience: 3
    metric: "val_loss"
    min_delta: 0.01
    target_val_loss: 0.3  # Stopp bei Plateau um 0.2-0.3
    
  # Disjunkt val set (kein overlap)
  val_strategy: "disjoint_facts"
```

---

## 5. LoRA Training Script

**Datei:** `scripts/lora/train_lora.py`

```python
"""
Universelles LoRA Training Script.
Funktioniert für Meta- und Topic-LoRAs.
"""

import argparse
import torch
from pathlib import Path
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
import json

from auralis.model import build_model
from auralis.tokenizer import HelixTokenizer
from auralis.lora.mora import MoRAConfig, apply_mora
from auralis.training.sft_dataset import SFTDataset
from auralis.training.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-name', required=True)
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    print("=" * 60)
    print(f"LoRA Training: {config.name}")
    print("=" * 60)
    
    # === Basis-Modell (frozen) ===
    print(f"\nLoading base model: {config.base_model}")
    model = build_model("configs/model/helix_v2_3b.yaml")
    
    checkpoint = torch.load(config.base_model, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.cuda()
    
    # === Freeze alles ===
    for p in model.parameters():
        p.requires_grad = False
    
    # === LoRA aufsetzen ===
    method = config.lora.method
    
    if method == "dora":
        # DoRA via PEFT library
        lora_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=config.lora.target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            use_dora=True,  # KRITISCH: DoRA
        )
        model = get_peft_model(model, lora_config)
    
    elif method == "mora":
        # MoRA (Matrix-Rank Adaptation) - Custom impl
        mora_config = MoRAConfig(
            r=config.lora.r,
            alpha=config.lora.alpha,
            target_modules=config.lora.target_modules,
        )
        model = apply_mora(model, mora_config)
    
    else:
        raise ValueError(f"Unknown LoRA method: {method}")
    
    # Print trainable
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable / 1e6:.1f}M / {total / 1e6:.1f}M ({100*trainable/total:.2f}%)")
    
    # === Data ===
    tokenizer = HelixTokenizer()
    
    train_dataset = SFTDataset(
        path=config.training.data,
        tokenizer=tokenizer,
        seq_length=2048,
    )
    val_dataset = SFTDataset(
        path=config.training.val,
        tokenizer=tokenizer,
        seq_length=2048,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
    )
    
    # === Optimizer ===
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.training.lr,
    )
    
    # === Training ===
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.training.epochs):
        print(f"\n=== Epoch {epoch+1}/{config.training.epochs} ===")
        
        model.train()
        epoch_loss = 0
        n_batches = 0
        
        for step, batch in enumerate(train_loader):
            batch = {k: v.cuda() for k, v in batch.items()}
            
            output = model(
                input_ids=batch['input_ids'],
                labels=batch['labels'],
                attention_mask=batch['attention_mask'],
            )
            
            loss = output['loss']
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            n_batches += 1
            
            if step % 20 == 0:
                print(f"  Step {step} | loss {loss.item():.4f}")
        
        avg_train_loss = epoch_loss / n_batches
        print(f"  Avg train loss: {avg_train_loss:.4f}")
        
        # === Eval ===
        val_loss = evaluate_lora(model, val_loader)
        print(f"  Val loss: {val_loss:.4f}")
        
        # === Early Stopping (Lessons aus v1!) ===
        if val_loss < best_val_loss - config.training.early_stopping.min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save LoRA
            output_path = Path(f"lora_adapters/{args.output_name}")
            output_path.mkdir(parents=True, exist_ok=True)
            
            if hasattr(model, 'save_pretrained'):
                # PEFT model
                model.save_pretrained(output_path)
            else:
                # Custom save
                torch.save({
                    'state_dict': {k: v for k, v in model.state_dict().items() if 'lora' in k.lower()},
                    'config': config,
                    'val_loss': val_loss,
                }, output_path / "adapter.pt")
            
            print(f"  ✓ Saved: {output_path}")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{config.training.early_stopping.patience}")
        
        # === CRITICAL: Val Loss Threshold Check ===
        # Aus v1 gelernt: Val Loss zu niedrig = Memorization
        if val_loss < 0.05:
            print(f"\n⚠️  Val Loss {val_loss:.4f} unter 0.05 - wahrscheinlich Memorization!")
            print("   Stoppe Training, prüfe Val-Set Disjunktheit.")
            break
        
        # Target Val Loss erreicht?
        target_loss = config.training.early_stopping.target_val_loss
        if val_loss <= target_loss:
            print(f"  ✓ Target val loss {target_loss} erreicht, stoppe")
            break
        
        # Patience überschritten?
        if patience_counter >= config.training.early_stopping.patience:
            print(f"\n⏹  Early stopping (patience exhausted)")
            break
    
    print(f"\n✓ LoRA Training complete. Best val: {best_val_loss:.4f}")


def evaluate_lora(model, val_loader):
    model.eval()
    total_loss = 0
    n = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            output = model(
                input_ids=batch['input_ids'],
                labels=batch['labels'],
            )
            total_loss += output['loss'].item()
            n += 1
    
    return total_loss / n


if __name__ == "__main__":
    main()
```

---

## 6. LoRA Runtime System

### 6.1 Hot-Swap Manager

**Datei:** `src/auralis/lora/manager.py`

```python
"""
LoRA Manager: Lädt Adapter on-demand.
"""

import torch
from pathlib import Path
from typing import Dict, Optional
from peft import PeftModel


class LoRAManager:
    """Manages LoRA adapters for runtime."""
    
    def __init__(self, base_model, base_model_path: str):
        self.base_model = base_model
        self.base_model_path = base_model_path
        
        # Loaded adapters
        self.loaded_adapters: Dict[str, dict] = {}
        self.active_adapter: Optional[str] = None
        
        # Discover available adapters
        self._discover_adapters()
    
    def _discover_adapters(self):
        """Find all LoRA adapters on disk."""
        adapter_dir = Path("lora_adapters")
        
        self.available = {}
        for path in adapter_dir.iterdir():
            if path.is_dir():
                self.available[path.name] = path
        
        print(f"Found {len(self.available)} adapters: {list(self.available.keys())}")
    
    def load(self, adapter_name: str):
        """Load adapter into memory (not yet active)."""
        if adapter_name in self.loaded_adapters:
            return  # Already loaded
        
        if adapter_name not in self.available:
            raise ValueError(f"Unknown adapter: {adapter_name}")
        
        path = self.available[adapter_name]
        print(f"Loading adapter: {adapter_name}")
        
        # Load via PEFT
        adapter = PeftModel.from_pretrained(
            self.base_model,
            path,
            adapter_name=adapter_name,
        )
        
        self.loaded_adapters[adapter_name] = {
            'path': path,
            'adapter': adapter,
        }
    
    def activate(self, adapter_name: str):
        """Make adapter active for next generation."""
        if adapter_name not in self.loaded_adapters:
            self.load(adapter_name)
        
        self.base_model.set_adapter(adapter_name)
        self.active_adapter = adapter_name
    
    def deactivate(self):
        """Disable all adapters (use base model)."""
        if self.active_adapter:
            self.base_model.disable_adapter()
            self.active_adapter = None
    
    def combine(self, adapter_names: list[str]):
        """Combine multiple adapters (e.g., Router + Topic + Denk)."""
        # Ensure all loaded
        for name in adapter_names:
            if name not in self.loaded_adapters:
                self.load(name)
        
        # Combine via PEFT's adapter composition
        # (requires PEFT 0.8+)
        self.base_model.set_adapter(adapter_names)
    
    def unload(self, adapter_name: str):
        """Free memory."""
        if adapter_name in self.loaded_adapters:
            del self.loaded_adapters[adapter_name]
```

### 6.2 Orchestrator (Pipeline)

**Datei:** `src/auralis/inference/orchestrator.py`

```python
"""
Auralis Orchestrator:
  Koordiniert Router → Topic → Denk → Logik → Output
"""

import json
from typing import Optional


class AuralisOrchestrator:
    """Main inference orchestrator."""
    
    def __init__(self, model, lora_manager, tokenizer):
        self.model = model
        self.lora_manager = lora_manager
        self.tokenizer = tokenizer
        
        # Always-load adapters
        self.lora_manager.load("router_lora")
        self.lora_manager.load("denk_lora")
        self.lora_manager.load("logik_lora")
    
    def respond(
        self,
        user_message: str,
        conversation_history: list[dict] = None,
    ) -> dict:
        """
        Main response pipeline.
        
        Returns:
            {
                'answer': str,
                'metadata': {
                    'level': int,
                    'topics': list,
                    'tools_used': list,
                    'thinking': str,
                    'reflection': str,
                    'inference_time_ms': int,
                }
            }
        """
        if conversation_history is None:
            conversation_history = []
        
        # === Step 1: Routing ===
        self.lora_manager.activate("router_lora")
        route_decision = self._route(user_message)
        
        level = route_decision['level']
        topics = route_decision['topics']
        needs_tools = route_decision['tools']
        
        # === Step 2: Topic-LoRAs laden ===
        for topic in topics:
            self.lora_manager.load(f"{topic}_lora")
        
        # === Step 3: Generate (mit Meta-LoRAs) ===
        adapters = []
        
        if level >= 2 and topics:
            adapters.extend([f"{t}_lora" for t in topics])
        
        if level >= 3:
            adapters.append("denk_lora")
        
        if adapters:
            self.lora_manager.combine(adapters)
        else:
            self.lora_manager.deactivate()
        
        # Generate main answer
        initial_answer = self._generate(user_message, conversation_history)
        
        # === Step 4: Self-Verification (bei Level 4+) ===
        reflection = None
        final_answer = initial_answer
        
        if level >= 4:
            self.lora_manager.activate("logik_lora")
            reflection_result = self._self_verify(user_message, initial_answer)
            
            if reflection_result['needs_correction']:
                # Regeneriere mit Feedback
                final_answer = reflection_result['corrected_answer']
                reflection = reflection_result['reflection']
        
        # === Step 5: Post-Processing ===
        # Strip <think>, <reflection>, <lora>, <route> tags
        final_answer = self._strip_meta_tags(final_answer)
        
        return {
            'answer': final_answer,
            'metadata': {
                'level': level,
                'topics': topics,
                'tools_used': [],
                'thinking': None,  # Extract from answer if needed
                'reflection': reflection,
            }
        }
    
    def _route(self, user_message: str) -> dict:
        """Router-LoRA: Komplexitäts-Entscheidung."""
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            add_generation_prompt=True,
        )
        
        output = self.model.generate(
            self.tokenizer.encode(prompt, return_tensor=True).cuda().unsqueeze(0),
            max_new_tokens=100,
            temperature=0.1,  # Deterministisch für Routing
        )
        
        text = self.tokenizer.decode(output[0])
        
        # Parse JSON
        try:
            # Extract JSON from generated text
            json_str = text[text.find('{'):text.rfind('}')+1]
            decision = json.loads(json_str)
        except (ValueError, json.JSONDecodeError):
            # Fallback
            decision = {"level": 1, "topics": [], "tools": False}
        
        return decision
    
    def _generate(self, user_message: str, history: list) -> str:
        """Generate answer with current LoRA setup."""
        messages = history + [{"role": "user", "content": user_message}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        
        output = self.model.generate(
            self.tokenizer.encode(prompt, return_tensor=True).cuda().unsqueeze(0),
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
        )
        
        return self.tokenizer.decode(output[0][len(prompt):])
    
    def _self_verify(self, user_message: str, initial_answer: str) -> dict:
        """Logik-LoRA verifiziert Antwort."""
        verification_prompt = (
            f"[USER FRAGE] {user_message}\n"
            f"[DRAFT ANTWORT] {initial_answer}\n"
            f"Prüfe den Draft auf Fehler, Widersprüche, fehlende Informationen."
        )
        
        messages = [{"role": "user", "content": verification_prompt}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        
        output_tokens = self.model.generate(
            self.tokenizer.encode(prompt, return_tensor=True).cuda().unsqueeze(0),
            max_new_tokens=512,
            temperature=0.3,
        )
        
        verification = self.tokenizer.decode(output_tokens[0][len(prompt):])
        
        # Check if correction needed
        needs_correction = "[CORRECTED]" in verification or "Fehler" in verification
        
        if needs_correction:
            # Extract corrected version
            if "[CORRECTED]" in verification:
                corrected = verification.split("[CORRECTED]")[1].strip()
            else:
                corrected = initial_answer  # Fallback
            
            return {
                'needs_correction': True,
                'corrected_answer': corrected,
                'reflection': verification,
            }
        
        return {
            'needs_correction': False,
            'corrected_answer': initial_answer,
            'reflection': verification,
        }
    
    def _strip_meta_tags(self, text: str) -> str:
        """Remove internal tags from user-facing output."""
        import re
        
        # Remove <think>...</think>, <reflection>...</reflection>, etc.
        patterns = [
            r'<think>.*?</think>',
            r'<reflection>.*?</reflection>',
            r'<lora>.*?</lora>',
            r'<route>.*?</route>',
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL)
        
        # Cleanup whitespace
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
        
        return text
```

---

## 7. Topic-LoRA Entwicklung (Template)

**Für jedes neue Topic den gleichen Workflow:**

```
1. YAML-Fakten-Spec erstellen
   - 100 atomare Fakten
   - 80 train + 20 val (disjunkt!)
   - Mit Quellenangaben

2. Samples generieren
   - 3 Paraphrasen pro Fact = 240 core samples
   - + 500-750 kontextuelle = ~1000 total
   - Kategorien abdecken

3. LoRA trainieren
   - MoRA-Method für Fakten
   - Early Stopping bei Val ~0.2-0.3
   - NICHT: Loss unter 0.05 (Memorization!)

4. Evaluieren
   - 50 disjunkte Test-Fragen
   - Accuracy > 70% anstreben
   - Baseline-Vergleich (ohne LoRA)

5. Deployen
   - In lora_adapters/ speichern
   - Router-LoRA Training updaten
   - In LoRAManager registrieren
```

---

## 8. Geplante Topic-LoRAs (Roadmap)

```
Phase 5a (Core): 4 Wochen
  ✓ Router-LoRA
  ✓ Denk-LoRA
  ✓ Logik-LoRA
  ✓ Medizin-LoRA (Proof-of-Concept)

Phase 5b (Expansion): 2-3 Wochen
  → Recht
  → Technik / Programmierung
  → Kochen / Ernährung
  → Reisen
  → Finanzen (Basics)

Phase 5c (Long-tail): on demand
  → Pflanzenpflege
  → Tierhaltung
  → Sport
  → Musik-Theorie
  → etc.
```

---

## 9. Akzeptanz-Kriterien

```
Meta-LoRAs:
  □ Router-LoRA: > 90% korrektes Level 0-5 Routing
  □ Denk-LoRA: Chain-of-Thought ist kohärent
  □ Logik-LoRA: Erkennt bekannte Fehler in Drafts

Topic-LoRAs (pro Adapter):
  □ Val Loss im Bereich 0.2-0.4 (nicht < 0.05 = Memorization!)
  □ Test-Fragen > 70% correct (disjunkt zum Training)
  □ Besser als Basis-Modell ohne LoRA
  □ Hot-Swap funktioniert ohne Crash
  □ Kombinierbar mit Meta-LoRAs

Orchestrator:
  □ Komplette Pipeline läuft stabil
  □ Level 0-5 Handling korrekt
  □ <think>, <reflection>, <lora> werden geschnitten
  □ Kein User-facing Leak von Internal Tags
  □ Latency-Ziele eingehalten:
    - Level 0: < 500ms
    - Level 1: < 1s
    - Level 2: < 3s
    - Level 3: < 7s
    - Level 4: < 12s
    - Level 5: < 20s

Deployment:
  □ LoRA-Manager stabil (kein Memory-Leak)
  □ Multi-LoRA-Composition funktioniert
  □ API-Endpoint für LoRA-Switch
  □ Monitoring (welcher LoRA wann aktiv)
```

---

## 10. Next Steps: Post-Phase 5

Nach diesen 5 Phasen ist Helix v2 "production-ready":

```
Post-Launch Arbeit:
  → Quantisierung (AWQ 4-bit)
  → vLLM Deployment
  → FastAPI Production-Server
  → Open WebUI Integration
  → User-Feedback-Loop
  → Continuous Topic-LoRA Erweiterung

Forschung/Experiment:
  → MoE aktivieren (8 Experten)
  → Multi-Token Prediction
  → Längerer Context (16k+)
  → Memory-LoRA (Persistent User Memory)
  → Live-Learning (Instant LoRA Creation)
```

---

## Zusammenfassung aller 6 Phasen

```
Phase 0:    Tokenizer                        (2-3 Tage)
Phase 0.5:  Model Architecture               (1 Woche)
Phase 1:    Pretraining (EN-heavy)           (3-4 Wochen, $500-800)
Phase 2:    Continued Bilingual + KL         (1-2 Wochen, $200-400)
Phase 3:    SFT (GaLore)                     (1 Woche, $100-200)
Phase 4:    ORPO Alignment                   (3-5 Tage, $50-100)
Phase 5:    LoRA System                      (2-3 Wochen, $50-100)

Total:      4-5 Monate, $900-1600
```

---

*Phase 5 Spec Version 1.0 — April 2026*
*Damit sind alle Phasen für Auralis v2 / Helix v2 vollständig spezifiziert.*
