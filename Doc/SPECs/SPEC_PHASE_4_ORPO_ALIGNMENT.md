# Phase 4: ORPO Alignment

**Projekt:** Auralis v2 / Helix v2
**Phase:** 4 (Präferenz-Alignment ohne komplexes RLHF)
**Dauer:** 3-5 Tage
**Ziel:** Modell bevorzugt hilfreiche, ehrliche Antworten
**Voraussetzung:** Phase 3 SFT Checkpoint
**Hardware:** H200 oder RTX Pro 5000
**Budget:** ~$50-100

---

## 1. Warum ORPO statt DPO/RLHF?

**Das Problem mit klassischem RLHF:**

```
RLHF-Pipeline (wie GPT-3 → GPT-4):
  1. SFT Modell trainieren
  2. Reward Model trainieren (Human Preferences)
  3. PPO mit Reward Model
  
Nachteile:
  ✗ 3 Modelle gleichzeitig (Policy, Ref, Reward) → VRAM-Hungrig
  ✗ PPO instabil, schwer zu tunen
  ✗ Reward Hacking
  ✗ Braucht Reward Model (zusätzliches Training)
```

**DPO (Direct Preference Optimization):**

```
Besser als RLHF, aber:
  ✗ Braucht 2 Modelle im Speicher (Policy + Reference)
  ✗ Reference muss identisch zu Initial Policy sein
  ✗ Separate SFT + DPO Schritte
```

**ORPO (Odds Ratio Preference Optimization):**

```
Vorteile:
  ✓ NUR EIN Modell im Speicher
  ✓ Kein Reference Model nötig
  ✓ SFT + Preference in EINEM Schritt möglich
  ✓ Stabiler als DPO
  ✓ Geringerer VRAM-Bedarf
  ✓ Einfacher zu implementieren

Formel (vereinfacht):
  loss = sft_loss(chosen) + λ * log_odds_ratio(chosen, rejected)
  
Wo:
  sft_loss: Standard Cross-Entropy auf chosen response
  log_odds_ratio: Erhöht Wahrscheinlichkeit chosen vs rejected
```

---

## 2. Präferenz-Daten Strategie

### 2.1 Daten-Mix

```
Total: 50k-100k Preference Pairs

Strategie:
  → Gleiche Frage, zwei Antworten
  → "Chosen" = gute Antwort (DeepSeek V3 / GPT-4)
  → "Rejected" = schlechtere Antwort
  
Quellen der Rejected-Samples:
  1. Llama 2 7B Generierungen (~40%)
     → Natürlich schlechter als DeepSeek V3
  
  2. Frühere Helix-Versionen (~20%)
     → "Helix v1 Artefakte" als negatives Beispiel
  
  3. Helix nach nur 500 SFT steps (~20%)
     → Vor-SFT "ehrlicher" Output
  
  4. Synthetisch schlechter gemacht (~20%)
     → Floskeln hinzugefügt
     → Kürzer/länger als optimal
     → Thema verfehlt
```

### 2.2 Automatisierte Generation

**Script:** `scripts/data/generate_preference_pairs.py`

```python
"""
Generiert Präferenz-Paare für ORPO.
"""

import json
from pathlib import Path
from openai import OpenAI  # DeepSeek V3 kompatibel
import random
from tqdm import tqdm


def generate_preference_pairs(
    prompts_file: str,
    output_file: str,
    n_pairs: int = 50000,
):
    """
    Generiert Preference Pairs aus einer Prompt-Liste.
    
    Pipeline:
      1. Für jeden Prompt:
         - Chosen: DeepSeek V3 generiert
         - Rejected: Schlechtere Quelle (siehe Strategien)
      2. Filter für Qualität
      3. Schreibe als JSONL
    """
    # Load prompts
    with open(prompts_file) as f:
        prompts = [json.loads(line) for line in f]
    
    print(f"Processing {len(prompts)} prompts...")
    
    deepseek = OpenAI(
        api_key="YOUR_KEY",
        base_url="https://api.deepseek.com",
    )
    
    helix_v1 = OpenAI(
        api_key="none",
        base_url="http://localhost:8000/v1",  # Helix v1 running
    )
    
    pairs = []
    
    for prompt in tqdm(prompts):
        # CHOSEN: DeepSeek V3 High-Quality
        try:
            chosen_response = deepseek.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "Du bist ein hilfreicher, klarer Assistent. Antworte präzise ohne Floskeln."},
                    {"role": "user", "content": prompt['question']},
                ],
                temperature=0.3,
            )
            chosen = chosen_response.choices[0].message.content
        except Exception as e:
            print(f"DeepSeek error: {e}")
            continue
        
        # REJECTED: Strategie wählen
        strategy = random.choice([
            "helix_v1",
            "floskel_injection",
            "too_short",
            "too_long",
            "topic_drift",
        ])
        
        if strategy == "helix_v1":
            # Helix v1 (natürlich schlechter)
            try:
                rejected_response = helix_v1.chat.completions.create(
                    model="helix-v1",
                    messages=[{"role": "user", "content": prompt['question']}],
                    temperature=0.7,
                )
                rejected = rejected_response.choices[0].message.content
            except Exception:
                continue
        
        elif strategy == "floskel_injection":
            # Chosen + Floskeln
            rejected = (
                "Natürlich! Das ist eine ausgezeichnete Frage. "
                "Gerne erkläre ich dir das. " + chosen + 
                " Ich hoffe, das hilft dir weiter!"
            )
        
        elif strategy == "too_short":
            # Viel zu kurz
            words = chosen.split()
            rejected = " ".join(words[:max(3, len(words)//5)]) + "..."
        
        elif strategy == "too_long":
            # Künstlich aufgebläht
            rejected = chosen + "\n\n" + chosen  # Repetition
        
        elif strategy == "topic_drift":
            # Thema verfehlen
            rejected = (
                "Das ist ein interessantes Thema. "
                "Es gibt viele Aspekte dazu. "
                "Man könnte sagen, dass es komplex ist."
            )
        
        # Pair speichern
        pair = {
            "prompt": prompt['question'],
            "chosen": chosen,
            "rejected": rejected,
            "strategy": strategy,
        }
        pairs.append(pair)
        
        if len(pairs) >= n_pairs:
            break
    
    # Filter
    pairs = [p for p in pairs if _is_valid_pair(p)]
    
    # Save
    with open(output_file, 'w') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    print(f"\n✓ {len(pairs)} preference pairs gespeichert: {output_file}")


def _is_valid_pair(pair):
    """Filter für Qualität."""
    chosen = pair['chosen']
    rejected = pair['rejected']
    
    # Mindestlänge
    if len(chosen) < 20 or len(rejected) < 10:
        return False
    
    # Maximallänge
    if len(chosen) > 2000 or len(rejected) > 2000:
        return False
    
    # Chosen und Rejected müssen unterschiedlich sein
    if chosen == rejected:
        return False
    
    # Chosen sollte NICHT mit Floskel starten
    floskeln = ["Natürlich", "Gerne", "Klar!", "Absolut"]
    if any(chosen.startswith(f) for f in floskeln):
        return False
    
    return True


if __name__ == "__main__":
    generate_preference_pairs(
        prompts_file="data/training/phase4/prompts.jsonl",
        output_file="data/training/phase4/preference_pairs.jsonl",
        n_pairs=50000,
    )
```

### 2.3 Datenformat

```jsonl
{"prompt": "Was ist Photosynthese?", "chosen": "Photosynthese ist der Prozess, bei dem Pflanzen Lichtenergie in chemische Energie umwandeln. Dabei wird CO2 und Wasser mithilfe von Chlorophyll zu Glukose und Sauerstoff umgewandelt.", "rejected": "Natürlich! Das ist eine ausgezeichnete Frage. Gerne erkläre ich dir das. Photosynthese ist wichtig für Pflanzen.", "strategy": "floskel_injection"}
{"prompt": "Wie kocht man Reis?", "chosen": "Reis kochst du so: 1 Teil Reis mit 2 Teilen Wasser in einen Topf, aufkochen, dann 15 Minuten auf niedriger Stufe köcheln lassen. Salz nach Geschmack dazu.", "rejected": "Kochen. Wasser. Warten.", "strategy": "too_short"}
```

---

## 3. ORPO Implementation

**Datei:** `src/auralis/training/orpo.py`

```python
"""
ORPO: Odds Ratio Preference Optimization.

Paper: https://arxiv.org/abs/2403.07691
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ORPOLoss:
    """ORPO Loss-Funktion."""
    
    def __init__(
        self,
        beta: float = 0.1,
        label_pad_token_id: int = -100,
    ):
        self.beta = beta
        self.label_pad_token_id = label_pad_token_id
    
    def compute_loss(
        self,
        model,
        batch: dict,
    ) -> dict:
        """
        Berechnet ORPO Loss.
        
        Batch muss enthalten:
          - chosen_input_ids
          - chosen_labels
          - chosen_attention_mask
          - rejected_input_ids
          - rejected_labels
          - rejected_attention_mask
        """
        # === Chosen Forward ===
        chosen_out = model(
            input_ids=batch['chosen_input_ids'],
            labels=batch['chosen_labels'],
            attention_mask=batch['chosen_attention_mask'],
        )
        chosen_logits = chosen_out['logits']
        chosen_nll = self._compute_nll(chosen_logits, batch['chosen_labels'])
        
        # === Rejected Forward ===
        rejected_out = model(
            input_ids=batch['rejected_input_ids'],
            labels=batch['rejected_labels'],
            attention_mask=batch['rejected_attention_mask'],
        )
        rejected_logits = rejected_out['logits']
        rejected_nll = self._compute_nll(rejected_logits, batch['rejected_labels'])
        
        # === Log-Probabilities ===
        # Negative NLL = Log-Prob
        chosen_logps = -chosen_nll
        rejected_logps = -rejected_nll
        
        # === Log Odds Ratio ===
        # log(sigmoid(logit)) = log(p) - log(1-p)
        # We want ratio of chosen being preferred over rejected
        log_odds_ratio = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) 
            - torch.log1p(-torch.exp(rejected_logps))
        )
        
        # Sigmoid Ratio Loss
        ratio_loss = -F.logsigmoid(log_odds_ratio).mean()
        
        # === Total Loss ===
        # SFT Loss auf Chosen + Scaled Ratio Loss
        sft_loss = chosen_nll.mean()
        total_loss = sft_loss + self.beta * ratio_loss
        
        # === Metrics ===
        # Accuracy: chosen > rejected
        accuracy = (chosen_logps > rejected_logps).float().mean()
        
        # Reward Margin
        margin = (chosen_logps - rejected_logps).mean()
        
        return {
            'loss': total_loss,
            'sft_loss': sft_loss.detach(),
            'ratio_loss': ratio_loss.detach(),
            'accuracy': accuracy.detach(),
            'margin': margin.detach(),
            'chosen_logps': chosen_logps.mean().detach(),
            'rejected_logps': rejected_logps.mean().detach(),
        }
    
    def _compute_nll(self, logits, labels):
        """Negative Log-Likelihood, maskiert Padding."""
        # Shift für next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Loss per token
        loss_fn = nn.CrossEntropyLoss(
            ignore_index=self.label_pad_token_id,
            reduction='none',
        )
        
        per_token_loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        
        # Reshape back
        per_token_loss = per_token_loss.view(shift_labels.size())
        
        # Mask and average per sequence
        valid_mask = (shift_labels != self.label_pad_token_id).float()
        per_seq_loss = (per_token_loss * valid_mask).sum(dim=-1) / valid_mask.sum(dim=-1).clamp(min=1)
        
        return per_seq_loss
```

---

## 4. Konfiguration

**Datei:** `configs/training/phase4_orpo.yaml`

```yaml
experiment:
  name: "helix_v2_phase4_orpo"
  version: "1.0"
  description: "ORPO Alignment without reward model"

model:
  config_path: "configs/model/helix_v2_3b.yaml"
  resume_from: "checkpoints/phase3_sft/best.pt"

data:
  train_path: "data/training/phase4/preference_pairs.jsonl"
  val_path: "data/training/phase4/preference_val.jsonl"
  seq_length: 2048

training:
  device: "cuda"
  dtype: "bfloat16"
  
  reset_optimizer: true  # Wichtig (Lektion aus v1!)
  
  optimizer:
    name: "adamw"
    lr: 1.0e-5  # Niedriger als SFT
    betas: [0.9, 0.95]
    weight_decay: 0.0
  
  scheduler:
    type: "cosine"
    warmup_steps: 100
    min_lr_ratio: 0.1
  
  batch_size_per_device: 2  # Klein wegen 2x forward (chosen + rejected)
  gradient_accumulation: 16
  
  gradient_clip_norm: 1.0
  
  # ORPO-spezifisch
  orpo:
    beta: 0.1  # Standard, kann 0.05-0.2 variieren
  
  # Duration
  num_epochs: 2
  
  # Early Stopping
  early_stopping:
    enabled: true
    patience: 3
    metric: "val_accuracy"
    mode: "max"
  
  gradient_checkpointing: true
  torch_compile: true

logging:
  log_every: 10
  eval_every: 200
  save_every: 500
  
  wandb:
    enabled: true
    project: "auralis-v2"
    tags: ["phase4", "orpo", "alignment"]

checkpointing:
  output_dir: "checkpoints/phase4_orpo"
  save_last_n: 3
  save_best: true
  
  best_metric: "val_accuracy"
  best_mode: "max"

evaluation:
  val_data: "data/training/phase4/preference_val.jsonl"
  max_val_batches: 100
  
  benchmarks:
    # Baseline bleibt wichtig (keine Regression!)
    - name: "baseline_50"
      frequency: 500
    
    # MT-Bench für Qualität
    - name: "mt_bench"
      path: "data/eval/mt_bench_10.jsonl"
      frequency: 1000
    
    # AlpacaEval 2 (automatisiert)
    - name: "alpaca_eval_subset"
      frequency: 1000
    
    # Helpfulness / Harmlessness Test
    - name: "helpful_harmless"
      path: "data/eval/hh_20.jsonl"
      frequency: 500
```

---

## 5. Training Script

**Datei:** `scripts/orpo/train_phase4.py`

```python
"""
Phase 4 ORPO Training.
"""

import argparse
import torch
from torch.utils.data import DataLoader, Dataset
import json
from pathlib import Path

from auralis.model import build_model
from auralis.tokenizer import HelixTokenizer
from auralis.training.orpo import ORPOLoss
from auralis.training.utils import load_config


class PreferenceDataset(Dataset):
    """Dataset für Preference Pairs."""
    
    def __init__(
        self,
        path: str,
        tokenizer,
        seq_length: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        self.pairs = []
        with open(path) as f:
            for line in f:
                self.pairs.append(json.loads(line))
        
        print(f"Loaded {len(self.pairs)} preference pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Format beide responses im Chat-Template
        chosen_messages = [
            {"role": "user", "content": pair['prompt']},
            {"role": "assistant", "content": pair['chosen']},
        ]
        rejected_messages = [
            {"role": "user", "content": pair['prompt']},
            {"role": "assistant", "content": pair['rejected']},
        ]
        
        # Tokenize beide
        chosen_item = self._tokenize_conversation(chosen_messages, pair['chosen'])
        rejected_item = self._tokenize_conversation(rejected_messages, pair['rejected'])
        
        return {
            'chosen_input_ids': chosen_item['input_ids'],
            'chosen_labels': chosen_item['labels'],
            'chosen_attention_mask': chosen_item['attention_mask'],
            'rejected_input_ids': rejected_item['input_ids'],
            'rejected_labels': rejected_item['labels'],
            'rejected_attention_mask': rejected_item['attention_mask'],
        }
    
    def _tokenize_conversation(self, messages, response_text):
        """Tokenisiert und maskiert Labels."""
        full_prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
        )
        tokens = self.tokenizer.encode(full_prompt, add_eos=True)
        
        # Truncate
        if len(tokens) > self.seq_length:
            tokens = tokens[:self.seq_length]
        
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()
        
        # TODO: Mask non-assistant tokens (wie in SFTDataset)
        # Vereinfacht: Mask für User/System Parts
        
        # Padding
        if len(input_ids) < self.seq_length:
            pad_len = self.seq_length - len(input_ids)
            input_ids = torch.cat([
                input_ids,
                torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long),
            ])
            labels = torch.cat([
                labels,
                torch.full((pad_len,), -100, dtype=torch.long),
            ])
        
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='configs/training/phase4_orpo.yaml',
    )
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    print("=" * 60)
    print("PHASE 4: ORPO Alignment")
    print("=" * 60)
    
    # === Model ===
    print(f"\nLoading from Phase 3: {config.model.resume_from}")
    model = build_model(config.model.config_path)
    
    checkpoint = torch.load(config.model.resume_from, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.cuda()
    
    # === Data ===
    tokenizer = HelixTokenizer()
    
    train_dataset = PreferenceDataset(
        path=config.data.train_path,
        tokenizer=tokenizer,
        seq_length=config.data.seq_length,
    )
    val_dataset = PreferenceDataset(
        path=config.data.val_path,
        tokenizer=tokenizer,
        seq_length=config.data.seq_length,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size_per_device,
        shuffle=True,
        num_workers=4,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size_per_device,
        shuffle=False,
    )
    
    # === Optimizer (Reset!) ===
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.optimizer.lr,
        betas=config.training.optimizer.betas,
        weight_decay=config.training.optimizer.weight_decay,
    )
    
    # === ORPO Loss ===
    orpo_loss = ORPOLoss(beta=config.training.orpo.beta)
    
    # === Training ===
    best_accuracy = 0
    patience_counter = 0
    
    for epoch in range(config.training.num_epochs):
        print(f"\n=== Epoch {epoch + 1} ===")
        
        model.train()
        
        for step, batch in enumerate(train_loader):
            batch = {k: v.cuda() for k, v in batch.items()}
            
            # ORPO Loss
            losses = orpo_loss.compute_loss(model, batch)
            
            loss = losses['loss'] / config.training.gradient_accumulation
            
            loss.backward()
            
            if (step + 1) % config.training.gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=config.training.gradient_clip_norm,
                )
                optimizer.step()
                optimizer.zero_grad()
            
            # Log
            if step % config.logging.log_every == 0:
                print(
                    f"  Step {step:5d} | "
                    f"total {losses['loss'].item():.4f} | "
                    f"sft {losses['sft_loss'].item():.4f} | "
                    f"ratio {losses['ratio_loss'].item():.4f} | "
                    f"acc {losses['accuracy'].item():.2%} | "
                    f"margin {losses['margin'].item():.3f}"
                )
            
            # Eval
            if step % config.logging.eval_every == 0 and step > 0:
                val_metrics = evaluate_orpo(model, val_loader, orpo_loss)
                print(f"    Val Accuracy: {val_metrics['accuracy']:.2%}")
                print(f"    Val Margin:   {val_metrics['margin']:.3f}")
                
                if val_metrics['accuracy'] > best_accuracy:
                    best_accuracy = val_metrics['accuracy']
                    patience_counter = 0
                    save_checkpoint(model, epoch, step, val_metrics, config, name='best')
                    print(f"    ✓ New best accuracy: {best_accuracy:.2%}")
                else:
                    patience_counter += 1
                
                if patience_counter >= config.training.early_stopping.patience:
                    print("\n⏹  Early stopping")
                    return
    
    print(f"\n✓ Training complete. Best accuracy: {best_accuracy:.2%}")


def evaluate_orpo(model, val_loader, orpo_loss):
    """Evaluate on val preferences."""
    model.eval()
    
    total_accuracy = 0
    total_margin = 0
    n = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            losses = orpo_loss.compute_loss(model, batch)
            
            total_accuracy += losses['accuracy'].item()
            total_margin += losses['margin'].item()
            n += 1
    
    model.train()
    return {
        'accuracy': total_accuracy / n,
        'margin': total_margin / n,
    }


def save_checkpoint(model, epoch, step, metrics, config, name):
    """Save checkpoint."""
    path = Path(config.checkpointing.output_dir) / f"{name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'step': step,
        'metrics': metrics,
        'model': model.state_dict(),
    }, path)


if __name__ == "__main__":
    main()
```

---

## 6. Erwartete Ergebnisse

```
Vor Phase 4 (Phase 3 SFT):
  Preference Accuracy:  ~55% (wenig besser als zufällig)
  MT-Bench:             6.5
  Baseline-50:          80%
  Floskel-Häufigkeit:   25% der Antworten

Nach Phase 4 ORPO:
  Preference Accuracy:  > 85%
  MT-Bench:             7.2+
  Baseline-50:          80% (keine Regression!)
  Floskel-Häufigkeit:   < 5%
  
Qualitative Verbesserung:
  → Weniger "Natürlich/Gerne/Klar" Floskeln
  → Direktere, fokussiertere Antworten
  → Bessere Länge (nicht zu lang, nicht zu kurz)
  → Klareres Deutsch
```

---

## 7. Akzeptanz-Kriterien

```
Data:
  □ 50k+ Preference Pairs generiert
  □ Validierungs-Set disjunkt
  □ Verschiedene Rejected-Strategien vertreten
  □ Qualitätsfilter angewandt

Training:
  □ ORPO Loss implementiert und stabil
  □ Preference Accuracy steigt stetig
  □ Reward Margin > 0 (positiv)
  □ --reset-optimizer aktiv
  □ Early Stopping greift sinnvoll

Quality:
  □ Val Preference Accuracy > 85%
  □ Baseline-50 nicht verschlechtert
  □ MT-Bench Score verbessert
  □ Keine Floskel-Regressionen
  □ Sampling-Tests qualitativ gut
```

---

## 8. Next Steps

Nach Phase 4 ist das Basismodell "fertig":
→ SPEC_PHASE_5_LORA_SYSTEM.md (LoRA-basiertes Erweiterungssystem)

---

*Phase 4 Spec Version 1.0 — April 2026*
