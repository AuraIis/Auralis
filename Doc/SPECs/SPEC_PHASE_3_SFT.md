# Phase 3: Supervised Fine-Tuning (SFT)

**Project:** Auralis v2 / Helix v2
**Phase:** 3 (learning instruction-following)
**Duration:** 1 week
**Goal:** Model follows instructions naturally in DE + EN + Code
**Prerequisite:** Phase 2 checkpoint exists
**Hardware:** H200 or RTX Pro 5000
**Budget:** ~$100-200

---

## 1. Goals

```
What the SFT achieves:
  ✓ Understand chat format (<|user|>, <|assistant|>)
  ✓ Follow instructions naturally
  ✓ Format consistency (clean answers)
  ✓ Politeness, helpfulness
  ✓ Able to express uncertainty
  ✓ Conduct small talk naturally

What it does NOT do:
  ✗ New world knowledge (comes from pretrain)
  ✗ Preferences (comes in Phase 4 ORPO)
  ✗ Domain expertise (comes in Phase 5 LoRAs)
```

---

## 2. Data Strategy (lesson from v1!)

**Distribution:**

```
Total: 100-200k samples

German:    45% (45-90k samples)
  → Own generation via DeepSeek V3
  → Gemini 2.0 Flash
  → Tülu 3 translated to German

English:   50% (50-100k samples)
  → Tülu 3 (best-quality)
  → UltraChat 200k
  → HelpSteer2

Code:      5% (5-10k samples)
  → MagiCoder
  → OpenCodeInterpreter
  → Tülu-Code
```

**Data categories (closing the gaps from v1):**

```
Q&A facts:             25%  - "What is X?"
Explanations:          15%  - "Explain Y"
Instructions:          15%  - "How do you do Z?"
Creative:              10%  - Texts, stories
Code help:              5%  - Debugging, refactoring

SMALL TALK:            10%  - "Hello", "Thanks", "How are you" ⚠ v1 gap!
UNCERTAINTY:            5%  - "I don't know" ⚠ v1 gap!
REASONING:             10%  - "Why/How come" with <think>
MULTI-TURN:             5%  - Longer conversations
```

**Quality filter (aggressive, learned from v1):**

```python
BLACKLIST_PHRASES = [
    # Floskeln
    "Natürlich!",
    "Gerne helfe ich",
    "Absolut!",
    "Das ist eine ausgezeichnete Frage",
    "Hervorragende Frage",
    "Als KI-Assistent",
    "Als Sprachmodell",
    
    # Englisch-Reste bei deutschen Antworten
    "Here is",
    "Let me",
    "I can help",
    
    # Artefakte aus v1
    "A:",
    "Q:",
    "Frage:",
    "Antwort:",
]

def is_quality_sample(sample):
    content = sample['messages'][-1]['content']
    
    # Zu kurz / zu lang?
    if len(content) < 10 or len(content) > 2000:
        return False
    
    # Blacklist
    for phrase in BLACKLIST_PHRASES:
        if content.lower().startswith(phrase.lower()):
            return False
    
    # Einleitungs-Floskeln
    if any(content.startswith(p) for p in ["Natürlich", "Gerne", "Klar!"]):
        return False
    
    return True
```

---

## 3. Configuration

**File:** `configs/training/phase3_sft.yaml`

```yaml
experiment:
  name: "helix_v2_phase3_sft"
  version: "1.0"
  description: "SFT with GaLore for full finetuning in LoRA VRAM budget"

model:
  config_path: "configs/model/helix_v2_3b.yaml"
  resume_from: "checkpoints/phase2_continued/best.pt"
  
  # SFT-specific modifications
  dropout:
    attention: 0.05   # Leichtes dropout für Regularisierung
    ffn: 0.05
    residual: 0.0

data:
  train_path: "data/training/phase3_sft/train.jsonl"
  val_path: "data/training/phase3_sft/val.jsonl"
  
  seq_length: 2048
  
  # Chat-Format MUSS identisch zum Training sein
  chat_template: "helix_v2_standard"
  
  # Loss nur auf Assistant-Tokens
  mask_user_tokens: true

training:
  device: "cuda"
  dtype: "bfloat16"
  
  # KRITISCH: Optimizer resetten (Lektion aus v1!)
  reset_optimizer: true
  
  # GaLore Setup
  optimizer:
    name: "galore_adamw"  # GaLore-wrapped AdamW
    lr: 2.0e-5  # Niedrig für SFT
    betas: [0.9, 0.95]
    weight_decay: 0.0  # Kein WD bei SFT
    
    # GaLore-specific
    galore_rank: 128
    galore_update_proj_gap: 200
    galore_scale: 0.25
    galore_proj_type: "std"
  
  scheduler:
    type: "cosine"
    warmup_steps: 100
    min_lr_ratio: 0.1
  
  batch_size_per_device: 4
  gradient_accumulation: 16
  # Effective batch: 64 sequences
  
  gradient_clip_norm: 1.0
  
  # Early Stopping (Lektion aus v1!)
  max_epochs: 3
  early_stopping:
    enabled: true
    patience: 3  # Eval mit Val-Loss
    min_delta: 0.01
  
  gradient_checkpointing: true
  torch_compile: true

logging:
  log_every: 10
  eval_every: 200   # Öfter als Pretrain (kurz SFT)
  save_every: 500
  
  wandb:
    enabled: true
    project: "auralis-v2"
    tags: ["phase3", "sft", "galore"]

checkpointing:
  output_dir: "checkpoints/phase3_sft"
  save_last_n: 3
  save_best: true

evaluation:
  val_data: "data/training/phase3_sft/val.jsonl"
  max_val_batches: 200
  
  # Baseline + neue Tests
  benchmarks:
    - name: "baseline_50_deutsch"
      path: "data/eval/baseline_questions.yaml"
      frequency: 200
    
    - name: "mt_bench_subset"
      path: "data/eval/mt_bench_10.jsonl"
      frequency: 500
    
    - name: "instruction_following_eval"
      path: "data/eval/ifeval_50.jsonl"
      frequency: 500
    
    - name: "smalltalk_quality"
      path: "data/eval/smalltalk_20.jsonl"
      frequency: 200
      metric: "human_like_score"
    
    - name: "uncertainty_eval"
      path: "data/eval/uncertainty_15.jsonl"
      frequency: 500
      note: "Checks if model says 'I don't know' appropriately"

monitoring:
  alert_on:
    - val_loss_increase: 3
    - val_loss_plateau: 5
    - nan_in_loss: true
```

---

## 4. GaLore Integration

**File:** `src/auralis/training/galore.py`

```python
"""
GaLore: Gradient Low-Rank Projection.
Ermöglicht volles Finetuning mit LoRA-artigem VRAM.
"""

import torch
from torch.optim import AdamW


class GaLoreAdamW(AdamW):
    """AdamW mit GaLore Gradient Projection.
    
    Projiziert Gradienten auf low-rank subspace für Memory-Effizienz.
    """
    
    def __init__(
        self,
        params,
        lr=1e-4,
        rank=128,
        update_proj_gap=200,
        scale=0.25,
        proj_type='std',
        **adamw_kwargs,
    ):
        super().__init__(params, lr=lr, **adamw_kwargs)
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.proj_type = proj_type
        
        # Projection matrices per parameter
        self.projectors = {}
    
    @torch.no_grad()
    def step(self, closure=None):
        """Step mit GaLore projection."""
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                
                # Get or init projector
                param_id = id(param)
                state = self.state[param]
                
                if 'step' not in state:
                    state['step'] = 0
                
                state['step'] += 1
                
                # Update projector periodically
                if state['step'] % self.update_proj_gap == 0 or param_id not in self.projectors:
                    self._update_projector(param)
                
                # Project gradient
                grad_original_shape = param.grad.shape
                projected_grad = self._project(param.grad, param_id)
                
                # AdamW step in projected space
                # (Simplified - real impl uses projected state too)
                
                # Project back
                param.grad = self._project_back(projected_grad, param_id, grad_original_shape)
        
        # Standard AdamW step with projected gradients
        super().step(closure)
    
    def _update_projector(self, param):
        """Update projection matrix via SVD."""
        if param.ndim < 2:
            return  # Only project 2D+ tensors
        
        # SVD on gradient
        try:
            U, S, V = torch.svd_lowrank(param.grad, q=self.rank)
            self.projectors[id(param)] = {
                'U': U,
                'V': V,
            }
        except Exception as e:
            print(f"SVD failed: {e}")
    
    def _project(self, grad, param_id):
        """Project gradient to low-rank subspace."""
        if param_id not in self.projectors:
            return grad
        
        proj = self.projectors[param_id]
        # grad_projected = U^T @ grad @ V
        return proj['U'].T @ grad @ proj['V']
    
    def _project_back(self, projected_grad, param_id, original_shape):
        """Project back to full space."""
        if param_id not in self.projectors:
            return projected_grad
        
        proj = self.projectors[param_id]
        # grad_full = U @ projected @ V^T
        full = proj['U'] @ projected_grad @ proj['V'].T
        return full.reshape(original_shape) * self.scale
```

**Note:** Production-ready GaLore uses the official package:

```bash
pip install galore-torch
```

```python
from galore_torch import GaLoreAdamW
```

---

## 5. Chat Template in SFT

**File:** `src/auralis/training/sft_dataset.py`

```python
"""
SFT Dataset mit korrektem Chat-Template.
Label-Masking nur auf Assistant-Tokens.
"""

import json
import torch
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    """SFT Dataset with chat-template tokenization."""
    
    def __init__(
        self,
        path: str,
        tokenizer,
        seq_length: int = 2048,
        mask_user_tokens: bool = True,
    ):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.mask_user_tokens = mask_user_tokens
        
        # Load samples
        self.samples = []
        with open(path) as f:
            for line in f:
                self.samples.append(json.loads(line))
        
        print(f"Loaded {len(self.samples)} SFT samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        messages = sample['messages']
        
        # WICHTIG: Diese Funktion ist der ONE-AND-ONLY Weg
        # um Prompts zu bauen (verhindert v1-Bug)
        full_prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
        )
        
        # Tokenize
        tokens = self.tokenizer.encode(full_prompt, add_eos=True)
        
        # Truncate
        if len(tokens) > self.seq_length:
            tokens = tokens[:self.seq_length]
        
        # Input IDs = Token Sequence
        input_ids = torch.tensor(tokens, dtype=torch.long)
        
        # Labels: Copy of input_ids with masking
        labels = input_ids.clone()
        
        # Mask user and system tokens (-100 = ignore in loss)
        if self.mask_user_tokens:
            labels = self._mask_non_assistant(labels, messages)
        
        # Padding to seq_length
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
        
        # Attention mask
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }
    
    def _mask_non_assistant(self, labels, messages):
        """Set labels to -100 for everything except assistant tokens.
        
        Loss is only computed where labels != -100.
        So we only compute loss on what the model should generate.
        """
        # Find assistant token positions
        assistant_token_id = self.tokenizer.assistant_token_id
        end_token_id = self.tokenizer.end_token_id
        
        # Start with everything masked
        new_labels = torch.full_like(labels, -100)
        
        # Find assistant sections and unmask them
        in_assistant = False
        for i, token_id in enumerate(labels):
            if token_id == assistant_token_id:
                in_assistant = True
                continue
            if token_id == end_token_id and in_assistant:
                in_assistant = False
                # Include end token in loss
                new_labels[i] = token_id
                continue
            if in_assistant:
                new_labels[i] = token_id
        
        return new_labels
```

---

## 6. Training Script

**File:** `scripts/sft/train_phase3.py`

```python
"""
Phase 3 SFT Training mit GaLore.
"""

import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from auralis.model import build_model
from auralis.tokenizer import HelixTokenizer
from auralis.training.sft_dataset import SFTDataset
from auralis.training.galore import GaLoreAdamW
from auralis.training.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='configs/training/phase3_sft.yaml',
    )
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    print("=" * 60)
    print("PHASE 3: Supervised Fine-Tuning")
    print("=" * 60)
    
    # === Model ===
    print(f"\nLoading model from {config.model.resume_from}")
    model = build_model(config.model.config_path)
    
    checkpoint = torch.load(config.model.resume_from, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.cuda()
    
    print(f"Model: {model.count_parameters() / 1e9:.2f}B params")
    
    # === Tokenizer ===
    tokenizer = HelixTokenizer()
    
    # === Data ===
    print(f"\nLoading training data...")
    train_dataset = SFTDataset(
        path=config.data.train_path,
        tokenizer=tokenizer,
        seq_length=config.data.seq_length,
        mask_user_tokens=config.data.mask_user_tokens,
    )
    
    val_dataset = SFTDataset(
        path=config.data.val_path,
        tokenizer=tokenizer,
        seq_length=config.data.seq_length,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size_per_device,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size_per_device,
        shuffle=False,
        num_workers=2,
    )
    
    # === Optimizer (GaLore!) ===
    # WICHTIG: --reset-optimizer = True in config
    # Optimizer-State von Phase 2 wird NICHT geladen
    
    try:
        from galore_torch import GaLoreAdamW as OfficialGaLore
        optimizer_cls = OfficialGaLore
        print("Using official galore_torch package")
    except ImportError:
        optimizer_cls = GaLoreAdamW
        print("⚠️  Using custom GaLore implementation")
    
    optimizer = optimizer_cls(
        model.parameters(),
        lr=config.training.optimizer.lr,
        rank=config.training.optimizer.galore_rank,
        update_proj_gap=config.training.optimizer.galore_update_proj_gap,
        scale=config.training.optimizer.galore_scale,
        proj_type=config.training.optimizer.galore_proj_type,
        betas=config.training.optimizer.betas,
        weight_decay=config.training.optimizer.weight_decay,
    )
    
    # === Training Loop ===
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.training.max_epochs):
        print(f"\n=== Epoch {epoch + 1}/{config.training.max_epochs} ===")
        
        model.train()
        
        for step, batch in enumerate(train_loader):
            batch = {k: v.cuda() for k, v in batch.items()}
            
            # Forward
            output = model(
                input_ids=batch['input_ids'],
                labels=batch['labels'],
                attention_mask=batch['attention_mask'],
            )
            
            loss = output['loss']
            
            # Check NaN
            if torch.isnan(loss):
                print("⚠️  NaN loss! Skipping batch.")
                optimizer.zero_grad()
                continue
            
            # Backward
            loss.backward()
            
            # Clip
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config.training.gradient_clip_norm,
            )
            
            # Step
            optimizer.step()
            optimizer.zero_grad()
            
            # Log
            if step % config.logging.log_every == 0:
                print(f"  Step {step:5d} | loss {loss.item():.4f}")
            
            # Eval
            if step % config.logging.eval_every == 0 and step > 0:
                val_loss = evaluate(model, val_loader)
                print(f"    Val Loss: {val_loss:.4f}")
                
                # Save best
                if val_loss < best_val_loss - config.training.early_stopping.min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    save_checkpoint(model, epoch, step, val_loss, config, name='best')
                    print(f"    ✓ New best: {val_loss:.4f}")
                else:
                    patience_counter += 1
                    print(f"    Patience: {patience_counter}/{config.training.early_stopping.patience}")
                
                # Early stopping
                if patience_counter >= config.training.early_stopping.patience:
                    print(f"\n⏹  Early stopping at epoch {epoch+1}, step {step}")
                    return
    
    print(f"\n✓ Training complete. Best val loss: {best_val_loss:.4f}")


def evaluate(model, val_loader):
    """Evaluate on val set."""
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
    
    model.train()
    return total_loss / n


def save_checkpoint(model, epoch, step, val_loss, config, name):
    """Save checkpoint."""
    path = Path(config.checkpointing.output_dir) / f"{name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'step': step,
        'val_loss': val_loss,
        'model': model.state_dict(),
    }, path)
    
    print(f"    Saved: {path}")


if __name__ == "__main__":
    main()
```

---

## 7. Expected Results

```
Starting point (Phase 2 best.pt):
  Baseline questions: ~30% correct
  Small talk:         poor (v1 problem)
  Uncertainty:        hallucinates

After Phase 3 SFT:
  Baseline questions: ~80% correct ✓
  Small talk:         natural ✓
  Uncertainty:        says "I don't know" ✓
  Format:             clean German ✓
  Multi-turn:         works ✓
  
Val Loss Target: < 1.0
```

---

## 8. Acceptance Criteria

```
Data Preparation:
  □ 100k+ train samples, quality filtered
  □ 2-5k val samples (disjoint!)
  □ Small-talk samples included
  □ Uncertainty samples included
  □ Multi-turn samples included

Training:
  □ GaLore runs (official package or custom)
  □ Val Loss < 1.0
  □ Early stopping kicks in (no overfitting)
  □ --reset-optimizer active
  □ Chat template consistent

Quality:
  □ Baseline-50 > 80% correct
  □ MT-Bench subset: good answers
  □ Small talk feels natural
  □ Uncertainty is expressed appropriately
  □ No <think>/<lora> leakage (post-processing!)
```

---

## 9. Next Steps

After Phase 3:
→ SPEC_PHASE_4_ORPO_ALIGNMENT.md

---

*Phase 3 Spec Version 1.0 — April 2026*
