# Phase 2: Bilingual Continued Pretraining

**Projekt:** Auralis v2 / Helix v2
**Phase:** 2 (Deutsch-Aufbau ohne Englisch zu vergessen)
**Dauer:** 1-2 Wochen
**Ziel:** Starkes Deutsch + Englisch-Retention > 95%
**Voraussetzung:** Phase 1 abgeschlossen (Phase-1-Checkpoint existiert)
**Hardware:** H200 (brauchen Student + Teacher gleichzeitig)
**Budget:** ~$200-400

---

## 1. Das Kernproblem

Continued Pretraining mit neuer Sprache führt zu **catastrophic forgetting**:

```
Phase 1 Modell (EN-stark):
  English MMLU:  45  ← gute Basis
  German eval:   20  ← schwach aber grundlegend

Naive Continued Training (100% Deutsch):
  English MMLU:  25  ← stark verloren!
  German eval:   50  ← gut gewachsen

Das ist inakzeptabel. Lösung: KL-Distillation.
```

Die Teacher-Student-Architektur erhält das Englisch-Wissen
während Deutsch dazu gelernt wird.

---

## 2. KL-Distillation Strategie

### 2.1 Konzept

```
Phase 1 Modell → kopieren in zwei Versionen:
  Teacher:  Frozen (nie trainiert)
  Student:  Trainierbar (lernt Phase 2)

Während Training:
  Student → lernt Deutsch (normaler Task Loss)
  Student → bleibt nah an Teacher (KL Loss)
  
  Total Loss = Task Loss + λ * KL Loss
  
  λ (lambda) balanciert:
    Niedrig (0.1-0.3): Mehr lernen, weniger bewahren
    Mittel (0.5):      Ausgewogen (empfohlen)
    Hoch (0.8-1.5):    Stark bewahren, weniger lernen
```

### 2.2 Temperature Parameter

```
T = 3.0 (Standard):
  → Weichere Verteilungen
  → KL sieht ganze Top-k Rangfolge
  → Bewährt in Papers

Niedriger T: KL fokussiert auf Top-1
Höherer T: KL verteilt sich breiter
```

---

## 3. Daten-Mix

```yaml
# configs/data/phase2_mix.yaml

english:
  target_tokens: 4_500_000_000  # 4.5B (30%)
  sources:
    # Hochwertige Replay-Samples von Phase 1
    - name: "phase1_golden_replay"
      path: "data/training/phase1/english_golden.bin"
      weight: 0.50
      note: "Top 5% quality samples from Phase 1"
    
    # Neue Englisch-Daten (Diversität)
    - name: "fineweb-edu-fresh"
      path: "HuggingFaceFW/fineweb-edu"
      config: "sample-10BT"
      weight: 0.30
    
    # Wissenschaftliche Artikel
    - name: "arxiv-papers"
      weight: 0.10
    
    # Reasoning-Daten (schützt Reasoning-Fähigkeit)
    - name: "proof-pile-2"
      weight: 0.10

german:
  target_tokens: 9_000_000_000  # 9B (60%)
  sources:
    # Modernere Daten als Phase 1
    - name: "fineweb-edu-de"
      weight: 0.40
    
    - name: "german-commons-best"
      path: "coral-nlp/german-commons"
      weight: 0.30
      filters:
        max_perplexity: 300  # Strenger als Phase 1
        cultural_subsample: 0.05  # Noch weniger historisch
    
    - name: "wikipedia-de-full"
      weight: 0.15
    
    - name: "german-news-2023-2024"
      weight: 0.10
    
    # DEUTSCHES Reasoning (wenn verfügbar)
    - name: "german-reasoning-dataset"
      weight: 0.05
      optional: true

code:
  target_tokens: 1_500_000_000  # 1.5B (10%)
  # Gleich wie Phase 1, um Code-Skills zu erhalten
  sources:
    - name: "the-stack-v2-subset"
      weight: 1.0

total_tokens: 15_000_000_000  # 15B total
```

---

## 4. Training-Konfiguration

**Datei:** `configs/training/phase2_continued.yaml`

```yaml
experiment:
  name: "helix_v2_phase2_continued"
  version: "1.0"
  description: "Continued Pretraining with KL-Distillation"

model:
  # Student startet von Phase 1 Checkpoint
  config_path: "configs/model/helix_v2_3b.yaml"
  student_resume_from: "checkpoints/phase1_pretrain/best.pt"
  
  # Teacher ist Phase 1 Checkpoint (frozen)
  teacher_checkpoint: "checkpoints/phase1_pretrain/best.pt"

data:
  config_path: "configs/data/phase2_mix.yaml"
  data_dir: "data/training/phase2"
  
  mix_ratios:
    english: 0.30
    german: 0.60
    code: 0.10
  
  seq_length: 2048

training:
  device: "cuda"
  dtype: "bfloat16"
  
  # WICHTIG: --reset-optimizer (Phase-1-Optimizer-State NICHT laden!)
  reset_optimizer: true
  
  optimizer:
    name: "adamw"
    lr: 3.0e-5  # 10x NIEDRIGER als Phase 1
    betas: [0.9, 0.95]
    weight_decay: 0.1
  
  scheduler:
    type: "cosine"
    warmup_steps: 1000  # Kurzer Warmup
    min_lr_ratio: 0.1
  
  batch_size_per_device: 8
  gradient_accumulation: 16
  
  gradient_clip_norm: 1.0
  
  total_steps: 60_000
  # 15B tokens / 262k per step = ~60k steps
  
  gradient_checkpointing: true
  torch_compile: true

# === KL-DISTILLATION CONFIG ===
kl_distillation:
  enabled: true
  
  # Teacher Setup
  teacher_checkpoint: "checkpoints/phase1_pretrain/best.pt"
  teacher_on_cpu: false  # H200 hat genug VRAM
  teacher_quantize: false  # Volle Präzision für gute Targets
  
  # Loss Parameters
  lambda_kd: 0.5  # Initial value, adaptive
  temperature: 3.0
  
  # Adaptive Lambda
  adaptive_lambda:
    enabled: true
    check_every_steps: 2000
    
    # Thresholds
    target_retention: [92.0, 98.0]  # 92-98%
    retention_benchmark: "english_mmlu_subset"
    
    # Adjustments
    if_below_target: "increase"  # retention < 92 → lambda *= 1.5
    if_above_target: "decrease"  # retention > 98 → lambda *= 0.7
    max_lambda: 2.0
    min_lambda: 0.1

logging:
  log_every: 10
  eval_every: 500
  save_every: 2500
  
  wandb:
    enabled: true
    project: "auralis-v2"
    tags: ["phase2", "continued", "kl-distill"]

checkpointing:
  output_dir: "checkpoints/phase2_continued"
  save_last_n: 3
  save_best: true
  
  external_backup:
    enabled: true
    path: "/mnt/external_backup/helix_v2/phase2"
    interval_steps: 5000

evaluation:
  val_data_dir: "data/eval/phase2_val"
  max_val_batches: 100
  
  # Benchmarks MÜSSEN beide Sprachen prüfen
  benchmarks:
    # Englisch (Retention kritisch!)
    - name: "english_mmlu_subset"
      path: "data/eval/mmlu_100.jsonl"
      frequency: 1000
      language: "en"
    
    - name: "hellaswag_subset"
      path: "data/eval/hellaswag_100.jsonl"
      frequency: 2500
      language: "en"
    
    # Deutsch (Fortschritt!)
    - name: "belebele_de_subset"
      path: "data/eval/belebele_de_100.jsonl"
      frequency: 1000
      language: "de"
    
    - name: "german_knowledge"
      path: "data/eval/german_qa_50.jsonl"
      frequency: 2500
      language: "de"
    
    # Code
    - name: "humaneval_subset"
      path: "data/eval/humaneval_20.jsonl"
      frequency: 5000
      language: "code"
    
    # Baseline
    - name: "baseline_50"
      path: "data/eval/baseline_questions.yaml"
      frequency: 2500

monitoring:
  alert_on:
    - english_retention_below: 90.0  # Kritisch!
    - val_loss_increase: 3
    - grad_norm_explosion: true
    - nan_in_loss: true
```

---

## 5. KL-Distillation Implementation

**Datei:** `src/auralis/training/kl_distillation.py`

```python
"""
KL-Distillation Training.
Student lernt Neues, Teacher bewahrt Altes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from dataclasses import dataclass


@dataclass
class KLDConfig:
    lambda_kd: float = 0.5
    temperature: float = 3.0
    teacher_on_cpu: bool = False
    teacher_quantize: bool = False


class KLDistillationWrapper:
    """Wrapper der KL-Loss zu Student hinzufügt."""
    
    def __init__(
        self,
        student: nn.Module,
        teacher_checkpoint_path: str,
        config: KLDConfig,
    ):
        self.student = student
        self.config = config
        self.current_lambda = config.lambda_kd
        
        # Teacher laden
        self.teacher = self._load_teacher(teacher_checkpoint_path)
        
        # Teacher einfrieren
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        
        # Optional: auf CPU
        if config.teacher_on_cpu:
            self.teacher = self.teacher.to('cpu')
        
        # Optional: quantize
        if config.teacher_quantize:
            self.teacher = self._quantize_teacher(self.teacher)
        
        print(f"Teacher initialized: "
              f"on_cpu={config.teacher_on_cpu}, "
              f"quantized={config.teacher_quantize}")
    
    def _load_teacher(self, checkpoint_path):
        """Lädt Teacher als Kopie des Student."""
        teacher = deepcopy(self.student)
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        teacher.load_state_dict(checkpoint['model'])
        
        # Teacher auf gleiches Device wie Student
        teacher_device = next(self.student.parameters()).device
        if not self.config.teacher_on_cpu:
            teacher = teacher.to(teacher_device)
        
        return teacher
    
    def _quantize_teacher(self, model):
        """8-bit Quantisierung."""
        # Mit bitsandbytes
        try:
            import bitsandbytes as bnb
            # Convert Linear layers to 8-bit
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    # Replace with 8-bit
                    pass  # Implementation
        except ImportError:
            print("⚠️  bitsandbytes nicht verfügbar, Teacher bleibt FP16")
        
        return model
    
    def compute_loss(
        self,
        input_ids,
        labels,
        attention_mask=None,
    ):
        """Berechnet Total Loss = Task + KL."""
        # === Student Forward ===
        student_out = self.student(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
        task_loss = student_out['loss']
        student_logits = student_out['logits']
        
        # === Teacher Forward ===
        with torch.no_grad():
            if self.config.teacher_on_cpu:
                # Move to CPU
                teacher_input = input_ids.cpu()
                mask = attention_mask.cpu() if attention_mask is not None else None
                
                teacher_out = self.teacher(
                    input_ids=teacher_input,
                    attention_mask=mask,
                )
                teacher_logits = teacher_out['logits'].to(student_logits.device)
            else:
                teacher_out = self.teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                teacher_logits = teacher_out['logits']
        
        # === KL-Divergence ===
        kl_loss = self._compute_kl(student_logits, teacher_logits, labels)
        
        # === Kombiniert ===
        total_loss = task_loss + self.current_lambda * kl_loss
        
        return {
            'total_loss': total_loss,
            'task_loss': task_loss.detach(),
            'kl_loss': kl_loss.detach(),
            'lambda_used': self.current_lambda,
        }
    
    def _compute_kl(self, student_logits, teacher_logits, labels):
        """KL-Divergence mit Temperature."""
        T = self.config.temperature
        
        # Nur valid labels (ignore -100)
        valid_mask = (labels != -100)
        
        # Flatten
        B, L, V = student_logits.shape
        s_flat = student_logits.reshape(-1, V)
        t_flat = teacher_logits.reshape(-1, V)
        mask_flat = valid_mask.reshape(-1)
        
        # Apply mask
        s_valid = s_flat[mask_flat]
        t_valid = t_flat[mask_flat]
        
        # KL(teacher || student) with temperature
        student_log_probs = F.log_softmax(s_valid / T, dim=-1)
        teacher_probs = F.softmax(t_valid / T, dim=-1)
        
        kl = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean',
            log_target=False,
        )
        
        # Rescale
        kl = kl * (T ** 2)
        
        return kl
    
    def adjust_lambda(self, new_lambda: float):
        """Adaptive Lambda-Anpassung."""
        self.current_lambda = max(
            min(new_lambda, 2.0),  # max
            0.1,  # min
        )
        print(f"  Lambda adjusted: {self.current_lambda}")
```

---

## 6. Adaptive Lambda Logic

**Datei:** `src/auralis/training/adaptive_lambda.py`

```python
"""
Adaptive Lambda basierend auf English-Retention.
"""


class AdaptiveLambda:
    def __init__(
        self,
        initial_lambda: float = 0.5,
        target_range: tuple[float, float] = (92.0, 98.0),
        increase_factor: float = 1.5,
        decrease_factor: float = 0.7,
        min_lambda: float = 0.1,
        max_lambda: float = 2.0,
    ):
        self.current_lambda = initial_lambda
        self.target_range = target_range
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        
        self.history = []
    
    def update(self, english_retention: float) -> float:
        """
        Passt lambda basierend auf Retention an.
        
        Args:
            english_retention: Prozent 0-100
        
        Returns:
            Neues lambda
        """
        self.history.append((english_retention, self.current_lambda))
        
        low, high = self.target_range
        
        if english_retention < low:
            # Zu wenig bewahrt -> lambda erhöhen
            new_lambda = self.current_lambda * self.increase_factor
            action = "INCREASE (retention too low)"
        elif english_retention > high:
            # Zu viel bewahrt, zu wenig gelernt -> lambda senken
            new_lambda = self.current_lambda * self.decrease_factor
            action = "DECREASE (retention too high, learning slow)"
        else:
            # Im optimalen Bereich
            new_lambda = self.current_lambda
            action = "KEEP (in target range)"
        
        # Clamp
        new_lambda = max(min(new_lambda, self.max_lambda), self.min_lambda)
        
        print(f"  Retention: {english_retention:.1f}% | Lambda: {self.current_lambda:.3f} -> {new_lambda:.3f} | {action}")
        
        self.current_lambda = new_lambda
        return new_lambda
```

---

## 7. Training Script

**Datei:** `scripts/pretrain/train_phase2.py`

```python
"""
Phase 2 Training mit KL-Distillation.
"""

import argparse
import torch
from pathlib import Path

from auralis.model import build_model
from auralis.training.kl_distillation import KLDistillationWrapper, KLDConfig
from auralis.training.adaptive_lambda import AdaptiveLambda
from auralis.training.dataset import MixedDataLoader
from auralis.training.utils import load_config, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='configs/training/phase2_continued.yaml',
    )
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_seed(42)
    
    print("=" * 60)
    print("PHASE 2: Continued Pretraining with KL-Distillation")
    print("=" * 60)
    
    # === Student Model (aus Phase 1) ===
    print("\nLoading Student from Phase 1...")
    student = build_model(config.model.config_path)
    
    phase1_checkpoint = torch.load(
        config.model.student_resume_from,
        map_location='cpu',
    )
    student.load_state_dict(phase1_checkpoint['model'])
    student = student.cuda()
    
    # === KL-Distillation Setup ===
    print("\nLoading Teacher...")
    kld_config = KLDConfig(
        lambda_kd=config.kl_distillation.lambda_kd,
        temperature=config.kl_distillation.temperature,
        teacher_on_cpu=config.kl_distillation.teacher_on_cpu,
        teacher_quantize=config.kl_distillation.teacher_quantize,
    )
    
    kld_wrapper = KLDistillationWrapper(
        student=student,
        teacher_checkpoint_path=config.kl_distillation.teacher_checkpoint,
        config=kld_config,
    )
    
    # === Adaptive Lambda ===
    adaptive = None
    if config.kl_distillation.adaptive_lambda.enabled:
        adaptive = AdaptiveLambda(
            initial_lambda=config.kl_distillation.lambda_kd,
            target_range=tuple(config.kl_distillation.adaptive_lambda.target_retention),
        )
    
    # === Data ===
    print("\nLoading Phase 2 data...")
    dataloader = MixedDataLoader(
        data_dir=config.data.data_dir,
        mix_ratios=config.data.mix_ratios,
        batch_size=config.training.batch_size_per_device,
        seq_length=config.data.seq_length,
    )
    
    # === Optimizer (RESET!) ===
    # WICHTIG: Optimizer-State von Phase 1 NICHT laden
    from auralis.training.optimizer import build_optimizer, build_scheduler
    optimizer = build_optimizer(student, config.training.optimizer)
    scheduler = build_scheduler(
        optimizer,
        config.training.scheduler,
        total_steps=config.training.total_steps,
    )
    
    # === Training Loop ===
    step = 0
    best_score = 0
    
    while step < config.training.total_steps:
        # Accumulation
        total_loss = 0
        task_losses = []
        kl_losses = []
        
        for micro_step in range(config.training.gradient_accumulation):
            batch = next(iter(dataloader))
            batch = {k: v.cuda() for k, v in batch.items()}
            
            # KL-Distillation Loss
            losses = kld_wrapper.compute_loss(
                input_ids=batch['input_ids'],
                labels=batch['labels'],
            )
            
            # Scale for gradient accumulation
            loss = losses['total_loss'] / config.training.gradient_accumulation
            
            loss.backward()
            
            total_loss += loss.item()
            task_losses.append(losses['task_loss'].item())
            kl_losses.append(losses['kl_loss'].item())
        
        # Clip + Step
        torch.nn.utils.clip_grad_norm_(
            student.parameters(),
            max_norm=config.training.gradient_clip_norm,
        )
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        step += 1
        
        # === Logging ===
        if step % config.logging.log_every == 0:
            avg_task = sum(task_losses) / len(task_losses)
            avg_kl = sum(kl_losses) / len(kl_losses)
            
            print(
                f"Step {step:6d} | "
                f"total {total_loss:.4f} | "
                f"task {avg_task:.4f} | "
                f"kl {avg_kl:.4f} | "
                f"λ {kld_wrapper.current_lambda:.2f}"
            )
        
        # === Evaluation ===
        if step % config.logging.eval_every == 0:
            # Run benchmarks
            eng_retention = evaluate_english_retention(
                student=student,
                teacher=kld_wrapper.teacher,
                benchmark_path="data/eval/mmlu_100.jsonl",
            )
            
            german_score = evaluate_german(
                student=student,
                benchmark_path="data/eval/belebele_de_100.jsonl",
            )
            
            print(f"  English Retention: {eng_retention:.1f}%")
            print(f"  German Score:      {german_score:.1f}")
            
            # Adaptive Lambda
            if adaptive:
                new_lambda = adaptive.update(eng_retention)
                kld_wrapper.adjust_lambda(new_lambda)
        
        # === Checkpointing ===
        if step % config.logging.save_every == 0:
            save_checkpoint(student, step, config)


def evaluate_english_retention(student, teacher, benchmark_path):
    """Misst wie viel % des Teacher-Wissens Student behalten hat."""
    student_score = evaluate_on_benchmark(student, benchmark_path)
    teacher_score = evaluate_on_benchmark(teacher, benchmark_path)
    
    if teacher_score == 0:
        return 0
    
    return (student_score / teacher_score) * 100


def evaluate_on_benchmark(model, path):
    """Evaluate model on benchmark (simplified)."""
    # Placeholder - implement with actual benchmarking
    model.eval()
    # Run benchmark
    return 50.0  # placeholder


def save_checkpoint(model, step, config):
    """Save checkpoint."""
    path = Path(config.checkpointing.output_dir) / f"step_{step}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'step': step,
        'model': model.state_dict(),
    }, path)
    
    print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
```

---

## 8. Erwartete Ergebnisse

```
Start (von Phase 1):
  English MMLU:        45
  Belebele-DE:         25
  Code HumanEval:      20

Nach 15k Steps (~25%):
  English MMLU:        43  (-4% retention loss - normal)
  Belebele-DE:         40  (+15)
  Code HumanEval:      22  (+2)

Nach 30k Steps (~50%):
  English MMLU:        42  (-7% retention - acceptable)
  Belebele-DE:         52  (+27)
  Code HumanEval:      25  (+5)

Nach 60k Steps (Ende):
  English MMLU:        41  (retention: 91% ✓)
  Belebele-DE:         60  (deutlich stärker ✓)
  Code HumanEval:      26  (stabil ✓)

Ziele:
  ✓ English MMLU > 40 (Retention > 88%)
  ✓ Belebele-DE > 55
  ✓ Code HumanEval > 25
```

---

## 9. Akzeptanz-Kriterien

```
Training Quality:
  □ English Retention > 90% throughout training
  □ German benchmarks zeigen stetigen Fortschritt
  □ Code-Skills nicht verschlechtert
  □ Val Loss sinkt stetig

KL-Distillation:
  □ Teacher korrekt geladen (frozen)
  □ KL-Loss berechnet sich korrekt
  □ Adaptive Lambda justiert sich sinnvoll
  □ Temperature = 3.0 bewährt

Output:
  □ best.pt basierend auf kombiniertem Score (EN + DE + Code)
  □ Full Benchmark Report
  □ KL-Distillation Lambda History geloggt
  □ Externes Backup existiert
```

---

## 10. Next Steps

Nach Phase 2 erfolgreich:
→ SPEC_PHASE_3_SFT.md (Supervised Fine-Tuning mit GaLore)

---

*Phase 2 Spec Version 1.0 — April 2026*
