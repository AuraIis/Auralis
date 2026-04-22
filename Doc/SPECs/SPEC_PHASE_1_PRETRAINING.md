# Phase 1: Pretraining (Englisch-Heavy)

**Projekt:** Auralis v2 / Helix v2
**Phase:** 1 (erstes echtes Training)
**Dauer:** 3-4 Wochen
**Ziel:** Starke englische Basis + schwaches Deutsch + Code-Grundlagen
**Hardware:** H200 (143GB) oder H100 (80GB)
**Budget:** ~$500-800 (RunPod)

---

## 1. Ziele

**Was dieses Training erreichen soll:**

```
Englisch:    Starke Basis (MMLU 40+, HellaSwag 65+)
Deutsch:     Grundlagen (verstehen + generieren, aber schwach)
Code:        Basis-Syntax beherrschen
Weltwissen:  Breite Konzepte gelernt
Reasoning:   Simple Logik-Ketten
```

**Was dieses Training NICHT erreichen soll:**

```
✗ Perfektes Deutsch (kommt in Phase 2)
✗ Instruktions-Folgen (kommt in Phase 3 SFT)
✗ Präferenzen/Alignment (kommt in Phase 4)
✗ Spezialisierte Fähigkeiten (kommt über LoRAs in Phase 5)
```

---

## 2. Daten-Mix

### 2.1 Verhältnisse

```
75% Englisch
20% Deutsch
 5% Code

Total Tokens: 30-50B
```

**Warum diese Gewichtung:**

- Englisch dominiert, weil:
  - Mehr und bessere Daten verfügbar
  - Weltwissen dort konzentriert
  - Englische Konzepte übertragen cross-lingual

- 20% Deutsch weil:
  - Tokenizer braucht Deutsch-Exposure
  - Basis-Syntax-Lernen
  - Zu wenig = Phase 2 hat nichts aufzubauen

- 5% Code weil:
  - Syntax-Grundlagen lernen
  - Nicht als Haupt-Fähigkeit, eher als "Domäne"

### 2.2 Konkrete Datenquellen

```yaml
# configs/data/phase1_mix.yaml

english:
  target_tokens: 30_000_000_000  # 30B
  sources:
    - name: "fineweb-edu"
      path: "HuggingFaceFW/fineweb-edu"
      config: "sample-100BT"
      weight: 0.70
      filters:
        min_score: 2.5
        min_length: 200
        max_length: 100000
    
    - name: "wikipedia-en"
      path: "wikipedia/20240401.en"
      weight: 0.15
      filters:
        min_length: 500
    
    - name: "stack-exchange-en"
      path: "HuggingFaceH4/stack-exchange-preferences"
      weight: 0.05
      filters:
        min_score: 5
    
    - name: "arxiv-abstracts"
      path: "CShorten/ML-ArXiv-Papers"
      weight: 0.05
    
    - name: "textbooks"
      path: "open-phi/textbooks"
      weight: 0.05

german:
  target_tokens: 8_000_000_000  # 8B
  sources:
    - name: "german-commons-filtered"
      path: "coral-nlp/german-commons"
      weight: 0.60
      filters:
        max_perplexity: 500
        cultural_subsample: 0.10  # Max 10% historisch
        min_length: 200
    
    - name: "wikipedia-de"
      path: "wikipedia/20240401.de"
      weight: 0.25
      filters:
        min_length: 500
    
    - name: "news-de"
      path: "oscar-corpus/OSCAR-2301"
      config: "de"
      weight: 0.10
      filters:
        year_min: 2020  # Nur modernes Deutsch
    
    - name: "fineweb-edu-de"
      path: "HuggingFaceFW/fineweb-edu-v2-DE"  # Falls verfügbar
      weight: 0.05
      optional: true

code:
  target_tokens: 2_000_000_000  # 2B
  sources:
    - name: "the-stack-v2"
      path: "bigcode/the-stack-v2"
      weight: 1.0
      languages:
        Python: 0.25
        JavaScript: 0.20
        TypeScript: 0.10
        Rust: 0.10
        "C++": 0.10
        Go: 0.08
        Java: 0.07
        C: 0.05
        Shell: 0.03
        SQL: 0.02
      filters:
        min_stars: 10
        min_length: 100
        max_length: 30000
```

### 2.3 Daten-Preparation

**Script:** `scripts/pretrain/prepare_phase1_data.py`

```python
"""
Bereitet Phase-1 Daten vor.
Tokenisiert, deduped, packt in Binary-Format.
"""

from pathlib import Path
from datasets import load_dataset, interleave_datasets
from auralis.tokenizer import HelixTokenizer
import numpy as np
from tqdm import tqdm


def prepare_phase1_data(
    output_dir: str = "data/training/phase1",
    config_path: str = "configs/data/phase1_mix.yaml",
):
    """
    Pipeline:
      1. Lade Datasets aus HuggingFace
      2. Filter nach Quality
      3. Tokenisiere mit Helix v2 Tokenizer
      4. Dedup (optional, für Memory)
      5. Packe in .bin Files (memmap-able)
      6. Schreibe manifest.json
    """
    tokenizer = HelixTokenizer()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Lese Config
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Pro Sprache/Domäne separate .bin files
    # (erleichtert Mixing beim Training)
    for domain_name in ['english', 'german', 'code']:
        print(f"\n=== {domain_name.upper()} ===")
        
        domain_config = config[domain_name]
        target_tokens = domain_config['target_tokens']
        
        domain_file = output_path / f"{domain_name}.bin"
        index_file = output_path / f"{domain_name}.idx"
        
        total_tokens = 0
        
        with open(domain_file, 'wb') as f_data, \
             open(index_file, 'wb') as f_idx:
            
            # Alle Sources für diese Domäne kombiniert
            for source in domain_config['sources']:
                print(f"Source: {source['name']}")
                
                dataset = load_dataset(
                    source['path'],
                    source.get('config', None),
                    split='train',
                    streaming=True,
                )
                
                for example in tqdm(dataset):
                    text = example.get('text', '') or example.get('content', '')
                    
                    # Filters anwenden
                    if not _apply_filters(text, example, source.get('filters', {})):
                        continue
                    
                    # Tokenize
                    tokens = tokenizer.encode(text, add_eos=True)
                    
                    # Write
                    tokens_arr = np.array(tokens, dtype=np.uint32)
                    tokens_arr.tofile(f_data)
                    
                    # Index: (offset, length) pairs
                    offset = f_data.tell() - len(tokens) * 4
                    idx_arr = np.array([offset // 4, len(tokens)], dtype=np.int64)
                    idx_arr.tofile(f_idx)
                    
                    total_tokens += len(tokens)
                    
                    if total_tokens >= target_tokens:
                        break
                
                if total_tokens >= target_tokens:
                    break
        
        print(f"  Total: {total_tokens / 1e9:.2f}B tokens")
        print(f"  File:  {domain_file.stat().st_size / 1024**3:.2f} GB")


def _apply_filters(text, example, filters):
    """Apply filter rules."""
    if not text:
        return False
    
    min_len = filters.get('min_length', 0)
    max_len = filters.get('max_length', float('inf'))
    if not (min_len <= len(text) <= max_len):
        return False
    
    min_score = filters.get('min_score', None)
    if min_score is not None:
        if example.get('score', 0) < min_score:
            return False
    
    return True


if __name__ == "__main__":
    prepare_phase1_data()
```

---

## 3. Training-Konfiguration

**Datei:** `configs/training/phase1_pretrain.yaml`

```yaml
# Phase 1 Pretraining Configuration

experiment:
  name: "helix_v2_phase1_pretrain"
  version: "1.0"
  description: "English-Heavy Pretraining, 30B tokens"

model:
  config_path: "configs/model/helix_v2_3b.yaml"
  # Keine Resume von Checkpoint (von Scratch)

data:
  config_path: "configs/data/phase1_mix.yaml"
  data_dir: "data/training/phase1"
  
  # Mix-Verhältnisse pro Batch
  mix_ratios:
    english: 0.75
    german: 0.20
    code: 0.05
  
  seq_length: 2048  # Training context
  
training:
  # Hardware
  device: "cuda"
  dtype: "bfloat16"
  
  # Optimizer
  optimizer:
    name: "adamw"  # oder "muon" (moderner)
    lr: 3.0e-4
    betas: [0.9, 0.95]
    weight_decay: 0.1
    eps: 1.0e-8
  
  # Schedule
  scheduler:
    type: "cosine"
    warmup_steps: 2000
    min_lr_ratio: 0.1
  
  # Batch
  batch_size_per_device: 8
  gradient_accumulation: 16
  # Effective batch: 8 * 16 = 128 sequences
  # At seq_len 2048: 262k tokens per batch
  
  # Gradient Handling
  gradient_clip_norm: 1.0
  
  # Duration
  total_steps: 115_000
  # 115k steps * 262k tokens = 30B tokens
  
  # Mixed Precision
  grad_accum_dtype: "bfloat16"
  
  # Memory
  gradient_checkpointing: true
  
  # Compilation (falls PyTorch 2.5+)
  torch_compile: true

logging:
  log_every: 10
  eval_every: 1000
  save_every: 2500
  
  wandb:
    enabled: true
    project: "auralis-v2"
    tags: ["phase1", "pretrain", "helix-v2"]

checkpointing:
  output_dir: "checkpoints/phase1_pretrain"
  save_last_n: 3  # Nur letzte 3 Checkpoints behalten
  save_best: true  # Plus best.pt nach val_loss
  
  # Externe Backups
  external_backup:
    enabled: true
    path: "/mnt/external_backup/helix_v2/phase1"
    interval_steps: 10000

evaluation:
  # Val-Loss auf gehaltenen Daten
  val_data_dir: "data/eval/phase1_val"
  max_val_batches: 100
  
  # Benchmarks (schnelle Subsets)
  benchmarks:
    - name: "hellaswag_subset"
      path: "data/eval/hellaswag_100.jsonl"
      frequency: 5000
    
    - name: "belebele_de_subset"
      path: "data/eval/belebele_de_100.jsonl"
      frequency: 5000
    
    - name: "humaneval_subset"
      path: "data/eval/humaneval_20.jsonl"
      frequency: 10000
    
    - name: "baseline_50"
      path: "data/eval/baseline_questions.yaml"
      frequency: 5000

monitoring:
  alert_on:
    - val_loss_increase: 3  # 3 consecutive evals
    - grad_norm_explosion: true
    - nan_in_loss: true
    - disk_full: true
```

---

## 4. Training-Script

**Datei:** `scripts/pretrain/train_phase1.py`

```python
"""
Phase 1 Pretraining Training Script.
"""

import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import wandb

from auralis.model import build_model
from auralis.training.dataset import PretrainDataset, MixedDataLoader
from auralis.training.trainer import PretrainTrainer
from auralis.training.optimizer import build_optimizer, build_scheduler
from auralis.training.utils import load_config, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='configs/training/phase1_pretrain.yaml',
    )
    parser.add_argument('--resume', default=None)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Reproducibility
    set_seed(42)
    
    # === Preflight Checks ===
    _preflight_checks(config)
    
    if args.dry_run:
        print("✓ Preflight passed, exiting (--dry-run)")
        return
    
    # === Wandb ===
    if config.logging.wandb.enabled:
        wandb.init(
            project=config.logging.wandb.project,
            name=config.experiment.name,
            config=config,
            tags=config.logging.wandb.tags,
        )
    
    # === Model ===
    print("Building model...")
    model = build_model(config.model.config_path)
    print(f"Parameters: {model.count_parameters() / 1e9:.2f}B")
    
    # === Resume if requested ===
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        start_step = checkpoint.get('step', 0)
    else:
        start_step = 0
    
    # === Move to GPU ===
    model = model.to(config.training.device)
    
    if config.training.torch_compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
    
    # === Data ===
    print("Loading data...")
    dataloader = MixedDataLoader(
        data_dir=config.data.data_dir,
        mix_ratios=config.data.mix_ratios,
        batch_size=config.training.batch_size_per_device,
        seq_length=config.data.seq_length,
    )
    
    # === Optimizer ===
    optimizer = build_optimizer(model, config.training.optimizer)
    scheduler = build_scheduler(
        optimizer,
        config.training.scheduler,
        total_steps=config.training.total_steps,
    )
    
    # === Trainer ===
    trainer = PretrainTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=dataloader,
        config=config,
        start_step=start_step,
    )
    
    # === Train ===
    trainer.train()
    
    # === Cleanup ===
    if config.logging.wandb.enabled:
        wandb.finish()
    
    print("\n✓ Training complete!")


def _preflight_checks(config):
    """Check alles ist bereit."""
    checks = []
    
    # Data exists?
    data_dir = Path(config.data.data_dir)
    required_files = ['english.bin', 'german.bin', 'code.bin']
    for fname in required_files:
        if not (data_dir / fname).exists():
            checks.append(f"❌ Missing: {data_dir / fname}")
    
    # Checkpoint dir writable?
    ckpt_dir = Path(config.checkpointing.output_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # GPU available?
    if not torch.cuda.is_available():
        checks.append("❌ No CUDA GPU available")
    
    # Disk space?
    import shutil
    free_gb = shutil.disk_usage(ckpt_dir).free / 1024**3
    if free_gb < 500:  # 500GB Reserve
        checks.append(f"⚠️  Only {free_gb:.0f}GB free (need ~500GB)")
    
    if checks:
        print("\n=== PREFLIGHT FAILED ===")
        for check in checks:
            print(check)
        raise SystemExit(1)
    
    print("✓ Preflight checks passed")


if __name__ == "__main__":
    main()
```

---

## 5. Trainer-Implementation

**Datei:** `src/auralis/training/trainer.py`

```python
"""
PretrainTrainer: Orchestriert das Training.
Mit Checkpoints, Eval, Alerting, Dashboard.
"""

import torch
import time
from pathlib import Path
from typing import Optional


class PretrainTrainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        dataloader,
        config,
        start_step: int = 0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.config = config
        self.step = start_step
        
        # Tracking
        self.best_val_loss = float('inf')
        self.consecutive_increases = 0
        
        # Paths
        self.ckpt_dir = Path(config.checkpointing.output_dir)
        
    def train(self):
        """Main training loop."""
        self.model.train()
        
        data_iter = iter(self.dataloader)
        t0 = time.time()
        
        while self.step < self.config.training.total_steps:
            # === Gradient Accumulation ===
            total_loss = 0
            
            for micro_step in range(self.config.training.gradient_accumulation):
                batch = next(data_iter)
                batch = {k: v.cuda() for k, v in batch.items()}
                
                # Forward
                with torch.cuda.amp.autocast(
                    dtype=torch.bfloat16,
                ):
                    output = self.model(
                        input_ids=batch['input_ids'],
                        labels=batch['labels'],
                    )
                    loss = output['loss'] / self.config.training.gradient_accumulation
                
                # Check NaN
                if torch.isnan(loss):
                    self._alert("NaN loss detected!")
                    raise RuntimeError("Training diverged")
                
                # Backward
                loss.backward()
                total_loss += loss.item()
            
            # Clip gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.training.gradient_clip_norm,
            )
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            self.step += 1
            
            # === Logging ===
            if self.step % self.config.logging.log_every == 0:
                self._log_step(total_loss, grad_norm, t0)
                t0 = time.time()
            
            # === Evaluation ===
            if self.step % self.config.logging.eval_every == 0:
                val_loss = self._evaluate()
                self._track_val_loss(val_loss)
            
            # === Checkpointing ===
            if self.step % self.config.logging.save_every == 0:
                self._save_checkpoint()
    
    def _log_step(self, loss, grad_norm, t0):
        """Log training step."""
        lr = self.scheduler.get_last_lr()[0]
        elapsed = time.time() - t0
        
        # Tokens per second
        total_tokens = (
            self.config.training.batch_size_per_device
            * self.config.training.gradient_accumulation
            * self.config.data.seq_length
            * self.config.logging.log_every
        )
        tps = total_tokens / elapsed
        
        msg = (
            f"Step {self.step:6d} | "
            f"loss {loss:.4f} | "
            f"lr {lr:.2e} | "
            f"grad_norm {grad_norm:.2f} | "
            f"tok/s {tps/1e3:.1f}k"
        )
        print(msg)
        
        if wandb.run:
            wandb.log({
                'train/loss': loss,
                'train/grad_norm': grad_norm,
                'train/lr': lr,
                'train/tokens_per_second': tps,
            }, step=self.step)
    
    def _evaluate(self) -> float:
        """Evaluate on val set."""
        self.model.eval()
        
        total_loss = 0
        n_batches = 0
        
        with torch.no_grad():
            for batch in self._get_val_dataloader():
                batch = {k: v.cuda() for k, v in batch.items()}
                output = self.model(
                    input_ids=batch['input_ids'],
                    labels=batch['labels'],
                )
                total_loss += output['loss'].item()
                n_batches += 1
                
                if n_batches >= self.config.evaluation.max_val_batches:
                    break
        
        val_loss = total_loss / n_batches
        
        print(f"  Val Loss: {val_loss:.4f}")
        
        if wandb.run:
            wandb.log({'eval/val_loss': val_loss}, step=self.step)
        
        # Baseline Benchmarks
        if self.step % 5000 == 0:
            self._run_benchmarks()
        
        self.model.train()
        return val_loss
    
    def _run_benchmarks(self):
        """Run baseline benchmarks."""
        # Placeholder - implementation
        pass
    
    def _track_val_loss(self, val_loss):
        """Track val loss, save best."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.consecutive_increases = 0
            # Save best
            self._save_checkpoint(name='best')
        else:
            self.consecutive_increases += 1
            
            if self.consecutive_increases >= 3:
                self._alert(
                    f"Val loss increased {self.consecutive_increases}x in a row. "
                    f"Consider stopping or adjusting."
                )
    
    def _save_checkpoint(self, name: Optional[str] = None):
        """Save checkpoint."""
        if name is None:
            name = f"step_{self.step}"
        
        path = self.ckpt_dir / f"{name}.pt"
        
        checkpoint = {
            'step': self.step,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        
        torch.save(checkpoint, path)
        print(f"  Saved: {path}")
        
        # External backup
        if self.config.checkpointing.external_backup.enabled:
            if self.step % self.config.checkpointing.external_backup.interval_steps == 0:
                backup_path = Path(
                    self.config.checkpointing.external_backup.path
                ) / f"{name}.pt"
                # Copy (not move)
                import shutil
                shutil.copy2(path, backup_path)
                print(f"  External backup: {backup_path}")
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """Keep only last N checkpoints + best.pt."""
        keep_last = self.config.checkpointing.save_last_n
        
        step_checkpoints = sorted(
            self.ckpt_dir.glob("step_*.pt"),
            key=lambda p: int(p.stem.split('_')[1]),
            reverse=True,
        )
        
        for ckpt in step_checkpoints[keep_last:]:
            ckpt.unlink()
    
    def _alert(self, message: str):
        """Alert on issue."""
        print(f"\n⚠️  ALERT: {message}\n")
        # Could send to Telegram/Email here
```

---

## 6. Akzeptanz-Kriterien

```
Start:
  □ Alle Daten-Files vorbereitet (.bin + .idx)
  □ Pretrain-Config valid
  □ Model baut erfolgreich
  □ Dry-Run läuft durch
  □ RunPod Guthaben > $800

Training Milestones:
  □ Step 1000: Loss sinkt stetig
  □ Step 5000: Erste Eval, val_loss < 7
  □ Step 25000: Benchmarks zeigen Fortschritt
  □ Step 50000: val_loss < 4, HellaSwag > 40%
  □ Step 100000: val_loss < 3, MMLU > 30%
  □ Step 115000: TRAINING COMPLETE

End State:
  □ best.pt gespeichert
  □ Externes Backup existiert
  □ Full Benchmark-Report
  □ HellaSwag > 55
  □ Belebele-DE > 45
  □ HumanEval > 15%
```

---

## 7. Fehlerbehebung

```
Problem: Training diverges (NaN loss)
  → LR zu hoch? Auf 2e-4 senken
  → Gradient Clipping zu hoch? Auf 0.5 setzen
  → Daten-Bug? Batch vor dem NaN prüfen

Problem: Training zu langsam
  → torch.compile aktivieren
  → Flash Attention prüfen
  → batch_size_per_device erhöhen
  → gradient_accumulation senken

Problem: VRAM OOM
  → batch_size senken
  → gradient_checkpointing aktivieren
  → seq_length auf 1024 senken

Problem: Val Loss stagniert
  → LR warm restart
  → Daten-Mix prüfen
  → Ggf. frühe Stoppen, continuen mit anderem Mix
```

---

## 8. Next Steps

Nach Phase 1 erfolgreich:
→ SPEC_PHASE_2_CONTINUED_BILINGUAL.md
   (mit KL-Distillation gegen Forgetting)

---

*Phase 1 Spec Version 1.0 — April 2026*
