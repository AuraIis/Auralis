# Multi-GPU Training Spec (Overlay)

**Projekt:** Auralis v2 / Helix v2
**Status:** Overlay - gilt für ALLE Phasen (1-5)
**Ziel:** Kostengünstiges Training auf RunPod mit Multi-GPU-Setups

---

## 1. Warum Multi-GPU?

### 1.1 Kosten-Vergleich (RunPod, Stand Anfang 2026)

```
Single-GPU (sehr teuer):
  1x H200 (143GB):        ~$3.50-4.50/h
  1x H100 (80GB):         ~$2.50-3.50/h
  1x A100 80GB:           ~$1.90-2.50/h

Multi-GPU (deutlich günstiger pro VRAM):
  2x A100 80GB (160GB):   ~$3.00-3.80/h   ← oft besser als 1x H200
  4x A6000 48GB (192GB):  ~$2.00-2.80/h   ← sweet spot!
  8x A40 48GB (384GB):    ~$3.50-5.00/h   ← für großes Pretraining
  4x A40 48GB (192GB):    ~$1.80-2.40/h   ← günstigste Option

Fazit:
  Für Phase 1 (3-4 Wochen Pretraining):
    1x H200: ~$2500-3500
    4x A40:  ~$1200-1800   ← spart 50%
```

### 1.2 Wann Multi-GPU nötig/sinnvoll?

```
MUSS Multi-GPU:
  Phase 1 Pretraining:    Ja (bei >3B Modell für Geschwindigkeit)
  Phase 2 + Teacher:      Ja (Teacher + Student + Optimizer State)

KANN Multi-GPU (aus Kostengründen):
  Phase 3 SFT:            Optional (passt oft auf 1 GPU)
  Phase 4 ORPO:           Optional (2x forward = 2x Memory)

OFT single-GPU:
  Phase 5 LoRA:           1x RTX 3090 reicht für LoRA-Training
  Tokenizer (Phase 0):    CPU/1x GPU
  Testing/Debugging:      1x GPU
```

---

## 2. Framework-Wahl: FSDP vs DeepSpeed ZeRO

### 2.1 Die beiden Hauptoptionen

```
DeepSpeed ZeRO (Microsoft):
  ✓ Älter, battle-tested
  ✓ ZeRO-3 + CPU/NVMe Offload
  ✓ Huge community
  ✗ Extra Dependency
  ✗ Config-JSON kann komplex werden

PyTorch FSDP (native):
  ✓ In PyTorch integriert
  ✓ Moderner, besseres API
  ✓ Weniger Dependencies
  ✓ Besseres Compilation-Support (torch.compile)
  ✗ Etwas jünger (stabil seit PyTorch 2.1+)

Empfehlung für Auralis v2: FSDP
  → Einfachere Integration
  → Kein externer JSON-Config
  → Moderne PyTorch-API
  → torch.compile kompatibel
```

### 2.2 Sharding-Strategien

```
NO_SHARD (entspricht DDP):
  → Jede GPU hat volles Modell
  → Keine Memory-Ersparnis
  → Nur für kleine Modelle / Sanity-Test

SHARD_GRAD_OP (entspricht ZeRO-2):
  → Gradient + Optimizer-State über GPUs verteilt
  → Model selbst: Copy pro GPU
  → Gute Balance zwischen Memory und Speed

FULL_SHARD (entspricht ZeRO-3):
  → Alles verteilt (Weights, Gradients, Optimizer)
  → Maximum Memory-Ersparnis
  → Etwas langsamer wegen Communication
  → Für große Modelle MANDATORY

HYBRID_SHARD:
  → FULL_SHARD innerhalb eines Nodes
  → SHARD_GRAD_OP zwischen Nodes
  → Für Multi-Node Setups
```

### 2.3 Welche Strategy für welche Phase?

```
Phase 1 Pretraining (3B Modell):
  4x A40 48GB: FULL_SHARD
  8x A40 48GB: HYBRID_SHARD oder FULL_SHARD
  2x A100 80GB: SHARD_GRAD_OP
  
Phase 2 (Student + Teacher):
  Student nimmt ~8GB, Teacher nimmt ~8GB (beide 3B FP16)
  + Optimizer State: ~24GB (AdamW fp32)
  → Mindestens 4x A40: FULL_SHARD + Teacher auf separatem Device
  
Phase 3 SFT (GaLore reduziert Memory):
  1x A100 80GB reicht meist
  Bei Multi-GPU: SHARD_GRAD_OP
  
Phase 4 ORPO:
  2x forward (chosen + rejected) = ~2x Memory
  2x A100: SHARD_GRAD_OP
  4x A40: FULL_SHARD
  
Phase 5 LoRA:
  LoRA Training ist leicht, 1x RTX 3090 reicht
  Bei Multi-GPU: NO_SHARD mit DDP
```

---

## 3. Empfohlene RunPod-Setups

### 3.1 Budget-Setup: 4x A40 48GB

```
Hardware:       4x NVIDIA A40 48GB
Total VRAM:     192GB
Kosten:         ~$1.80-2.40/h
Beste für:      Phase 1, 2, 3 (komplett)

Vorteile:
  ✓ Günstigster Multi-GPU Preis
  ✓ Genug VRAM für 3B Modell mit FULL_SHARD
  ✓ NVLink Bandwith ausreichend für FSDP
  
Nachteile:
  ✗ Älter (Ampere), langsamer als Hopper
  ✗ Keine FP8, nur BF16
```

### 3.2 Sweet-Spot: 4x A6000 48GB

```
Hardware:       4x NVIDIA RTX A6000 48GB
Total VRAM:     192GB
Kosten:         ~$2.00-2.80/h
Beste für:      Alle Phasen (Pretraining bis LoRA)

Vorteile:
  ✓ Moderne Ampere, besserer Compute
  ✓ 48GB pro GPU bequem
  ✓ Oft verfügbar auf RunPod
```

### 3.3 Performance-Setup: 2x A100 80GB

```
Hardware:       2x NVIDIA A100 80GB SXM
Total VRAM:     160GB
Kosten:         ~$3.00-3.80/h
Beste für:      Phase 1 wenn Speed > Kosten

Vorteile:
  ✓ NVLink (900GB/s) - extrem schnell
  ✓ FP8 Support möglich
  ✓ Maximum tok/s pro Dollar bei großem Batch
  
Nachteile:
  ✗ 2x GPUs = kleinere FSDP Sharding
```

### 3.4 Bare-Bones: 8x A40 für großes Pretraining

```
Hardware:       8x NVIDIA A40 48GB
Total VRAM:     384GB
Kosten:         ~$3.50-5.00/h
Beste für:      Phase 1 bei sehr großem Modell (7B+)

Für Helix v2 (3B) nicht nötig, aber:
  → Schnelleres Pretraining (bessere Throughput)
  → Kann größere Batch Size fahren
  → HYBRID_SHARD sinnvoll
```

### 3.5 Kosten-Kalkulation pro Phase

```
Phase 1 (30B Tokens Pretraining):

1x H200:
  Zeit: ~3 Wochen (21 Tage)
  Kosten: 21 × 24 × $4.00 = $2016

4x A40:
  Zeit: ~2.5 Wochen (17 Tage, linearer Speedup ~3.5x)
  Kosten: 17 × 24 × $2.20 = $898

Ersparnis: ~55% bei 4x A40

ABER: Engineering-Overhead für Multi-GPU beachten!
Realistische Rechnung:
  + 1 Woche für Multi-GPU Setup + Debugging
  + Erhöhtes Risiko von Out-of-Memory
  + Checkpoint-Size größer (alle Ranks)
  
Trotzdem: Multi-GPU lohnt sich für Phase 1 klar.
```

---

## 4. FSDP Integration

### 4.1 Core FSDP Wrapper

**Datei:** `src/auralis/training/distributed.py`

```python
"""
Multi-GPU Training Setup via FSDP.
Wrapper für alle Trainings-Phasen.
"""

import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    CPUOffload,
    MixedPrecision,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from functools import partial


def setup_distributed():
    """Initialize distributed environment.
    
    Expects these env vars (set by torchrun):
      RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
    """
    # From torchrun
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        torch.cuda.set_device(local_rank)
        
        if rank == 0:
            print(f"Distributed setup: world_size={world_size}")
    
    return {
        'rank': rank,
        'local_rank': local_rank,
        'world_size': world_size,
        'is_main': rank == 0,
    }


def cleanup_distributed():
    """Cleanup at end of training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_with_fsdp(
    model: torch.nn.Module,
    sharding_strategy: str = "FULL_SHARD",
    cpu_offload: bool = False,
    mixed_precision_dtype: str = "bfloat16",
    transformer_layer_cls: type = None,  # HelixBlock
    min_num_params: int = 1_000_000,
) -> FSDP:
    """Wrap model with FSDP.
    
    Args:
        model: Model to wrap
        sharding_strategy: FULL_SHARD | SHARD_GRAD_OP | NO_SHARD | HYBRID_SHARD
        cpu_offload: Offload optimizer states to CPU (slow but saves VRAM)
        mixed_precision_dtype: bfloat16 or float16
        transformer_layer_cls: Class to wrap (e.g., HelixBlock)
        min_num_params: Min params for size-based auto-wrap
    
    Returns:
        FSDP-wrapped model
    """
    # Sharding strategy
    strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    }
    shard_strat = strategy_map[sharding_strategy]
    
    # Mixed Precision
    mp_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[mixed_precision_dtype]
    
    mixed_precision = MixedPrecision(
        param_dtype=mp_dtype,
        reduce_dtype=mp_dtype,
        buffer_dtype=mp_dtype,
    )
    
    # CPU Offload
    cpu_offload_cfg = None
    if cpu_offload:
        cpu_offload_cfg = CPUOffload(offload_params=True)
    
    # Auto-wrap policy
    if transformer_layer_cls is not None:
        # Wrap each transformer block
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={transformer_layer_cls},
        )
    else:
        # Size-based fallback
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=min_num_params,
        )
    
    # Wrap!
    model = FSDP(
        model,
        sharding_strategy=shard_strat,
        mixed_precision=mixed_precision,
        cpu_offload=cpu_offload_cfg,
        auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        forward_prefetch=True,
        use_orig_params=True,  # Needed for optimizer state handling
        device_id=torch.cuda.current_device(),
    )
    
    return model


def is_main_process() -> bool:
    """True wenn rank == 0."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size() -> int:
    """Anzahl der Processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def barrier():
    """Sync all processes."""
    if dist.is_initialized():
        dist.barrier()


def reduce_scalar(value: float, op: str = "mean") -> float:
    """Reduce scalar across all ranks."""
    if not dist.is_initialized():
        return value
    
    tensor = torch.tensor(value, device="cuda")
    
    if op == "mean":
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    elif op == "sum":
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    elif op == "max":
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    else:
        raise ValueError(f"Unknown op: {op}")
    
    return tensor.item()
```

### 4.2 Distributed DataLoader

**Datei:** `src/auralis/training/distributed_data.py`

```python
"""
Distributed DataLoader mit DistributedSampler.
"""

import torch
from torch.utils.data import DataLoader, DistributedSampler, Dataset


def build_distributed_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    drop_last: bool = True,
    seed: int = 42,
) -> DataLoader:
    """Create DataLoader that splits data across ranks.
    
    Jeder Rank sieht nur seinen Teil der Daten.
    """
    import torch.distributed as dist
    
    if dist.is_initialized():
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        # Don't shuffle in DataLoader (sampler does it)
        shuffle_loader = False
    else:
        sampler = None
        shuffle_loader = shuffle
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_loader,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )
    
    return loader


class DistributedStreamingDataset(torch.utils.data.IterableDataset):
    """Streaming Dataset für Pretraining.
    
    Splittet automatisch auf Ranks.
    """
    
    def __init__(
        self,
        data_files: list[str],
        seq_length: int = 2048,
        buffer_size: int = 1000,
        seed: int = 42,
    ):
        super().__init__()
        self.data_files = data_files
        self.seq_length = seq_length
        self.buffer_size = buffer_size
        self.seed = seed
    
    def __iter__(self):
        import torch.distributed as dist
        
        # Determine rank
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
        
        # Worker-within-rank
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1
        
        # Combined split
        total_workers = world_size * num_workers
        worker_global_id = rank * num_workers + worker_id
        
        # Each worker gets different files
        files_per_worker = self.data_files[worker_global_id::total_workers]
        
        # Stream through files
        import numpy as np
        for file in files_per_worker:
            data = np.fromfile(file, dtype=np.uint32)
            
            # Slice into sequences
            n_sequences = len(data) // self.seq_length
            
            # Shuffle indices
            rng = np.random.default_rng(self.seed + worker_global_id)
            indices = rng.permutation(n_sequences)
            
            for idx in indices:
                start = idx * self.seq_length
                end = start + self.seq_length
                seq = data[start:end]
                
                yield {
                    'input_ids': torch.tensor(seq[:-1], dtype=torch.long),
                    'labels': torch.tensor(seq[1:], dtype=torch.long),
                }
```

### 4.3 Distributed Checkpoint Handling

**Datei:** `src/auralis/training/distributed_checkpoint.py`

```python
"""
Checkpoint Handling für FSDP.
Die State-Dicts sind verteilt — braucht spezielle Behandlung.
"""

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)
from pathlib import Path


def save_fsdp_checkpoint(
    model: FSDP,
    optimizer,
    step: int,
    output_dir: str,
    name: str = "checkpoint",
    save_full: bool = True,
):
    """Save FSDP checkpoint.
    
    Args:
        save_full: If True, gathers full state on rank 0 (memory-intensive).
                   If False, saves sharded checkpoints (faster).
    """
    output_path = Path(output_dir)
    
    if save_full:
        # Full state dict on rank 0
        save_policy = FullStateDictConfig(
            offload_to_cpu=True,
            rank0_only=True,
        )
        
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            save_policy,
        ):
            cpu_state = model.state_dict()
        
        if dist.get_rank() == 0:
            output_path.mkdir(parents=True, exist_ok=True)
            torch.save({
                'step': step,
                'model': cpu_state,
            }, output_path / f"{name}.pt")
            print(f"✓ Saved full checkpoint: {output_path / f'{name}.pt'}")
    
    else:
        # Sharded checkpoint (each rank saves its part)
        from torch.distributed.checkpoint import save
        
        shard_dir = output_path / f"{name}_sharded"
        if dist.get_rank() == 0:
            shard_dir.mkdir(parents=True, exist_ok=True)
        
        dist.barrier()
        
        state_dict = {
            'model': model.state_dict(),
            'step': step,
        }
        
        save(
            state_dict=state_dict,
            checkpoint_id=str(shard_dir),
        )
        
        if dist.get_rank() == 0:
            print(f"✓ Saved sharded checkpoint: {shard_dir}")
    
    # Barrier: all ranks wait until save done
    dist.barrier()


def load_fsdp_checkpoint(
    model: FSDP,
    checkpoint_path: str,
    sharded: bool = False,
):
    """Load FSDP checkpoint."""
    if sharded:
        from torch.distributed.checkpoint import load
        load(
            state_dict={'model': model.state_dict()},
            checkpoint_id=checkpoint_path,
        )
    else:
        # Load full state
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        load_policy = FullStateDictConfig(
            offload_to_cpu=True,
            rank0_only=False,  # All ranks need state
        )
        
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            load_policy,
        ):
            model.load_state_dict(checkpoint['model'])


def save_for_inference(
    model: FSDP,
    output_path: str,
):
    """Export FSDP model as single file for inference.
    
    After training, use this to get a standard PyTorch checkpoint
    that can be loaded without FSDP for inference.
    """
    save_policy = FullStateDictConfig(
        offload_to_cpu=True,
        rank0_only=True,
    )
    
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        save_policy,
    ):
        full_state = model.state_dict()
    
    if dist.get_rank() == 0:
        torch.save({'model': full_state}, output_path)
        print(f"✓ Inference checkpoint: {output_path}")
```

---

## 5. Training Script (Phase 1 Multi-GPU Example)

**Datei:** `scripts/pretrain/train_phase1_distributed.py`

```python
"""
Phase 1 Pretraining mit FSDP.

Launch mit:
  torchrun --nproc_per_node=4 scripts/pretrain/train_phase1_distributed.py \
    --config configs/training/phase1_pretrain_multi_gpu.yaml
"""

import argparse
import os
import torch
import torch.distributed as dist
from pathlib import Path

from auralis.model import build_model
from auralis.model.helix_model import HelixBlock
from auralis.training.distributed import (
    setup_distributed,
    cleanup_distributed,
    wrap_model_with_fsdp,
    is_main_process,
    reduce_scalar,
    barrier,
)
from auralis.training.distributed_data import build_distributed_dataloader
from auralis.training.distributed_checkpoint import (
    save_fsdp_checkpoint,
    save_for_inference,
)
from auralis.training.utils import load_config, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume', default=None)
    args = parser.parse_args()
    
    # === Setup Distributed ===
    dist_info = setup_distributed()
    
    if dist_info['is_main']:
        print("=" * 60)
        print("PHASE 1: Multi-GPU Pretraining")
        print("=" * 60)
        print(f"World Size: {dist_info['world_size']}")
        print(f"GPUs: {torch.cuda.device_count()}")
    
    # === Config ===
    config = load_config(args.config)
    set_seed(42 + dist_info['rank'])
    
    # === Model ===
    if dist_info['is_main']:
        print("\nBuilding model...")
    
    # Build on each rank (but FSDP will shard)
    model = build_model(config.model.config_path)
    
    # Move to local GPU
    model = model.to(torch.cuda.current_device())
    
    # === FSDP Wrap ===
    if dist_info['is_main']:
        print(f"Wrapping with FSDP: {config.training.distributed.sharding_strategy}")
    
    model = wrap_model_with_fsdp(
        model,
        sharding_strategy=config.training.distributed.sharding_strategy,
        cpu_offload=config.training.distributed.cpu_offload,
        mixed_precision_dtype=config.training.dtype,
        transformer_layer_cls=HelixBlock,
    )
    
    if dist_info['is_main']:
        # Count trainable params (approximation)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total params (shard): {total_params / 1e9:.2f}B")
    
    # === Data ===
    from auralis.training.distributed_data import DistributedStreamingDataset
    
    train_dataset = DistributedStreamingDataset(
        data_files=[
            f"{config.data.data_dir}/english.bin",
            f"{config.data.data_dir}/german.bin",
            f"{config.data.data_dir}/code.bin",
        ],
        seq_length=config.data.seq_length,
    )
    
    # Standard DataLoader — IterableDataset handles distribution
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size_per_device,
        num_workers=config.training.num_workers,
        pin_memory=True,
    )
    
    # === Optimizer ===
    # WICHTIG: Optimizer wird NACH FSDP-Wrap erstellt!
    optimizer = torch.optim.AdamW(
        model.parameters(),  # FSDP parameters
        lr=config.training.optimizer.lr,
        betas=config.training.optimizer.betas,
        weight_decay=config.training.optimizer.weight_decay,
    )
    
    # Scheduler
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.training.total_steps,
        eta_min=config.training.optimizer.lr * 0.1,
    )
    
    # === Resume ===
    start_step = 0
    if args.resume:
        # Load checkpoint
        pass  # Implementation depends on sharded vs full
    
    # === Training Loop ===
    model.train()
    
    step = start_step
    accumulated_loss = 0
    
    data_iter = iter(train_loader)
    
    while step < config.training.total_steps:
        # === Gradient Accumulation ===
        for micro_step in range(config.training.gradient_accumulation):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            
            # Move to GPU
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            
            # Forward
            output = model(
                input_ids=batch['input_ids'],
                labels=batch['labels'],
            )
            loss = output['loss'] / config.training.gradient_accumulation
            
            # Backward
            loss.backward()
            
            accumulated_loss += loss.item()
        
        # === Gradient Clip ===
        grad_norm = model.clip_grad_norm_(
            max_norm=config.training.gradient_clip_norm,
        )
        
        # === Optimizer Step ===
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        step += 1
        
        # === Logging (only rank 0) ===
        if step % config.logging.log_every == 0:
            # Reduce loss across ranks for accurate metric
            avg_loss = reduce_scalar(accumulated_loss, op="mean")
            
            if dist_info['is_main']:
                lr = scheduler.get_last_lr()[0]
                print(
                    f"Step {step:6d} | "
                    f"loss {avg_loss:.4f} | "
                    f"lr {lr:.2e} | "
                    f"grad_norm {grad_norm:.2f}"
                )
            
            accumulated_loss = 0
        
        # === Eval (only rank 0) ===
        if step % config.logging.eval_every == 0:
            # All ranks participate in eval forward passes
            # But only rank 0 aggregates/prints
            val_loss = _distributed_eval(model, config)
            
            if dist_info['is_main']:
                print(f"  Val Loss: {val_loss:.4f}")
        
        # === Checkpoint ===
        if step % config.logging.save_every == 0:
            save_fsdp_checkpoint(
                model=model,
                optimizer=optimizer,
                step=step,
                output_dir=config.checkpointing.output_dir,
                name=f"step_{step}",
                save_full=config.checkpointing.save_full,
            )
    
    # === Final Save ===
    if dist_info['is_main']:
        print("\nTraining complete. Saving final model...")
    
    save_for_inference(
        model,
        output_path=f"{config.checkpointing.output_dir}/final.pt",
    )
    
    cleanup_distributed()


def _distributed_eval(model, config):
    """Eval on val set, averaged across ranks."""
    model.eval()
    
    # Placeholder - implementiere mit eigenem val loader
    total_loss = 0.0
    n = 1
    
    model.train()
    return total_loss / n


if __name__ == "__main__":
    main()
```

---

## 6. Launch Scripts

### 6.1 Single-Node Multi-GPU (häufigster Fall)

**Datei:** `scripts/launch/train_multi_gpu.sh`

```bash
#!/bin/bash
# Launch Multi-GPU Training on Single Node (RunPod Pod)

# Auto-detect GPUs
NGPUS=$(nvidia-smi --list-gpus | wc -l)

echo "Starting training with $NGPUS GPUs"

# NCCL Tuning (important for performance!)
export NCCL_IB_DISABLE=0          # Enable InfiniBand if available
export NCCL_P2P_DISABLE=0          # Enable P2P
export NCCL_DEBUG=WARN             # Only warnings (set to INFO for debugging)
export OMP_NUM_THREADS=8           # OpenMP threads per process

# Memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# torchrun Launcher
torchrun \
    --standalone \
    --nproc_per_node=$NGPUS \
    scripts/pretrain/train_phase1_distributed.py \
    --config configs/training/phase1_pretrain_multi_gpu.yaml \
    "$@"
```

### 6.2 Variante mit spezifischer GPU-Anzahl

```bash
#!/bin/bash
# scripts/launch/train_4gpu.sh

torchrun \
    --standalone \
    --nproc_per_node=4 \
    scripts/pretrain/train_phase1_distributed.py \
    --config configs/training/phase1_pretrain_4gpu.yaml
```

### 6.3 Resumption

```bash
# Resume from checkpoint
bash scripts/launch/train_multi_gpu.sh \
    --resume checkpoints/phase1_pretrain/step_50000.pt
```

---

## 7. Config-Anpassungen

### 7.1 Neues Multi-GPU Config-Pattern

```yaml
# configs/training/phase1_pretrain_4gpu.yaml

experiment:
  name: "helix_v2_phase1_pretrain_4gpu"

model:
  config_path: "configs/model/helix_v2_3b.yaml"

data:
  config_path: "configs/data/phase1_mix.yaml"
  data_dir: "data/training/phase1"
  seq_length: 2048

training:
  dtype: "bfloat16"
  
  # === DISTRIBUTED CONFIG ===
  distributed:
    sharding_strategy: "FULL_SHARD"   # oder SHARD_GRAD_OP / NO_SHARD
    cpu_offload: false                # true nur wenn VRAM knapp
    backward_prefetch: true
  
  # Batch sizing
  batch_size_per_device: 4            # Pro GPU
  gradient_accumulation: 8
  # Effective Batch = 4 (GPUs) * 4 (batch) * 8 (accum) = 128 sequences
  
  num_workers: 4
  
  optimizer:
    name: "adamw"
    lr: 3.0e-4
    betas: [0.9, 0.95]
    weight_decay: 0.1
  
  scheduler:
    type: "cosine"
    warmup_steps: 2000
    min_lr_ratio: 0.1
  
  gradient_clip_norm: 1.0
  total_steps: 115_000

logging:
  log_every: 10
  eval_every: 1000
  save_every: 2500

checkpointing:
  output_dir: "checkpoints/phase1_pretrain"
  save_full: true                     # Full state (rank 0 aggregates)
  save_last_n: 3
```

### 7.2 Batch-Size Scaling bei Multi-GPU

```
Konzept: "Effective Batch Size" muss konstant bleiben

Single GPU Baseline:
  batch_per_device: 8
  gradient_accumulation: 16
  effective_batch: 8 * 16 = 128

4-GPU Äquivalent (gleiche effective batch):
  batch_per_device: 8
  gradient_accumulation: 4    # 16/4 = 4
  effective_batch: 4 * 8 * 4 = 128 ✓

8-GPU Äquivalent:
  batch_per_device: 8
  gradient_accumulation: 2    # 16/8 = 2
  effective_batch: 8 * 8 * 2 = 128 ✓

Regel: 
  effective_batch = world_size * batch_per_device * gradient_accumulation
```

---

## 8. Per-Phase Anpassungen

### 8.1 Phase 1 Pretraining

```yaml
# 4x A40 48GB Setup
training:
  distributed:
    sharding_strategy: "FULL_SHARD"
  batch_size_per_device: 4
  gradient_accumulation: 8
  # Effective: 128

# 2x A100 80GB Setup
training:
  distributed:
    sharding_strategy: "SHARD_GRAD_OP"  # Weniger Sharding, mehr Speed
  batch_size_per_device: 8
  gradient_accumulation: 8
  # Effective: 128
```

### 8.2 Phase 2 Continued mit Teacher (tricky!)

**Problem:** Student + Teacher + Optimizer = viel Memory

**Lösung A:** Teacher auf separatem Device / CPU

```python
# In train_phase2.py
student = build_model(...).to("cuda:0")
teacher = build_model(...).to("cuda:0")  # Same device with FSDP

# OR mit Tensor Parallel
student = wrap_with_fsdp(student, sharding=FULL_SHARD)
teacher = teacher.eval()  # Frozen, kein FSDP nötig
teacher = teacher.to("cpu")  # Offload wenn VRAM knapp
```

**Lösung B:** Beide FSDP-Wrapped

```python
# Auf 4x A40:
student = wrap_with_fsdp(student, sharding=FULL_SHARD)
teacher = wrap_with_fsdp(teacher, sharding=FULL_SHARD)

# Teacher state bleibt frozen
for p in teacher.parameters():
    p.requires_grad = False
```

### 8.3 Phase 3 SFT mit GaLore

```yaml
# GaLore + FSDP funktioniert, aber:
# Initial: mit SHARD_GRAD_OP statt FULL_SHARD starten
training:
  distributed:
    sharding_strategy: "SHARD_GRAD_OP"
  
  optimizer:
    name: "galore_adamw"
    # GaLore-Rank auf Multi-GPU: per-rank reduziert sich automatisch
    galore_rank: 128  # Gleich wie single-GPU
```

### 8.4 Phase 4 ORPO (2x forward)

```yaml
# ORPO braucht 2x Memory wegen chosen + rejected
# Daher: kleinere batch_per_device
training:
  distributed:
    sharding_strategy: "FULL_SHARD"
  batch_size_per_device: 1   # Klein wegen 2x forward
  gradient_accumulation: 32
  # Effective: world_size * 1 * 32
```

### 8.5 Phase 5 LoRA

```yaml
# LoRA training ist leicht, einfaches DDP reicht
training:
  distributed:
    sharding_strategy: "NO_SHARD"  # = DDP
  batch_size_per_device: 8
  gradient_accumulation: 2
```

---

## 9. Fehlerbehebung

### 9.1 Häufige Probleme

```
Problem: "Address already in use"
Ursache: Vorheriger torchrun nicht sauber beendet
Lösung:
  pkill -f torchrun
  pkill -f python

Problem: "NCCL connection failure"
Ursache: Port-Konflikt oder Netzwerk
Lösung:
  export MASTER_PORT=12357  # Anderer Port
  export NCCL_DEBUG=INFO    # Debug-Infos

Problem: "CUDA OOM on one rank"
Ursache: Load imbalance
Lösung: Kleinere batch_per_device, oder CPU offload

Problem: "Deadlock in DataLoader"
Ursache: num_workers Problem mit multiprocessing
Lösung: num_workers=0 oder persistent_workers=True
```

### 9.2 Performance-Debugging

```bash
# GPU Utilization prüfen
nvidia-smi dmon -s pucvmet -d 1

# NCCL Communication profiling
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Python-level profiling
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Deadlock detection
export TORCH_SHOW_CPP_STACKTRACES=1
```

### 9.3 Speed-Optimierungen

```yaml
# 1. Gradient Checkpointing (trade compute for memory)
advanced:
  gradient_checkpointing: true
  
# 2. Flash Attention 3 wenn möglich
advanced:
  use_flash_attention: true

# 3. torch.compile (PyTorch 2.5+)
advanced:
  torch_compile: true
  compile_mode: "max-autotune"  # oder "reduce-overhead"

# 4. Activation Offloading
distributed:
  activation_cpu_offload: true  # Nur wenn VRAM super knapp
```

---

## 10. Checkliste pro Training-Launch

```
Vor dem Start:
  □ RunPod Pod gemietet (mit geplanter GPU-Anzahl)
  □ NCCL funktioniert (Test mit kleinem Script)
  □ Alle Daten synced auf Pod (oder remote mounted)
  □ Config YAML prüft effective_batch_size
  □ Disk-Space > 500GB frei
  □ Auto-Refill aktiv auf RunPod Account
  □ WandB Run erstellt und getaggt

Sanity Check (vor großem Run):
  □ 100 Steps Test-Run auf 1 GPU klappt
  □ 100 Steps Test-Run auf Multi-GPU klappt
  □ Checkpoint-Save klappt (inkl. Load zurück)
  □ Loss-Kurven zwischen Single und Multi ähnlich

Während des Trainings:
  □ GPU Utilization > 80% (sonst: Bottleneck)
  □ VRAM < 90% (sonst: kurz vor OOM)
  □ Loss sinkt monoton
  □ Grad Norm stabil
  □ Checkpoints werden gespeichert

Nach Training:
  □ Final-Inference-Checkpoint gespeichert (full state, rank 0)
  □ Full State Validierung: Model lädt standalone
  □ External Backup auf S3/lokal
  □ Pod gestoppt (Kosten!)
```

---

## 11. Beispiel: Kompletter Phase 1 Multi-GPU Workflow

```bash
# === 1. Pod starten ===
# RunPod Web UI: 4x A40 48GB Pod mit PyTorch 2.5 Template

# === 2. SSH in Pod ===
ssh root@pod-xxxxx.runpod.io

# === 3. Environment ===
cd /workspace
git clone https://github.com/yourname/auralis-v2.git
cd auralis-v2
pip install -e ".[dev]"

# === 4. Daten laden (Syncthing/rsync von BitBastion) ===
rsync -avP bitbastion:/data/training/phase1/ data/training/phase1/
rsync -avP bitbastion:/tokenizer/ tokenizer/

# === 5. Sanity Check ===
bash scripts/launch/train_multi_gpu.sh \
    --config configs/training/phase1_pretrain_4gpu.yaml \
    --dry-run

# === 6. Echter Start (mit tmux!) ===
tmux new -s training

# Inside tmux:
bash scripts/launch/train_multi_gpu.sh \
    --config configs/training/phase1_pretrain_4gpu.yaml \
    2>&1 | tee logs/phase1_$(date +%s).log

# Detach: Ctrl+B, D

# === 7. Monitoring ===
# In separatem Terminal:
ssh root@pod-xxxxx.runpod.io
tmux attach -t training

# Oder via WandB Web-UI

# === 8. Backup während Training ===
# Cron job every 6h:
*/6 * * * * rsync -avP checkpoints/ bitbastion:/backup/auralis-v2/checkpoints/

# === 9. Training beendet ===
# Final Export für Phase 2:
python scripts/utils/extract_inference_checkpoint.py \
    --input checkpoints/phase1_pretrain/final.pt \
    --output checkpoints/phase1_final_inference.pt

# Backup
rsync -avP checkpoints/phase1_final_inference.pt bitbastion:/backup/

# === 10. Pod stoppen ===
# RunPod UI: Stop Pod (um Kosten zu sparen)
```

---

## 12. Summary: Multi-GPU Best Practices

```
DO:
  ✓ FSDP nutzen statt DeepSpeed (für Auralis v2)
  ✓ FULL_SHARD für > 3B Modelle
  ✓ BF16 Mixed Precision (A40/A100/H100)
  ✓ Gradient Checkpointing aktivieren wenn VRAM knapp
  ✓ torch.compile für Speed
  ✓ NCCL Environment Variables setzen
  ✓ Effective Batch Size konstant halten
  ✓ Sanity Check vor großem Run
  ✓ Regelmäßige Backups
  ✓ tmux/screen für lange Runs

DON'T:
  ✗ Multi-GPU für LoRA Training (Overhead > Benefit)
  ✗ FULL_SHARD bei <3B Modell (unnötige Communication)
  ✗ CPU Offload wenn VRAM reicht (sehr langsam!)
  ✗ num_workers=0 bei großen Datasets (CPU Bottleneck)
  ✗ Ohne Sanity-Check Phase 1 starten
  ✗ Checkpoint ohne Full-State-Extract für Phase 2
```

---

## 13. Anpassungen an bestehende Phasen-Specs

Diese Multi-GPU Spec gilt als **Overlay** über Phase 1-5. Konkret:

**Phase 1 (Pretraining):** Standardmäßig Multi-GPU, siehe 4x A40 Config oben

**Phase 2 (Continued + KL):** Multi-GPU mit Teacher-Handling (Sektion 8.2)

**Phase 3 (SFT):** Single oder Multi-GPU möglich, SHARD_GRAD_OP empfohlen

**Phase 4 (ORPO):** Multi-GPU wegen doppeltem Memory-Bedarf

**Phase 5 (LoRA):** Single-GPU reicht, außer bei großen Meta-LoRAs

**Hinweis für Claude Code:**

Die Trainings-Scripts in den Phase-Specs sind single-GPU Versionen. Für Multi-GPU diesen Overlay nutzen und die Scripts entsprechend anpassen (setup_distributed, FSDP-Wrap, DistributedSampler).

---

*Multi-GPU Training Spec Version 1.0 — April 2026*
*Overlay für alle Phasen von Auralis v2 / Helix v2*
