# Phase 0.5: Model Architecture

> Current note (2026-05-17): This is the main architecture spec for Helix v2.
> It describes the intended hybrid stack. Some production-size examples
> mention 2B/3B, while current canary work uses smaller 100M/500M configs.
> For live run state, use `STATUS.md`.

**Project:** Auralis v2 / Helix v2
**Phase:** 0.5 (between tokenizer and pretraining)
**Duration:** 1 week
**Goal:** Modular, scalable architecture implemented + tested

---

## 1. Goals

- Modular architecture that later supports MoE, MTP, various attention types
- Heterogeneous layer stack (Mamba + GLA + Sparse Attention)
- Fully config-driven (no hardcoding)
- Small version (100M) for tests, large version (3B) for production
- 100% unit-test coverage of the core components

---

## 2. Deliverables

```
/src/auralis/model/
    __init__.py
    config.py                   # AuralisConfig dataclass
    helix_model.py              # HelixModel (Main Class)
    
    layers/
        __init__.py
        mamba_layer.py          # Mamba-2 Implementation
        gla_layer.py            # Gated Linear Attention
        sparse_attn_layer.py    # Sparse Attention
        ffn.py                  # Dense FFN + MoE-Ready
        embedding.py            # Token + Position Embeddings
        norm.py                 # RMSNorm, LayerNorm
    
    utils/
        __init__.py
        rotary.py               # Rotary Position Embeddings
        init.py                 # Weight Initialization
        kv_cache.py             # KV-Cache für Inference
        
    experimental/
        moe.py                  # MoE (für später aktivierbar)
        mtp.py                  # Multi-Token Prediction

/configs/model/
    helix_v2_100m.yaml         # Test-Modell
    helix_v2_2b.yaml           # Production-Modell
    helix_v2_3b.yaml           # Alternative
    
/tests/model/
    test_config.py
    test_layers.py
    test_helix_model.py
    test_forward_backward.py
    test_inference.py
```

---

## 3. Architecture Details

### 3.1 Layer Stack (for 28-layer model)

```
Layer 0-5:    Mamba-2 (6 Layers)
              → Lokaler Kontext, günstig
              → Beste für frühe Features
              → Kein Quadratisches Memory

Layer 6-21:   GLA (16 Layers)
              → Hauptkörper, effizient
              → Gated Linear Attention
              → Du hast Custom Kernel aus v1

Layer 22-27:  Sparse Attention (6 Layers)
              → Long-context retrieval
              → Volle Attention nur in letzten Layers
              → Needle-in-Haystack fähig
```

**Why this stack:**

- Early layers learn tokenized patterns (SSM perfect)
- Middle layers build features (GLA efficient)
- Late layers do "global reasoning" (attention needed)

### 3.2 Dimensions (Helix v2 Standard)

```yaml
# configs/model/helix_v2_3b.yaml

name: "helix_v2_3b"
version: "2.0.0"

# Tokenizer
tokenizer:
  path: "tokenizer/helix_v2_tokenizer.model"
  vocab_size: 200000

# Core Dimensions
model:
  d_model: 2048              # Hidden size
  n_layers: 28                # Total layers
  n_heads: 16                 # Attention heads (wo attention)
  d_head: 128                 # Dimension per head
  d_ffn: 5632                 # FFN hidden (2.75x d_model)
  
  # Activation
  activation: "silu"          # SiLU / Swish
  
  # Normalization
  norm_type: "rmsnorm"        # RMSNorm > LayerNorm
  norm_eps: 1.0e-6

# Layer Configuration (pro Layer anpassbar!)
layers:
  # Layer 0-5: Mamba
  - type: "mamba"
    d_state: 128
    d_conv: 4
    expand_factor: 2
    dt_min: 0.001
    dt_max: 0.1
  - type: "mamba"  # Repeat config oder per-layer anders
  - type: "mamba"
  - type: "mamba"
  - type: "mamba"
  - type: "mamba"
  
  # Layer 6-21: GLA (16 layers - kurz schreiben)
  - {type: "gla", d_state: 128}  # repeat 16x via config loader
  # ...
  
  # Layer 22-27: Sparse Attention
  - type: "sparse_attention"
    window_size: 1024
    global_tokens: 32
    use_rope: true
  # ...

# Position Encoding
position_encoding:
  type: "rope"                # Rotary Position Embeddings
  theta: 10000.0
  max_seq_length: 8192

# FFN Configuration
ffn:
  type: "dense"               # "dense" or "moe"
  activation: "silu_gated"    # SwiGLU

# MoE Configuration (default: off, für später)
moe:
  enabled: false              # Switch on/off
  n_experts: 8
  n_experts_per_token: 2
  capacity_factor: 1.25
  load_balance_loss_weight: 0.01

# Multi-Token Prediction (default: off)
mtp:
  enabled: false
  n_heads: 1                  # 1 = normal, 3 = DeepSeek-style

# Initialization
init:
  scheme: "scaled_normal"     # Scaled normal init
  init_std: 0.02
  embedding_init_std: 0.02
  output_init_scale: 0.5      # Output Layer skalieren

# Dropout (training only)
dropout:
  embedding: 0.0
  attention: 0.0
  ffn: 0.0
  residual: 0.0

# Advanced
advanced:
  tie_embeddings: false       # lm_head shared with embeddings
  use_flash_attention: true   # Wenn verfügbar
  gradient_checkpointing: false  # Für Training aktivierbar
```

---

## 4. Modular Config Class

**File:** `src/auralis/model/config.py`

```python
"""
Config für Helix v2 Modell.
Vollständig YAML-driven, keine Hardcoded Werte.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class LayerConfig:
    """Konfiguration für einzelne Layer."""
    type: str  # "mamba", "gla", "sparse_attention"
    
    # Common
    d_state: Optional[int] = None
    
    # Mamba-specific
    d_conv: Optional[int] = None
    expand_factor: Optional[int] = None
    dt_min: Optional[float] = None
    dt_max: Optional[float] = None
    
    # Attention-specific
    window_size: Optional[int] = None
    global_tokens: Optional[int] = None
    use_rope: Optional[bool] = None


@dataclass
class FFNConfig:
    type: str = "dense"  # "dense" or "moe"
    activation: str = "silu_gated"


@dataclass
class MoEConfig:
    enabled: bool = False
    n_experts: int = 8
    n_experts_per_token: int = 2
    capacity_factor: float = 1.25
    load_balance_loss_weight: float = 0.01


@dataclass
class MTPConfig:
    enabled: bool = False
    n_heads: int = 1


@dataclass
class PositionEncodingConfig:
    type: str = "rope"
    theta: float = 10000.0
    max_seq_length: int = 8192


@dataclass
class InitConfig:
    scheme: str = "scaled_normal"
    init_std: float = 0.02
    embedding_init_std: float = 0.02
    output_init_scale: float = 0.5


@dataclass
class DropoutConfig:
    embedding: float = 0.0
    attention: float = 0.0
    ffn: float = 0.0
    residual: float = 0.0


@dataclass
class AdvancedConfig:
    tie_embeddings: bool = False
    use_flash_attention: bool = True
    gradient_checkpointing: bool = False


@dataclass
class AuralisConfig:
    """Gesamt-Konfiguration für Helix v2."""
    
    name: str
    version: str
    
    # Dimensions
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    d_head: int
    d_ffn: int
    
    # Activation/Norm
    activation: str = "silu"
    norm_type: str = "rmsnorm"
    norm_eps: float = 1e-6
    
    # Layer-wise config
    layers: list[LayerConfig] = field(default_factory=list)
    
    # Modular components
    ffn: FFNConfig = field(default_factory=FFNConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    mtp: MTPConfig = field(default_factory=MTPConfig)
    position_encoding: PositionEncodingConfig = field(
        default_factory=PositionEncodingConfig
    )
    init: InitConfig = field(default_factory=InitConfig)
    dropout: DropoutConfig = field(default_factory=DropoutConfig)
    advanced: AdvancedConfig = field(default_factory=AdvancedConfig)
    
    # Tokenizer
    tokenizer_path: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "AuralisConfig":
        """Lädt Config aus YAML-Datei."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        
        # Extract tokenizer info
        tokenizer_path = data.get('tokenizer', {}).get('path')
        vocab_size = data.get('tokenizer', {}).get('vocab_size', 200000)
        
        # Model section
        model_cfg = data.get('model', {})
        
        # Layer configs (handle shortcuts)
        layers_raw = data.get('layers', [])
        layers = []
        for layer_spec in layers_raw:
            # Skip malformed entries
            if isinstance(layer_spec, dict) and 'type' in layer_spec:
                layers.append(LayerConfig(**layer_spec))
        
        # Validate: layers count must match n_layers
        n_layers = model_cfg.get('n_layers', 28)
        if len(layers) != n_layers:
            raise ValueError(
                f"Config error: {len(layers)} layers specified "
                f"but n_layers={n_layers}"
            )
        
        return cls(
            name=data.get('name', 'helix_v2'),
            version=data.get('version', '2.0.0'),
            vocab_size=vocab_size,
            d_model=model_cfg['d_model'],
            n_layers=model_cfg['n_layers'],
            n_heads=model_cfg['n_heads'],
            d_head=model_cfg['d_head'],
            d_ffn=model_cfg['d_ffn'],
            activation=model_cfg.get('activation', 'silu'),
            norm_type=model_cfg.get('norm_type', 'rmsnorm'),
            norm_eps=model_cfg.get('norm_eps', 1e-6),
            layers=layers,
            ffn=FFNConfig(**data.get('ffn', {})),
            moe=MoEConfig(**data.get('moe', {})),
            mtp=MTPConfig(**data.get('mtp', {})),
            position_encoding=PositionEncodingConfig(
                **data.get('position_encoding', {})
            ),
            init=InitConfig(**data.get('init', {})),
            dropout=DropoutConfig(**data.get('dropout', {})),
            advanced=AdvancedConfig(**data.get('advanced', {})),
            tokenizer_path=tokenizer_path,
        )
    
    def to_yaml(self, path: str | Path) -> None:
        """Speichert Config als YAML."""
        # Implementation ausgelassen für Kürze
        pass
    
    def count_parameters(self) -> dict[str, int]:
        """Schätzt Parameter-Anzahl pro Komponente."""
        # Embedding: vocab * d_model
        emb_params = self.vocab_size * self.d_model
        
        # Per-Layer (grob)
        layer_params = 0
        for layer in self.layers:
            if layer.type == "mamba":
                # Rough: d_state * d_model * expand * 4
                layer_params += (
                    (layer.d_state or 128) 
                    * self.d_model 
                    * (layer.expand_factor or 2) 
                    * 4
                )
            elif layer.type == "gla":
                # GLA: q, k, v, o projections
                layer_params += 4 * self.d_model * self.d_model
            elif layer.type == "sparse_attention":
                # Standard attention
                layer_params += 4 * self.d_model * self.d_model
        
        # FFN per layer: 3x gated
        ffn_params = 3 * self.d_model * self.d_ffn * self.n_layers
        
        # Output head
        lm_head_params = 0 if self.advanced.tie_embeddings \
                          else self.vocab_size * self.d_model
        
        total = emb_params + layer_params + ffn_params + lm_head_params
        
        return {
            "embedding": emb_params,
            "layers": layer_params,
            "ffn": ffn_params,
            "lm_head": lm_head_params,
            "total": total,
        }
```

---

## 5. Layer Implementations

### 5.1 Mamba-2 Layer

**File:** `src/auralis/model/layers/mamba_layer.py`

```python
"""
Mamba-2 Layer (Selective State Space).
Modernisierte Version von Mamba-1, besser skalierbar.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mamba2Layer(nn.Module):
    """Mamba-2 mit Selective State Space.
    
    Wesentlich effizienter als Attention für:
      - Lokalen Kontext
      - Frühe Feature-Extraction
      - Lange Sequenzen
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand_factor: int = 2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand_factor * d_model
        
        # Input projection
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        
        # Conv layer
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        
        # dt initialization
        dt_init_std = self.d_inner ** -0.5
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # Initialize dt in a specific range
        dt = torch.exp(
            torch.rand(self.d_inner) * (torch.log(torch.tensor(dt_max)) 
                                         - torch.log(torch.tensor(dt_min)))
            + torch.log(torch.tensor(dt_min))
        )
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        
        # State Space A (learnable)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        A = A.repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        
        # D (skip connection in SSM)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
    
    def forward(
        self,
        x: torch.Tensor,  # [B, L, D]
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            state: Optional previous state for inference
        
        Returns:
            output: [batch, seq_len, d_model]
            state: Updated state (for inference)
        """
        B, L, D = x.shape
        
        # Input projection
        xz = self.in_proj(x)  # [B, L, 2*d_inner]
        x_in, z = xz.chunk(2, dim=-1)
        
        # Conv (transpose for Conv1d)
        x_in = x_in.transpose(1, 2)  # [B, d_inner, L]
        x_in = self.conv1d(x_in)[..., :L]
        x_in = x_in.transpose(1, 2)  # [B, L, d_inner]
        x_in = F.silu(x_in)
        
        # SSM
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]
        
        # Project to dt, B, C
        x_dbl = self.x_proj(x_in)  # [B, L, d_state*2 + d_inner]
        dt, B_ssm, C_ssm = x_dbl.split(
            [self.d_inner, self.d_state, self.d_state], dim=-1
        )
        
        dt = F.softplus(self.dt_proj(dt))  # [B, L, d_inner]
        
        # Selective SSM (simplified for clarity — real impl uses CUDA kernel)
        y = self._selective_scan(x_in, dt, A, B_ssm, C_ssm, self.D, state)
        
        # Gate
        y = y * F.silu(z)
        
        # Output
        output = self.out_proj(y)
        
        return output, state  # state handling simplified
    
    def _selective_scan(self, x, dt, A, B, C, D, state=None):
        """Simplified selective scan. Use mamba_ssm library for production."""
        # Placeholder - use torch implementation or mamba_ssm package
        # This is a pedagogical version, NOT optimized
        B_size, L, d_inner = x.shape
        d_state = A.shape[1]
        
        # Discretize
        dA = torch.exp(dt.unsqueeze(-1) * A)  # [B, L, d_inner, d_state]
        dB = dt.unsqueeze(-1) * B.unsqueeze(-2)  # [B, L, d_inner, d_state]
        
        # Initial state
        if state is None:
            h = torch.zeros(B_size, d_inner, d_state, device=x.device, dtype=x.dtype)
        else:
            h = state
        
        # Scan
        outputs = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            y_t = (h * C[:, t].unsqueeze(-2)).sum(dim=-1)
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)
        
        # Skip connection
        y = y + x * D
        
        return y
```

**Note:** Production-ready Mamba-2 uses the `mamba_ssm` package.
The code above is pedagogical — replace with a library import:

```python
from mamba_ssm import Mamba2
```

### 5.2 GLA Layer (port from v1)

You have this from Helix v1 — just transfer + clean up.

```python
# src/auralis/model/layers/gla_layer.py

"""
Gated Linear Attention — aus Helix v1 portiert.
Custom Kernel gibt 19-30x Speedup.
"""

import torch
import torch.nn as nn
# from auralis.kernels.gla_kernel import gla_attention  # Custom Kernel


class GLALayer(nn.Module):
    """GLA mit Gated Linear Attention."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 16,
        d_head: int = 128,
        d_state: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state
        
        # QKV projections
        self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        
        # Gate projection
        self.g_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        
        # Alpha (decay rate) - learnable
        self.alpha_proj = nn.Linear(d_model, n_heads, bias=True)
        
        # Output
        self.out_proj = nn.Linear(n_heads * d_head, d_model, bias=False)
    
    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_head)
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head)
        g = torch.sigmoid(self.g_proj(x).view(B, L, self.n_heads, self.d_head))
        
        alpha = torch.sigmoid(self.alpha_proj(x))  # [B, L, n_heads]
        
        # GLA Attention (use custom kernel in production)
        output = self._gla_attention(q, k, v, alpha, state)
        
        # Gate
        output = output * g
        
        # Reshape + output projection
        output = output.view(B, L, -1)
        output = self.out_proj(output)
        
        return output, state
    
    def _gla_attention(self, q, k, v, alpha, state):
        """GLA Attention berechnung.
        
        Siehe: auralis.kernels.gla_kernel für Custom CUDA Impl.
        """
        # Placeholder - portiere von v1
        raise NotImplementedError("Port from Helix v1")
```

### 5.3 Sparse Attention Layer

```python
# src/auralis/model/layers/sparse_attn_layer.py

"""
Sparse Attention für long-context retrieval.
Window + Global Tokens Pattern.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from auralis.model.utils.rotary import apply_rotary_pos_emb


class SparseAttentionLayer(nn.Module):
    """Sliding Window + Global Tokens."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 16,
        d_head: int = 128,
        window_size: int = 1024,
        global_tokens: int = 32,
        use_rope: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.window_size = window_size
        self.global_tokens = global_tokens
        self.use_rope = use_rope
        
        self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.out_proj = nn.Linear(n_heads * d_head, d_model, bias=False)
    
    def forward(self, x, kv_cache=None, rope_cache=None):
        B, L, D = x.shape
        
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_head)
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head)
        
        if self.use_rope and rope_cache is not None:
            q, k = apply_rotary_pos_emb(q, k, rope_cache)
        
        # Sparse Attention: window + globals
        # Simplified - use flash_attention for production
        output = self._sparse_attention(q, k, v)
        
        output = output.view(B, L, -1)
        return self.out_proj(output), kv_cache
    
    def _sparse_attention(self, q, k, v):
        """Sparse Attention Pattern.
        
        Für Production: flash_attn_with_kvcache
        Mit sliding window + global token handling.
        """
        # Placeholder
        # Production: from flash_attn import flash_attn_func
        B, L, H, D = q.shape
        
        q = q.transpose(1, 2)  # [B, H, L, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scale = D ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Causal mask
        mask = torch.triu(torch.ones(L, L, device=q.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
        
        # TODO: Apply window + global token pattern here
        
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)  # [B, H, L, D]
        
        return output.transpose(1, 2)  # [B, L, H, D]
```

### 5.4 FFN (with MoE-Ready Interface)

```python
# src/auralis/model/layers/ffn.py

"""
Feed-Forward Network.
Dense (default) oder MoE (optional aktivierbar).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseFFN(nn.Module):
    """Standard SwiGLU FFN."""
    
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        activation: str = "silu_gated",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        
        # SwiGLU: gated variant
        self.gate_proj = nn.Linear(d_model, d_ffn, bias=False)
        self.up_proj = nn.Linear(d_model, d_ffn, bias=False)
        self.down_proj = nn.Linear(d_ffn, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class MoEFFN(nn.Module):
    """Mixture of Experts FFN.
    
    Aktivierbar via config.moe.enabled = True.
    Für Phase 5 (wenn MoE gewünscht).
    """
    
    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        n_experts: int = 8,
        n_experts_per_token: int = 2,
        capacity_factor: float = 1.25,
    ):
        super().__init__()
        # Placeholder - aktivierbar später
        self.dense_fallback = DenseFFN(d_model, d_ffn)
    
    def forward(self, x):
        # Falls MoE aktiviert: Experten-Routing
        # Sonst: Fallback zu Dense
        return self.dense_fallback(x)


def build_ffn(config) -> nn.Module:
    """Factory für FFN basierend auf Config."""
    if config.ffn.type == "moe" and config.moe.enabled:
        return MoEFFN(
            d_model=config.d_model,
            d_ffn=config.d_ffn,
            n_experts=config.moe.n_experts,
            n_experts_per_token=config.moe.n_experts_per_token,
        )
    else:
        return DenseFFN(
            d_model=config.d_model,
            d_ffn=config.d_ffn,
            activation=config.ffn.activation,
        )
```

---

## 6. Main Model

**File:** `src/auralis/model/helix_model.py`

```python
"""
Helix v2 Hauptmodell.
Heterogener Hybrid-Stack: Mamba + GLA + Sparse Attention.
"""

import torch
import torch.nn as nn
from auralis.model.config import AuralisConfig, LayerConfig
from auralis.model.layers.mamba_layer import Mamba2Layer
from auralis.model.layers.gla_layer import GLALayer
from auralis.model.layers.sparse_attn_layer import SparseAttentionLayer
from auralis.model.layers.ffn import build_ffn
from auralis.model.layers.norm import RMSNorm
from auralis.model.utils.rotary import RotaryEmbedding


class HelixBlock(nn.Module):
    """Ein Transformer-artiger Block.
    
    Pre-Norm Architektur:
        x = x + attention(norm1(x))
        x = x + ffn(norm2(x))
    """
    
    def __init__(self, config: AuralisConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_config = config.layers[layer_idx]
        
        # Normalization
        self.norm1 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.norm2 = RMSNorm(config.d_model, eps=config.norm_eps)
        
        # Attention/SSM Layer
        self.attn = self._build_attention_layer(config)
        
        # FFN
        self.ffn = build_ffn(config)
    
    def _build_attention_layer(self, config: AuralisConfig):
        """Wählt Layer-Typ basierend auf Config."""
        layer_cfg = self.layer_config
        
        if layer_cfg.type == "mamba":
            return Mamba2Layer(
                d_model=config.d_model,
                d_state=layer_cfg.d_state or 128,
                d_conv=layer_cfg.d_conv or 4,
                expand_factor=layer_cfg.expand_factor or 2,
                dt_min=layer_cfg.dt_min or 0.001,
                dt_max=layer_cfg.dt_max or 0.1,
            )
        
        elif layer_cfg.type == "gla":
            return GLALayer(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_head=config.d_head,
                d_state=layer_cfg.d_state or 128,
            )
        
        elif layer_cfg.type == "sparse_attention":
            return SparseAttentionLayer(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_head=config.d_head,
                window_size=layer_cfg.window_size or 1024,
                global_tokens=layer_cfg.global_tokens or 32,
                use_rope=layer_cfg.use_rope or True,
            )
        
        else:
            raise ValueError(f"Unknown layer type: {layer_cfg.type}")
    
    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None = None,
        rope_cache=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Attention with residual
        attn_out, state = self.attn(self.norm1(x), state=state)
        x = x + attn_out
        
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        
        return x, state


class HelixModel(nn.Module):
    """Helix v2 — Main Model."""
    
    def __init__(self, config: AuralisConfig):
        super().__init__()
        self.config = config
        
        # Token Embedding
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )
        
        # Position Encoding (RoPE)
        if config.position_encoding.type == "rope":
            self.rope = RotaryEmbedding(
                dim=config.d_head,
                max_seq_len=config.position_encoding.max_seq_length,
                theta=config.position_encoding.theta,
            )
        
        # Transformer Blocks
        self.blocks = nn.ModuleList([
            HelixBlock(config, layer_idx=i)
            for i in range(config.n_layers)
        ])
        
        # Final Norm
        self.norm_out = RMSNorm(config.d_model, eps=config.norm_eps)
        
        # LM Head
        if config.advanced.tie_embeddings:
            self.lm_head = None  # Use embedding weights
        else:
            self.lm_head = nn.Linear(
                config.d_model,
                config.vocab_size,
                bias=False,
            )
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Scaled Normal Initialization."""
        std = self.config.init.init_std
        
        # Embedding
        nn.init.normal_(self.embedding.weight, mean=0.0, std=std)
        
        # Layers
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=std)
        
        # Output layer - scaled
        if self.lm_head is not None:
            nn.init.normal_(
                self.lm_head.weight,
                mean=0.0,
                std=std * self.config.init.output_init_scale,
            )
    
    def forward(
        self,
        input_ids: torch.Tensor,  # [B, L]
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> dict:
        """Forward pass.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            labels: Optional labels für Loss [batch, seq_len]
            attention_mask: Optional mask
        
        Returns:
            dict mit 'loss', 'logits', ...
        """
        # Embedding
        x = self.embedding(input_ids)
        
        # RoPE cache
        if hasattr(self, 'rope'):
            rope_cache = self.rope(x.size(1), device=x.device)
        else:
            rope_cache = None
        
        # Blocks
        for block in self.blocks:
            x, _ = block(x, rope_cache=rope_cache)
        
        # Final norm
        x = self.norm_out(x)
        
        # LM Head
        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            # Tied embeddings
            logits = torch.matmul(x, self.embedding.weight.T)
        
        # Loss
        loss = None
        if labels is not None:
            loss = self._compute_loss(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
        }
    
    def _compute_loss(self, logits, labels):
        """Cross-Entropy Loss auf Assistant-Tokens."""
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        
        return loss
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(config_path: str | Path) -> HelixModel:
    """Factory function — lädt Config und baut Modell."""
    config = AuralisConfig.from_yaml(config_path)
    model = HelixModel(config)
    
    # Info
    n_params = model.count_parameters()
    print(f"Modell geladen: {config.name}")
    print(f"  Parameter: {n_params / 1e9:.2f}B ({n_params / 1e6:.1f}M)")
    
    return model
```

---

## 7. Tests

**File:** `tests/model/test_helix_model.py`

```python
"""
Tests für Helix v2 Modell.
Test mit 100M-Variante (klein, schnell).
"""

import pytest
import torch
from auralis.model import build_model, AuralisConfig


@pytest.fixture(scope="module")
def small_model():
    """100M Helix für Tests."""
    return build_model("configs/model/helix_v2_100m.yaml")


class TestModelBuild:
    def test_model_builds(self, small_model):
        """Modell baut ohne Fehler."""
        assert small_model is not None
    
    def test_parameter_count(self, small_model):
        """~100M Parameter."""
        n_params = small_model.count_parameters()
        assert 80_000_000 < n_params < 150_000_000
    
    def test_config_loaded(self, small_model):
        assert small_model.config.n_layers > 0
        assert small_model.config.d_model > 0


class TestForwardPass:
    def test_forward_runs(self, small_model):
        """Forward pass läuft ohne Fehler."""
        small_model.eval()
        input_ids = torch.randint(0, 200000, (2, 128))
        
        with torch.no_grad():
            output = small_model(input_ids)
        
        assert 'logits' in output
        assert output['logits'].shape == (2, 128, 200000)
    
    def test_forward_with_labels(self, small_model):
        """Loss wird berechnet."""
        small_model.eval()
        input_ids = torch.randint(0, 200000, (2, 128))
        labels = torch.randint(0, 200000, (2, 128))
        
        with torch.no_grad():
            output = small_model(input_ids, labels=labels)
        
        assert output['loss'] is not None
        assert output['loss'].item() > 0
    
    def test_no_nan_in_output(self, small_model):
        """Keine NaN im Output."""
        small_model.eval()
        input_ids = torch.randint(0, 200000, (2, 128))
        
        with torch.no_grad():
            output = small_model(input_ids)
        
        assert not torch.isnan(output['logits']).any()


class TestBackwardPass:
    def test_backward_runs(self, small_model):
        """Backward pass läuft, Gradienten werden berechnet."""
        small_model.train()
        input_ids = torch.randint(0, 200000, (2, 64))
        labels = torch.randint(0, 200000, (2, 64))
        
        output = small_model(input_ids, labels=labels)
        output['loss'].backward()
        
        # Check gradients exist
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in small_model.parameters()
        )
        assert has_grad


class TestLayerTypes:
    def test_mamba_in_early_layers(self, small_model):
        """Frühe Layers sind Mamba."""
        first_layer_type = small_model.blocks[0].layer_config.type
        assert first_layer_type == "mamba"
    
    def test_sparse_attn_in_late_layers(self, small_model):
        """Späte Layers sind Sparse Attention."""
        last_layer_type = small_model.blocks[-1].layer_config.type
        assert last_layer_type == "sparse_attention"
```

---

## 8. Acceptance Criteria

```
□ AuralisConfig lädt YAML korrekt
□ HelixModel baut aus Config ohne Fehler
□ 100M Test-Modell läuft Forward + Backward
□ Keine NaN/Inf in Outputs
□ Parameter-Count stimmt mit Erwartung überein (±10%)
□ Alle Layer-Typen (Mamba, GLA, Sparse) testbar
□ Mamba: nutzt mamba_ssm Library in Production
□ GLA: Custom Kernel von v1 portiert
□ Sparse Attention: nutzt flash_attn in Production
□ MoE-Hooks vorhanden (default: aus)
□ MTP-Hooks vorhanden (default: aus)
□ Alle Tests grün
□ Config-Files existieren (100m, 2b, 3b)
```

---

## 9. Duration & Effort

```
Tag 1: Config + Helper Utilities
Tag 2: Mamba Layer + Tests
Tag 3: GLA Layer (Port + Tests)
Tag 4: Sparse Attention + FFN + Main Model
Tag 5: Integration-Tests, Debugging
Tag 6: Optimierungen, Dokumentation
Tag 7: Code Review, Commit, Tag
```

---

## 10. Next Steps

After Phase 0.5:
→ SPEC_PHASE_1_PRETRAINING.md (English-heavy pretraining)

---

*Phase 0.5 Spec Version 1.0 — April 2026*
