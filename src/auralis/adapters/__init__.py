"""Parameter-efficient adapters (LoRA / DoRA) for modular skills on a frozen base."""
from .lora import (
    LoRALinear,
    DoRALinear,
    inject_adapters,
    freeze_base,
    adapter_state_dict,
    load_adapter_state_dict,
    enable_input_require_grads,
    set_adapter_scale,
    DEFAULT_TARGETS,
    ADAPTER_KEYS,
)

__all__ = [
    "LoRALinear", "DoRALinear", "inject_adapters", "freeze_base",
    "adapter_state_dict", "load_adapter_state_dict", "enable_input_require_grads",
    "set_adapter_scale", "DEFAULT_TARGETS", "ADAPTER_KEYS",
]
