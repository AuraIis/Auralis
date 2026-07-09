"""Parameter-efficient adapters (LoRA / DoRA) for modular skills on a frozen base."""

from .lora import (
    ADAPTER_KEYS,
    DEFAULT_TARGETS,
    DoRALinear,
    LoRALinear,
    adapter_state_dict,
    enable_input_require_grads,
    freeze_base,
    inject_adapters,
    load_adapter_state_dict,
    set_adapter_scale,
)

__all__ = [
    "ADAPTER_KEYS",
    "DEFAULT_TARGETS",
    "DoRALinear",
    "LoRALinear",
    "adapter_state_dict",
    "enable_input_require_grads",
    "freeze_base",
    "inject_adapters",
    "load_adapter_state_dict",
    "set_adapter_scale",
]
