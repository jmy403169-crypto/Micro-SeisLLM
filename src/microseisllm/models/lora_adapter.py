"""LoRA adapter configuration helpers for GPT-2 and Qwen backbones."""

from __future__ import annotations

from peft import LoraConfig


def lora_setting(
    target_modules,
    r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    bias: str = "lora_only",
) -> LoraConfig:
    return LoraConfig(
        target_modules=target_modules,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
    )


# GPT-2: adapt all linear projections.
GPT2_LORA = lora_setting("all-linear")

# Qwen2 / Qwen3 attention + MLP linear layers.
QWEN_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
QWEN_LORA = lora_setting(QWEN_LORA_TARGETS)

# Backward-compatible aliases from source repos.
GPT2_lora = GPT2_LORA
Qwen2_lora = QWEN_LORA


def build_lora_config(
    llm_name: str,
    r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
) -> LoraConfig:
    name = (llm_name or "GPT2").strip().upper()
    if name.startswith("QWEN"):
        return lora_setting(
            QWEN_LORA_TARGETS, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
        )
    return lora_setting(
        "all-linear", r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
    )
