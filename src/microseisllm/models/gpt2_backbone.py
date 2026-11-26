"""GPT-2 backbone loader (first N layers) with optional LoRA."""

from __future__ import annotations

import os
import warnings
from typing import Optional

import torch
import torch.nn as nn
import transformers.models.gpt2 as GPT2
from peft import get_peft_model


def _wrap_from_pretrained_torch_load_error(fn, path_desc: str = "weights"):
    try:
        return fn()
    except ValueError as e:
        msg = str(e)
        if "torch.load" in msg or "2.6" in msg or "safetensors" in msg.lower():
            raise RuntimeError(
                "Loading pretrained weights failed due to a transformers/torch "
                f"security restriction while reading {path_desc}. "
                "Upgrade PyTorch (>=2.6) or use safetensors checkpoints."
            ) from e
        raise


def build_gpt2_backbone(
    start_layer: int = 0,
    end_layer: int = 3,
    pretrain: bool = True,
    freeze: bool = True,
    llm_path: Optional[str] = None,
    lora_config=None,
    d_model: int = 768,
    n_layer_cfg: Optional[int] = None,
) -> nn.Module:
    """
    Build a GPT-2 encoder truncated to ``[start_layer, end_layer)``.

    When ``pretrain=False``, a randomly initialized GPT-2 with ``n_embd=d_model``
    is created (no network download).
    """
    n_layers = end_layer - start_layer
    env_path = os.environ.get("MICROSEISLLM_GPT2_PATH", "")
    gpt_path = (llm_path or env_path or "").replace("\\", "/")

    if pretrain and gpt_path and os.path.isdir(gpt_path):

        def _load_local():
            return GPT2.GPT2Model.from_pretrained(
                gpt_path,
                output_hidden_states=True,
                vocab_size=0,
                ignore_mismatched_sizes=True,
            )

        llm = _wrap_from_pretrained_torch_load_error(
            _load_local, path_desc=f"GPT2 dir {gpt_path!r}"
        )
    elif pretrain:

        def _load_hub():
            return GPT2.GPT2Model.from_pretrained(
                "gpt2",
                output_hidden_states=True,
                vocab_size=0,
                ignore_mismatched_sizes=True,
            )

        llm = _wrap_from_pretrained_torch_load_error(_load_hub, path_desc="gpt2")
    else:
        cfg = GPT2.configuration_gpt2.GPT2Config(
            vocab_size=0,
            n_embd=d_model,
            n_layer=n_layer_cfg or max(n_layers, end_layer),
            n_head=max(1, d_model // 64),
        )
        llm = GPT2.GPT2Model(cfg)

    llm.h = llm.h[start_layer:end_layer]

    if freeze and pretrain and lora_config is not None:
        llm = get_peft_model(llm, lora_config)
        for name, param in llm.named_parameters():
            if "ln" in name or "wpe" in name or "lora" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        for name, param in llm.named_parameters():
            if "wte" in name:
                param.requires_grad = False

    return llm


def gpt2_hidden_state(out) -> torch.Tensor:
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)) and len(out) > 0 and out[0] is not None:
        return out[0]
    h = getattr(out, "last_hidden_state", None)
    if h is not None:
        return h
    hs = getattr(out, "hidden_states", None)
    if hs is not None and len(hs) > 0:
        return hs[-1]
    raise RuntimeError(f"Cannot parse GPT-2 hidden state from {type(out)}")
