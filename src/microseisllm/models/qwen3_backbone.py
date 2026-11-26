"""Qwen3 / Qwen2 backbone loader (first N layers) with optional LoRA."""

from __future__ import annotations

import json
import os
import warnings
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
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


def qwen_pretrained_dir_ok(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    has_config = os.path.isfile(os.path.join(path, "config.json"))
    try:
        files = os.listdir(path)
        has_safetensors = any(
            f.endswith(".safetensors")
            and (
                f.startswith("model-")
                or "model.safetensors-" in f
                or f == "model.safetensors"
            )
            for f in files
        )
    except OSError:
        return False
    return has_config and has_safetensors


def qwen_model_type(path: str) -> Optional[str]:
    cfg_path = os.path.join(path, "config.json")
    if not os.path.isfile(cfg_path):
        return None
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("model_type") or cfg.get("text_config", {}).get("model_type")
    except Exception:
        return None


_QWEN35_LANG_CKPT_PREFIX = "model.language_model."
_QWEN35_LANG_MODULE_PREFIX = "model."


def _load_qwen35_language_state_dict(path: str):
    try:
        from safetensors import safe_open
    except ImportError as e:
        raise ImportError("Loading Qwen3.5 text submodule requires safetensors") from e

    index_path = os.path.join(path, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"Missing {index_path}")

    with open(index_path, "r", encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]

    lang_items = {
        k: shard
        for k, shard in weight_map.items()
        if k.startswith(_QWEN35_LANG_CKPT_PREFIX)
    }
    if not lang_items:
        raise RuntimeError(
            f"No keys with prefix {_QWEN35_LANG_CKPT_PREFIX!r} in {path}"
        )

    shard_to_keys = {}
    for key, shard in lang_items.items():
        shard_to_keys.setdefault(shard, []).append(key)

    state = OrderedDict()
    for shard, keys in shard_to_keys.items():
        shard_path = os.path.join(path, shard)
        with safe_open(shard_path, framework="pt", device="cpu") as st:
            for key in keys:
                new_key = _QWEN35_LANG_MODULE_PREFIX + key[len(_QWEN35_LANG_CKPT_PREFIX) :]
                state[new_key] = st.get_tensor(key)
    return state


def _load_qwen35_language_model(path: str, torch_dtype=torch.float32):
    try:
        from transformers import AutoConfig, Qwen3_5TextConfig, Qwen3_5TextModel
    except ImportError as e:
        raise RuntimeError(
            "Current transformers does not expose Qwen3_5TextModel. "
            "Upgrade transformers to a version that supports Qwen3.5."
        ) from e

    full_cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    text_cfg = full_cfg.text_config
    if isinstance(text_cfg, dict):
        text_cfg = Qwen3_5TextConfig(**text_cfg)
    elif not isinstance(text_cfg, Qwen3_5TextConfig):
        text_cfg = Qwen3_5TextConfig(**text_cfg.to_dict())

    def _build():
        model = Qwen3_5TextModel(text_cfg)
        state = _load_qwen35_language_state_dict(path)
        if torch_dtype is not None:
            state = {k: v.to(dtype=torch_dtype) for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        return model

    return _wrap_from_pretrained_torch_load_error(
        _build, path_desc=f"Qwen3.5 language_model @ {path!r}"
    )


def _slice_qwen_layers(llm: nn.Module, start_layer: int, end_layer: int) -> None:
    layers_attr = getattr(llm, "model", llm)
    layers = getattr(layers_attr, "layers", None)
    if layers is not None:
        layers_attr.layers = nn.ModuleList(layers[start_layer:end_layer])


def build_qwen_backbone(
    start_layer: int = 0,
    end_layer: int = 3,
    pretrain: bool = True,
    freeze: bool = True,
    llm_path: Optional[str] = None,
    lora_config=None,
    d_model: int = 2560,
) -> nn.Module:
    """
    Build a Qwen2/Qwen3 encoder truncated to ``[start_layer, end_layer)``.

    Default ``d_model=2560`` matches Qwen3-4B. When ``pretrain=False``, a
    randomly initialized Qwen2Model is created (no download).
    """
    n_layers = end_layer - start_layer
    env_path = os.environ.get("MICROSEISLLM_QWEN3_PATH", "")
    path = (llm_path or env_path or "").replace("\\", "/")

    if pretrain:
        if not qwen_pretrained_dir_ok(path):
            raise FileNotFoundError(
                f"Valid Qwen pretrained directory not found: {path!r}. "
                "Download Qwen3-4B (or Qwen2.5-3B) locally and set "
                "MICROSEISLLM_QWEN3_PATH / --llm-path. "
                "Required: config.json and model-*.safetensors."
            )
        model_type = qwen_model_type(path)
        if model_type == "qwen3_5":
            llm = _load_qwen35_language_model(path, torch_dtype=torch.float32)
            _slice_qwen_layers(llm, start_layer, end_layer)
        elif model_type == "qwen3":
            from transformers import AutoModel

            def _load():
                full = AutoModel.from_pretrained(
                    path, trust_remote_code=True, torch_dtype=torch.float32
                )
                return getattr(full, "text_model", getattr(full, "model", full))

            llm = _wrap_from_pretrained_torch_load_error(
                _load, path_desc=f"Qwen3 dir {path!r}"
            )
            _slice_qwen_layers(llm, start_layer, end_layer)
        else:
            from transformers import Qwen2Model

            def _load_qwen2():
                return Qwen2Model.from_pretrained(path, torch_dtype=torch.float32)

            llm = _wrap_from_pretrained_torch_load_error(
                _load_qwen2, path_desc=f"Qwen dir {path!r}"
            )
            _slice_qwen_layers(llm, start_layer, end_layer)
    else:
        from transformers import Qwen2Config, Qwen2Model

        n_heads = max(1, d_model // 128)
        while d_model % n_heads != 0 and n_heads > 1:
            n_heads -= 1
        cfg = Qwen2Config(
            hidden_size=d_model,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            intermediate_size=d_model * 4,
            vocab_size=0,
        )
        llm = Qwen2Model(cfg)

    if getattr(llm, "gradient_checkpointing_enable", None) is not None:
        llm.gradient_checkpointing_enable()

    if freeze and pretrain and lora_config is not None:
        llm = get_peft_model(llm, lora_config)
        for name, param in llm.named_parameters():
            if "lora" in name or "layernorm" in name.lower() or "norm" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif not freeze and pretrain:
        for name, param in llm.named_parameters():
            if "embed" in name:
                param.requires_grad = False

    return llm


def qwen_hidden_state(out) -> torch.Tensor:
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
    raise RuntimeError(f"Cannot parse Qwen hidden state from {type(out)}")
