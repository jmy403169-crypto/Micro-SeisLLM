"""Shared CLI helpers for Micro-SeisLLM scripts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

from microseisllm.utils.config import load_config
from microseisllm.utils.io import repo_root, resolve_path
from microseisllm.utils.seed import set_seed


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return True
    if s in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


def resolve_env_path(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1], "")
    return value


def prepare_cfg(args) -> Dict[str, Any]:
    cfg = load_config(args.config)
    if getattr(args, "pretrain", None) is not None:
        cfg["pretrain"] = args.pretrain
    if getattr(args, "seed", None) is not None:
        cfg["random_seed"] = args.seed
    if getattr(args, "llm_path", None):
        cfg["llm_path"] = args.llm_path
    cfg["llm_path"] = resolve_env_path(cfg.get("llm_path", ""))
    set_seed(int(cfg.get("random_seed", 0)))
    os.chdir(repo_root())
    return cfg


def model_kwargs_from_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "llm": cfg.get("llm", cfg.get("backbone", "GPT2")),
        "llm_layers": int(cfg.get("num_transformer_layers", 3)),
        "LLM_start_layer": int(cfg.get("LLM_start_layer", 0)),
        "d_model": int(cfg.get("d_model", cfg.get("embedding_dim", 768))),
        "patch_size": int(cfg.get("patch_size", 8)),
        "pretrain": bool(cfg.get("pretrain", False)),
        "freeze": bool(cfg.get("freeze", True)),
        "llm_path": cfg.get("llm_path") or None,
        "lora_rank": int(cfg.get("lora_rank", 16)),
        "lora_alpha": int(cfg.get("lora_alpha", 16)),
    }
