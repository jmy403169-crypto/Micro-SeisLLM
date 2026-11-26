"""YAML config loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import yaml

from .io import resolve_path


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    cfg_path = resolve_path(path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {cfg_path}")
    cfg["_config_path"] = str(cfg_path)
    return cfg
