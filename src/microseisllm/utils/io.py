"""Path helpers (relative to repository root; no absolute Windows paths)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union


def repo_root() -> Path:
    """Return Micro-SeisLLM repository root (two levels above this file's package)."""
    return Path(__file__).resolve().parents[3]


def resolve_path(path: Union[str, Path], base: Union[str, Path, None] = None) -> Path:
    """Resolve ``path`` relative to ``base`` (default: repo root)."""
    p = Path(path)
    if p.is_absolute():
        return p
    root = Path(base) if base is not None else repo_root()
    return (root / p).resolve()


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
