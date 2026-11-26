"""Latent patching: unfold feature maps into LLM token embeddings and reverse."""

from __future__ import annotations

import torch
from einops import rearrange


def latent_patch(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Unfold ``(B, C, L)`` feature maps into ``(B, N, C*P)`` token embeddings.

    This is the operation applied immediately before the LLM backbone.
    """
    x = x.unfold(dimension=-1, size=patch_size, step=patch_size)
    return rearrange(x, "b c n p -> b n (c p)")


def latent_unpatch(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Fold ``(B, N, C*P)`` token embeddings back to ``(B, C, L)``."""
    return rearrange(x, "b n (c p) -> b c (n p)", p=patch_size)
