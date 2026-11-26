"""Huber loss for source localization (loc4)."""

from __future__ import annotations

import torch
import torch.nn as nn


class LocalizationHuberLoss(nn.Module):
    def __init__(self, delta: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.loss = nn.HuberLoss(delta=delta, reduction=reduction)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss(preds, targets)
