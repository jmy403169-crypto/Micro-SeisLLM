"""MSE loss for waveform reconstruction / denoising."""

from __future__ import annotations

import torch
import torch.nn as nn


class DenoisingMSELoss(nn.Module):
    def __init__(self, weight=None) -> None:
        super().__init__()
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float32)
        else:
            weight = torch.tensor(1.0, dtype=torch.float32)
        self.register_buffer("weight", weight)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = (preds - targets) ** 2
        loss = loss * self.weight
        return loss.mean()
