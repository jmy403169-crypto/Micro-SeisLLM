"""Weighted binary cross-entropy for detection / phase picking."""

from __future__ import annotations

import torch
import torch.nn as nn


class BCELoss(nn.Module):
    """Binary cross entropy for phase-picking and detection (shape N,C,L)."""

    _epsilon = 1e-6

    def __init__(self, weight=None) -> None:
        super().__init__()
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float32)
        else:
            weight = torch.tensor(1.0, dtype=torch.float32)
        self.register_buffer("weight", weight)

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = -(
            targets * torch.log(preds + self._epsilon)
            + (1 - targets) * torch.log(1 - preds + self._epsilon)
        )
        loss = loss * self.weight
        return loss.mean()


PickingBCELoss = BCELoss
