"""Phase-picking MAE in samples and milliseconds."""

from __future__ import annotations

import numpy as np
import torch


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def pick_from_prob(prob: np.ndarray, threshold: float = 0.5) -> float:
    """Return argmax sample index if max >= threshold else NaN."""
    prob = np.asarray(prob).reshape(-1)
    idx = int(np.argmax(prob))
    if prob[idx] < threshold:
        return float("nan")
    return float(idx)


def picking_mae_samples(pred_idx, true_idx) -> float:
    pred = _to_numpy(pred_idx).astype(np.float64).reshape(-1)
    true = _to_numpy(true_idx).astype(np.float64).reshape(-1)
    mask = np.isfinite(pred) & np.isfinite(true)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(pred[mask] - true[mask])))


def picking_mae_ms(pred_idx, true_idx, sampling_rate_hz: float = 500.0) -> float:
    mae_s = picking_mae_samples(pred_idx, true_idx)
    if not np.isfinite(mae_s):
        return float("nan")
    return float(mae_s * 1000.0 / float(sampling_rate_hz))
