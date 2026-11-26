"""SNR metrics for noise suppression."""

from __future__ import annotations

import numpy as np
import torch


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def snr_db(signal, noise, eps: float = 1e-12) -> float:
    signal = _to_numpy(signal).astype(np.float64)
    noise = _to_numpy(noise).astype(np.float64)
    ps = np.mean(signal ** 2)
    pn = np.mean(noise ** 2)
    return float(10.0 * np.log10((ps + eps) / (pn + eps)))


def snr_improvement(clean, noisy, denoised, eps: float = 1e-12) -> float:
    """SNR(denoised) - SNR(noisy), both relative to clean residual noise."""
    clean = _to_numpy(clean)
    noisy = _to_numpy(noisy)
    denoised = _to_numpy(denoised)
    snr_in = snr_db(clean, noisy - clean, eps=eps)
    snr_out = snr_db(clean, denoised - clean, eps=eps)
    return float(snr_out - snr_in)
