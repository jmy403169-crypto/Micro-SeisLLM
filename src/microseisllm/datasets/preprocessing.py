"""Shared waveform preprocessing helpers."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


SAMPLING_RATE_HZ = 500.0


def normalize_event_layout(arr: np.ndarray) -> np.ndarray:
    """
    Normalize event arrays to ``(N_sta, T, 3)`` float32.

    Accepts ``(N,T,3)``, ``(T,3,N)``, or ``(N,3,T)``.
    """
    a = np.asarray(arr)
    if a.ndim != 3:
        raise ValueError(f"Expected 3-D array, got shape {a.shape}")
    if a.shape[-1] == 3:
        return np.ascontiguousarray(a.astype(np.float32))
    if a.shape[1] == 3:
        # (N, 3, T) -> (N, T, 3)
        return np.ascontiguousarray(np.transpose(a, (0, 2, 1)).astype(np.float32))
    if a.shape[0] == 3:
        # (3, T, N) unlikely; try (T, 3, N)
        return np.ascontiguousarray(np.transpose(a, (2, 0, 1)).astype(np.float32))
    raise ValueError(f"Cannot infer layout for shape {a.shape}")


def to_bct(arr_nt3: np.ndarray) -> np.ndarray:
    """``(N,T,3)`` -> ``(N,3,T)`` for model input."""
    return np.ascontiguousarray(np.transpose(arr_nt3, (0, 2, 1)).astype(np.float32))


def add_gaussian_noise(
    clean: np.ndarray,
    snr_db: float,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Add Gaussian noise so that SNR ≈ ``snr_db`` (signal power / noise power).

    Returns ``(noisy, noise)`` with the same shape as ``clean``.
    """
    rng = rng or np.random.default_rng(0)
    clean = clean.astype(np.float32)
    power = np.mean(clean.astype(np.float64) ** 2) + 1e-12
    noise_power = power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=clean.shape).astype(np.float32)
    return clean + noise, noise


def gaussian_pick_label(
    length: int,
    arrival_sample: float,
    sigma: float = 10.0,
) -> np.ndarray:
    """Create a soft Gaussian label peak at the arrival sample."""
    t = np.arange(length, dtype=np.float32)
    y = np.exp(-0.5 * ((t - float(arrival_sample)) / sigma) ** 2)
    return y.astype(np.float32)


def zscore_per_trace(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-channel z-score for ``(C,T)`` or ``(N,C,T)``."""
    x = x.astype(np.float32)
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)
