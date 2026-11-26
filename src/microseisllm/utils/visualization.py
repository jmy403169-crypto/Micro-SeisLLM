"""Lightweight plotting helpers (optional; not used by core tests)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np


def plot_waveform_compare(
    clean: np.ndarray,
    noisy: np.ndarray,
    denoised: Optional[np.ndarray] = None,
    sampling_rate: float = 500.0,
    out_path: Union[str, Path, None] = None,
    channel: int = 2,
):
    """Plot one channel of clean / noisy / denoised traces."""
    import matplotlib.pyplot as plt

    t = np.arange(clean.shape[-1]) / sampling_rate
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, noisy[channel], label="noisy", alpha=0.7, linewidth=0.8)
    ax.plot(t, clean[channel], label="clean", alpha=0.9, linewidth=0.8)
    if denoised is not None:
        ax.plot(t, denoised[channel], label="denoised", alpha=0.9, linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend(frameon=False)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return fig
