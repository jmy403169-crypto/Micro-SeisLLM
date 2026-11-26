#!/usr/bin/env python
"""Smoke denoising on bundled synthetic samples; writes results/sample_outputs/denoised.npz."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from microseisllm.datasets import DenoisingDataset
from microseisllm.metrics import snr_improvement
from microseisllm.models.seismollm import SeisMoLLM_denoise


def main():
    out_dir = ROOT / "results" / "sample_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = DenoisingDataset(
        ROOT / "data/sample/denoising",
        ROOT / "data/splits/denoising_split.json",
        split="test",
    )
    batch = ds[0]
    model = SeisMoLLM_denoise(llm="GPT2", d_model=768, pretrain=False, llm_layers=2)
    model.eval()
    with torch.no_grad():
        pred = model(batch["noisy"].unsqueeze(0))[0]
    snr = float(snr_improvement(batch["clean"], batch["noisy"], pred))
    print("event", batch["event_id"])
    print("snr_improvement_db", snr)

    np.savez_compressed(
        out_dir / "denoised.npz",
        noisy=batch["noisy"].cpu().numpy(),
        clean=batch["clean"].cpu().numpy(),
        denoise=pred.cpu().numpy(),
        event_id=str(batch["event_id"]),
        sampling_rate_hz=500,
    )
    (out_dir / "denoising_metrics.json").write_text(
        json.dumps(
            {
                "task": "denoising",
                "event_id": str(batch["event_id"]),
                "snr_improvement_db": snr,
                "output": "denoised.npz",
                "note": "Smoke run without trained weights.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Denoised waveform saved to:", out_dir / "denoised.npz")


if __name__ == "__main__":
    main()
