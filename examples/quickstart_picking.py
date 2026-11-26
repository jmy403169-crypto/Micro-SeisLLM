#!/usr/bin/env python
"""Smoke picking on bundled synthetic samples; writes results/sample_outputs/picks.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from microseisllm.datasets import PickingDataset
from microseisllm.models.seismollm import SeisMoLLM_dpk


def main():
    out_dir = ROOT / "results" / "sample_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = PickingDataset(
        ROOT / "data/sample/phase_picking",
        ROOT / "data/splits/phase_picking_split.json",
        split="test",
    )
    batch = ds[0]
    model = SeisMoLLM_dpk(llm="GPT2", d_model=768, pretrain=False, llm_layers=2)
    model.eval()
    with torch.no_grad():
        out = model(batch["waveform"].unsqueeze(0))[0]
    p_pred = int(out[1].argmax())
    s_pred = int(out[2].argmax()) if out.shape[0] > 2 else None
    payload = {
        "event_id": str(batch["event_id"]),
        "sampling_rate_hz": 500,
        "p_true_sample": float(batch["p_arrival_sample"]),
        "s_true_sample": float(batch.get("s_arrival_sample", -1)),
        "p_pred_sample": p_pred,
        "s_pred_sample": s_pred,
        "note": "Smoke run without trained weights.",
    }
    print("event", payload["event_id"])
    print("true P (samples):", payload["p_true_sample"])
    print("pred P (samples):", p_pred)
    (out_dir / "picks.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Phase arrivals saved to:", out_dir / "picks.json")


if __name__ == "__main__":
    main()
