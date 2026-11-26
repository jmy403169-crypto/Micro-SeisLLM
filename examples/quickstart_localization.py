#!/usr/bin/env python
"""Smoke localization on bundled synthetic samples; writes results/sample_outputs/location.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from microseisllm.datasets import LocalizationDataset
from microseisllm.metrics import localization_mae
from microseisllm.models.seismollm import SeisMoLLM_loc_event


def main():
    out_dir = ROOT / "results" / "sample_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = LocalizationDataset(
        ROOT / "data/sample/source_localization",
        ROOT / "data/splits/source_localization_split.json",
        split="test",
    )
    batch = ds[0]
    model = SeisMoLLM_loc_event(llm="GPT2", d_model=768, pretrain=False, llm_layers=2)
    model.eval()
    with torch.no_grad():
        pred = model(batch["waveform"].unsqueeze(0), batch["station_mask"].unsqueeze(0))[0]
    true = [float(x) for x in batch["target"].tolist()]
    pred_list = [float(x) for x in pred.tolist()]
    mae = localization_mae(pred, batch["target"])
    if hasattr(mae, "items"):
        mae = {k: float(v) for k, v in mae.items()}
    else:
        mae = float(mae)
    payload = {
        "event_id": str(batch["event_id"]),
        "true_source": true,
        "pred_source": pred_list,
        "mae": mae,
        "note": "Smoke run without trained weights. Coordinates follow sample catalogue units.",
    }
    print("event", payload["event_id"])
    print("true", true)
    print("pred", pred_list)
    print(mae)
    (out_dir / "location.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Source location saved to:", out_dir / "location.json")


if __name__ == "__main__":
    main()
