#!/usr/bin/env python
"""Evaluate phase-picking checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import model_kwargs_from_cfg, prepare_cfg, str2bool  # noqa: E402
from microseisllm.datasets import PickingDataset  # noqa: E402
from microseisllm.metrics import pick_from_prob, picking_mae_ms, picking_mae_samples  # noqa: E402
from microseisllm.models.seismollm import SeisMoLLM_dpk  # noqa: E402
from microseisllm.utils.io import ensure_dir, resolve_path  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default="", help="Optional .pt; empty = smoke eval")
    p.add_argument("--pretrain", type=str2bool, default=False)
    p.add_argument("--llm-path", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = prepare_cfg(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = cfg.get("data", {})
    ds = PickingDataset(
        data_dir=data_cfg.get("data_dir", "data/sample/phase_picking"),
        split_file=data_cfg.get("split_file", "data/splits/phase_picking_split.json"),
        split=data_cfg.get("test_split", "test"),
    )
    loader = DataLoader(ds, batch_size=1)
    model = SeisMoLLM_dpk(**model_kwargs_from_cfg(cfg)).to(device)
    if args.checkpoint:
        ckpt = torch.load(resolve_path(args.checkpoint), map_location=device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
    model.eval()

    pred_idx, true_idx = [], []
    fs = float(cfg.get("sampling_rate_hz", 500.0))
    with torch.no_grad():
        for batch in loader:
            x = batch["waveform"].to(device)
            pred = model(x)[0].cpu().numpy()  # (C,T)
            p_prob = pred[1] if pred.shape[0] > 1 else pred[0]
            pred_idx.append(pick_from_prob(p_prob, threshold=0.1))
            true_idx.append(float(batch["p_arrival_sample"].item()))
            fs = float(batch["sampling_rate_hz"]) if "sampling_rate_hz" in batch else fs

    metrics = {
        "task": "phase_picking",
        "backbone": cfg.get("backbone", cfg.get("llm")),
        "n": len(pred_idx),
        "p_mae_samples": picking_mae_samples(pred_idx, true_idx),
        "p_mae_ms": picking_mae_ms(pred_idx, true_idx, sampling_rate_hz=fs),
        "note": "Smoke metrics on data/sample only. Paper numbers are in results/expected_metrics.json.",
    }
    out_dir = ensure_dir(resolve_path(cfg.get("results_dir", "results/sample_outputs")))
    out_path = out_dir / "phase_picking_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
