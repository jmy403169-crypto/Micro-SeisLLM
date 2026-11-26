#!/usr/bin/env python
"""Evaluate denoising checkpoint on sample / configured split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import model_kwargs_from_cfg, prepare_cfg, str2bool  # noqa: E402
from microseisllm.datasets import DenoisingDataset  # noqa: E402
from microseisllm.metrics import snr_improvement  # noqa: E402
from microseisllm.models.seismollm import (  # noqa: E402
    SeisMoLLM_denoise,
    SeisMoLLM_denoise_event,
)
from microseisllm.utils.io import ensure_dir, resolve_path  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument(
        "--checkpoint",
        default="",
        help="Optional .pt checkpoint; empty = random-init smoke eval",
    )
    p.add_argument("--pretrain", type=str2bool, default=False)
    p.add_argument("--llm-path", default=None)
    return p.parse_args()


def _use_event_model(cfg) -> bool:
    name = str(cfg.get("model_name", "SeisMoLLM_denoise_event")).lower()
    return "event" in name


def main():
    args = parse_args()
    cfg = prepare_cfg(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = cfg.get("data", {})
    event_mode = _use_event_model(cfg)
    flatten = bool(data_cfg.get("flatten_stations", not event_mode))
    ds = DenoisingDataset(
        data_dir=data_cfg.get("data_dir", "data/sample/denoising"),
        split_file=data_cfg.get("split_file", "data/splits/denoising_split.json"),
        split=data_cfg.get("test_split", "test"),
        flatten_stations=flatten,
    )
    loader = DataLoader(ds, batch_size=1)

    mk = model_kwargs_from_cfg(cfg)
    model = (
        SeisMoLLM_denoise_event(**mk) if event_mode else SeisMoLLM_denoise(**mk)
    ).to(device)
    if args.checkpoint:
        ckpt = torch.load(resolve_path(args.checkpoint), map_location=device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        if isinstance(ckpt, dict) and "model_dict" in ckpt and "model" not in ckpt:
            state = ckpt["model_dict"]
        model.load_state_dict(state, strict=False)
    model.eval()

    improvements = []
    with torch.no_grad():
        for batch in loader:
            noisy = batch["noisy"].to(device)
            clean = batch["clean"].to(device)
            pred = model(noisy)
            # Event-level: average SNR improvement over stations
            if pred.dim() == 4:
                vals = []
                for s in range(pred.size(1)):
                    vals.append(
                        snr_improvement(clean[0, s], noisy[0, s], pred[0, s])
                    )
                improvements.append(sum(vals) / max(len(vals), 1))
            else:
                improvements.append(snr_improvement(clean[0], noisy[0], pred[0]))

    metrics = {
        "task": "denoising",
        "model_name": cfg.get("model_name", "SeisMoLLM_denoise_event"),
        "backbone": cfg.get("backbone", cfg.get("llm")),
        "n": len(improvements),
        "snr_improvement_db_mean": float(
            sum(improvements) / max(len(improvements), 1)
        ),
        "note": "Smoke metrics on data/sample only. Paper numbers are in results/expected_metrics.json.",
    }
    out_dir = ensure_dir(resolve_path(cfg.get("results_dir", "results/sample_outputs")))
    out_path = out_dir / "denoising_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
