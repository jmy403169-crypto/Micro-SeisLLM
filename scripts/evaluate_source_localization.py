#!/usr/bin/env python
"""Evaluate source localization checkpoint."""

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
from microseisllm.datasets import LocalizationDataset  # noqa: E402
from microseisllm.metrics import localization_mae, localization_rmse  # noqa: E402
from microseisllm.models.seismollm import SeisMoLLM_loc_event  # noqa: E402
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
    ds = LocalizationDataset(
        data_dir=data_cfg.get("data_dir", "data/sample/source_localization"),
        split_file=data_cfg.get(
            "split_file", "data/splits/source_localization_split.json"
        ),
        split=data_cfg.get("test_split", "test"),
    )
    loader = DataLoader(ds, batch_size=1)
    mk = model_kwargs_from_cfg(cfg)
    mk["in_channels"] = int(cfg.get("in_channels", 3))
    model = SeisMoLLM_loc_event(**mk).to(device)
    if args.checkpoint:
        ckpt = torch.load(resolve_path(args.checkpoint), map_location=device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
    model.eval()

    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["waveform"].to(device)
            mask = batch["station_mask"].to(device)
            y = batch["target"].cpu().numpy()[0]
            pred = model(x, station_mask=mask).cpu().numpy()[0]
            preds.append(pred)
            targets.append(y)

    preds = np.asarray(preds)
    targets = np.asarray(targets)
    metrics = {
        "task": "source_localization",
        "backbone": cfg.get("backbone", cfg.get("llm")),
        "n": int(len(preds)),
        **localization_mae(preds, targets),
        **localization_rmse(preds, targets),
        "note": "Smoke metrics on data/sample only. Paper numbers are in results/expected_metrics.json.",
    }
    out_dir = ensure_dir(resolve_path(cfg.get("results_dir", "results/sample_outputs")))
    out_path = out_dir / "source_localization_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
