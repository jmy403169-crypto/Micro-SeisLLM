#!/usr/bin/env python
"""Single-sample inference for any of the three Micro-SeisLLM tasks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import model_kwargs_from_cfg, prepare_cfg, str2bool  # noqa: E402
from microseisllm.datasets.preprocessing import normalize_event_layout, to_bct  # noqa: E402
from microseisllm.models.seismollm import (  # noqa: E402
    SeisMoLLM_denoise,
    SeisMoLLM_dpk,
    SeisMoLLM_loc_event,
)
from microseisllm.utils.io import resolve_path  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--input", required=True, help="NPZ sample path")
    p.add_argument("--pretrain", type=str2bool, default=False)
    p.add_argument("--llm-path", default=None)
    p.add_argument("--output", default="")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = prepare_cfg(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mk = model_kwargs_from_cfg(cfg)
    task = str(cfg.get("task", "denoising")).lower()

    if task in ("denoising", "denoise"):
        model = SeisMoLLM_denoise(**mk)
    elif task in ("phase_picking", "picking"):
        model = SeisMoLLM_dpk(**mk)
    else:
        model = SeisMoLLM_loc_event(**mk)
    model = model.to(device)

    if args.checkpoint:
        ckpt = torch.load(resolve_path(args.checkpoint), map_location=device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
    model.eval()

    arr = np.load(resolve_path(args.input), allow_pickle=False)
    with torch.no_grad():
        if task in ("denoising", "denoise"):
            x = to_bct(normalize_event_layout(arr["noisy"]))[0]
            pred = model(torch.from_numpy(x).unsqueeze(0).to(device)).cpu().numpy()[0]
            result = {
                "task": "denoising",
                "denoised_shape": list(pred.shape),
                "denoised_mean_abs": float(np.mean(np.abs(pred))),
            }
        elif task in ("phase_picking", "picking"):
            x = to_bct(normalize_event_layout(arr["waveform"]))[0]
            pred = model(torch.from_numpy(x).unsqueeze(0).to(device)).cpu().numpy()[0]
            result = {
                "task": "phase_picking",
                "p_sample": int(np.argmax(pred[1])),
                "s_sample": int(np.argmax(pred[2])),
                "label_unit": "sample_index",
            }
        else:
            x = to_bct(normalize_event_layout(arr["waveform"]))
            pred = model(torch.from_numpy(x).unsqueeze(0).to(device)).cpu().numpy()[0]
            result = {"task": "source_localization", "pred_source": pred.tolist()}

    print(json.dumps(result, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
