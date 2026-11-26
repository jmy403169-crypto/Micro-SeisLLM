#!/usr/bin/env python
"""Train Micro-SeisLLM for noise suppression (paper default: event-level)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import model_kwargs_from_cfg, prepare_cfg, str2bool  # noqa: E402
from microseisllm.datasets import DenoisingDataset  # noqa: E402
from microseisllm.losses import DenoisingMSELoss  # noqa: E402
from microseisllm.models.seismollm import (  # noqa: E402
    SeisMoLLM_denoise,
    SeisMoLLM_denoise_event,
)
from microseisllm.utils.io import ensure_dir, resolve_path  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Train denoising model")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--pretrain", type=str2bool, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--llm-path", default=None)
    p.add_argument("--max-steps", type=int, default=None, help="Smoke-test step cap")
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

    train_ds = DenoisingDataset(
        data_dir=data_cfg.get("data_dir", "data/sample/denoising"),
        split_file=data_cfg.get("split_file", "data/splits/denoising_split.json"),
        split=data_cfg.get("train_split", "train"),
        flatten_stations=flatten,
    )
    val_ds = DenoisingDataset(
        data_dir=data_cfg.get("data_dir", "data/sample/denoising"),
        split_file=data_cfg.get("split_file", "data/splits/denoising_split.json"),
        split=data_cfg.get("val_split", "validation"),
        flatten_stations=flatten,
    )
    train_loader = DataLoader(
        train_ds, batch_size=int(cfg.get("batch_size", 2)), shuffle=True
    )
    val_loader = DataLoader(val_ds, batch_size=int(cfg.get("batch_size", 2)))

    mk = model_kwargs_from_cfg(cfg)
    model = (
        SeisMoLLM_denoise_event(**mk) if event_mode else SeisMoLLM_denoise(**mk)
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", 1e-4)))
    criterion = DenoisingMSELoss()
    ckpt_dir = ensure_dir(
        resolve_path(cfg.get("checkpoint_dir", "checkpoints/lora_adapters/denoising"))
    )

    best = float("inf")
    patience = int(cfg.get("early_stopping", {}).get("patience", 10))
    bad = 0
    max_epochs = int(cfg.get("max_epochs", 50))
    max_steps = args.max_steps
    global_step = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        n = 0
        for batch in train_loader:
            noisy = batch["noisy"].to(device)
            clean = batch["clean"].to(device)
            pred = model(noisy)
            loss = criterion(pred, clean)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += float(loss.item()) * noisy.size(0)
            n += noisy.size(0)
            global_step += 1
            if max_steps is not None and global_step >= max_steps:
                break
        train_loss /= max(n, 1)

        model.eval()
        val_loss = 0.0
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                noisy = batch["noisy"].to(device)
                clean = batch["clean"].to(device)
                pred = model(noisy)
                loss = criterion(pred, clean)
                val_loss += float(loss.item()) * noisy.size(0)
                vn += noisy.size(0)
        val_loss /= max(vn, 1)
        print(f"[denoising] epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if val_loss < best:
            best = val_loss
            bad = 0
            torch.save(
                {"model": model.state_dict(), "cfg": cfg, "epoch": epoch},
                ckpt_dir / "best.pt",
            )
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break
        if max_steps is not None and global_step >= max_steps:
            torch.save(
                {"model": model.state_dict(), "cfg": cfg, "epoch": epoch},
                ckpt_dir / "smoke.pt",
            )
            break


if __name__ == "__main__":
    main()
