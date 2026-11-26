#!/usr/bin/env python
"""Train Micro-SeisLLM for source localization."""

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
from microseisllm.datasets import LocalizationDataset  # noqa: E402
from microseisllm.losses import LocalizationHuberLoss  # noqa: E402
from microseisllm.models.seismollm import SeisMoLLM_loc_event  # noqa: E402
from microseisllm.utils.io import ensure_dir, resolve_path  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Train source localization model")
    p.add_argument("--config", required=True)
    p.add_argument("--pretrain", type=str2bool, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--llm-path", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = prepare_cfg(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = cfg.get("data", {})

    train_ds = LocalizationDataset(
        data_dir=data_cfg.get("data_dir", "data/sample/source_localization"),
        split_file=data_cfg.get(
            "split_file", "data/splits/source_localization_split.json"
        ),
        split=data_cfg.get("train_split", "train"),
    )
    val_ds = LocalizationDataset(
        data_dir=data_cfg.get("data_dir", "data/sample/source_localization"),
        split_file=data_cfg.get(
            "split_file", "data/splits/source_localization_split.json"
        ),
        split=data_cfg.get("val_split", "validation"),
    )
    train_loader = DataLoader(train_ds, batch_size=int(cfg.get("batch_size", 2)), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(cfg.get("batch_size", 2)))

    mk = model_kwargs_from_cfg(cfg)
    mk["in_channels"] = int(cfg.get("in_channels", 3))
    model = SeisMoLLM_loc_event(**mk).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", 1e-4)))
    criterion = LocalizationHuberLoss(delta=float(cfg.get("huber_delta", 1.0)))
    ckpt_dir = ensure_dir(
        resolve_path(
            cfg.get("checkpoint_dir", "checkpoints/lora_adapters/source_localization")
        )
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
            x = batch["waveform"].to(device)
            y = batch["target"].to(device)
            mask = batch["station_mask"].to(device)
            pred = model(x, station_mask=mask)
            loss = criterion(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += float(loss.item()) * x.size(0)
            n += x.size(0)
            global_step += 1
            if max_steps is not None and global_step >= max_steps:
                break
        train_loss /= max(n, 1)

        model.eval()
        val_loss = 0.0
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["waveform"].to(device)
                y = batch["target"].to(device)
                mask = batch["station_mask"].to(device)
                pred = model(x, station_mask=mask)
                loss = criterion(pred, y)
                val_loss += float(loss.item()) * x.size(0)
                vn += x.size(0)
        val_loss /= max(vn, 1)
        print(f"[localization] epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

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
