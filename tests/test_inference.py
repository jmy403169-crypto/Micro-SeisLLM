"""Inference shape + loss computation tests."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from microseisllm.datasets import DenoisingDataset, PickingDataset, LocalizationDataset
from microseisllm.losses import DenoisingMSELoss, BCELoss, LocalizationHuberLoss
from microseisllm.models.seismollm import (
    SeisMoLLM_denoise,
    SeisMoLLM_dpk,
    SeisMoLLM_loc_event,
)


def test_denoising_loss_and_infer():
    ds = DenoisingDataset(data_dir=ROOT / "data/sample/denoising", split_file=None)
    item = ds[0]
    model = SeisMoLLM_denoise(llm="GPT2", d_model=768, pretrain=False, llm_layers=2)
    pred = model(item["noisy"].unsqueeze(0))
    loss = DenoisingMSELoss()(pred, item["clean"].unsqueeze(0))
    assert pred.shape == (1, 3, item["noisy"].shape[-1])
    assert torch.isfinite(loss)


def test_picking_loss_and_infer():
    ds = PickingDataset(data_dir=ROOT / "data/sample/phase_picking", split_file=None)
    item = ds[0]
    model = SeisMoLLM_dpk(llm="GPT2", d_model=768, pretrain=False, llm_layers=2)
    pred = model(item["waveform"].unsqueeze(0))
    loss = BCELoss(weight=[[0.0], [1.0], [1.0]])(pred, item["labels"].unsqueeze(0))
    assert pred.shape[1] == 3
    assert torch.isfinite(loss)


def test_localization_loss_and_infer():
    ds = LocalizationDataset(
        data_dir=ROOT / "data/sample/source_localization", split_file=None
    )
    item = ds[0]
    model = SeisMoLLM_loc_event(llm="GPT2", d_model=768, pretrain=False, llm_layers=2)
    pred = model(item["waveform"].unsqueeze(0), item["station_mask"].unsqueeze(0))
    loss = LocalizationHuberLoss()(pred, item["target"].unsqueeze(0))
    assert pred.shape == (1, 4)
    assert torch.isfinite(loss)
