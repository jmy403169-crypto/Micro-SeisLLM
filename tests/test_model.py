"""Model initialization tests (no pretrained LLM weights required)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from microseisllm.models.seismollm import (
    SeisMoLLM_denoise,
    SeisMoLLM_denoise_event,
    SeisMoLLM_dpk,
    SeisMoLLM_loc_event,
)


def test_denoising_init_forward():
    model = SeisMoLLM_denoise(llm="GPT2", d_model=768, pretrain=False, llm_layers=2)
    x = torch.randn(2, 3, 1024)
    y = model(x)
    assert y.shape == x.shape


def test_denoising_event_init_forward():
    model = SeisMoLLM_denoise_event(llm="GPT2", d_model=768, pretrain=False, llm_layers=2)
    x = torch.randn(1, 4, 3, 1024)
    y = model(x)
    assert y.shape == x.shape


def test_picking_init_forward():
    model = SeisMoLLM_dpk(llm="GPT2", d_model=768, pretrain=False, llm_layers=2)
    x = torch.randn(2, 3, 1024)
    y = model(x)
    assert y.shape == (2, 3, 1024)
    assert torch.all(y >= 0) and torch.all(y <= 1)


def test_localization_init_forward():
    model = SeisMoLLM_loc_event(llm="GPT2", d_model=768, pretrain=False, llm_layers=2)
    x = torch.randn(2, 8, 3, 1024)
    y = model(x)
    assert y.shape == (2, 4)
