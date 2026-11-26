"""Sample dataset loading tests."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from microseisllm.datasets import DenoisingDataset, PickingDataset, LocalizationDataset
from microseisllm.utils.seed import set_seed


def test_denoising_dataset():
    set_seed(0)
    ds = DenoisingDataset(
        data_dir=ROOT / "data/sample/denoising",
        split_file=ROOT / "data/splits/denoising_split.json",
        split="test",
    )
    assert len(ds) >= 1
    item = ds[0]
    assert item["noisy"].shape[0] == 3
    assert item["clean"].shape == item["noisy"].shape


def test_picking_dataset():
    ds = PickingDataset(
        data_dir=ROOT / "data/sample/phase_picking",
        split_file=ROOT / "data/splits/phase_picking_split.json",
        split="test",
    )
    item = ds[0]
    assert item["waveform"].shape[0] == 3
    assert item["labels"].shape[0] == 3
    assert item["labels"].shape[-1] == item["waveform"].shape[-1]


def test_localization_dataset():
    ds = LocalizationDataset(
        data_dir=ROOT / "data/sample/source_localization",
        split_file=ROOT / "data/splits/source_localization_split.json",
        split="test",
    )
    item = ds[0]
    assert item["waveform"].ndim == 3
    assert item["target"].numel() == 4
    assert (ROOT / "data/sample/source_localization/stations.csv").is_file()
    assert (ROOT / "data/sample/source_localization/velocity_model.json").is_file()
