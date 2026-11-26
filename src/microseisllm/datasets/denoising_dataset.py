"""Denoising dataset over sample NPZ files (noisy + clean pairs)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.io import resolve_path
from .preprocessing import normalize_event_layout, to_bct, zscore_per_trace


class DenoisingDataset(Dataset):
    """
    Loads ``data/sample/denoising/*.npz`` with keys:
      - noisy: (N_sta, T, 3) float32
      - clean: (N_sta, T, 3) float32
      - event_id: str
      - sampling_rate_hz: float (500)
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "data/sample/denoising",
        split_file: Optional[Union[str, Path]] = "data/splits/denoising_split.json",
        split: str = "train",
        station_indices: Optional[Sequence[int]] = None,
        normalize: bool = True,
        flatten_stations: bool = True,
    ):
        self.data_dir = resolve_path(data_dir)
        self.split = split
        self.normalize = normalize
        self.flatten_stations = flatten_stations
        self.station_indices = (
            list(station_indices) if station_indices is not None else None
        )

        self.event_ids = self._load_split_ids(split_file, split)
        self.files: List[Path] = []
        for eid in self.event_ids:
            p = self.data_dir / f"{eid}.npz"
            if p.is_file():
                self.files.append(p)
        if not self.files:
            # Fall back to all NPZ in directory.
            self.files = sorted(self.data_dir.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No denoising NPZ files under {self.data_dir}")

    @staticmethod
    def _load_split_ids(split_file, split: str) -> List[str]:
        if split_file is None:
            return []
        path = resolve_path(split_file)
        if not path.is_file():
            return []
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        key = {
            "train": "train_event_ids",
            "validation": "validation_event_ids",
            "val": "validation_event_ids",
            "test": "test_event_ids",
        }.get(split, f"{split}_event_ids")
        return [str(x) for x in manifest.get(key, [])]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.files[idx]
        with np.load(path, allow_pickle=False) as z:
            noisy = normalize_event_layout(z["noisy"])
            clean = normalize_event_layout(z["clean"])
            event_id = str(z["event_id"]) if "event_id" in z.files else path.stem
            fs = float(z["sampling_rate_hz"]) if "sampling_rate_hz" in z.files else 500.0

        if self.station_indices is not None:
            noisy = noisy[self.station_indices]
            clean = clean[self.station_indices]

        noisy_bct = to_bct(noisy)  # (N,3,T)
        clean_bct = to_bct(clean)
        if self.normalize:
            noisy_bct = zscore_per_trace(noisy_bct)
            clean_bct = zscore_per_trace(clean_bct)

        if self.flatten_stations:
            # Training default: one station as one sample → (3,T)
            # Use station 0 for deterministic smoke tests; randomize via DataLoader if needed.
            s = 0
            return {
                "noisy": torch.from_numpy(noisy_bct[s]),
                "clean": torch.from_numpy(clean_bct[s]),
                "event_id": event_id,
                "sampling_rate_hz": fs,
            }
        return {
            "noisy": torch.from_numpy(noisy_bct),
            "clean": torch.from_numpy(clean_bct),
            "event_id": event_id,
            "sampling_rate_hz": fs,
        }
