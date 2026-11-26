"""Phase-picking dataset over sample NPZ + optional labels CSV."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.io import resolve_path
from .preprocessing import (
    gaussian_pick_label,
    normalize_event_layout,
    to_bct,
    zscore_per_trace,
)


class PickingDataset(Dataset):
    """
    Loads ``data/sample/phase_picking/*.npz`` with keys:
      - waveform: (N_sta, T, 3) float32
      - p_arrival_sample: (N_sta,) float32  [sample indices at 500 Hz]
      - s_arrival_sample: (N_sta,) float32
      - event_id, sampling_rate_hz
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "data/sample/phase_picking",
        split_file: Optional[Union[str, Path]] = "data/splits/phase_picking_split.json",
        split: str = "train",
        station_index: int = 0,
        normalize: bool = True,
        label_sigma: float = 10.0,
        out_channels: int = 3,
    ):
        self.data_dir = resolve_path(data_dir)
        self.split = split
        self.station_index = int(station_index)
        self.normalize = normalize
        self.label_sigma = float(label_sigma)
        self.out_channels = int(out_channels)

        self.event_ids = self._load_split_ids(split_file, split)
        self.files: List[Path] = []
        for eid in self.event_ids:
            p = self.data_dir / f"{eid}.npz"
            if p.is_file():
                self.files.append(p)
        if not self.files:
            self.files = sorted(self.data_dir.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No picking NPZ files under {self.data_dir}")

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
            wave = normalize_event_layout(z["waveform"])
            p_arr = np.asarray(z["p_arrival_sample"], dtype=np.float32)
            s_arr = (
                np.asarray(z["s_arrival_sample"], dtype=np.float32)
                if "s_arrival_sample" in z.files
                else np.full(wave.shape[0], np.nan, dtype=np.float32)
            )
            event_id = str(z["event_id"]) if "event_id" in z.files else path.stem
            fs = float(z["sampling_rate_hz"]) if "sampling_rate_hz" in z.files else 500.0

        s = min(self.station_index, wave.shape[0] - 1)
        x = to_bct(wave[s : s + 1])[0]  # (3,T)
        if self.normalize:
            x = zscore_per_trace(x)

        t_len = x.shape[-1]
        det = gaussian_pick_label(t_len, float(p_arr[s]), self.label_sigma)
        p_lab = gaussian_pick_label(t_len, float(p_arr[s]), self.label_sigma)
        if np.isfinite(s_arr[s]):
            s_lab = gaussian_pick_label(t_len, float(s_arr[s]), self.label_sigma)
            det = np.maximum(det, s_lab)
        else:
            s_lab = np.zeros(t_len, dtype=np.float32)

        # Channel order: detection, P, S (matches SeisMoLLM_dpk out_channels=3)
        labels = np.stack([det, p_lab, s_lab], axis=0).astype(np.float32)
        if self.out_channels < 3:
            labels = labels[: self.out_channels]

        return {
            "waveform": torch.from_numpy(x),
            "labels": torch.from_numpy(labels),
            "p_arrival_sample": torch.tensor(float(p_arr[s]), dtype=torch.float32),
            "s_arrival_sample": torch.tensor(
                float(s_arr[s]) if np.isfinite(s_arr[s]) else -1.0,
                dtype=torch.float32,
            ),
            "event_id": event_id,
            "sampling_rate_hz": fs,
        }
