"""Source localization dataset (multi-station waveforms + catalogue)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..utils.io import resolve_path
from .preprocessing import normalize_event_layout, to_bct, zscore_per_trace


class LocalizationDataset(Dataset):
    """
    Loads ``data/sample/source_localization/``:
      - events/{id}.npz : waveform (N,T,3)
      - stations.csv
      - catalogue.csv : event_id, lat, lon, depth, mag  (local metres-like units)
      - velocity_model.json
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "data/sample/source_localization",
        split_file: Optional[Union[str, Path]] = "data/splits/source_localization_split.json",
        split: str = "train",
        max_stations: Optional[int] = None,
        normalize: bool = True,
    ):
        self.data_dir = resolve_path(data_dir)
        self.split = split
        self.normalize = normalize
        self.max_stations = max_stations

        self.stations = pd.read_csv(self.data_dir / "stations.csv")
        self.catalogue = pd.read_csv(self.data_dir / "catalogue.csv")
        if "event_id" not in self.catalogue.columns:
            # Accept first column as event id.
            first = self.catalogue.columns[0]
            self.catalogue = self.catalogue.rename(columns={first: "event_id"})
        self.catalogue["event_id"] = self.catalogue["event_id"].astype(str)

        vel_path = self.data_dir / "velocity_model.json"
        self.velocity_model = {}
        if vel_path.is_file():
            with open(vel_path, "r", encoding="utf-8") as f:
                self.velocity_model = json.load(f)

        ids = self._load_split_ids(split_file, split)
        events_dir = self.data_dir / "events"
        self.rows: List[dict] = []
        cat_by_id = {str(r["event_id"]): r for _, r in self.catalogue.iterrows()}
        candidate_ids = ids if ids else [str(x) for x in self.catalogue["event_id"].tolist()]
        for eid in candidate_ids:
            npz = events_dir / f"{eid}.npz"
            if not npz.is_file():
                continue
            if eid not in cat_by_id:
                continue
            self.rows.append({"event_id": eid, "path": npz, "meta": cat_by_id[eid]})
        if not self.rows:
            raise FileNotFoundError(
                f"No localization events under {events_dir} matching catalogue/split"
            )

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
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        with np.load(row["path"], allow_pickle=False) as z:
            wave = normalize_event_layout(z["waveform"])
            fs = float(z["sampling_rate_hz"]) if "sampling_rate_hz" in z.files else 500.0

        n_sta = wave.shape[0]
        if self.max_stations is not None:
            n_sta = min(n_sta, int(self.max_stations))
            wave = wave[:n_sta]

        x = to_bct(wave)  # (N,3,T)
        if self.normalize:
            x = zscore_per_trace(x)

        meta = row["meta"]
        target = np.array(
            [
                float(meta["lat"]),
                float(meta["lon"]),
                float(meta["depth"]),
                float(meta["mag"]),
            ],
            dtype=np.float32,
        )
        station_mask = np.ones(x.shape[0], dtype=np.bool_)

        return {
            "waveform": torch.from_numpy(x),
            "target": torch.from_numpy(target),
            "station_mask": torch.from_numpy(station_mask),
            "event_id": row["event_id"],
            "sampling_rate_hz": fs,
        }
