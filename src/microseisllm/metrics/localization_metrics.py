"""Localization MAE / RMSE in metres (synthetic catalogue local coords)."""

from __future__ import annotations

import numpy as np
import torch


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def localization_mae(pred, target, dims=(0, 1, 2)) -> dict:
    """
    MAE per coordinate and 3-D hypocenter distance.

    Synthetic catalogues in this repo use local Cartesian-like units in metres.
    """
    pred = _to_numpy(pred).astype(np.float64)
    target = _to_numpy(target).astype(np.float64)
    if pred.ndim == 1:
        pred = pred[None, :]
        target = target[None, :]
    names = ["lat", "lon", "depth", "mag"]
    out = {}
    for i in dims:
        name = names[i] if i < len(names) else f"dim{i}"
        out[f"mae_{name}"] = float(np.mean(np.abs(pred[:, i] - target[:, i])))
    hyp_err = np.sqrt(
        np.sum((pred[:, :3] - target[:, :3]) ** 2, axis=1)
    )
    out["mae_hypocenter_m"] = float(np.mean(hyp_err))
    out["mae_hypocenter_km"] = out["mae_hypocenter_m"] / 1000.0
    return out


def localization_rmse(pred, target, dims=(0, 1, 2)) -> dict:
    pred = _to_numpy(pred).astype(np.float64)
    target = _to_numpy(target).astype(np.float64)
    if pred.ndim == 1:
        pred = pred[None, :]
        target = target[None, :]
    names = ["lat", "lon", "depth", "mag"]
    out = {}
    for i in dims:
        name = names[i] if i < len(names) else f"dim{i}"
        out[f"rmse_{name}"] = float(
            np.sqrt(np.mean((pred[:, i] - target[:, i]) ** 2))
        )
    hyp_err = np.sqrt(np.sum((pred[:, :3] - target[:, :3]) ** 2, axis=1))
    out["rmse_hypocenter_m"] = float(np.sqrt(np.mean(hyp_err ** 2)))
    out["rmse_hypocenter_km"] = out["rmse_hypocenter_m"] / 1000.0
    return out
