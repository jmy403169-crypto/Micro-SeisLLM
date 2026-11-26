"""Task heads: waveform reconstruction, detection/picking, localization (loc4)."""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multiscale_embedder import auto_pad_1d


def _head_apply_temporal_convs(convs: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
    for conv in convs:
        if x.size(-1) < conv.kernel_size[0]:
            break
        x = conv(x)
    return x


class HeadDetectionPicking(nn.Module):
    """Detection and phase-picking head (sigmoid probabilities along time)."""

    def __init__(
        self,
        feature_channels,
        layer_channels,
        layer_kernel_sizes,
        act_layer,
        norm_layer,
        out_act_layer=nn.Identity,
        out_channels=1,
        **kwargs,
    ):
        super().__init__()
        assert len(layer_channels) == len(layer_kernel_sizes)
        self.depth = len(layer_channels)
        self.up_layers = nn.ModuleList()
        for inc, outc, kers in zip(
            [feature_channels] + layer_channels[:-1],
            layer_channels[:-1] + [out_channels * 2],
            layer_kernel_sizes,
        ):
            conv = nn.Conv1d(in_channels=inc, out_channels=outc, kernel_size=kers)
            norm = norm_layer(outc)
            act = act_layer()
            self.up_layers.append(
                nn.Sequential(
                    OrderedDict([("conv", conv), ("norm", norm), ("act", act)])
                )
            )
        self.out_conv = nn.Conv1d(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            kernel_size=7,
            padding=3,
        )
        self.out_act = out_act_layer()

    def _upsampling_sizes(self, in_size: int, out_size: int):
        sizes = [out_size] * self.depth
        factor = (out_size / in_size) ** (1 / self.depth)
        for i in range(self.depth - 2, -1, -1):
            sizes[i] = int(sizes[i + 1] / factor)
        return sizes

    def forward(self, x, x0):
        up_sizes = self._upsampling_sizes(in_size=x.size(-1), out_size=x0.size(-1))
        for i, layer in enumerate(self.up_layers):
            x = F.interpolate(x, size=up_sizes[i], mode="linear")
            x = auto_pad_1d(x, layer.conv.kernel_size[0], layer.conv.stride[0])
            x = layer(x)
        x = self.out_conv(x)
        x = self.out_act(x)
        return x


class HeadWaveformReconstruction(nn.Module):
    """Three-component waveform reconstruction head (denoising)."""

    def __init__(
        self,
        feature_channels,
        layer_channels,
        layer_kernel_sizes,
        act_layer,
        norm_layer,
        out_act_layer=nn.Identity,
        out_channels=3,
        **kwargs,
    ):
        super().__init__()
        assert len(layer_channels) == len(layer_kernel_sizes)
        self.depth = len(layer_channels)
        self.up_layers = nn.ModuleList()
        for inc, outc, kers in zip(
            [feature_channels] + layer_channels[:-1],
            layer_channels[:-1] + [out_channels * 2],
            layer_kernel_sizes,
        ):
            conv = nn.Conv1d(in_channels=inc, out_channels=outc, kernel_size=kers)
            norm = norm_layer(outc)
            act = act_layer()
            self.up_layers.append(
                nn.Sequential(
                    OrderedDict([("conv", conv), ("norm", norm), ("act", act)])
                )
            )
        self.out_conv = nn.Conv1d(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            kernel_size=7,
            padding=3,
        )
        self.out_act = out_act_layer()

    def _upsampling_sizes(self, in_size: int, out_size: int):
        sizes = [out_size] * self.depth
        factor = (out_size / in_size) ** (1 / self.depth)
        for i in range(self.depth - 2, -1, -1):
            sizes[i] = int(sizes[i + 1] / factor)
        return sizes

    def forward(self, x, x0):
        up_sizes = self._upsampling_sizes(in_size=x.size(-1), out_size=x0.size(-1))
        for i, layer in enumerate(self.up_layers):
            x = F.interpolate(x, size=up_sizes[i], mode="linear")
            x = auto_pad_1d(x, layer.conv.kernel_size[0], layer.conv.stride[0])
            x = layer(x)
        x = self.out_conv(x)
        x = self.out_act(x)
        return x


class HeadLoc4(nn.Module):
    """Source localization head: lat, lon, depth, magnitude (4-D regression)."""

    def __init__(self, feature_channels, out_act_layer=nn.Identity, **kwargs):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(feature_channels, feature_channels, 16, 4) for _ in range(2)]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(1, -1)
        self.lin = nn.Linear(feature_channels, 4)
        self.out_act = out_act_layer()

    def forward(self, x, _: torch.Tensor = None):
        x = _head_apply_temporal_convs(self.convs, x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.lin(x)
        x = self.out_act(x)
        return x


class HeadClassification(nn.Module):
    def __init__(self, feature_channels, num_classes, out_act_layer, **kwargs):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(feature_channels, feature_channels, 16, 4) for _ in range(2)]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(1, -1)
        self.lin = nn.Linear(feature_channels, num_classes)
        self.out_act = out_act_layer()

    def forward(self, x, _: torch.Tensor = None):
        x = _head_apply_temporal_convs(self.convs, x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.out_act(self.lin(x))


class HeadRegression(nn.Module):
    def __init__(self, feature_channels, out_act_layer, **kwargs):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(feature_channels, feature_channels, 16, 4) for _ in range(2)]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(1, -1)
        self.lin = nn.Linear(feature_channels, 1)
        self.out_act = out_act_layer()

    def forward(self, x, _: torch.Tensor = None):
        x = _head_apply_temporal_convs(self.convs, x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.out_act(self.lin(x))
