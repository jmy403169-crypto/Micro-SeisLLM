"""Multi-scale convolutional embedder for three-component waveforms."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def auto_pad_1d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int = 1,
    dim: int = -1,
    padding_value: float = 0.0,
) -> torch.Tensor:
    """Pad so that Conv1d output length is ``ceil(L / stride)``."""
    assert kernel_size >= stride, (
        f"`kernel_size` must be >= `stride`, got {kernel_size}, {stride}"
    )
    pos_dim = dim if dim >= 0 else x.dim() + dim
    pds = (stride - (x.size(dim) % stride)) % stride + kernel_size - stride
    padding = (0, 0) * (x.dim() - pos_dim - 1) + (pds // 2, pds - pds // 2)
    return F.pad(x, padding, "constant", padding_value)


# Backward-compatible alias used by original SeisMoLLM code.
_auto_pad_1d = auto_pad_1d


class ScaledActivation(nn.Module):
    def __init__(self, act_layer: nn.Module, scale_factor: float):
        super().__init__()
        self.scale_factor = scale_factor
        self.act = act_layer()

    def forward(self, x):
        return self.act(x) * self.scale_factor


class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, act_layer, norm_layer):
        super().__init__()
        self.in_proj = nn.Conv1d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1, bias=False
        )
        self.conv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )
        self.norm = norm_layer(out_dim)
        self.act = act_layer()

    def forward(self, x):
        x = self.in_proj(x)
        x = auto_pad_1d(x, self.conv.kernel_size[0], self.conv.stride[0])
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class MultiScaleConvBlock(nn.Module):
    """Parallel multi-kernel Conv1d branches fused by 1x1 projection."""

    def __init__(
        self,
        scale_num,
        scale_stride,
        in_dim,
        out_dim,
        kernel_size,
        stride,
        act_layer,
        norm_layer,
    ):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                ConvBlock(
                    in_dim,
                    out_dim,
                    kernel_size + int(scale_stride * scale),
                    stride,
                    act_layer,
                    norm_layer,
                )
                for scale in range(scale_num)
            ]
        )
        self.out_proj = nn.Conv1d(
            in_channels=scale_num * out_dim,
            out_channels=out_dim,
            kernel_size=1,
            bias=False,
        )
        self.norm = norm_layer(out_dim)

    def forward(self, x):
        outs = [conv(x) for conv in self.convs]
        x = torch.cat(outs, dim=1)
        x = self.out_proj(x)
        x = self.norm(x)
        return x


# Alias matching source naming.
Multi_Scale_Conv_Block = MultiScaleConvBlock


class MultiScaleEmbedder(nn.Module):
    """Stack of multi-scale conv blocks mapping waveforms to LLM feature channels."""

    def __init__(
        self,
        in_channels=3,
        conv_scale_num=4,
        conv_scale_strides=(8, 6, 4, 2),
        conv_channels=(16, 48, 96),
        conv_kernel_sizes=(16, 8, 6, 1),
        conv_strides=(2, 2, 2, 1),
        d_model=768,
        patch_size=8,
        act_layer=nn.GELU,
        norm_layer=nn.BatchNorm1d,
    ):
        super().__init__()
        channels = list(conv_channels) + [d_model // patch_size]
        assert len(channels) == len(conv_kernel_sizes) == len(conv_strides)
        self.feature_channels = channels[-1]
        self.convs = nn.Sequential(
            *[
                MultiScaleConvBlock(
                    scale_num=conv_scale_num,
                    scale_stride=ss,
                    in_dim=inc,
                    out_dim=outc,
                    kernel_size=kers,
                    stride=strd,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
                for ss, inc, outc, kers, strd in zip(
                    conv_scale_strides,
                    [in_channels] + channels[:-1],
                    channels,
                    conv_kernel_sizes,
                    conv_strides,
                )
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convs(x)
