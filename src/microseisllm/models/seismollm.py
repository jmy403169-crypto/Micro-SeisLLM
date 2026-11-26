"""SeisMoLLM / Micro-SeisLLM assembler and task factories."""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gpt2_backbone import build_gpt2_backbone, gpt2_hidden_state
from .latent_patching import latent_patch, latent_unpatch
from .lora_adapter import build_lora_config
from .multiscale_embedder import MultiScaleConvBlock
from .qwen3_backbone import build_qwen_backbone, qwen_hidden_state
from .task_heads import (
    HeadDetectionPicking,
    HeadLoc4,
    HeadWaveformReconstruction,
)


class LLM_Block(nn.Module):
    """Latent patching + truncated LLM backbone (GPT-2 or Qwen3)."""

    def __init__(
        self,
        start_layer,
        end_layer,
        patch_size,
        lora_config,
        pretrain=True,
        freeze=True,
        llm_name="GPT2",
        llm_path=None,
        d_model=768,
    ):
        super().__init__()
        self.pretrain = pretrain
        self.freeze = freeze
        self.lora_config = lora_config
        self.patch_size = patch_size
        self.llm_name = (llm_name or "GPT2").strip()
        self._is_qwen = self.llm_name.upper().startswith("QWEN")
        self.d_model = d_model

        if self._is_qwen:
            self.llm = build_qwen_backbone(
                start_layer=start_layer,
                end_layer=end_layer,
                pretrain=pretrain,
                freeze=freeze,
                llm_path=llm_path,
                lora_config=lora_config,
                d_model=d_model,
            )
        else:
            self.llm = build_gpt2_backbone(
                start_layer=start_layer,
                end_layer=end_layer,
                pretrain=pretrain,
                freeze=freeze,
                llm_path=llm_path,
                lora_config=lora_config,
                d_model=d_model,
            )

    def forward(self, x):
        x = latent_patch(x, self.patch_size)
        out = self.llm(inputs_embeds=x)
        if self._is_qwen:
            x = qwen_hidden_state(out)
        else:
            x = gpt2_hidden_state(out)
        x = latent_unpatch(x, self.patch_size)
        return x


class SeisMoLLM(nn.Module):
    """Single-trace Micro-SeisLLM: multi-scale embedder + LLM + task head."""

    def __init__(
        self,
        in_channels=3,
        conv_scale_num=4,
        conv_scale_strides=(8, 6, 4, 2),
        conv_channels=(16, 48, 96),
        conv_kernel_sizes=(16, 8, 6, 1),
        conv_strides=(2, 2, 2, 1),
        llm_layers=3,
        d_model=768,
        patch_size=8,
        dp_head_channels=(128, 160, 192, 224),
        path_drop_rate=0.2,
        mlp_drop_rate=0.2,
        mlp_ratio=4,
        mlp_bias=True,
        act_layer=nn.GELU,
        norm_layer=nn.BatchNorm1d,
        use_checkpoint=False,
        output_head=None,
        llm="GPT2",
        pretrain=True,
        freeze=True,
        LLM_start_layer=0,
        LLM_layer_num=None,
        llm_path=None,
        lora_rank=16,
        lora_alpha=16,
        **kwargs,
    ):
        super().__init__()
        if output_head is None:
            output_head = HeadLoc4
        if LLM_layer_num is not None:
            llm_layers = LLM_layer_num
        end_layer = LLM_start_layer + llm_layers
        lora_config = build_lora_config(
            llm, r=lora_rank, lora_alpha=lora_alpha
        )

        assert len(conv_channels) + 1 == len(conv_kernel_sizes) == len(conv_strides)
        conv_channels = list(conv_channels)
        conv_channels.append(d_model // patch_size)

        self.use_checkpoint = use_checkpoint
        self.patch_size = patch_size
        self.feature_channels = conv_channels[-1]

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
                    [in_channels] + conv_channels[:-1],
                    conv_channels,
                    conv_kernel_sizes,
                    conv_strides,
                )
            ]
        )

        self.llm_blocks = LLM_Block(
            start_layer=LLM_start_layer,
            end_layer=end_layer,
            patch_size=patch_size,
            lora_config=lora_config,
            pretrain=pretrain,
            freeze=freeze,
            llm_name=llm,
            llm_path=llm_path,
            d_model=d_model,
        )

        if (output_head in [HeadDetectionPicking, HeadWaveformReconstruction]) or (
            isinstance(output_head, partial)
            and (
                output_head.func
                in [HeadDetectionPicking, HeadWaveformReconstruction]
            )
        ):
            out_layer_channels = []
            out_layer_kernel_sizes = []
            for channel, kernel in zip(dp_head_channels, conv_kernel_sizes):
                out_layer_channels.insert(0, channel)
                out_layer_kernel_sizes.insert(0, kernel)
            self.out_head = output_head(
                in_channels=in_channels,
                feature_channels=self.feature_channels,
                layer_channels=out_layer_channels,
                layer_kernel_sizes=out_layer_kernel_sizes,
                act_layer=act_layer,
                norm_layer=norm_layer,
                path_drop_rate=path_drop_rate,
                mlp_drop_rate=mlp_drop_rate,
                mlp_ratio=mlp_ratio,
                mlp_bias=mlp_bias,
            )
        else:
            self.out_head = output_head(
                feature_channels=self.feature_channels,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )

    def forward(self, x):
        x_input = x
        x = self.convs(x)
        x = self.llm_blocks(x)
        x = self.out_head(x, x_input)
        return x


def SeisMoLLM_dpk(**kwargs):
    """Detection and phase-picking factory."""
    model = SeisMoLLM(
        path_drop_rate=0.3,
        mlp_drop_rate=0.3,
        output_head=partial(
            HeadDetectionPicking, out_act_layer=nn.Sigmoid, out_channels=3
        ),
        **kwargs,
    )
    return model


def SeisMoLLM_denoise(**kwargs):
    """Single-trace noise suppression / waveform reconstruction."""
    model = SeisMoLLM(
        path_drop_rate=0.2,
        mlp_drop_rate=0.2,
        output_head=partial(
            HeadWaveformReconstruction, out_act_layer=nn.Identity, out_channels=3
        ),
        **kwargs,
    )
    return model


def SeisMoLLM_loc(**kwargs):
    """Single-trace localization (loc4)."""
    model = SeisMoLLM(
        path_drop_rate=0.2,
        mlp_drop_rate=0.2,
        output_head=partial(HeadLoc4, out_act_layer=nn.Identity),
        **kwargs,
    )
    return model


class SeisMoLLMDenoiseEvent(nn.Module):
    """Event-level multi-station denoising with cross-station attention."""

    def __init__(
        self,
        in_channels=3,
        conv_scale_num=4,
        conv_scale_strides=(8, 6, 4, 2),
        conv_channels=(16, 48, 96),
        conv_kernel_sizes=(16, 8, 6, 1),
        conv_strides=(2, 2, 2, 1),
        llm_layers=3,
        d_model=768,
        patch_size=8,
        dp_head_channels=(128, 160, 192, 224),
        path_drop_rate=0.2,
        mlp_drop_rate=0.2,
        mlp_ratio=4,
        mlp_bias=True,
        act_layer=nn.GELU,
        norm_layer=nn.BatchNorm1d,
        use_checkpoint=False,
        llm="GPT2",
        pretrain=True,
        freeze=True,
        LLM_start_layer=0,
        LLM_layer_num=None,
        llm_path=None,
        station_attn_heads=8,
        lora_rank=16,
        lora_alpha=16,
        **kwargs,
    ):
        super().__init__()
        if LLM_layer_num is not None:
            llm_layers = LLM_layer_num
        end_layer = LLM_start_layer + llm_layers
        lora_config = build_lora_config(llm, r=lora_rank, lora_alpha=lora_alpha)

        conv_channels = list(conv_channels)
        conv_channels.append(d_model // patch_size)
        self.patch_size = patch_size
        self.feature_channels = conv_channels[-1]

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
                    [in_channels] + conv_channels[:-1],
                    conv_channels,
                    conv_kernel_sizes,
                    conv_strides,
                )
            ]
        )
        self.llm_blocks = LLM_Block(
            start_layer=LLM_start_layer,
            end_layer=end_layer,
            patch_size=patch_size,
            lora_config=lora_config,
            pretrain=pretrain,
            freeze=freeze,
            llm_name=llm,
            llm_path=llm_path,
            d_model=d_model,
        )

        out_layer_channels = []
        out_layer_kernel_sizes = []
        for channel, kernel in zip(dp_head_channels, conv_kernel_sizes):
            out_layer_channels.insert(0, channel)
            out_layer_kernel_sizes.insert(0, kernel)
        self.out_head = HeadWaveformReconstruction(
            feature_channels=self.feature_channels,
            layer_channels=out_layer_channels,
            layer_kernel_sizes=out_layer_kernel_sizes,
            act_layer=act_layer,
            norm_layer=norm_layer,
            out_act_layer=nn.Identity,
            out_channels=3,
        )

        feat_c = self.feature_channels
        n_heads = int(station_attn_heads)
        while feat_c % n_heads != 0 and n_heads > 1:
            n_heads -= 1
        self.station_attn = nn.MultiheadAttention(
            embed_dim=feat_c, num_heads=n_heads, dropout=0.1, batch_first=True
        )
        self.station_norm = nn.LayerNorm(feat_c)

    def forward(self, x: torch.Tensor, station_mask=None) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"SeisMoLLMDenoiseEvent expects (B,N,3,T), got {tuple(x.shape)}"
            )
        b, n, c, t = x.shape
        x_in = x.reshape(b * n, c, t)
        feat = self.convs(x_in)
        feat = self.llm_blocks(feat)
        cf, length = feat.shape[1], feat.shape[2]
        feat = feat.view(b, n, cf, length)
        h = feat.permute(0, 3, 1, 2).reshape(b * length, n, cf)
        key_pad = None
        if station_mask is not None:
            key_pad = (
                ~station_mask.bool()
                .unsqueeze(1)
                .expand(b, length, n)
                .reshape(b * length, n)
            )
        attn_out, _ = self.station_attn(
            h, h, h, key_padding_mask=key_pad, need_weights=False
        )
        h = self.station_norm(h + attn_out)
        feat = h.view(b, length, n, cf).permute(0, 2, 3, 1).reshape(b * n, cf, length)
        out = self.out_head(feat, x_in)
        return out.view(b, n, c, t)


class SeisMoLLMLocEvent(nn.Module):
    """Multi-station localization: shared encoder + station attention → loc4."""

    def __init__(
        self,
        in_channels=3,
        conv_scale_num=4,
        conv_scale_strides=(8, 6, 4, 2),
        conv_channels=(16, 48, 96),
        conv_kernel_sizes=(16, 8, 6, 1),
        conv_strides=(2, 2, 2, 1),
        llm_layers=3,
        d_model=768,
        patch_size=8,
        dp_head_channels=(128, 160, 192, 224),
        path_drop_rate=0.2,
        mlp_drop_rate=0.2,
        mlp_ratio=4,
        mlp_bias=True,
        act_layer=nn.GELU,
        norm_layer=nn.BatchNorm1d,
        use_checkpoint=False,
        llm="GPT2",
        pretrain=True,
        freeze=True,
        LLM_start_layer=0,
        LLM_layer_num=None,
        llm_path=None,
        station_attn_heads=8,
        loc_out_dim=4,
        lora_rank=16,
        lora_alpha=16,
        **kwargs,
    ):
        super().__init__()
        self.loc_out_dim = int(loc_out_dim)
        backbone = SeisMoLLM(
            in_channels=in_channels,
            conv_scale_num=conv_scale_num,
            conv_scale_strides=conv_scale_strides,
            conv_channels=conv_channels,
            conv_kernel_sizes=conv_kernel_sizes,
            conv_strides=conv_strides,
            llm_layers=llm_layers,
            d_model=d_model,
            patch_size=patch_size,
            dp_head_channels=dp_head_channels,
            path_drop_rate=path_drop_rate,
            mlp_drop_rate=mlp_drop_rate,
            mlp_ratio=mlp_ratio,
            mlp_bias=mlp_bias,
            act_layer=act_layer,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            output_head=partial(HeadLoc4, out_act_layer=nn.Identity),
            llm=llm,
            pretrain=pretrain,
            freeze=freeze,
            LLM_start_layer=LLM_start_layer,
            LLM_layer_num=LLM_layer_num,
            llm_path=llm_path,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )
        self.convs = backbone.convs
        self.llm_blocks = backbone.llm_blocks
        self.patch_size = backbone.patch_size
        feat_c = backbone.feature_channels
        n_heads = int(station_attn_heads)
        while feat_c % n_heads != 0 and n_heads > 1:
            n_heads -= 1
        self.station_attn = nn.MultiheadAttention(
            embed_dim=feat_c, num_heads=n_heads, dropout=0.1, batch_first=True
        )
        self.station_norm = nn.LayerNorm(feat_c)
        self.loc_head = nn.Sequential(
            nn.Linear(feat_c, feat_c),
            act_layer(),
            nn.Dropout(mlp_drop_rate),
            nn.Linear(feat_c, self.loc_out_dim),
        )

    def _encode_traces(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c, t = x.shape
        x = x.reshape(b * n, c, t)
        x = self.convs(x)
        x = self.llm_blocks(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        return x.view(b, n, -1)

    def forward(self, x: torch.Tensor, station_mask=None) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"SeisMoLLMLocEvent expects (B,N,C,T), got {tuple(x.shape)}"
            )
        h = self._encode_traces(x)
        key_pad = None
        if station_mask is not None:
            key_pad = ~station_mask.bool()
        attn_out, _ = self.station_attn(
            h, h, h, key_padding_mask=key_pad, need_weights=False
        )
        h = self.station_norm(h + attn_out)
        if station_mask is not None:
            m = station_mask.float().unsqueeze(-1)
            h = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        else:
            h = h.mean(dim=1)
        return self.loc_head(h)


def SeisMoLLM_loc_event(**kwargs):
    kwargs.pop("in_samples", None)
    kwargs.pop("output_head", None)
    return SeisMoLLMLocEvent(**kwargs)


def SeisMoLLM_denoise_event(**kwargs):
    kwargs.pop("in_samples", None)
    kwargs.pop("output_head", None)
    return SeisMoLLMDenoiseEvent(**kwargs)


def build_model(task: str, backbone: str = "gpt2", **kwargs) -> nn.Module:
    """Factory used by training / evaluation scripts."""
    llm = "Qwen3" if str(backbone).lower().startswith("qwen") else "GPT2"
    kwargs.setdefault("llm", llm)
    kwargs.setdefault("pretrain", False)
    task = task.lower().replace("-", "_")
    if task in ("denoise", "denoising", "noise_suppression"):
        return SeisMoLLM_denoise(**kwargs)
    if task in ("pick", "picking", "phase_picking", "dpk"):
        return SeisMoLLM_dpk(**kwargs)
    if task in ("loc", "localization", "source_localization", "loc_event"):
        return SeisMoLLM_loc_event(**kwargs)
    raise ValueError(f"Unknown task: {task}")
