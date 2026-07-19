"""Physics-Constrained Unfolding Transformer (PUT)."""

import math
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.init import _calculate_fan_in_and_fan_out

__all__ = ["PUT"]


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


def shift_back(inputs, step=2):  # input [bs,28,256,310]  output [bs, 28, 256, 256]
    [bs, nC, row, col] = inputs.shape
    down_sample = 256 // row
    step = float(step) / float(down_sample * down_sample)
    out_col = row
    for i in range(nC):
        inputs[:, i, :, :out_col] = \
            inputs[:, i, :, int(step * i):int(step * i) + out_col]
    return inputs[:, :, :, :out_col]


class MS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.qkv = nn.Conv2d(heads, dim_head * heads*3, kernel_size=1, bias=None)
        self.qkv_dwconv = nn.Conv2d(dim_head * heads*3, dim_head * heads*3, kernel_size=3, stride=1, padding=1, groups=dim_head * heads*3, bias=None)
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_qy = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_ky = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.rescaley = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, z, ms,pan):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = z.shape
        b, hy, wy, cy = ms.shape
        z = z.reshape(b, h * w, c)
        pan = self.qkv_dwconv(self.qkv(pan.permute(0,3,1,2))).permute(0,2,3,1)
        panq = pan[:, :, :, 0:c].reshape(b, h * w, c)
        pank = pan[:, :, :, c:c*2].reshape(b, h * w, c)
        panv = pan[:, :, :, c*2:c * 3].reshape(b, h * w, c)
        # panv = pan.reshape(b, h * w, c)

        ms = ms.reshape(b, hy * wy, cy)
        q_inp = self.to_q(z)
        k_inp = self.to_k(z)
        v_inp = self.to_v(z)
        q_inp_ = q_inp*panq
        k_inp_ = k_inp*pank
        v_inp_ = v_inp*panv
        q_inpy = self.to_qy(ms)
        k_inpy = self.to_ky(ms)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q_inp_, k_inp_, v_inp_))
        qy, ky = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q_inpy, k_inpy))
        v = v
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        qy = qy.transpose(-2, -1)
        ky = ky.transpose(-2, -1)

        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)

        qy = F.normalize(qy, dim=-1, p=2)
        ky = F.normalize(ky, dim=-1, p=2)

        attny = (ky @ qy.transpose(-2, -1))  # A = K^T*Q
        attny = attny * self.rescaley
        attny = attny.softmax(dim=-1)

        attn = (k @ q.transpose(-2, -1))  # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attny @ attn @ v  # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)  # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, heads, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim + heads, (dim * mult) + heads, 1, 1, bias=False),
            GELU(),
            nn.Conv2d((dim * mult) + heads, (dim * mult) + heads, 3, 1, 1, bias=False, groups=dim * mult + heads),
            GELU(),
            nn.Conv2d((dim * mult) + heads, dim, 1, 1, bias=False),
        )

    def forward(self, x, pan):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(torch.cat([x.permute(0, 3, 1, 2), pan.permute(0, 3, 1, 2)], dim=1))
        return out.permute(0, 2, 3, 1)


class MSAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            num_blocks,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MS_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim, heads=heads))
            ]))

    def forward(self, z, ms, pan):

        z = z.permute(0, 2, 3, 1)
        ms = ms.permute(0, 2, 3, 1)
        pan = pan.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            z = attn(z, ms,pan) + z
            z = ff(z, pan) + z
        out = z.permute(0, 3, 1, 2)
        return out




class CustomModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.final_conv = nn.Conv2d(in_dim + 1, out_dim, kernel_size=1)

    def forward(self, lms, ms, pan):
        """Construct the initial HR-HSI estimate from LR-HSI and HR-PAN."""
        if lms is None:
            lms = F.interpolate(ms, size=pan.shape[-2:], mode="bilinear", align_corners=False)
        if lms.shape[-2:] != pan.shape[-2:]:
            raise ValueError("The upsampled HSI and PAN must have the same spatial size.")
        return self.final_conv(torch.cat((lms, pan), dim=1))





class MST(nn.Module):
    def __init__(self, in_dim=64, out_dim=64, dim=64, stage=2, num_blocks=[2, 4, 4]):
        super(MST, self).__init__()
        self.dim = dim
        self.stage = stage

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                MSAB(
                    dim=dim_stage, num_blocks=num_blocks[i], dim_head=dim, heads=dim_stage // dim),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_stage // dim, (dim_stage // dim) * 2, 4, 2, 1, bias=False),
            ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = MSAB(
            dim=dim_stage, dim_head=dim, heads=dim_stage // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.ConvTranspose2d(dim_stage // dim, (dim_stage // dim) // 2, stride=2, kernel_size=2, padding=0,
                                   output_padding=0),
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                MSAB(
                    dim=dim_stage // 2, num_blocks=num_blocks[stage - 1 - i], dim_head=dim,
                    heads=(dim_stage // 2) // dim),
            ]))
            dim_stage //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, z, ms, pan):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        # Embedding
        fea = self.embedding(z)
        ms = self.embedding(ms)

        # Encoder
        fea_encoder = []
        for (MSAB, FeaDownSample, pandown) in self.encoder_layers:
            fea = MSAB(fea, ms, pan)
            fea_encoder.append(fea)
            # y_list.append(y)
            fea = FeaDownSample(fea)
            ms = FeaDownSample(ms)
            pan = pandown(pan)

        # Bottleneck
        fea = self.bottleneck(fea, ms, pan)

        # Decoder
        for i, (FeaUpSample, panup, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            ms = FeaUpSample(ms)
            pan = panup(pan)
            fea = Fution(torch.cat([fea, fea_encoder[self.stage - 1 - i]], dim=1))
            # fea = LeWinBlcok(fea,y_list[self.stage-1-i])
            fea = LeWinBlcok(fea, ms, pan)

        # Mapping
        out = self.mapping(fea) + z

        return out


class SPED(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.spe = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=None), nn.ReLU())

    def forward(self, fea):
        out = self.spe(fea)
        return out


class SPAD(nn.Module):
    def __init__(self, ratio, in_dim, out_dim):
        super().__init__()
        self.ratio = ratio
        self.spa = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1), nn.ReLU())

    def forward(self, fea, output_size=None):
        if output_size is None:
            output_size = (fea.shape[-2] // self.ratio, fea.shape[-1] // self.ratio)
        out = self.spa(F.interpolate(fea, size=output_size, mode='bilinear', align_corners=False))
        return out


class DATA(nn.Module):
    """Data-fidelity update using a truncated Neumann-style recurrence."""

    def __init__(self, ratio, in_dim, spad_module, sped_module, neumann_terms=3):
        super().__init__()
        self.ratio = ratio
        self.spaT = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1), nn.ReLU())
        self.speT = nn.Sequential(nn.Conv2d(1, in_dim, kernel_size=1, bias=None), nn.ReLU())
        self.spaD = spad_module
        self.speD = sped_module
        self.mu = nn.Parameter(torch.tensor(10.0))
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.neumann_terms = int(neumann_terms)
        if self.neumann_terms < 1:
            raise ValueError("neumann_terms must be at least 1")

    def forward(self, lms, ms, pan):
        out_spa = self.spaT(F.interpolate(ms, size=lms.shape[-2:], mode='bilinear', align_corners=False))
        out_spae = self.speT(pan) + out_spa + self.mu * lms
        mu = torch.clamp(self.mu, min=1e-6)
        x_next = (1 / mu) * out_spae
        for _ in range(self.neumann_terms):
            degraded = self.spaD(x_next, output_size=ms.shape[-2:])
            Ax = self.spaT(F.interpolate(degraded, size=lms.shape[-2:], mode='bilinear', align_corners=False))
            Rx = self.gamma * self.speT(self.speD(x_next))
            x_next = (1 / mu) * (out_spae - Ax - Rx)
        return x_next


class PUT(nn.Module):
    """Physics-Constrained Unfolding Transformer for HSI-PAN fusion.

    Args:
        ratio: Spatial scale between the HR-PAN and LR-HSI.
        in_channels: Number of PAN channels (one in the paper).
        out_channels: Number of hyperspectral bands.
        stage: Number of unfolding stages.
        channels: Feature width. The released experiments use one feature
            channel per hyperspectral band.
        neumann_terms: Number of recurrence terms in each data-fidelity block.

    The forward pass returns ``(hr_hsi, reconstructed_pan,
    reconstructed_lr_hsi)``.
    """

    def __init__(
        self,
        ratio: int = 4,
        in_channels: int = 1,
        out_channels: int = 128,
        stage: int = 3,
        channels: int = 128,
        neumann_terms: int = 3,
    ):
        super().__init__()
        if ratio < 1:
            raise ValueError("ratio must be a positive integer")
        if stage < 1:
            raise ValueError("stage must be at least 1")
        if channels != out_channels:
            raise ValueError("channels must equal out_channels for this checkpoint-compatible implementation")

        self.ratio = int(ratio)
        self.stage = int(stage)
        self.channels = int(channels)
        self.out_channels = int(out_channels)
        self.init_model = CustomModel(in_dim=out_channels, out_dim=out_channels)

        self.conv_inx = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.conv_iny = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.shared_sped = SPED(out_channels, 1)
        self.shared_spad = SPAD(ratio, out_channels, out_channels)

        self.bodies = nn.ModuleList()
        for _ in range(stage):
            self.bodies.append(
                MST(
                    in_dim=out_channels,
                    out_dim=out_channels,
                    dim=self.channels,
                    stage=2,
                    num_blocks=[1, 1, 1]
                )
            )

        self.datas = nn.ModuleList()
        for _ in range(stage):
            self.datas.append(
                DATA(
                    ratio,
                    out_channels,
                    self.shared_spad,
                    self.shared_sped,
                    neumann_terms=neumann_terms,
                )
            )

        self.conv_out = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

    def forward(
        self,
        ms: torch.Tensor,
        pan: torch.Tensor,
        lms: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if ms.ndim != 4 or pan.ndim != 4:
            raise ValueError("ms and pan must be BCHW tensors")
        if ms.shape[0] != pan.shape[0]:
            raise ValueError("ms and pan must have the same batch size")
        if ms.shape[1] != self.out_channels:
            raise ValueError(f"Expected {self.out_channels} HSI bands, got {ms.shape[1]}")
        if pan.shape[1] != 1:
            raise ValueError(f"Expected a single PAN channel, got {pan.shape[1]}")
        expected = (ms.shape[-2] * self.ratio, ms.shape[-1] * self.ratio)
        if pan.shape[-2:] != expected:
            raise ValueError(
                f"PAN spatial size must be ratio x LR-HSI size: expected {expected}, got {pan.shape[-2:]}"
            )

        init_features = self.init_model(lms, ms, pan)

        processed_ms = self.conv_iny(ms)
        processed_pan = self.conv_inx(pan)

        features = [init_features]
        for i in range(self.stage):
            current_body = self.bodies[i]
            current_sup = self.datas[i]

            h = current_body(features[i], processed_ms, processed_pan)
            v = current_sup(h, ms, pan)
            features.append(v)

        output_features = self.conv_out(features[-1])
        pan_d = self.shared_sped(output_features)
        hsi_d = self.shared_spad(output_features)
        return output_features, pan_d, hsi_d
