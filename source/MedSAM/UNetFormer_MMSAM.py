import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from typing import Optional
from timm.models.layers import DropPath, trunc_normal_

class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalLocalAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3*dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1,  padding=(window_size//2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size//2 - 1))

        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.ws - 1
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape

        local = self.local2(x) + self.local1(x)

        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)

        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]

        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))

        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        out = out[:, :, :H, :W]

        return out

class Block(nn.Module):
    def __init__(self, dim=256, num_heads=16,  mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

        self.beta_phi = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, phi_map=None):
        if phi_map is not None:
            x = x * (1.0 + self.beta_phi * (1.0 - phi_map.clamp(0.0, 1.0)))

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class WF(nn.Module):
    """Gated skip fusion modulated by Φ gradient + raw Φ.

    ‖∇Φ‖ → boundary strength ("where to sharpen")
    Φ     → cross-modal agreement ("real boundary or artifact?")

    Raw Φ is fed as extra channel into the gate conv — the gate learns to
    distinguish true edges (low Φ + high ‖∇Φ‖) from shadows (low Φ + low ‖∇Φ‖).
    λ_phi init=0 so behaviour at epoch 1 is identical to Φ-free baseline.
    """
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)
        self.gate_conv = nn.Sequential(
            nn.Conv2d(decode_channels * 2 + 2, decode_channels // 4, 1, bias=False),
            nn.BatchNorm2d(decode_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(decode_channels // 4, 1, 1, bias=False),
        )
        self.lambda_u = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, res, U=None, Phi=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        res_proj = self.pre_conv(res)
        h, w = x.shape[2:]

        u_ch = F.interpolate(U, size=(h, w), mode='bilinear', align_corners=False) \
               if U is not None else torch.zeros(1, 1, h, w, device=x.device)
        p_ch = F.interpolate(Phi.clamp(0.0, 1.0), size=(h, w), mode='bilinear', align_corners=False) \
               if Phi is not None else torch.zeros(1, 1, h, w, device=x.device)

        gate_logits = self.gate_conv(torch.cat([res_proj, x, u_ch, p_ch], dim=1))

        if U is not None:
            gate_logits = gate_logits * (1.0 + self.lambda_u * _normalize_U(u_ch, q=0.95))

        g = torch.sigmoid(gate_logits)
        x = g * res_proj + (1 - g) * x
        x = self.post_conv(x)
        return x


class FeatureRefinementHead(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

        self.pa = nn.Sequential(nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
                                nn.Sigmoid())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                Conv(decode_channels, decode_channels//16, kernel_size=1),
                                nn.ReLU6(),
                                Conv(decode_channels//16, decode_channels, kernel_size=1),
                                nn.Sigmoid())

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)

        return x

class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(64, 128, 256, 512),
                 decode_channels=64,
                 dropout=0.1,
                 window_size=8,
                 num_classes=6):
        super(Decoder, self).__init__()

        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.b4 = Block(dim=decode_channels, num_heads=8, window_size=window_size)

        self.b3 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p3 = WF(encoder_channels[-2], decode_channels)

        self.b2 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p2 = WF(encoder_channels[-3], decode_channels)

        self.p1 = FeatureRefinementHead(encoder_channels[-4], decode_channels)

        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))
        self.init_weight()

    def forward(self, res1, res2, res3, res4, h, w, U_list=None, Phi_list=None):
        """Forward with Φ+‖∇Φ‖ modulation via WF gates."""
        def _get(l, idx):
            return l[idx] if l is not None and len(l) > idx else None
        U3, U2, U1 = _get(U_list, 2), _get(U_list, 1), _get(U_list, 0)
        P3, P2, P1 = _get(Phi_list, 2), _get(Phi_list, 1), _get(Phi_list, 0)

        def _resize_phi(p, target):
            if p is None: return None
            return F.interpolate(p, size=target.shape[2:], mode='bilinear', align_corners=False)

        x = self.b4(self.pre_conv(res4))
        x = self.p3(x, res3, U=U3, Phi=P3)
        x = self.b3(x, phi_map=_resize_phi(P3, x))

        x = self.p2(x, res2, U=U2, Phi=P2)
        x = self.b2(x, phi_map=_resize_phi(P2, x))

        x = self.p1(x, res1)

        x = self.segmentation_head(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class SegFormerDecoder(nn.Module):
    """SegFormer-style MLP decoder — lightweight multi-scale aggregation.

    Projects each of the 4 (same-channel) encoder outputs to a common dim,
    upsamples all to the highest resolution (res1 = H/4), concatenates, then
    fuses with an MLP. Replaces the attention-heavy UNetFormer decoder when
    scale aggregation matters more than boundary refinement (e.g. land cover).

    Same signature as Decoder.forward (U_list/Phi_list accepted but ignored),
    so the call site is unchanged.
    """

    def __init__(self, encoder_channels=(256, 256, 256, 256),
                 decoder_dim=256, dropout=0.1, num_classes=6):
        super().__init__()
        self.linear_c = nn.ModuleList([
            nn.Conv2d(c, decoder_dim, 1, bias=False) for c in encoder_channels
        ])
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * len(encoder_channels), decoder_dim, 1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(p=dropout)
        self.linear_pred = nn.Conv2d(decoder_dim, num_classes, 1)
        self.init_weight()

    def forward(self, res1, res2, res3, res4, h, w, U_list=None, Phi_list=None):
        feats = [res1, res2, res3, res4]
        out = []
        for i, f in enumerate(feats):
            f = self.linear_c[i](f)
            f = F.interpolate(f, size=feats[0].shape[2:], mode='bilinear', align_corners=False)
            out.append(f)
        x = torch.cat(out, dim=1)
        x = self.linear_fuse(x)
        x = self.dropout(x)
        x = self.linear_pred(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        return x

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def _normalize_U(U, q=0.95):
    """Percentile-based normalisation — robust to extreme outliers.

    Uses the q-th quantile (default 95%) as the divisor instead of max(),
    so a single noisy pixel doesn't squash the entire edge map to near-zero.
    Floored at 1e-3 to prevent noise amplification on flat images.
    Clamped to [0, 1] for stable modulation range.

    Args: U: [B, 1, H, W], q: quantile in (0, 1]
    Returns: U_norm: [B, 1, H, W] ∈ [0, 1]
    """
    B = U.size(0)
    flat = U.view(B, -1)                                           # [B, H*W]
    k = max(1, int(flat.shape[1] * q))
    U_ref = torch.kthvalue(flat, k, dim=1, keepdim=True)[0]        # [B, 1]
    U_ref = torch.clamp(U_ref.view(B, 1, 1, 1), min=1e-3)
    return (U / U_ref).clamp(0.0, 1.0)


class PatchConfidenceFusion(nn.Module):
    """Potential-field-gradient-driven feature fusion.

    Replaces variational uncertainty with the gradient magnitude of the
    cross-modal potential field Φ = cos_sim(f_0, f_1):

      U(x,y) = ||∇Φ(x,y)||  = sqrt(∂Φ/∂x² + ∂Φ/∂y²)

    ∇Φ is a natural, annotation-free edge detector:
      ||∇Φ|| large → modality boundary (RGB & DSM disagree) → amplify detail
      ||∇Φ|| ≈ 0  → flat region (modalities agree) → smooth fusion

    Computed via 3×3 Sobel — 2 convolution ops, no learnable parameters.
    Pixel-precise, unlike the old 7×7 box-filter variance that smeared edges.
    """

    def __init__(self, n_modalities, dim=256):
        super().__init__()
        self.n_modalities = n_modalities
        self.dim = dim

        sx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.0
        sy = torch.tensor([[-1.,-2.,-1.], [ 0., 0., 0.], [ 1., 2., 1.]]) / 8.0
        self.register_buffer('sobel_x', sx.view(1, 1, 3, 3), persistent=False)
        self.register_buffer('sobel_y', sy.view(1, 1, 3, 3), persistent=False)
        self.sobel_dilations = [1, 3, 5]

        self.sobel_fusion = nn.Conv2d(len(self.sobel_dilations), 1, 1, bias=False)
        nn.init.constant_(self.sobel_fusion.weight, 1.0 / len(self.sobel_dilations))

        self.conf_heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dim, dim // 4, 1, bias=False),
                          nn.BatchNorm2d(dim // 4), nn.ReLU(inplace=True),
                          nn.Conv2d(dim // 4, 1, 1, bias=False))
            for _ in range(n_modalities)
        ])

        self.lambda_param = nn.Parameter(torch.tensor(0.5))

    def _cos_sim_c2d(self, f0, f1):
        """Φ = cos_sim over channel dim.  f0,f1 ∈ [B,C,H,W] → [B,1,H,W]."""
        B, C, H, W = f0.shape
        flat0 = f0.reshape(B, C, -1)
        flat1 = f1.reshape(B, C, -1)
        dot = (flat0 * flat1).sum(dim=1)
        n0 = flat0.norm(dim=1).clamp(min=1e-6)
        n1 = flat1.norm(dim=1).clamp(min=1e-6)
        return (dot / (n0 * n1 + 1e-8)).reshape(B, 1, H, W)

    def _compute_potential_field(self, feats, low_pass=False):
        """If low_pass=True, pools features first → Φ_low (geometric)."""
        if len(feats) < 2:
            return None
        f0, f1 = feats[0], feats[1]
        if low_pass:
            f0 = F.avg_pool2d(f0, 3, stride=1, padding=1)
            f1 = F.avg_pool2d(f1, 3, stride=1, padding=1)
        return self._cos_sim_c2d(f0, f1)

    def _gradient_magnitude(self, Phi):
        """Multi-scale gradient magnitude via dilated Sobel (rates 1,3,5).

        Fine edges (rate 1) + medium transitions (rate 3) + coarse boundaries (rate 5)
        are fused by a learned 1×1 convolution.
        """
        if Phi is None:
            return None
        device = Phi.device
        grads = []
        for d in self.sobel_dilations:
            gx = F.conv2d(Phi, self.sobel_x, padding=d, dilation=d)
            gy = F.conv2d(Phi, self.sobel_y, padding=d, dilation=d)
            grads.append(torch.sqrt(gx * gx + gy * gy + 1e-8))
        multi = torch.cat(grads, dim=1)                                 # [B, 3, H, W]
        U = self.sobel_fusion(multi)                                    # [B, 1, H, W]
        return U

    def forward(self, feats, return_weights=False):
        B, C, H, W = feats[0].shape

        Phi_full = self._compute_potential_field(feats, low_pass=False)
        Phi_low  = self._compute_potential_field(feats, low_pass=True)
        U_low  = self._gradient_magnitude(Phi_low)
        U_full = self._gradient_magnitude(Phi_full)
        U = (U_low + U_full) / 2.0 if U_low is not None and U_full is not None else U_full
        self._last_U = U
        self._last_Phi = Phi_full

        raw_scores = [head(f) for f, head in zip(feats, self.conf_heads)]
        scores = torch.cat(raw_scores, dim=1)  # [B, N, H, W]
        weights = F.softmax(scores, dim=1)

        fused = sum(weights[:, i:i + 1] * feats[i]
                    for i in range(self.n_modalities))

        if U is not None:
            U_norm = _normalize_U(U, q=0.95)
            modulation = 1.0 + self.lambda_param * U_norm
            fused = fused * modulation

        return (fused, weights) if return_weights else fused

    def get_U(self) -> Optional[torch.Tensor]:
        """Return last computed ‖∇Φ‖ for decoder WF."""
        return getattr(self, '_last_U', None)

    def get_Phi(self) -> Optional[torch.Tensor]:
        """Return last computed raw Φ for decoder WF."""
        return getattr(self, '_last_Phi', None)

    def get_kl_loss(self) -> torch.Tensor:
        """No more variational KL — returns zero for backward compatibility."""
        return torch.tensor(0.0)


class _SqueezeAndExcitationMFNet(nn.Module):
    """Verbatim SE block from the official MFNet (TGRS'25) implementation,
    with the last conv zero-initialised so σ(0)=0.5 makes the fusion start
    EXACTLY at the mean — the neutral start, gates learned from there."""

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1),
            nn.Sigmoid())
        nn.init.zeros_(self.fc[2].weight)
        nn.init.zeros_(self.fc[2].bias)

    def forward(self, x):
        return x * self.fc(F.adaptive_avg_pool2d(x, 1))


class _SEFusionMFNet(nn.Module):
    """Official MFNet fusion: per-modality SE gates then element-wise add."""

    def __init__(self, channels):
        super().__init__()
        self.se0 = _SqueezeAndExcitationMFNet(channels)
        self.se1 = _SqueezeAndExcitationMFNet(channels)

    def forward(self, x0, x1):
        return self.se0(x0) + self.se1(x1)


class UNetFormer(nn.Module):
    """Multimodal semantic segmentation with DINOv3 and LoRA."""

    def __init__(self,
                 decode_channels=64,
                 dropout=0.1,
                 window_size=8,
                 num_classes=6,
                 n_modalities=2,
                 encoder_type='dinov3',
                 encoder_size='large',
                 decoder_type='unetformer',
                 dataset=None,
                 ):
        super().__init__()
        self.dataset = dataset
        self.n_modalities = n_modalities
        self.encoder_type = encoder_type
        self.encoder_size = encoder_size
        self.decoder_type = decoder_type

        from mm_encoder import build_mm_encoder

        private_r, shared_r = 4, 4

        if os.environ.get('MFNET_EVD') == '1':
            os.environ['MFNET_EVD_K'] = str(num_classes)

        self.image_encoder = build_mm_encoder(
            encoder_type=encoder_type, size=encoder_size,
            n_modalities=n_modalities, private_r=private_r, shared_r=shared_r,
        )

        encoder_channels = (256, 256, 256, 256)

        self.optical_proj = nn.Conv2d(4, 3, 1, bias=False)
        with torch.no_grad():
            self.optical_proj.weight.zero_()
            if dataset == 'WHU-OPT-SAR':
                self.optical_proj.weight[0, 2] = 1.0
                self.optical_proj.weight[1, 1] = 1.0
                self.optical_proj.weight[2, 0] = 1.0
            else:
                self.optical_proj.weight[0, 0] = 1.0
                self.optical_proj.weight[1, 1] = 1.0
                self.optical_proj.weight[2, 2] = 1.0

        self.decfuse = os.environ.get('MFNET_DECFUSE') == '1'

        self.single_stream = os.environ.get('MFNET_SINGLE') == '1'
        if self.single_stream:
            self.stack_proj = nn.Conv2d(6, 3, 1, bias=False)
            with torch.no_grad():
                self.stack_proj.weight.zero_()
                self.stack_proj.weight[0, 0] = 1.0
                self.stack_proj.weight[1, 1] = 1.0
                self.stack_proj.weight[2, 2] = 1.0

        self.arch = os.environ.get('MFNET_ARCH', 'light')
        if self.arch not in ('light', 'official'):
            raise ValueError('CDM-LoRA uses a decoder that takes DINOv3 feature taps')
        if self.arch == 'official':
            embed_dim = getattr(self.image_encoder, 'embed_dim', 1024)
            try:
                from .official_head import Feature2Pyramid, UPerHead
            except ImportError:
                from official_head import Feature2Pyramid, UPerHead
            self.tap_fuse = nn.Conv2d(embed_dim * n_modalities, embed_dim,
                                      kernel_size=1, bias=False)
            self.pyramid = Feature2Pyramid(embed_dim, rescales=(4, 2, 1, 0.5))
            self.official_head = UPerHead(in_ch=embed_dim, ch=512,
                                          num_classes=num_classes)
        elif self.arch == 'light':
            try:
                from .official_head import LightFPNHead
            except ImportError:
                from official_head import LightFPNHead
            self.depa = os.environ.get('MFNET_DEPA') == '1'
            self.dual_fusion = os.environ.get('MFNET_DUALFUSION') == '1'
            if self.depa:
                self.arb_gate = nn.Conv2d(2048, 1, kernel_size=1)
            self.cmpt = os.environ.get('MFNET_CMPT') == '1'
            if self.cmpt:
                n_cls = num_classes
                self.proto0 = nn.Parameter(torch.zeros(n_cls, 256), requires_grad=False)
                self.proto1 = nn.Parameter(torch.zeros(n_cls, 256), requires_grad=False)
                self.proto_mom = 0.99
                self.cmpt_pt = nn.Conv2d(512, 256, kernel_size=1)
                nn.init.zeros_(self.cmpt_pt.weight); nn.init.zeros_(self.cmpt_pt.bias)
            self.bjt = int(os.environ.get('MFNET_BJT', '0'))
            _four_tap = True
            if self.bjt:
                _r = int(os.environ.get('MFNET_BJT_RANK', '64'))
                self.bjt_conv = os.environ.get('MFNET_BJT_CONV') == '1'
                self.bjt_ctx = os.environ.get('MFNET_BJT_CTX') == '1'
                self.bjt_drop = float(os.environ.get('MFNET_BJT_DROP', '0'))
                self.bjt_jgate = float(os.environ.get('MFNET_JGATE', '0'))
                self.bjt_jgate_rel = os.environ.get('MFNET_JGATE_REL') == '1'
                if self.bjt_jgate_rel:
                    self.register_buffer('jgate_mu', torch.tensor(4.0))
                    self.register_buffer('jgate_sd', torch.tensor(0.5))
                    self.jgate_momentum = 0.99
                if self.bjt_conv:
                    self.joint_Uc = nn.Conv2d(1024, _r, 3, padding=1, bias=False)
                    self.joint_Vc = nn.Conv2d(1024, _r, 3, padding=1, bias=False)
                else:
                    self.joint_U = nn.Parameter(torch.randn(_r, 1024) * 0.02)
                    self.joint_V = nn.Parameter(torch.randn(_r, 1024) * 0.02)
                self.bjt_fuse = os.environ.get('MFNET_BJT_FUSE') == '1'
                if self.bjt_ctx and not self.bjt_fuse:
                    self.joint_out_ctx = nn.ModuleList([
                        nn.Sequential(
                            nn.Conv2d(2 * _r, 256, 1, bias=False),
                            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                            nn.Conv2d(256, 256, 3, padding=1, bias=False),
                            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                            nn.Conv2d(256, 1024, 1, bias=False))
                        for _ in range(4)])
                    for _s in self.joint_out_ctx:
                        nn.init.zeros_(_s[-1].weight)
                self.joint_out = nn.ModuleList([
                    nn.Conv2d(2 * _r, 1024, 1, bias=False) for _ in range(4)])
                for _c in self.joint_out:
                    nn.init.zeros_(_c.weight)
            self.use_xgl = os.environ.get('MFNET_XGL') == '1'
            _xk = os.environ.get('MFNET_XOR_KILL') == '1'
            _ak = os.environ.get('MFNET_AND_KILL') == '1'
            if _xk or _ak:
                for m in self.modules():
                    if getattr(m, '_use_xgl', False):
                        if _xk:
                            m.xgl_xor_kill = True
                        if _ak:
                            m.xgl_and_kill = True
            if self.use_xgl and _four_tap:
                try:
                    from .official_head import XGLHead
                except ImportError:
                    from official_head import XGLHead
                self.light_head = XGLHead(in_ch=1024, ch=256,
                                          num_classes=num_classes, n_taps=4,
                                          ev_ch=8)
                _wg_r = int(os.environ.get('MFNET_XGL_WG_R', '16'))
                self.xgl_Wg = nn.Parameter(torch.zeros(num_classes, _wg_r))
                self.xgl_d = os.environ.get('MFNET_XGL_D') == '1'
                if self.xgl_d:
                    _4r = 32
                    self.xgl_d_form = os.environ.get('MFNET_XGL_D_FORM', 'cls')
                    if self.xgl_d_form == 'scalar':
                        self.xgl_d_w = nn.Parameter(torch.zeros(_4r))
                        self.xgl_d_b = nn.Parameter(torch.tensor(-2.2))
                    else:
                        self.xgl_d_w = nn.Parameter(torch.zeros(num_classes, _4r))
                        self.xgl_d_b = nn.Parameter(
                            torch.full((num_classes,), -2.2))
            elif _four_tap and getattr(self, 'bjt_fuse', False):
                try:
                    from .official_head import LightFuseHead
                except ImportError:
                    from official_head import LightFuseHead
                self.light_head = LightFuseHead(in_ch=getattr(self.image_encoder, "embed_dim", 1024), ch=256,
                                                num_classes=num_classes, n_taps=4)
            elif _four_tap:
                self.light_head = LightFPNHead(in_ch=getattr(self.image_encoder, "embed_dim", 1024), ch=256,
                                               num_classes=num_classes, n_taps=4,
                                               extra=4 if self.bjt else 0)
                if os.environ.get('MFNET_XGA') == '1':
                    self.xga = True
                    self.xga2 = os.environ.get('MFNET_XGA2') == '1'
                    self.evd_pure = os.environ.get('MFNET_EVD_PURE') == '1'
                    _hch = getattr(self.image_encoder, 'embed_dim', 1024) \
                        + (64 if (self.xga2 and not self.evd_pure) else 0)
                    self.eref = os.environ.get('MFNET_EREF') == '1'
                    _evch = (2 * int(os.environ.get('MFNET_EVD_K', '6'))
                             + (2 if os.environ.get('MFNET_EVD_DS16') == '1' else 0)
                             if os.environ.get('MFNET_EVD_EDL') == '1'
                             else 4 * int(os.environ.get('MFNET_EVD_K', '6'))
                             if os.environ.get('MFNET_EVD') == '1' else 32)
                    self.xga_head0 = LightFPNHead(in_ch=_hch, ch=256,
                                                  num_classes=num_classes, n_taps=4,
                                                  ev_gate=self.eref, ev_ch=_evch)
                    self.xga_head1 = LightFPNHead(in_ch=_hch, ch=256,
                                                  num_classes=num_classes, n_taps=4,
                                                  ev_gate=self.eref, ev_ch=_evch)
                    if self.xga2:
                        _nec = (4 * int(os.environ.get('MFNET_EVD_K', '6'))
                                if os.environ.get('MFNET_EVD') == '1' else 32)
                        self.xgl_e2c = nn.Conv2d(_nec, 64, 1, bias=False)
                        nn.init.zeros_(self.xgl_e2c.weight)
                    self.xga_mdrop = float(os.environ.get('MFNET_MDROP', '0'))
                    _aec = (4 * int(os.environ.get('MFNET_EVD_K', '6'))
                            if os.environ.get('MFNET_EVD') == '1' else 32)
                    _K_ = int(os.environ.get('MFNET_EVD_K', '6'))
                    self._alpha_ev = os.environ.get('MFNET_ALPHA_EV', '')
                    self.evd_edl = os.environ.get('MFNET_EVD_EDL') == '1'
                    self.evd_ds14 = os.environ.get('MFNET_EVD_DS14') == '1'
                    self.evd_ds16 = os.environ.get('MFNET_EVD_DS16') == '1'
                    if self.evd_edl:
                        _xs11 = 4 if os.environ.get('MFNET_EVD_XS') == '1' else 0
                        _aroute = 2 * _K_ + 2 + _xs11
                    elif self._alpha_ev == 'log':
                        _aroute = 2 * _K_          # [log a; log-odds d]
                    elif self._alpha_ev == '1':
                        _aroute = _aec
                    else:
                        _aroute = _aec + 2 * num_classes + 2
                    _axst = 0 if self.evd_edl else (4 if getattr(self, 'xga2', False) else 0)
                    self.xgl_aw = nn.Parameter(
                        torch.zeros(_aroute + _axst))
                    self.xgl_ab = nn.Parameter(torch.zeros(1))
                    self.evd_fuse = os.environ.get('MFNET_EVD_FUSE') == '1'
                    if self.evd_fuse:
                        self.evd_zscore = False
                        _fuse_in = (2 * int(os.environ.get('MFNET_EVD_K', '6'))
                                    if os.environ.get('MFNET_EVD_EDL') == '1'
                                    else _aec)
                        self.xgl_fuse = nn.Conv2d(_fuse_in, num_classes, 1, bias=False)
                        nn.init.zeros_(self.xgl_fuse.weight)
                    self.evd_alpha_fix = os.environ.get('MFNET_ALPHA_FIX') == '1'
            else:
                self.light_head = LightFPNHead(in_ch=256, ch=256,
                                               num_classes=num_classes, n_taps=1)
        else:
            self.arch = 'legacy'

        if self.arch == 'legacy':
            self.fpn_branches = nn.ModuleList([
                nn.ModuleList([
                    nn.Sequential(
                        nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
                        Norm2d(256) if i == 0 else nn.BatchNorm2d(256),
                        nn.GELU(),
                        nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
                    ) if i == 0 else
                    nn.Sequential(
                        nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
                    ) if i == 1 else
                    nn.Identity() if i == 2 else
                    nn.MaxPool2d(kernel_size=2, stride=2)
                    for i in range(4)
                ])
                for _ in range(n_modalities)
            ])

            self.fusions = nn.ModuleList([
                PatchConfidenceFusion(n_modalities, encoder_channels[i])
                for i in range(4)
            ])

        for n, value in self.image_encoder.named_parameters():
            value.requires_grad = ('lora_' in n or 'alloc_' in n or 'pa_' in n or 'lcx_' in n or 'xgl_' in n
                                   or 'evd_wc_shared' in n)

        if self.arch == 'legacy':
            if decoder_type == 'segformer':
                self.decoder = SegFormerDecoder(encoder_channels, decoder_dim=256,
                                                dropout=dropout, num_classes=num_classes)
            elif decoder_type == 'unetformer':
                self.decoder = Decoder(encoder_channels, decode_channels, dropout, window_size, num_classes)
            else:
                raise ValueError(f"Unknown decoder_type: {decoder_type} (use 'unetformer' or 'segformer')")

    def forward(self, *inputs, mode='Train'):
        """Forward pass supporting 1-N modalities.

        Input parsing (Bug 3 fix): extracts all Tensor arguments,
        ignoring string mode tags.  Works for any N ≥ 1.
        """
        modalities = []
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                modalities.append(inp)
            elif isinstance(inp, (list, tuple)):
                modalities.extend([t for t in inp if isinstance(t, torch.Tensor)])
        if not modalities:
            raise ValueError("No input tensors found")

        h, w = modalities[0].size()[-2:]

        processed = []
        for mod in modalities:
            if mod.dim() == 3:
                mod = mod.unsqueeze(1)
            C = mod.shape[1]
            if C == 1:
                mod = mod.repeat(1, 3, 1, 1)
            elif C == 4:
                mod = self.optical_proj(mod)
            processed.append(mod)

        if getattr(self, 'single_stream', False) and len(processed) == 2:
            x3 = self.stack_proj(torch.cat(processed, dim=1))   # [B,3,H,W]
            deep = self.image_encoder([x3])
            taps = deep[0] if isinstance(deep[0], (list, tuple)) else [deep[0]]
            l0 = self.xga_head0(taps)
            if self.training:
                self._xga_lg = (l0,)
            out = F.interpolate(l0, size=(h, w), mode='bilinear',
                                align_corners=False)
            return out

        deep_features = self.image_encoder(processed)

        if isinstance(deep_features, torch.Tensor):
            deep_features = [deep_features]
        elif isinstance(deep_features, tuple):
            deep_features = list(deep_features)

        if self.arch == 'official':
            f0, f1 = deep_features[0], deep_features[1]
            feats = [self.tap_fuse(torch.cat([a, b], dim=1))
                     for a, b in zip(f0, f1)]
            pyr = self.pyramid(feats)
            x = self.official_head(pyr)                       # [B,cls,h/4,w/4]
            x = F.interpolate(x, size=(h, w), mode='bilinear',
                              align_corners=False)
            return x

        if self.arch == 'light':
            streams = [f if isinstance(f, (list, tuple)) else [f]
                       for f in deep_features]
            n_taps = len(streams[0])
            if getattr(self, 'decfuse', False) and len(streams) == 2:
                fused = [(streams[0][k] + streams[1][k]) / 2
                         for k in range(n_taps)]
                l = self.light_head(fused)
                if self.training:
                    self._xga_lg = (l,)
                return F.interpolate(l, size=(h, w), mode='bilinear',
                                     align_corners=False)
            if getattr(self, 'cmpt', False) and len(streams) == 2:
                head = self.light_head
                e0 = head.reduce(torch.cat(streams[0], dim=1))
                e1 = head.reduce(torch.cat(streams[1], dim=1))
                e_f = head.reduce(torch.cat([ (streams[0][k] + streams[1][k]) / 2
                                              for k in range(n_taps) ], dim=1))
                def post(e):
                    u = F.interpolate(e, scale_factor=4, mode='bilinear',
                                      align_corners=False)
                    return torch.softmax(head.conv_seg(head.refine(u)), dim=1)
                p0 = F.adaptive_avg_pool2d(post(e0).detach(), e0.shape[-2:])
                p1 = F.adaptive_avg_pool2d(post(e1).detach(), e1.shape[-2:])
                if self.training:
                    with torch.no_grad():
                        for p_i, e_i, P in ((p0, e0, self.proto0), (p1, e1, self.proto1)):
                            wgt = p_i.permute(0, 2, 3, 1).reshape(-1, p_i.shape[1])  # [Nw,C]
                            f_ = e_i.permute(0, 2, 3, 1).reshape(-1, e_i.shape[1])
                            num = (wgt.t() @ f_)                                   # [C,256]
                            den = wgt.sum(0).clamp(min=1e-6).unsqueeze(1)
                            P.mul_(self.proto_mom).add_((1 - self.proto_mom) * num / den)
                t01 = torch.einsum('bchw,cd->bdhw', p0, self.proto1)
                t10 = torch.einsum('bchw,cd->bdhw', p1, self.proto0)
                e_f = e_f + self.cmpt_pt(torch.cat([t01, t10], dim=1))
                u = F.interpolate(e_f, scale_factor=4, mode='bilinear',
                                  align_corners=False)
                x = head.conv_seg(head.refine(u))
            elif getattr(self, 'depa', False) and len(streams) == 2:
                l0 = self.light_head(streams[0])
                l1 = self.light_head(streams[1])
                g = torch.sigmoid(self.arb_gate(
                    torch.cat([streams[0][0], streams[1][0]], dim=1)))
                g = F.interpolate(g, size=l0.shape[-2:], mode='bilinear',
                                  align_corners=False)
                x = g * l0 + (1.0 - g) * l1
            elif getattr(self, 'dual_fusion', False) and len(streams) == 2:
                head = self.light_head
                l0 = head(streams[0])                     # [B,cls,h/4,w/4]
                l1 = head(streams[1])
                H_, W_ = l0.shape[-2:]
                blk = self.image_encoder.blocks[-1].mlp.lin1
                bm = getattr(blk, '_belief_map', None)
                w_list = []
                if bm is not None and 0 in bm and 1 in bm:
                    for v in (0, 1):
                        b = bm[v]                          # [B,N,C]
                        if b.dim() == 4 and b.shape[1] == 1:
                            b = b.squeeze(1)
                        B_, N_, C_ = b.shape
                        side = int(N_ ** 0.5)
                        n_pre = N_ - side * side
                        if side > 1 and 0 <= n_pre <= 10:
                            b = b[:, n_pre:, :].reshape(B_, side, side, C_)
                            b = b.permute(0, 3, 1, 2)      # [B,C,s,s]
                            b = F.interpolate(b, size=(H_, W_), mode='bilinear',
                                              align_corners=False)
                            b = torch.softmax(b, dim=1)
                        ent = -(b.clamp(min=1e-8) * b.clamp(min=1e-8).log()
                                ).sum(1, keepdim=True).detach()   # [B,1,H,W]
                        w_list.append(1.0 / ent.clamp(min=0.05))
                if len(w_list) == 2:
                    w0 = w_list[0] / (w_list[0] + w_list[1])
                else:
                    w0 = torch.full_like(l0[:, :1], 0.5)
                w1 = 1.0 - w0
                x = w0 * F.log_softmax(l0, dim=1) + \
                    w1 * F.log_softmax(l1, dim=1)
            elif (getattr(self, 'bjt', 0) or getattr(self, 'use_xgl', False)) and len(streams) == 2 and n_taps == 4:
                feats = [torch.stack([s[k] for s in streams]).mean(dim=0)
                         for k in range(n_taps)]
                jfeats = []
                j_raw = []
                for k in range(n_taps) if not getattr(self, 'use_xgl', False) else():
                    f0 = streams[0][k]                     # [B,1024,h,w]
                    f1 = streams[1][k]
                    if getattr(self, 'bjt_conv', False):
                        p0 = self.joint_Uc(f0)
                        p1 = self.joint_Vc(f1)
                        th, tw = f0.shape[-2:]
                    else:
                        th, tw = f0.shape[-2:]
                        p0 = torch.matmul(self.joint_U, f0.flatten(2))\
                            .reshape(f0.shape[0], -1, th, tw)
                        p1 = torch.matmul(self.joint_V, f1.flatten(2))\
                            .reshape(f1.shape[0], -1, th, tw)
                    j = torch.cat([p0 * p1, p0 - p1], dim=1)
                    if getattr(self, 'bjt_jgate_rel', False):
                        dE = (p0 - p1).norm(dim=1, keepdim=True)  # [B,1,h,w]
                        if self.training:
                            with torch.no_grad():
                                mu_b = dE.mean()
                                sd_b = dE.std().clamp(min=0.05)
                                self.jgate_mu.mul_(self.jgate_momentum).add_(
                                    (1 - self.jgate_momentum) * mu_b)
                                self.jgate_sd.mul_(self.jgate_momentum).add_(
                                    (1 - self.jgate_momentum) * sd_b)
                        w_gate = torch.sigmoid(
                            (self.jgate_mu + self.jgate_sd - dE.detach())
                            / (0.5 * self.jgate_sd))
                        j = j * w_gate
                    elif getattr(self, 'bjt_jgate', 0.0) > 0:
                        dE = (p0 - p1).norm(dim=1, keepdim=True)
                        w_gate = torch.sigmoid((self.bjt_jgate - dE) / 0.15)
                        j = j * w_gate
                    j_raw.append(j)
                    if getattr(self, 'bjt_fuse', False):
                        pass
                    elif getattr(self, 'bjt_ctx', False):
                        jfeats.append(self.joint_out_ctx[k](j))
                    else:
                        jfeats.append(self.joint_out[k](j))
                _jctrl = j_raw if getattr(self, 'bjt_fuse', False) else jfeats
                if self.training and getattr(self, 'bjt_drop', 0.0) > 0:
                    keep = (torch.rand(_jctrl[0].shape[0], 1, 1, 1,
                                       device=_jctrl[0].device)
                            >= self.bjt_drop).float() / (1.0 - self.bjt_drop)
                    _jctrl = [j * keep for j in _jctrl]
                if getattr(self, 'bjt', 0) == 2:
                    _jctrl = [torch.zeros_like(j) for j in _jctrl]
                if getattr(self, 'use_xgl', False):
                    blkL = self.image_encoder.blocks[-1].mlp.lin1
                    cache = getattr(blkL, '_xgl_cache', {})
                    B_, _, H_, W_ = streams[0][0].shape
                    side = H_
                    ntok = side * side
                    e0 = cache.get(0)
                    e1 = cache.get(1)
                    if e0 is None:
                        e0 = torch.zeros(B_, 1, ntok + 5, 8, device=streams[0][0].device)
                    if e1 is None:
                        e1 = torch.zeros(B_, 1, ntok + 5, 8, device=streams[0][0].device)
                    if e0.dim() == 4:
                        if e0.shape[-1] < 256:
                            e0 = e0.flatten(1, 2)
                            e1 = e1.flatten(1, 2)
                        else:
                            e0 = e0[:, 0]
                            e1 = e1[:, 0]
                    npre = e0.shape[1] - ntok
                    e0p = e0[:, npre:, :].reshape(B_, side, side, -1).permute(0, 3, 1, 2)
                    e1p = e1[:, npre:, :].reshape(B_, side, side, -1).permute(0, 3, 1, 2)
                    ev_cat = torch.cat([e0p, e1p], dim=1)  # [B, 4r, h, w]
                    if getattr(self, 'xgl_kill_all', False):
                        ev_cat = torch.zeros_like(ev_cat)
                        x = self.light_head(streams[0], streams[1], ev_cat)
                    else:
                        x = self.light_head(streams[0], streams[1], ev_cat)
                        if os.environ.get('MFNET_XGL_NOGUIDE') == '1':
                            out_sz = modalities[0].shape[-2:]
                            x = F.interpolate(x, size=out_sz, mode='bilinear',
                                              align_corners=False)
                            return x
                        _wg_r = self.xgl_Wg.shape[1]
                        efull = torch.cat([e0, e1], dim=-1)[:, npre:, :]
                        g = efull[:, :, :_wg_r] @ self.xgl_Wg.T
                        g = g.reshape(B_, side, side, -1).permute(0, 3, 1, 2)
                        if getattr(self, 'xgl_d', False):
                            eh = ev_cat.permute(0, 2, 3, 1)          # [B,h,w,4r]
                            if getattr(self, 'xgl_d_form', 'cls') == 'scalar':
                                d_logit = eh @ self.xgl_d_w.view(-1, 1) \
                                    + self.xgl_d_b                    # [B,h,w,1]
                            else:
                                d_logit = eh @ self.xgl_d_w.T \
                                    + self.xgl_d_b                    # [B,h,w,C]
                            d_map = torch.sigmoid(d_logit)
                            d_map = d_map.permute(0, 3, 1, 2)    # [B,C|1,h,w]
                            self._xgl_d_map = d_map.detach()
                            if self.training:
                                with torch.no_grad():
                                    base_l = F.interpolate(
                                        x, size=(side, side), mode='bilinear')
                                    self._xgl_d_dbg = (d_logit.permute(0, 3, 1, 2),
                                                       base_l)
                            g = g * F.interpolate(d_map, size=g.shape[-2:],
                                                  mode='bilinear',
                                                  align_corners=False)
                        g = F.interpolate(g, size=x.shape[-2:],
                                          mode='bilinear', align_corners=False)
                        if not getattr(self, 'xgl_noguide', False):
                            x = x + g
                elif getattr(self, 'bjt_fuse', False):
                    x = self.light_head(streams[0], streams[1], _jctrl)
                else:
                    x = self.light_head(feats + jfeats)    # [B,cls,h/4,w/4]
            else:
                feats = [torch.stack([s[k] for s in streams]).mean(dim=0)
                         for k in range(n_taps)]
                if getattr(self, 'xga', False) and len(streams) == 2:
                    s_in0, s_in1 = streams[0], streams[1]
                    xstats = []
                    if getattr(self, 'xga2', False):
                        side = streams[0][0].shape[-1]
                        ntok = side * side
                        s_in0, s_in1 = [], []
                        self._evd_taps = []
                        self._evd_log_taps = []
                        self._evd_unc_taps = []
                        self._evd_e_taps = []
                        for t in getattr(self.image_encoder, 'TAPS', (5, 11, 17, 23)):
                            c = getattr(self.image_encoder.blocks[t].mlp.lin1,
                                        '_xgl_cache', {})
                            e0t, e1t = c.get(0), c.get(1)
                            if e0t is not None and e0t.dim() == 4 and e0t.shape[-1] < 256:
                                e0t = e0t.flatten(1, 2)
                                e1t = e1t.flatten(1, 2)
                            if e0t is None:
                                s_in0 = streams[0]; s_in1 = streams[1]; xstats = []
                                break
                            if e0t.dim() == 4:
                                e0t, e1t = e0t[:, 0], e1t[:, 0]
                            npf = e0t.shape[1] - ntok
                            e0m = e0t[:, npf:, :].reshape(-1, side, side, e0t.shape[-1]).permute(0, 3, 1, 2)
                            e1m = e1t[:, npf:, :].reshape(-1, side, side, e1t.shape[-1]).permute(0, 3, 1, 2)
                            ev = torch.cat([e0m, e1m], dim=1)          # [B,32,s,s]
                            emb = self.xgl_e2c(ev)                     # [B,64,s,s]
                            if getattr(self, 'evd_pure', False):
                                s_in0.append(streams[0][len(s_in0)])
                                s_in1.append(streams[1][len(s_in1)])
                            else:
                                s_in0.append(torch.cat([streams[0][(5, 11, 17, 23).index(t)] if False else streams[0][len(s_in0)], emb], dim=1))
                                s_in1.append(torch.cat([streams[1][len(s_in1)], emb], dim=1))
                            r_ = e0t.shape[-1] // 2
                            if getattr(self, 'evd_edl', False):
                                _lg = getattr(self.image_encoder.blocks[t].mlp.lin1,
                                              '_evd_logu', None)
                                if _lg is not None:
                                    _d = _lg[0] - _lg[1]
                                    if _d.dim() == 4:
                                        _d = _d.flatten(1, 2)
                                    _d = _d[:, npf:, :]
                                    xstats.append(_d.abs().mean(-1).reshape(-1, 1, side, side))
                            else:
                                xstats.append(0.5 * (e0t[:, npf:, r_:].abs() + e1t[:, npf:, r_:].abs()).mean(-1).reshape(-1, 1, side, side))
                            _u = getattr(self.image_encoder.blocks[t].mlp.lin1,
                                         '_evd_u', None)
                            if _u is not None:
                                self._evd_taps.append((_u[0], _u[1]))
                                _lu = getattr(self.image_encoder.blocks[t].mlp.lin1,
                                              '_evd_logu', None)
                                if _lu is not None:
                                    self._evd_log_taps.append((_lu[0], _lu[1]))
                                _un = getattr(self.image_encoder.blocks[t].mlp.lin1,
                                              '_evd_unc', None)
                                if _un is not None:
                                    self._evd_unc_taps.append((_un[0], _un[1]))
                                _ee = getattr(self.image_encoder.blocks[t].mlp.lin1,
                                              '_evd_e', None)
                                if _ee is not None:
                                    self._evd_e_taps.append((_ee[0], _ee[1]))
                    _ev_g = None
                    _edl11 = getattr(self, 'evd_edl', False)
                    if _edl11:
                        _lt = getattr(self, '_evd_log_taps', None) or []
                        le0, le1 = _lt[-1]
                        if le0.dim() == 4:
                            le0 = le0.flatten(1, 2)
                            le1 = le1.flatten(1, 2)
                        _s11 = streams[0][0].shape[-1]
                        _n11 = le0.shape[1] - _s11 * _s11
                        _k11 = le0.shape[-1]
                        le0 = le0[:, _n11:, :] \
                            .reshape(-1, _s11, _s11, _k11).permute(0, 3, 1, 2)
                        le1 = le1[:, _n11:, :] \
                            .reshape(-1, _s11, _s11, _k11).permute(0, 3, 1, 2)
                        ev_cons = torch.cat([le0 + le1, le0 - le1], dim=1)
                        _ut = getattr(self, '_evd_unc_taps', None) or []
                        un0, un1 = _ut[-1]
                        if un0.dim() == 3:
                            un0 = un0.flatten(1, 2)
                            un1 = un1.flatten(1, 2)
                        un0 = un0[:, _n11:].reshape(-1, 1, _s11, _s11)
                        un1 = un1[:, _n11:].reshape(-1, 1, _s11, _s11)
                        ev_unc = torch.cat([un0, un1], dim=1)   # [B,2,s,s]
                        _kill = os.environ.get('MFNET_EVD_KILL', '')
                        if _kill:
                            _h = ev_cons.shape[1] // 2
                            if 'cons' in _kill:
                                ev_cons = torch.cat([torch.zeros_like(
                                    ev_cons[:, :_h]), ev_cons[:, _h:]], 1)
                            if 'dis' in _kill:
                                ev_cons = torch.cat([ev_cons[:, :_h], torch.zeros_like(
                                    ev_cons[:, _h:])], 1)
                            if 'both' in _kill:
                                ev_cons = torch.zeros_like(ev_cons)
                            if 'unc' in _kill:
                                ev_unc = torch.zeros_like(ev_unc)
                        if getattr(self, 'eref', False) and \
                                not getattr(self, 'evd_ds14', False):
                            _ev_g = torch.cat([ev_cons, ev_unc], dim=1) \
                                if getattr(self, 'evd_ds16', False) else ev_cons
                    elif getattr(self, 'eref', False):
                        _c = getattr(self.image_encoder.blocks[
                            getattr(self.image_encoder, 'TAPS', (5, 11, 17, 23))[-1]
                            if hasattr(self.image_encoder, 'blocks') and len(
                                getattr(self.image_encoder, 'TAPS', (5, 11, 17, 23))) > 0
                            else -1].mlp.lin1, '_xgl_cache', {})
                        _e0g, _e1g = _c.get(0), _c.get(1)
                        if _e0g is not None:
                            _side = streams[0][0].shape[-1]
                            _nt = _side * _side
                            if _e0g.dim() == 4 and _e0g.shape[-1] < 256:
                                _e0g = _e0g.flatten(1, 2); _e1g = _e1g.flatten(1, 2)
                            elif _e0g.dim() == 4:
                                _e0g = _e0g[:, 0]; _e1g = _e1g[:, 0]
                            _np_ = _e0g.shape[1] - _nt
                            _ev_g = torch.cat([_e0g[:, _np_:, :], _e1g[:, _np_:, :]], -1)
                            _ev_g = _ev_g.reshape(-1, _side, _side, _ev_g.shape[-1]).permute(0, 3, 1, 2)
                    l0 = self.xga_head0(s_in0, ev=_ev_g)
                    l1 = self.xga_head1(s_in1, ev=_ev_g)
                    blkL = self.image_encoder.blocks[-1].mlp.lin1
                    cache = getattr(blkL, '_xgl_cache', {})
                    e0, e1 = cache.get(0), cache.get(1)
                    if e0 is not None:
                        side = streams[0][0].shape[-1]
                        ntok = side * side
                        if e0.dim() == 4:
                            if e0.shape[-1] < 256:
                                e0 = e0.flatten(1, 2)
                                e1 = e1.flatten(1, 2)
                            else:
                                e0 = e0[:, 0]; e1 = e1[:, 0]
                        npre = e0.shape[1] - ntok
                        B_ = e0.shape[0]
                        ev = torch.cat([e0[:, npre:, :], e1[:, npre:, :]], -1)
                        ev = ev.reshape(B_, side, side, -1).permute(0, 3, 1, 2)
                        lp0 = F.interpolate(l0, size=(side, side),
                                            mode='bilinear').log_softmax(1)
                        lp1 = F.interpolate(l1, size=(side, side),
                                            mode='bilinear').log_softmax(1)
                        cf0 = lp0.max(1, keepdim=True).values
                        cf1 = lp1.max(1, keepdim=True).values
                        _aev = getattr(self, '_alpha_ev', '')
                        if _edl11:
                            af = torch.cat([ev_cons, ev_unc], dim=1)
                        elif _aev == 'log':
                            _lt = getattr(self, '_evd_log_taps', None) or []
                            lu0t, lu1t = _lt[-1]
                            if lu0t.dim() == 4:
                                lu0t = lu0t.flatten(1, 2)
                                lu1t = lu1t.flatten(1, 2)
                            _npf = lu0t.shape[1] - side * side
                            _Ku = lu0t.shape[-1]
                            lu0 = lu0t[:, _npf:, :] \
                                .reshape(B_, side, side, _Ku).permute(0, 3, 1, 2)
                            lu1 = lu1t[:, _npf:, :] \
                                .reshape(B_, side, side, _Ku).permute(0, 3, 1, 2)
                            af = torch.cat([lu0 + lu1, lu0 - lu1], dim=1)
                        elif _aev == '1':
                            af = ev
                        else:
                            af = torch.cat([ev, lp0, lp1, cf0, cf1], dim=1)
                        if xstats and (not _edl11 or os.environ.get('MFNET_EVD_XS') == '1'):
                            for st_ in xstats:
                                af = torch.cat([af, F.interpolate(st_, size=ev.shape[-2:], mode='bilinear')], dim=1)
                        if getattr(self, 'evd_ds16', False) and _edl11:
                            x = 0.5 * (l0 + l1)
                            _a0 = le0.exp(); _a1 = le1.exp()
                            _K16 = float(le0.shape[1])
                            _tr = 1.0 - (_K16 / _a0.sum(1, keepdim=True)) \
                                        * (_K16 / _a1.sum(1, keepdim=True))
                            a = F.interpolate(_tr, size=l0.shape[-2:],
                                              mode='bilinear', align_corners=False)
                        elif getattr(self, 'evd_ds14', False) and _edl11:
                            a0 = le0.exp()
                            a1 = le1.exp()
                            b12 = a0 * a1
                            b12 = b12 / b12.sum(1, keepdim=True).clamp_min(1e-8)
                            w14 = 1.0 - (b12.shape[1] / a0.sum(1, keepdim=True)) \
                                      * (b12.shape[1] / a1.sum(1, keepdim=True))
                            x = 0.5 * (l0 + l1)
                            if not getattr(self, 'evd_ds14_kill', False):
                                x = x + F.interpolate(
                                    w14 * b12.clamp_min(1e-8).log(),
                                    size=l0.shape[-2:], mode='bilinear',
                                    align_corners=False)
                            a = F.interpolate(w14, size=l0.shape[-2:],
                                              mode='bilinear', align_corners=False)
                        else:
                            a = torch.sigmoid(
                                (af * self.xgl_aw.view(1, -1, 1, 1)).sum(1, keepdim=True)
                                + self.xgl_ab)                    # [B,1,s,s]
                            a = F.interpolate(a, size=l0.shape[-2:],
                                              mode='bilinear', align_corners=False)
                            if getattr(self, 'evd_alpha_fix', False):
                                a = torch.full_like(a, 0.5)
                            x = (1.0 - a) * l0 + a * l1
                            if getattr(self, 'evd_fuse', False):
                                _fuse_in = ev_cons if _edl11 else ev
                                x = x + self.xgl_fuse(F.interpolate(
                                    _fuse_in, size=l0.shape[-2:], mode='bilinear',
                                    align_corners=False))
                        if self.training:
                            self._xga_lg = (l0, l1)
                        self._xga_ag = a
                        if getattr(self, 'evd_ds16', False):
                            self._xga_ag = None
                        if self.training and getattr(self, 'xga_mdrop', 0) > 0:
                            drop = (torch.rand(x.shape[0], 1, 1, 1,
                                               device=x.device) < self.xga_mdrop)
                            sel = (torch.rand(x.shape[0], 1, 1, 1,
                                              device=x.device) < 0.5).float()
                            solo = sel * l0 + (1 - sel) * l1
                            x = torch.where(drop, solo, x)
                        self._xga_map = a.detach()
                        self._xga_l = (l0.detach(), l1.detach())
                    else:
                        x = 0.5 * (l0 + l1)
                else:
                    x = self.light_head(feats)                # [B,cls,h/4,w/4]
            x = F.interpolate(x, size=(h, w), mode='bilinear',
                              align_corners=False)
            return x

        level_features = {0: [], 1: [], 2: [], 3: []}

        for mod_idx in range(min(len(deep_features), self.n_modalities)):
            for level in range(4):
                src = deep_features[mod_idx]
                f_in = src[level] if isinstance(src, (list, tuple)) else src
                f = self.fpn_branches[mod_idx][level](f_in)
                level_features[level].append(f)

        res1, w1 = self.fusions[0](level_features[0], return_weights=True)
        res2, w2 = self.fusions[1](level_features[1], return_weights=True)
        res3, w3 = self.fusions[2](level_features[2], return_weights=True)
        res4, w4 = self.fusions[3](level_features[3], return_weights=True)
        self._last_fusion_weights = [w1, w2, w3, w4]

        U_list = [fusion.get_U() for fusion in self.fusions]
        Phi_list = [fusion.get_Phi() for fusion in self.fusions]
        x = self.decoder(res1, res2, res3, res4, h, w, U_list=U_list, Phi_list=Phi_list)

        return x

    def per_class_modality_contribution(self, target, num_classes):
        """Fusion-based modality importance per class.

        Uses stored _last_fusion_weights from the most recent forward pass.
        Weights are softmax probabilities [B, N, H, W] — the model's own
        per-patch decisions about how much to trust each modality.

        Returns: dict[c] = [contrib_mod0, ..., contrib_modN-1]
        """
        if not hasattr(self, '_last_fusion_weights') or self._last_fusion_weights is None:
            return {}

        w = self._last_fusion_weights[0]  # [B, N, H_l, W_l]
        B, N, Hw, Ww = w.shape
        t = target.long()
        w_up = F.interpolate(w, size=t.shape[1:], mode='bilinear', align_corners=False)

        class_contrib = {}
        for c in range(num_classes):
            mask = (t == c)
            if mask.sum() > 0:
                for m in range(N):
                    contrib = w_up[:, m, :, :][mask].mean().item()
                    class_contrib.setdefault(c, []).append(contrib)
            else:
                class_contrib[c] = [0.0] * N

        return class_contrib

    def get_orthogonal_loss(self) -> torch.Tensor:
        """Parameter-space: mean ||A_i A_j^T||² across all MM layers."""
        loss = torch.tensor(0.0, device=next(self.parameters()).device)
        total = 0
        for mod in self.image_encoder.modules():
            name = type(mod).__name__
            if name in ('MMLinear', 'MMMergedLinear'):
                loss = loss + mod.get_orthogonal_loss()
                total += 1
        return loss / max(total, 1)

    def get_kl_loss(self) -> torch.Tensor:
        """KL regularizer — now returns zero (variational KL replaced by potential-field variance).

        The ARCHITECTURE.md framework replaces per-modality variational variance heads
        with the cross-modal potential field variance U = Var_Ω(Φ). This is computed
        in PatchConfidenceFusion and WF, not via KL divergence.
        """
        return torch.tensor(0.0, device=next(self.parameters()).device)
