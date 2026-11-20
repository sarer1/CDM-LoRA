"""DINOv3 backbone with paired low-rank adaptation and four feature taps."""
import math
import os
import sys
from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .mm_lora_block import MMAttention, MMMLPBlock

_DINOV3_REPO = '/root/autodl-tmp/dinov3'


def _import_dinov3_builder():
    if os.path.isdir(os.path.join(_DINOV3_REPO, 'dinov3')):
        if _DINOV3_REPO not in sys.path:
            sys.path.insert(0, _DINOV3_REPO)
        from dinov3.hub.backbones import dinov3_vitl16
        return dinov3_vitl16
    raise RuntimeError(
        f"DINOv3 repo not found at {_DINOV3_REPO} — clone "
        "facebookresearch/dinov3 there (code only, no weights needed).")


def build_rope_sincos(periods: torch.Tensor, H: int, W: int
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eval-mode RoPE (sin, cos) for an H×W patch grid — replicates
    dinov3/layers/rope_position_encoding.py with normalize_coords='separate'
    and no train-time shift/jitter/rescale."""
    dd = dict(device=periods.device, dtype=periods.dtype)
    coords_h = torch.arange(0.5, H, **dd) / H
    coords_w = torch.arange(0.5, W, **dd) / W
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'), -1)
    coords = coords.flatten(0, 1)                    # [HW, 2]
    coords = 2.0 * coords - 1.0
    angles = 2 * math.pi * coords[:, :, None] / periods[None, None, :]
    angles = angles.flatten(1, 2).tile(2)            # [HW, D]
    return torch.sin(angles), torch.cos(angles)


class MMDinoV3Block(nn.Module):
    """DINOv3 SelfAttentionBlock with MM-LoRA: LayerScale residual +
    RoPE-backed MMAttention (LoRA on Q,V) + MMMLPBlock (BC-LoRA quadratic
    terms + allocation gate live here — same MMLinear as the SAM encoder)."""

    def __init__(self, dim: int, num_heads: int, n_modalities: int = 2,
                 private_r: int = 4, shared_r: int = 2,
                 layerscale_init: float = 1e-5):
        super().__init__()
        self.n_modalities = n_modalities
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = MMAttention(dim, num_heads=num_heads,
                                n_modalities=n_modalities,
                                private_r=private_r, shared_r=shared_r,
                                qkv_bias=True, use_rel_pos=False)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.mlp = MMMLPBlock(embedding_dim=dim, mlp_dim=int(dim * 4),
                              n_modalities=n_modalities,
                              private_r=private_r, shared_r=shared_r,
                              act=nn.GELU)
        self.ls1 = nn.Parameter(layerscale_init * torch.ones(dim))
        self.ls2 = nn.Parameter(layerscale_init * torch.ones(dim))

    def forward(self, *modalities: torch.Tensor,
                rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                n_prefix_tokens: int = 0,
                ) -> Tuple[torch.Tensor, ...]:
        normed = [self.norm1(m) for m in modalities]
        attended = self.attn(*normed, rope=rope, n_prefix_tokens=n_prefix_tokens)
        x_attn = [x + self.ls1 * a for x, a in zip(modalities, attended)]
        mlp_out = self.mlp(*[self.norm2(t) for t in x_attn])
        return tuple(t + self.ls2 * m for t, m in zip(x_attn, mlp_out))


class DINOv3MMEncoder(nn.Module):
    """Frozen DINOv3 ViT with LoRA; returns four feature maps per modality."""

    TAPS = (5, 11, 17, 23)

    def __init__(self, n_modalities: int = 2, private_r: int = 4,
                 shared_r: int = 2, embed_dim: int = 1024, depth: int = 24,
                 num_heads: int = 16, n_storage_tokens: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.TAPS = (5, 11, 17, 23) if depth >= 24 else (2, 5, 8, 11)
        self.patch_size = 16
        self.n_prefix_tokens = 1 + n_storage_tokens
        self.patch_proj = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.storage_tokens = nn.Parameter(torch.zeros(1, n_storage_tokens, embed_dim))
        self.register_buffer('rope_periods',
                             torch.zeros(embed_dim // num_heads // 4),
                             persistent=True)
        self.blocks = nn.ModuleList([
            MMDinoV3Block(embed_dim, num_heads, n_modalities, private_r, shared_r)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-5)
        import os as _os
        self.ibi = _os.environ.get('MFNET_IBI') == '1'
        if self.ibi:
            self.IBI_DEPTHS = tuple(int(d) for d in
                                    _os.environ.get('MFNET_IBI_DEPTHS', '5,11,17').split(','))
            _r = int(_os.environ.get('MFNET_IBI_RANK', '64'))
            self.lora_ibu_U = nn.Parameter(torch.randn(_r, self.embed_dim) * 0.02)
            self.lora_ibu_V = nn.Parameter(torch.randn(_r, self.embed_dim) * 0.02)
            def _mk_inject():
                seq = nn.Sequential(
                    nn.Conv2d(2 * _r, 256, 1, bias=False),
                    nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, padding=1, bias=False),
                    nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                    nn.Conv2d(256, self.embed_dim, 1, bias=False))
                nn.init.zeros_(seq[-1].weight)
                return seq
            self.lora_ibu_W = nn.ModuleList([
                nn.ModuleList([_mk_inject(), _mk_inject()])
                for _ in self.IBI_DEPTHS])

        self._tap_cache = {}

    def _hook_tap(self, idx: int):
        def h(mod, inp, out):
            self._tap_cache[idx] = [o.reshape(o.shape[0], -1, self.embed_dim)
                                    for o in out]
        return h

    def load_from_dinov3(self, stock: nn.Module):
        """Copy frozen weights from a stock DINOv3 model (with or without a
        loaded checkpoint) into this MM structure. LoRA stays zero-init, so
        the encoder starts as the frozen backbone."""
        with torch.no_grad():
            self.patch_proj.weight.copy_(stock.patch_embed.proj.weight)
            self.patch_proj.bias.copy_(stock.patch_embed.proj.bias)
            self.cls_token.copy_(stock.cls_token)
            self.storage_tokens.copy_(stock.storage_tokens)
            self.rope_periods.copy_(stock.rope_embed.periods.reshape(-1))
            self.norm.weight.copy_(stock.norm.weight)
            self.norm.bias.copy_(stock.norm.bias)
            for i, src in enumerate(stock.blocks):
                dst = self.blocks[i]
                dst.norm1.weight.copy_(src.norm1.weight)
                dst.norm1.bias.copy_(src.norm1.bias)
                dst.norm2.weight.copy_(src.norm2.weight)
                dst.norm2.bias.copy_(src.norm2.bias)
                dst.ls1.copy_(src.ls1.gamma if hasattr(src.ls1, 'gamma')
                              else src.ls1.weight)
                dst.ls2.copy_(src.ls2.gamma if hasattr(src.ls2, 'gamma')
                              else src.ls2.weight)
                dst.attn.qkv.weight.copy_(src.attn.qkv.weight)
                bias = src.attn.qkv.bias
                mask = getattr(src.attn.qkv, 'bias_mask', None)
                dst.attn.qkv.bias.copy_(bias if mask is None else bias * mask)
                dst.attn.proj.weight.copy_(src.attn.proj.weight)
                dst.attn.proj.bias.copy_(src.attn.proj.bias)
                dst.mlp.lin1.weight.copy_(src.mlp.fc1.weight)
                dst.mlp.lin1.bias.copy_(src.mlp.fc1.bias)
                dst.mlp.lin2.weight.copy_(src.mlp.fc2.weight)
                dst.mlp.lin2.bias.copy_(src.mlp.fc2.bias)

    def forward(self, modalities: List[torch.Tensor]) -> List[List[torch.Tensor]]:
        B, _, H, W = modalities[0].shape
        pH, pW = H // self.patch_size, W // self.patch_size
        sin, cos = build_rope_sincos(self.rope_periods, pH, pW)
        rope = (sin, cos)
        npre = self.n_prefix_tokens

        embs = []
        for mod in modalities:
            patch = self.patch_proj(mod).flatten(2).transpose(1, 2)   # [B,HW,D]
            prefix = torch.cat([self.cls_token, self.storage_tokens], 1)  # [1,5,D]
            seq = torch.cat([prefix.expand(B, -1, -1), patch], dim=1)     # [B,5+HW,D]
            embs.append(seq.reshape(B, 1, npre + pH * pW, self.embed_dim))

        handles = [self.blocks[t].register_forward_hook(self._hook_tap(t))
                   for t in self.TAPS]
        for bi, blk in enumerate(self.blocks):
            embs = list(blk(*embs, rope=rope, n_prefix_tokens=npre))
            if getattr(self, 'ibi', False) and bi in self.IBI_DEPTHS:
                ti = self.IBI_DEPTHS.index(bi)
                f0 = embs[0][:, 0, npre:, :]
                f1 = embs[1][:, 0, npre:, :]
                p0 = torch.einsum('rd,bnd->bnr', self.lora_ibu_U, f0)
                p1 = torch.einsum('rd,bnd->bnr', self.lora_ibu_V, f1)
                j = torch.cat([p0 * p1, p0 - p1], dim=-1)     # [B,HW,2r]
                jm = j.reshape(B, pH, pW, j.shape[-1]).permute(0, 3, 1, 2)
                for v in range(len(embs)):
                    add = self.lora_ibu_W[ti][v](jm)          # [B,D,h,w]
                    add = add.flatten(2).transpose(1, 2)      # [B,HW,D]
                    prefix = embs[v][:, 0, :npre, :]
                    patched = embs[v][:, 0, npre:, :] + add
                    embs[v] = torch.cat([prefix, patched], dim=1) \
                        .reshape(B, 1, npre + pH * pW, self.embed_dim)
        for h in handles:
            h.remove()

        outputs = []
        for m in range(len(modalities)):
            maps = []
            for t in self.TAPS:
                tok = self.norm(self._tap_cache[t][m])
                feat = tok[:, npre:, :].reshape(B, pH, pW, self.embed_dim)
                maps.append(feat.permute(0, 3, 1, 2))
            outputs.append(maps)
        return outputs

    def get_orthogonal_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.rope_periods.device)


def build_dinov3_mm_encoder(size: str = 'large', n_modalities: int = 2,
                            private_r: int = 4, shared_r: int = 2,
                            checkpoint: Optional[str] = None):
    """Build the DINOv3 MM encoder; weights auto-discovered under weights/."""
    # Load the official DINOv3 package before constructing the encoder.
    _import_dinov3_builder()
    import dinov3.hub.backbones as _hub
    from dinov3.hub.backbones import Weights
    sat_variant = checkpoint is not None and (
        'sat493m' in os.path.basename(checkpoint) or 'eadcf0ff' in os.path.basename(checkpoint))
    _CFG = {'small': ('dinov3_vits16', 384, 12, 6),
            'base':  ('dinov3_vitb16', 768, 12, 12),
            'large': ('dinov3_vitl16', 1024, 24, 16)}
    _fn, _dim, _dep, _heads = _CFG[size]
    stock = getattr(_hub, _fn)(
        pretrained=False,
        weights=(Weights.SAT493M if sat_variant else Weights.LVD1689M))

    if checkpoint is not None and os.path.isfile(checkpoint):
        sd = torch.load(checkpoint, map_location='cpu', weights_only=True)
        stock.load_state_dict(sd, strict=True)
        print(f"Loaded DINOv3 {size} checkpoint: {checkpoint}")
    elif not os.environ.get('MFNET_DINO3_RANDOM'):
        raise FileNotFoundError(
            "DINOv3 checkpoint not found. Released weights are gated — request "
            "them at https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/ "
            f"and place the .pth at {checkpoint or 'weights/'} "
            "(set MFNET_DINO3_RANDOM=1 only for shape tests).")

    enc = DINOv3MMEncoder(n_modalities=n_modalities, private_r=private_r,
                          shared_r=shared_r, embed_dim=_dim, depth=_dep,
                          num_heads=_heads)
    enc.load_from_dinov3(stock)
    return enc
