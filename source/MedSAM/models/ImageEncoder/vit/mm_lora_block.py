"""Multi-Modal LoRA Transformer Block for N modalities.

Extends the LoraBlock design to support an arbitrary number of modalities
with shared + private LoRA and orthogonal regularization.
"""

from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...common import loralib as lora


def _bc_coupling_default():
    """Read the BC-LoRA coupling switch from utils.py's global BC_COUPLING.

    Falls back to True (full BC-LoRA) if utils is unavailable. Train.py does
    `from utils import *` before building the model, so utils is already in
    sys.modules and this is a cheap cached lookup with no import side-effects.
    """
    try:
        from utils import BC_COUPLING
        return bool(BC_COUPLING)
    except Exception:
        return True


def rope_rotate(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor,
                prefix: int = 0) -> torch.Tensor:
    """Apply rotary position embedding to [B*heads, N, head_dim] tokens.

    sin/cos: [N-prefix, head_dim] for the PATCH tokens (DINOv3 convention —
    half-dim paired rotation, matches official rope_apply). The leading
    `prefix` tokens (CLS + storage/registers) are left un-rotated, exactly
    like facebookresearch/dinov3 layers/attention.py:apply_rope."""
    if prefix == 0:
        sin, cos = sin.to(x.dtype), cos.to(x.dtype)
        return (x * cos) + (rope_rotate_half(x) * sin)
    head = x[:, :prefix, :]
    tail = rope_rotate(x[:, prefix:, :], sin, cos)
    return torch.cat((head, tail), dim=1)


def rope_rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class MMLoraBlock(nn.Module):
    """Transformer block with MM-LoRA for N modalities.

    Structure (per modality in parallel):
        x_i = x_i + attention_i(norm1(x_i), norm1(x_j for j!=i))
        x_i = x_i + mlp_i(norm2(x_i))

    Attention uses MMMergedLinear (LoRA on Q, V).
    MLP uses MMLinear (LoRA on lin1, lin2).
    """

    def __init__(
        self,
        args,
        dim: int,
        num_heads: int,
        n_modalities: int = 2,
        private_r: int = 4,
        shared_r: int = 2,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.n_modalities = n_modalities
        self.window_size = window_size

        self.norm1 = norm_layer(dim)

        self.attn = MMAttention(
            dim,
            num_heads=num_heads,
            n_modalities=n_modalities,
            private_r=private_r,
            shared_r=shared_r,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=(64, 64) if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MMMLPBlock(
            embedding_dim=dim,
            mlp_dim=int(dim * mlp_ratio),
            n_modalities=n_modalities,
            private_r=private_r,
            shared_r=shared_r,
            act=act_layer,
        )

    def forward(self, *modalities: torch.Tensor,
                rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, ...]:
        """Forward pass for N modalities.

        Args:
            *modalities: N tensors of shape [B, H, W, C]
            rope: optional (sin, cos) [N_tokens, head_dim] for RoPE-backed
                backbones (DINOv3); None for SAM's rel-pos / none.

        Returns:
            Tuple of N tensors of shape [B, H, W, C]
        """
        shortcuts = modalities
        normed = [self.norm1(m) for m in modalities]

        if self.window_size > 0:
            H, W = normed[0].shape[1], normed[0].shape[2]
            padded = []
            pad_hws = []
            for m in normed:
                m_pad, pad_hw = window_partition(m, self.window_size)
                padded.append(m_pad)
                pad_hws.append(pad_hw)
            normed = padded

        attended = self.attn(*normed, rope=rope)

        if self.window_size > 0:
            unpartitioned = []
            for i, (att, (ph, pw)) in enumerate(zip(attended, pad_hws)):
                att = window_unpartition(att, self.window_size, (ph, pw), (H, W))
                unpartitioned.append(att)
            attended = unpartitioned

        mlp_inputs = [self.norm2(shortcuts[i] + attended[i]) for i in range(self.n_modalities)]
        mlp_outputs = self.mlp(*mlp_inputs)
        outputs = [shortcuts[i] + attended[i] + mlp_outputs[i] for i in range(self.n_modalities)]

        return tuple(outputs)



class MMMLPBlock(nn.Module):
    """MLP block with MM-LoRA for N modalities, supporting Δ-gated routing."""

    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        n_modalities: int = 2,
        private_r: int = 4,
        shared_r: int = 2,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.n_modalities = n_modalities
        bc = _bc_coupling_default()
        self.lin1 = lora.MMLinear(embedding_dim, mlp_dim,
                                  n_modalities=n_modalities,
                                  private_r=private_r,
                                  shared_r=shared_r,
                                  bc_coupling=bc)
        self.lin2 = lora.MMLinear(mlp_dim, embedding_dim,
                                  n_modalities=n_modalities,
                                  private_r=private_r,
                                  shared_r=shared_r,
                                  bc_coupling=bc)
        self.act = act()

    def forward(self, *x_list: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        outputs = self.lin1(list(x_list))
        outputs = [self.act(o) for o in outputs]
        outputs = self.lin2(outputs)
        return tuple(outputs)



class MMAttention(nn.Module):
    """Multi-head Attention with MM-LoRA for N modalities.

    Uses MMMergedLinear for QKV projection (LoRA on Q, V only).
    Each modality has independent Q, K, V but shares attention pattern.

    Optional RoPE (rotary position embedding): pass `rope=(sin, cos)` of shape
    [N_tokens, head_dim] to forward — applied to q/k only, patch-grid tokens
    (no CLS prefix; used by the DINOv3 encoder whose backbone is RoPE-based).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        n_modalities: int = 2,
        private_r: int = 4,
        shared_r: int = 2,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.n_modalities = n_modalities
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = lora.MMMergedLinear(
            dim, dim * 3,
            n_modalities=n_modalities,
            private_r=private_r,
            shared_r=shared_r,
            bias=qkv_bias,
            enable_lora=[True, False, True],
        )
        self.proj = nn.Linear(dim, dim)

        self.use_xab = __import__('os').environ.get('MFNET_XAB') == '1'
        if self.use_xab and n_modalities == 2:
            _r = 8
            self.xab_r = _r
            self.xgl_aq = nn.ParameterList(
                [nn.Parameter(torch.randn(_r, dim) * 0.02) for _ in range(2)])
            self.xgl_ak = nn.ParameterList(
                [nn.Parameter(torch.randn(_r, dim) * 0.02) for _ in range(2)])
            self.xgl_xr = nn.Parameter(torch.zeros(2, self.num_heads))

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, \
                "Input size must be provided if using relative positional encoding."
            self.rel_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, *x_list: torch.Tensor,
                rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                n_prefix_tokens: int = 0,
                ) -> Tuple[torch.Tensor, ...]:
        """Forward for N modalities, each producing its own attended output.

        n_prefix_tokens: leading tokens (CLS+storage) carried by some
        backbones; they are excluded from RoPE (patch grid only)."""
        B, H, W, C = x_list[0].shape

        flat_list = [x.reshape(B, H * W, C) for x in x_list]

        qkv_list = self.qkv(flat_list)

        if getattr(self, 'use_xab', False) and self.n_modalities == 2:
            zq_x = [f @ self.xgl_aq[m].T for m, f in enumerate(flat_list)]
            zk_x = [f @ self.xgl_ak[m].T for m, f in enumerate(flat_list)]

        outputs = []
        for mi, qkv in enumerate(qkv_list):
            qkv = qkv.reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

            if rope is not None:
                q = rope_rotate(q, *rope, prefix=n_prefix_tokens)
                k = rope_rotate(k, *rope, prefix=n_prefix_tokens)

            attn = (q * self.scale) @ k.transpose(-2, -1)

            if self.use_rel_pos:
                attn = add_decomposed_rel_pos(attn, q, self.rel_h, self.rel_w, (H, W), (H, W))

            if getattr(self, 'use_xab', False) and self.n_modalities == 2:
                bias = torch.einsum('bir,bjr->bij', zq_x[mi], zk_x[1 - mi])
                bias = bias / (self.xab_r ** 0.5)
                bias = bias[:, None] * self.xgl_xr[mi].view(1, self.num_heads, 1, 1)
                attn = attn + bias.reshape(B * self.num_heads, H * W, H * W)

            attn = attn.softmax(dim=-1)
            x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
            x = self.proj(x)
            outputs.append(x)

        return tuple(outputs)




def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)
    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)
    return attn
