#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import math
import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer():
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.merged = False
        self.merge_weights = merge_weights


class Embedding(nn.Embedding, LoRALayer):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0,
                           merge_weights=merge_weights)
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if self.merge_weights and self.merged:
            if self.r > 0:
                self.weight.data -= (self.lora_B @ self.lora_A).T * self.scaling
            self.merged = False

    def eval(self):
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            if self.r > 0:
                self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            if self.r > 0:
                after_A = F.embedding(
                    x, self.lora_A.T, self.padding_idx, self.max_norm,
                    self.norm_type, self.scale_grad_by_freq, self.sparse
                )
                result += (after_A @ self.lora_B.T) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)


class Linear(nn.Linear, LoRALayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        if r > 0:
            self.lora_Ax = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_Ay = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_Bx = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.lora_By = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
        self.reset_parameters()
        self.lora_wx = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.lora_wy = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.lora_wx.data.fill_(0.5)
        self.lora_wy.data.fill_(0.5)
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_Ax'):
            nn.init.kaiming_uniform_(self.lora_Ax, a=math.sqrt(5))
            nn.init.zeros_(self.lora_Bx)
        if hasattr(self, 'lora_Ay'):
            nn.init.kaiming_uniform_(self.lora_Ay, a=math.sqrt(5))
            nn.init.zeros_(self.lora_By)

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)

    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            resultx = F.linear(x, T(self.weight), bias=self.bias)
            resulty = F.linear(y, T(self.weight), bias=self.bias)
            if self.r > 0:
                lorax = self.lora_dropout(x) @ self.lora_Ax.T @ self.lora_Bx.T
                loray = self.lora_dropout(y) @ self.lora_Ay.T @ self.lora_By.T
                resultx += self.lora_wx * lorax + (1 - self.lora_wx) * loray
                resulty += self.lora_wy * loray + (1 - self.lora_wy) * lorax

            return resultx, resulty
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

class MergedLinear(nn.Linear, LoRALayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        if r > 0 and any(enable_lora):
            self.lora_Ax = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_Bx = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            )
            self.lora_Ay = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_By = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            )
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
            self.lora_ind = self.weight.new_zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_Ax'):
            nn.init.kaiming_uniform_(self.lora_Ax, a=math.sqrt(5))
            nn.init.zeros_(self.lora_Bx)
        if hasattr(self, 'lora_Ay'):
            nn.init.kaiming_uniform_(self.lora_Ay, a=math.sqrt(5))
            nn.init.zeros_(self.lora_By)

    def zero_pad(self, x):
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(
            -1, self.out_features // len(self.enable_lora) * sum(self.enable_lora)
        )
        return result.view((*x.shape[:-1], self.out_features))

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)

    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            resultx = F.linear(x, T(self.weight), bias=self.bias)
            resulty = F.linear(y, T(self.weight), bias=self.bias)
            if self.r > 0:
                after_Ax = F.linear(self.lora_dropout(x), self.lora_Ax)
                after_Bx = F.conv1d(
                    after_Ax.transpose(-2, -1),
                    self.lora_Bx.unsqueeze(-1),
                    groups=sum(self.enable_lora)
                ).transpose(-2, -1)
                resultx += self.zero_pad(after_Bx) * self.scaling

                after_Ay = F.linear(self.lora_dropout(y), self.lora_Ay)
                after_By = F.conv1d(
                    after_Ay.transpose(-2, -1),
                    self.lora_By.unsqueeze(-1),
                    groups=sum(self.enable_lora)
                ).transpose(-2, -1)
                resulty += self.zero_pad(after_By) * self.scaling
            return resultx, resulty

class Conv2d(nn.Conv2d, LoRALayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert type(kernel_size) is int
        if r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r*kernel_size, in_channels*kernel_size))
            )
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_channels*kernel_size, r*kernel_size))
            )
            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Conv2d.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Conv2d.train(self, mode)
        if self.merge_weights and self.merged:
            self.weight.data -= (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
            self.merged = False

    def eval(self):
        nn.Conv2d.eval(self)
        if self.merge_weights and not self.merged:
            self.weight.data += (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            return F.conv2d(
                x,
                self.weight + (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling,
                self.bias, self.stride, self.padding, self.dilation, self.groups
            )
        return nn.Conv2d.forward(self, x)



class MMLinear(nn.Linear, LoRALayer):
    _SHARED_E = None
    _SHARED_PSI = None
    _SHARED_M = None
    _SHARED_AT = None
    _SHARED_WC = None
    _SHARED_PSI_T = None
    """Bilinear Coupled LoRA (BC-LoRA) for multi-modal adaptation.

    Breaks the *joint* linearity of standard LoRA -- the one constraint a
    single-modality LoRA can never touch -- by adding a cross-modal
    second-order term inside the low-rank bottleneck:

        z_i = A_i x_i                                  (private projection)
        dW.x = B_x . sigma(z_x) + B_c . (z_x (o) z_y)
        dW.y = B_y . sigma(z_y) + B_c . (z_x (o) z_y)

    - sigma(z_i): non-linear private branch, breaks linearity in x_i
      (GELU -- the same nonlinearity that makes Adapter beat LoRA in MFNet).
    - z_x (o) z_y: Hadamard coupling, breaks joint linearity in (x, y). It is
      a bilinear (2nd-order) term that a single-modality LoRA cannot form, and
      expresses "AND" semantics (RGB texture AND DSM height) that the additive
      fusion of MFNet (lambda1 z_x + (1-lambda1) z_y) cannot.

    A shared B_c reads the symmetric coupling (z_x (o) z_y == z_y (o) z_x);
    the private branches keep per-modality asymmetry. With coupling disabled
    (bc_coupling=False) this degenerates to non-linear LoRA, and with sigma
    removed it degenerates to standard LoRA.
    """

    DEFAULT_RANK = int(os.environ.get('MFNET_RANK', '8'))

    def __init__(self, in_features, out_features, n_modalities=2,
                 private_r=4, shared_r=2,
                 lora_alpha=8, lora_dropout=0.,
                 self_dropout=0.1,
                 fan_in_fan_out=False, merge_weights=True,
                 bc_coupling=True, **kwargs):
        self.n_modalities = n_modalities
        self.fan_in_fan_out = fan_in_fan_out
        self.bc_coupling = bc_coupling
        _terms = {t.strip() for t in
                  os.environ.get('MFNET_TERMS', 'priv,self,cross').split(',') if t.strip()}
        self._use_priv = 'priv' in _terms
        self._use_cross = bc_coupling and 'cross' in _terms
        self._use_cons = bc_coupling and 'cons' in _terms
        self._use_self = bc_coupling and 'self' in _terms
        self._use_cube = bc_coupling and 'cube' in _terms
        self._use_mix = bc_coupling and 'mix' in _terms
        self._use_xattn = bc_coupling and 'xattn' in _terms
        self._use_logic = bc_coupling and 'logic' in _terms
        self._use_ptl = bc_coupling and 'ptl' in _terms
        self._use_lcx = bc_coupling and 'lcx' in _terms
        self._use_asbi = bc_coupling and 'asbi' in _terms
        self._use_poe = bc_coupling and 'poe' in _terms
        self._use_ev = bc_coupling and 'ev' in _terms
        self._use_xgl = bc_coupling and 'xgl' in _terms
        self._use_xgb = bc_coupling and 'xgb' in _terms
        self.coupling_mode = os.environ.get('MFNET_COUPLING', 'bilinear')
        if self.coupling_mode not in ('bilinear', 'add'):
            raise ValueError(f"unknown MFNET_COUPLING={self.coupling_mode!r}")
        self.record_heatmaps = False
        self._heatmaps = {}
        self._ablate_cross = False
        self._ablate_self = False
        self._ablate_priv = False
        R = self.DEFAULT_RANK
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=R, lora_alpha=lora_alpha,
                           lora_dropout=lora_dropout, merge_weights=merge_weights)
        self.scaling = self.lora_alpha / self.r
        self.self_dropout = nn.Dropout(self_dropout) if self_dropout > 0 else nn.Identity()
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        self.lora_A_list = nn.ParameterList([
            nn.Parameter(torch.zeros((R, in_features))) for _ in range(n_modalities)
        ])
        self.lora_B_list = nn.ParameterList([
            nn.Parameter(torch.zeros((out_features, R))) for _ in range(n_modalities)
        ])

        self.lora_Bc = nn.Parameter(torch.zeros((out_features, R))) \
            if (self._use_cross or self._use_cons) else None
        self.lora_Bs_list = nn.ParameterList([
            nn.Parameter(torch.zeros((out_features, R))) for _ in range(n_modalities)
        ]) if self._use_self else None
        self.lora_Bs3_list = nn.ParameterList([
            nn.Parameter(torch.zeros((out_features, R))) for _ in range(n_modalities)
        ]) if self._use_cube else None
        self.lora_Bm3_list = nn.ParameterList([
            nn.Parameter(torch.zeros((out_features, R))) for _ in range(n_modalities)
        ]) if self._use_mix else None
        if self._use_xattn:
            self.xa_Wq = nn.Parameter(torch.randn(R, R) * 0.02)
            self.xa_Wk = nn.Parameter(torch.randn(R, R) * 0.02)
            self.xa_Wv = nn.Parameter(torch.randn(R, R) * 0.02)
            self.lora_Bxa_list = nn.ParameterList([
                nn.Parameter(torch.zeros((out_features, R))) for _ in range(n_modalities)
            ])
        else:
            self.xa_Wq = self.xa_Wk = self.xa_Wv = None
            self.lora_Bxa_list = None
        if self._use_logic:
            self.le_W_list = nn.ParameterList([
                nn.Parameter(torch.randn(R, R) * 0.02) for _ in range(n_modalities)
            ])
            self.le_b_list = nn.ParameterList([
                nn.Parameter(torch.zeros(R)) for _ in range(n_modalities)
            ])
            self.lora_Band = nn.Parameter(torch.zeros((out_features, R)))
            self.lora_Bor = nn.Parameter(torch.zeros((out_features, R)))
            self.lora_Bxor = nn.Parameter(torch.zeros((out_features, R)))
        else:
            self.le_W_list = self.le_b_list = None
            self.lora_Band = self.lora_Bor = self.lora_Bxor = None
        if self._use_ptl or self._use_asbi or self._use_poe or self._use_ev:
            import os as _os
            _C = int(_os.environ.get('MFNET_PT_CLASSES', '6'))
            self.ptl_C = _C
            _D = int(_os.environ.get('MFNET_PA_DIM', '64'))
            if self._use_ptl or self._use_asbi:
                if MMLinear._SHARED_E is None:
                    MMLinear._SHARED_E = nn.Parameter(torch.randn(_C, _D) * 0.02)
                    MMLinear._SHARED_PSI = nn.ParameterList([
                        nn.Parameter(torch.randn(_D, R) * 0.02) for _ in range(n_modalities)
                    ])
                self.pa_E = MMLinear._SHARED_E
                self.pa_psi_list = MMLinear._SHARED_PSI
            else:
                self.pa_E = None
                self.pa_psi_list = None
            if self._use_poe:
                if MMLinear._SHARED_M is None:
                    MMLinear._SHARED_M = nn.Parameter(torch.randn(_C, R) * 0.02)
                self.pa_M = MMLinear._SHARED_M
            else:
                self.pa_M = None
            if self._use_ev:
                self.lora_Be = nn.Parameter(
                    torch.zeros((out_features, self.ptl_C)))
            else:
                self.lora_Be = None

            self._at_decouple = _os.environ.get('MFNET_AT_DECOUPLE') == '1'
            if self._at_decouple:
                _Rt = int(_os.environ.get('MFNET_AT_RANK', '16'))
                if MMLinear._SHARED_AT is None:
                    MMLinear._SHARED_AT = {}
                    MMLinear._SHARED_PSI_T = nn.ParameterList([
                        nn.Parameter(torch.randn(_D, _Rt) * 0.02)
                        for _ in range(n_modalities)])
                if in_features not in MMLinear._SHARED_AT:
                    MMLinear._SHARED_AT[in_features] = nn.ParameterList([
                        nn.Parameter(torch.randn(_Rt, in_features) * 0.02)
                        for _ in range(n_modalities)])
                self.pa_At_list = MMLinear._SHARED_AT[in_features]
                self.pa_psi_t_list = MMLinear._SHARED_PSI_T
            else:
                self.pa_At_list = None
                self.pa_psi_t_list = None
            self.pa_tau = 0.1
            _eps = float(_os.environ.get('MFNET_PT_EPS', '0'))
            _init = (torch.randn((out_features, R)) * _eps
                     if _eps > 0 else torch.zeros((out_features, R)))
            self.lora_Bt = nn.Parameter(_init)
        else:
            self.pa_E = None
            self.pa_psi_list = None
            self.pa_M = None
            self.lora_Be = None
            self.lora_Bt = None
        if self._use_xgl:
            self._use_evd = os.environ.get('MFNET_EVD') == '1'
            self._use_evd_edl = os.environ.get('MFNET_EVD_EDL') == '1'
            if self._use_evd:
                _K = int(os.environ.get('MFNET_EVD_K', '6'))
                if MMLinear._SHARED_WC is None or \
                        MMLinear._SHARED_WC.shape[0] != _K:
                    MMLinear._SHARED_WC = nn.Parameter(torch.zeros(_K, R))
                self.register_parameter('evd_wc_shared', MMLinear._SHARED_WC)
                self.evd_K = _K
                self.xgl_Wx = None
                self._use_ds16 = os.environ.get('MFNET_EVD_DS16') == '1'
                self._ds16_gconst = os.environ.get('MFNET_EVD16_GCONST') == '1'
                if self._use_ds16:
                    self.xgl_Wx = nn.Parameter(torch.eye(R) * 0.1)
                    self.xgl_Bx = nn.Parameter(torch.zeros(out_features, 2 * R))
                self._use_evd_v3 = os.environ.get('MFNET_EVD_V3') == '1'
                if self._use_evd_v3:
                    self.xgl_Be = nn.Parameter(torch.zeros(out_features, _K))
                    if os.environ.get('MFNET_EVD_NOWRITE') == '1':
                        self.xgl_Be.requires_grad = False
                else:
                    self.xgl_Be = nn.Parameter(torch.zeros(out_features, 2 * _K))
            else:
                self.xgl_Wx = nn.Parameter(torch.eye(R) * 0.1)
                self.xgl_Be = nn.Parameter(torch.zeros(out_features, 2 * R))
        else:
            self._use_evd = False
            self.xgl_Wx = None
            self.xgl_Be = None
        if self._use_xgb:
            self.xgl_bx = nn.Parameter(torch.eye(R) * 0.1)
            self.xgl_bg = nn.Parameter(torch.zeros(2 * R, R))
        self._use_xmb = bc_coupling and 'xmb' in _terms
        if self._use_xmb:
            self.xgl_mb0 = nn.Parameter(torch.zeros(R, R))
            self.xgl_mb1 = nn.Parameter(torch.zeros(R, R))
        self._use_xgc = bc_coupling and 'xgc' in _terms and self._use_xgl
        if self._use_xgc:
            self.xgl_xa = nn.Parameter(torch.zeros(R))
        self.lcx_M = nn.Parameter(torch.zeros(R, R)) if self._use_lcx else None

        self.alloc_coupling = (bc_coupling and self._use_self and self._use_cross
                               and os.environ.get('MFNET_ALLOC') == '1')
        if os.environ.get('MFNET_ALLOC') == '1' and not self.alloc_coupling:
            raise ValueError("MFNET_ALLOC=1 requires both quadratic terms "
                             "(MFNET_TERMS='priv,self,cross')")
        self.alloc_active = True
        if self.alloc_coupling:
            self.alloc_w = nn.Parameter(torch.zeros(2, 2))  # [term, feat(φ,φ²)]
            self.alloc_b = nn.Parameter(torch.zeros(2))

        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
        self.reset_parameters()

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if not hasattr(self, 'lora_A_list'):
            return
        for i in range(self.n_modalities):
            nn.init.kaiming_uniform_(self.lora_A_list[i], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_list[i])
        if self.lora_Bc is not None:
            nn.init.zeros_(self.lora_Bc)
        if self.lora_Bs_list is not None:
            for i in range(self.n_modalities):
                nn.init.zeros_(self.lora_Bs_list[i])

    def _T(self, w): return w.T if self.fan_in_fan_out else w
    def train(self, m=True): nn.Linear.train(self, m)
    def eval(self): nn.Linear.eval(self)

    def forward(self, x_list: list) -> list:
        z_list = [
            self.lora_dropout(x) @ self.lora_A_list[i].T
            for i, x in enumerate(x_list)
        ]

        if self._use_xgb and self.n_modalities == 2:
            zs = (z_list[0], z_list[1])
            for i in range(2):
                zj = zs[1 - i] @ self.xgl_bx.T
                e = torch.cat([zs[i] * zj, (zs[i] - zj).abs()], dim=-1)
                z_list[i] = zs[i] * (1.0 + torch.tanh(e @ self.xgl_bg))
                if not hasattr(self, '_xgb_cache'):
                    self._xgb_cache = {}
                self._xgb_cache[i] = e

        if self._use_xmb and self.n_modalities == 2:
            zs = (z_list[0], z_list[1])
            z_list[0] = zs[0] + zs[1] @ self.xgl_mb0.T
            z_list[1] = zs[1] + zs[0] @ self.xgl_mb1.T

        if self._use_xgc and self.n_modalities == 2:
            d = z_list[0] - z_list[1] @ self.xgl_Wx.T
            self._xgc_d = d.detach()
            z_list[0] = z_list[0] + self.xgl_xa * d
            z_list[1] = z_list[1] - self.xgl_xa * d

        results = []
        for i, x in enumerate(x_list):
            result = F.linear(x, self._T(self.weight), bias=self.bias)

            h_priv = z_list[i] @ self.lora_B_list[i].T \
                if self._use_priv and not self._ablate_priv \
                else torch.zeros_like(z_list[i] @ self.lora_B_list[i].T)

            if (self._use_cross or self._use_self or self._use_cube
                    or self._use_mix or self._use_xattn
                    or self._use_logic or self._use_ptl
                    or self._use_lcx or self._use_asbi
                    or self._use_poe or self._use_ev
                    or self._use_xgl) and self.n_modalities >= 2:
                if self._use_cross and self.coupling_mode == 'add':
                    c = sum(z_list) / self.n_modalities
                else:
                    c = sum(z_list[i] * z_list[j]
                            for j in range(self.n_modalities) if j != i)
                s = z_list[i] * z_list[i]

                gated = False
                phi = g = None
                if self.alloc_coupling and self.alloc_active and self.n_modalities == 2:
                    phi = F.cosine_similarity(z_list[i], z_list[1 - i], dim=-1)
                    phi = phi.detach()
                    feat = torch.stack([phi, phi * phi], dim=-1)               # [B,N,2]
                    logits = 0.25 * torch.tanh(feat @ self.alloc_w.T + self.alloc_b)
                    g = 2.0 * torch.softmax(logits, dim=-1)
                    c = c * g[..., 1:2]
                    s = s * g[..., 0:1]
                    gated = True
                elif self.record_heatmaps:
                    phi = F.cosine_similarity(z_list[i], z_list[1 - i],
                                              dim=-1).detach()

                h_cross = c @ self.lora_Bc.T \
                    if self._use_cross and not self._ablate_cross \
                    else torch.zeros_like(h_priv)

                h_self = self.self_dropout(s) @ self.lora_Bs_list[i].T \
                    if self._use_self and not self._ablate_self \
                    else torch.zeros_like(h_priv)

                h_cube = (s * z_list[i]) @ self.lora_Bs3_list[i].T \
                    if self._use_cube else torch.zeros_like(h_priv)
                zij = z_list[i] * z_list[1 - i] if self.n_modalities == 2 else None
                h_mix = (s * z_list[1 - i] + zij * z_list[1 - i]) @ self.lora_Bm3_list[i].T \
                    if self._use_mix and self.n_modalities == 2 \
                    else torch.zeros_like(h_priv)

                if self._use_xattn and self.n_modalities == 2:
                    q = z_list[i] @ self.xa_Wq                       # [B,N,R]
                    k = z_list[1 - i] @ self.xa_Wk                   # [B,N,R]
                    v = z_list[1 - i] @ self.xa_Wv                   # [B,N,R]
                    att = torch.softmax(q @ k.transpose(-2, -1) / (self.r ** 0.5),
                                        dim=-1)                     # [B,N,N]
                    h_xattn = (att @ v) @ self.lora_Bxa_list[i].T
                else:
                    h_xattn = torch.zeros_like(h_priv)

                if self._use_logic and self.n_modalities == 2:
                    u_i = torch.sigmoid(z_list[i] @ self.le_W_list[i] + self.le_b_list[i])
                    u_j = torch.sigmoid(z_list[1 - i] @ self.le_W_list[1 - i]
                                        + self.le_b_list[1 - i])
                    e_and = u_i * u_j
                    e_or = u_i + u_j - e_and
                    e_xor = (u_i - u_j).abs()
                    h_logic = (e_and @ self.lora_Band.T
                               + e_or @ self.lora_Bor.T
                               + e_xor @ self.lora_Bxor.T)
                else:
                    h_logic = torch.zeros_like(h_priv)

                if self._use_ptl and self.n_modalities == 2:
                    e_j = z_list[1 - i] @ self.pa_psi_list[1 - i].T
                    b_j = torch.softmax(e_j @ self.pa_E.T / self.pa_tau, dim=-1)
                    t_i = (b_j @ self.pa_E) @ self.pa_psi_list[i]
                    h_ptl = t_i @ self.lora_Bt.T
                else:
                    h_ptl = torch.zeros_like(h_priv)

                if self._use_lcx and self.n_modalities == 2:
                    h_lcx = (z_list[1 - i] @ self.lcx_M.T) @ self.lora_B_list[i].T
                else:
                    h_lcx = torch.zeros_like(h_priv)

                if self._use_asbi and self.n_modalities == 2:
                    e_i = z_list[i] @ self.pa_psi_list[i].T          # [B,N,D]
                    e_j = z_list[1 - i] @ self.pa_psi_list[1 - i].T  # [B,N,D]
                    x = e_i * e_j
                    t_i = x @ self.pa_psi_list[i]
                    h_asbi = t_i @ self.lora_Bt.T
                    if self._at_decouple:
                        eb_j = (x_list[1 - i] @ self.pa_At_list[1 - i].T) \
                            @ self.pa_psi_t_list[1 - i].T          # [B,N,D]
                    else:
                        eb_j = e_j
                    bel_j = torch.softmax(
                        eb_j @ self.pa_E.T / self.pa_tau, dim=-1)  # [B,N,C]
                    self._last_belief = bel_j
                    if not hasattr(self, '_belief_map'):
                        self._belief_map = {}
                    self._belief_map[1 - i] = bel_j
                else:
                    h_asbi = torch.zeros_like(h_priv)

                if self._use_poe and self.n_modalities == 2:
                    b_i = torch.softmax(z_list[i] @ self.pa_M.T / self.pa_tau,
                                        dim=-1)                     # [B,N,C]
                    b_j = torch.softmax(z_list[1 - i] @ self.pa_M.T / self.pa_tau,
                                        dim=-1)                     # [B,N,C]
                    h_poe = ((b_i * b_j) @ self.pa_M) @ self.lora_Bt.T
                    self._last_belief = (b_i * b_j) / \
                        (b_i * b_j).sum(-1, keepdim=True).clamp(min=1e-8)
                else:
                    h_poe = torch.zeros_like(h_priv)

                if self._use_ev and self.n_modalities == 2:
                    e_ev = z_list[i] @ self.pa_psi_list[i].T       # [B,N,D]
                    h_ev = (e_ev @ self.pa_E.T / self.pa_tau) \
                        @ self.lora_Be.T                            # [B,N,out]
                else:
                    h_ev = torch.zeros_like(h_priv)

                if self._use_xgl and self.n_modalities == 2:
                    if self._use_evd:
                        # Use the same class readout for both modalities at all adapted sites.
                        wcT = MMLinear._SHARED_WC.t().contiguous()
                        lg0 = z_list[0] @ wcT
                        lg1 = z_list[1] @ wcT
                        if getattr(self, '_use_evd_edl', False):
                            e0 = F.softplus(lg0) - 0.6931471805599453
                            e1 = F.softplus(lg1) - 0.6931471805599453
                            S0 = e0.sum(-1, keepdim=True) + lg0.shape[-1]
                            S1 = e1.sum(-1, keepdim=True) + lg0.shape[-1]
                            u0 = e0 / S0
                            u1 = e1 / S1
                            if not hasattr(self, '_evd_e'):
                                self._evd_e = {}
                            self._evd_e[0] = e0
                            self._evd_e[1] = e1
                            if not hasattr(self, '_evd_unc'):
                                self._evd_unc = {}
                            self._evd_unc[0] = (lg0.shape[-1] / S0).squeeze(-1)
                            self._evd_unc[1] = (lg0.shape[-1] / S1).squeeze(-1)
                        else:
                            u0 = F.softmax(lg0, dim=-1)
                            u1 = F.softmax(lg1, dim=-1)
                        if not hasattr(self, '_evd_logu'):
                            self._evd_logu = {}
                        if getattr(self, '_use_evd_edl', False):
                            self._evd_logu[0] = torch.log(e0 + 1.0)
                            self._evd_logu[1] = torch.log(e1 + 1.0)
                        else:
                            self._evd_logu[0] = F.log_softmax(lg0, dim=-1)
                            self._evd_logu[1] = F.log_softmax(lg1, dim=-1)
                        e_and = u0 * u1
                        e_xor = u0 - u1
                        if not hasattr(self, '_evd_u'):
                            self._evd_u = {}
                        self._evd_u[0] = u0
                        self._evd_u[1] = u1
                        if getattr(self, '_use_evd_v3', False):
                            if getattr(self, '_use_ds16', False):
                                # Use softmax probabilities from this modality for its class feedback.
                                self._evd_priv = F.softmax(
                                    lg0 if i == 0 else lg1, dim=-1)
                            else:
                                self._evd_priv = u0 if i == 0 else u1
                    else:
                        _xc = getattr(self, '_xgc_cache', {})
                        if i in _xc:
                            _e0 = _xc[i][0]
                            _half = _e0.shape[-1] // 2
                            e_and, e_xor = _e0[..., :_half], _e0[..., _half:]
                        else:
                            zj_aligned = z_list[1 - i] @ self.xgl_Wx.T
                            e_and = z_list[i] * zj_aligned
                            e_xor = (z_list[i] - zj_aligned).abs()
                    if self._use_evd and i == 1:
                        e_xor = -e_xor
                    e_xgl = torch.cat([e_and, e_xor], dim=-1)  # [B,N,2r or 2K]
                    if getattr(self, '_use_evd_v3', False):
                        _bel = self._evd_priv @ self.xgl_Be.T
                        if not getattr(self, 'xgl_belief_kill', False):
                            h_xgl = _bel
                        else:
                            h_xgl = torch.zeros_like(_bel)
                        if getattr(self, '_use_ds16', False):
                            # Use the product and absolute difference of the aligned codes.
                            uj = self._evd_unc[1 - i]                # [B,N]
                            zj = z_list[1 - i] @ self.xgl_Wx.T
                            e_and_c = z_list[i] * zj
                            e_xor_c = (z_list[i] - zj).abs()
                            _cross = torch.cat([e_and_c, e_xor_c],
                                               dim=-1) @ self.xgl_Bx.T \
                                * (1.0 - uj).unsqueeze(-1)
                            # Use a constant coefficient in the fixed-coefficient ablation.
                            if getattr(self, '_ds16_gconst', False):
                                _cross = torch.cat([e_and_c, e_xor_c],
                                                   dim=-1) @ self.xgl_Bx.T
                            _paper_override = getattr(self, 'paper_cross_override', None)
                            if _paper_override is not None:
                                _cross = _paper_override(self, i, z_list, h_priv, _cross)
                            if not getattr(self, 'xgl_cross_kill', False):
                                h_xgl = h_xgl + _cross
                            _paper_probe = getattr(self, 'paper_probe', None)
                            if _paper_probe is not None:
                                _paper_probe(self, i, z_list, h_priv, _bel, _cross,
                                             not getattr(self, 'xgl_cross_kill', False))
                            if not hasattr(self, '_ds16_dbg'):
                                self._ds16_dbg = {}
                            self._ds16_dbg.setdefault(i, []).append((
                                (1.0 - uj).mean().item(),
                                _cross.abs().mean().item(),
                                _bel.abs().mean().item(),
                                h_priv.abs().mean().item()))
                    else:
                        h_xgl = e_xgl @ self.xgl_Be.T
                    if not hasattr(self, '_xgl_cache'):
                        self._xgl_cache = {}
                    self._xgl_cache[i] = e_xgl

                else:
                    h_xgl = torch.zeros_like(h_priv)

                if self.record_heatmaps:
                    entry = {
                        'private': (h_priv * h_priv).sum(-1).detach(),   # [B,H,W]
                        'cross':   (h_cross * h_cross).sum(-1).detach(),
                        'self':    (h_self * h_self).sum(-1).detach(),
                    }
                    if phi is not None:
                        entry['phi'] = phi
                        if g is not None:
                            entry['g_self'] = g[..., 0].detach()
                            entry['g_cross'] = g[..., 1].detach()
                    self._heatmaps[i] = entry
                    if os.environ.get('DEBUG_BRANCHES'):
                        import os as _os
                        print(f"[debug] mod{i} h_priv shape={tuple(h_priv.shape)} "
                              f"mean={h_priv.abs().mean().item():.6f} "
                              f"cross={h_cross.abs().mean().item():.6f} "
                              f"self={h_self.abs().mean().item():.6f}")

                result = result + self.scaling * (h_priv + h_cross + h_self
                                                 + h_cube + h_mix + h_xattn
                                                 + h_logic + h_ptl + h_lcx
                                                 + h_asbi + h_poe + h_ev
                                                 + h_xgl)
            else:
                result = result + self.scaling * h_priv

            results.append(result)
        return results

    def get_kl_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.weight.device)

    def get_orthogonal_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.weight.device)


class MMMergedLinear(nn.Linear, LoRALayer):
    """N-modality MergedLinear for QKV projection with shared + private LoRA.

    Like MergedLinear but generalized to N modalities. LoRA is only applied
    to the dimensions specified by enable_lora (e.g., Q and V for attention).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_modalities: int = 2,
        private_r: int = 4,
        shared_r: int = 2,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [True, False, True],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        self.n_modalities = n_modalities
        self.private_r = private_r
        self.shared_r = shared_r
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        self.n_lora_segments = sum(enable_lora)
        self.segment_dim = out_features // len(enable_lora) if out_features % len(enable_lora) == 0 else out_features // max(len(enable_lora), 1)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=private_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        self.lora_ind = self.weight.new_zeros((out_features,), dtype=torch.bool)
        self.lora_ind = self.lora_ind.view(len(enable_lora), -1)
        self.lora_ind[enable_lora, :] = True
        self.lora_ind = self.lora_ind.view(-1)

        shared_dim_in = private_r * self.n_lora_segments
        shared_dim_out = self.segment_dim * self.n_lora_segments
        if shared_r > 0:
            self.lora_shared_A = nn.Parameter(torch.zeros((shared_r * self.n_lora_segments, in_features)))
            self.lora_shared_B = nn.Parameter(torch.zeros((shared_dim_out, shared_r)))
            self.scaling_shared = self.lora_alpha / shared_r
        else:
            self.lora_shared_A = None
            self.lora_shared_B = None

        if private_r > 0:
            self.lora_A_list = nn.ParameterList([
                nn.Parameter(torch.zeros((private_r * self.n_lora_segments, in_features)))
                for _ in range(n_modalities)
            ])
            self.lora_B_list = nn.ParameterList([
                nn.Parameter(torch.zeros((shared_dim_out, private_r)))
                for _ in range(n_modalities)
            ])
            self.scaling_private = self.lora_alpha / private_r
        else:
            self.lora_A_list = nn.ParameterList([])
            self.lora_B_list = nn.ParameterList([])

        self.lora_w_shared = nn.Parameter(torch.ones(1) * 0.5)
        self.lora_w_private = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 0.5)
            for _ in range(n_modalities)
        ])

        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        self.reset_parameters()

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if self.shared_r > 0 and hasattr(self, 'lora_shared_A') and self.lora_shared_A is not None:
            nn.init.kaiming_uniform_(self.lora_shared_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_shared_B)
        if self.private_r > 0 and hasattr(self, 'lora_A_list'):
            for i in range(self.n_modalities):
                nn.init.kaiming_uniform_(self.lora_A_list[i], a=math.sqrt(5))
                nn.init.zeros_(self.lora_B_list[i])

    def _T(self, w):
        return w.T if self.fan_in_fan_out else w

    def zero_pad(self, x):
        """Pad LoRA output to full out_features, filling non-LoRA dims with zeros."""
        result = x.new_zeros((*x.shape[:-1], self.out_features))
        result = result.view(-1, self.out_features)
        result[:, self.lora_ind] = x.reshape(
            -1, self.out_features // len(self.enable_lora) * self.n_lora_segments
        )
        return result.view((*x.shape[:-1], self.out_features))

    def _apply_lora(self, x: torch.Tensor, A: torch.Tensor, B: torch.Tensor, scaling: float) -> torch.Tensor:
        """Apply LoRA: x -> B @ (A @ x^T)^T * scaling, with conv1d for grouped output."""
        after_A = F.linear(self.lora_dropout(x), A)  # [B, N, r*sum(enable)]
        after_B = F.conv1d(
            after_A.transpose(-2, -1),
            B.unsqueeze(-1),
            groups=self.n_lora_segments
        ).transpose(-2, -1)
        return self.zero_pad(after_B) * scaling

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)

    def eval(self):
        nn.Linear.eval(self)

    def forward(self, x_list: list) -> list:
        """Forward for QKV. No cross-modal."""
        results = []
        for i, x in enumerate(x_list):
            result = F.linear(x, self._T(self.weight), bias=self.bias)
            if self.shared_r > 0:
                result = result + self.lora_w_shared * self._apply_lora(x, self.lora_shared_A, self.lora_shared_B, self.scaling_shared)
            if self.private_r > 0:
                result = result + self.lora_w_private[i] * self._apply_lora(x, self.lora_A_list[i], self.lora_B_list[i], self.scaling_private)
            results.append(result)
        return results

    def get_orthogonal_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.weight.device)
