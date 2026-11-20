"""Official-recipe neck+decoder for DINOv3 segmentation (MFNET_ARCH=official).

Faithful ports of mmseg's Feature2Pyramid and UPerHead (the canonical
frozen-ViT segmentation recipe, cf. the official DINOv3 segmentation
tutorial: ViT backbone -> Feature2Pyramid -> UPerHead). Only deviation:
no FCNHead auxiliary head (the training loop consumes a single logit map).

Design intent (user 09-24): align the entire decoder side with the official
single-modal recipe so the ONLY custom components are (a) SO-LoRA in the
frozen backbone's MLP bottleneck and (b) one linear 1x1 fusion of the two
modalities' feature taps. No PFF, no confidence fusion, no Phi-gated skips.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Feature2Pyramid(nn.Module):
    """mmseg ViT pyramid neck: 4 input stages rescaled to strides [4,8,16,32].

    rescales (4,2,1,0.5): >1 = upsample path (extra 3x3 conv on the first
    level per mmseg), 1 = identity, <1 = strided conv + avg-pool.
    """

    def __init__(self, embed_dim=1024, rescales=(4, 2, 1, 0.5)):
        super().__init__()
        self.stages = nn.ModuleList()
        for i, s in enumerate(rescales):
            if s == 1:
                self.stages.append(nn.Identity())
            elif s > 1:
                mods = [nn.Sequential(
                    nn.Conv2d(embed_dim, embed_dim, 1, bias=False),
                    nn.BatchNorm2d(embed_dim),
                    nn.ReLU(inplace=True),
                )]
                if i == 0:
                    mods.append(nn.Sequential(
                        nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False),
                        nn.BatchNorm2d(embed_dim),
                        nn.ReLU(inplace=True),
                    ))
                mods.append(nn.Upsample(scale_factor=s, mode='bilinear',
                                        align_corners=False))
                self.stages.append(nn.Sequential(*mods))
            else:
                s = int(1 / s)
                self.stages.append(nn.Sequential(
                    nn.Conv2d(embed_dim, embed_dim, 3, stride=s, padding=1,
                              bias=False),
                    nn.BatchNorm2d(embed_dim),
                    nn.ReLU(inplace=True),
                    nn.AvgPool2d(kernel_size=s, stride=s),
                ))

    def forward(self, inputs):
        return [st(x) for st, x in zip(self.stages, inputs)]


class PPM(nn.ModuleList):
    """Pyramid pooling module on the deepest (stride-32) level."""

    def __init__(self, pool_scales, in_ch, ch):
        super().__init__([nn.Sequential(
            nn.AdaptiveAvgPool2d(scale),
            nn.Conv2d(in_ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        ) for scale in pool_scales])

    def forward(self, x):
        outs = [x]
        h, w = x.shape[-2:]
        for m in self:
            outs.append(F.interpolate(m(x), size=(h, w), mode='bilinear',
                                      align_corners=False))
        return torch.cat(outs, dim=1)


class UPerHead(nn.Module):
    """UNet-style Perceptual parser head (PSP + top-down FPN), mmseg port."""

    def __init__(self, in_ch=1024, ch=512, num_classes=6,
                 pool_scales=(1, 2, 3, 6)):
        super().__init__()
        self.ppm = PPM(pool_scales, in_ch, ch)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_ch + ch * len(pool_scales), ch, 3, padding=1,
                      bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )
        self.lateral = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, ch, 1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            ) for _ in range(3)
        ])
        self.fpn_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            ) for _ in range(4)
        ])
        self.fusion = nn.Sequential(
            nn.Conv2d(ch * 4, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )
        self.conv_seg = nn.Conv2d(ch, num_classes, 1)

    def forward(self, inputs):
        """inputs: 4 levels, strides [4,8,16,32]. Returns stride-4 logits."""
        laterals = [l(x) for l, x in zip(self.lateral, inputs[:3])]
        laterals.append(self.bottleneck(self.ppm(inputs[3])))
        for i in range(3, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:],
                mode='bilinear', align_corners=False)
        outs = [f(l) for f, l in zip(self.fpn_conv, laterals)]
        target = outs[0].shape[-2:]
        outs = [outs[0]] + [
            F.interpolate(o, size=target, mode='bilinear', align_corners=False)
            for o in outs[1:]
        ]
        return self.conv_seg(self.fusion(torch.cat(outs, dim=1)))


class LightPSPHead(nn.Module):
    """Light head + PSP context + evidence stream (v2, user 10-11). Keeps the
    LightFPNHead backbone (joint concat of all taps -> 1x1 -> 3x3, proven on
    this data) and adds exactly two things at the reduced level: PSP global
    context and the 32-ch AND/XOR evidence via a zero-init projection (the
    evidence on/off ablation toggle). ~4M per expert."""
    def __init__(self, in_ch=1024, ch=256, num_classes=6, n_taps=4, ev_ch=32):
        super().__init__()
        red = in_ch * n_taps
        self.reduce = nn.Sequential(
            nn.Conv2d(red, ch, 1, bias=False), nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True))
        self.ppm = nn.ModuleList([nn.Sequential(
            nn.AdaptiveAvgPool2d(s), nn.Conv2d(ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True)) for s in (1, 2, 3, 6)])
        self.ppm_fuse = nn.Conv2d(ch * 5, ch, 1, bias=False)
        self.ev_proj = nn.Conv2d(ev_ch, ch, 1)
        nn.init.zeros_(self.ev_proj.weight)
        nn.init.zeros_(self.ev_proj.bias)
        self.fusion = nn.Sequential(
            nn.Conv2d(ch * 3, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.refine = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True))
        self.conv_seg = nn.Conv2d(ch, num_classes, 1)

    def forward(self, inputs, ev=None):
        h = self.reduce(torch.cat(inputs, dim=1))
        ps = [h] + [F.interpolate(p(h), size=h.shape[-2:], mode='bilinear',
                                  align_corners=False) for p in self.ppm]
        c = self.ppm_fuse(torch.cat(ps, dim=1))
        e = self.ev_proj(ev) if ev is not None else torch.zeros_like(h)
        x = self.fusion(torch.cat([h, c, e], dim=1))
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return self.conv_seg(self.refine(x))


class EviGateHead(nn.Module):
    """D2+ multi-stage evidence-gated refinement (user 10-11): progressive
    x2 upsampling; each stage's residual is opened by the XOR (disagreement)
    gate and shielded by the AND (consensus) gate, reading the SAME 32-ch
    evidence the arbitrator consumes. All gate/refine convs zero-initialised,
    so the head starts exactly as the ungated LightFPNHead. +1.2M/expert."""
    def __init__(self, in_ch=1024, ch=256, num_classes=6, n_taps=4, ev_ch=32):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_ch * n_taps, ch, 1, bias=False), nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True))
        self.n_gate = 2
        self.gref = nn.ModuleList([nn.Conv2d(ch, ch, 3, padding=1, bias=False)
                                   for _ in range(self.n_gate)])
        self.gxor = nn.ModuleList([nn.Conv2d(ev_ch, 1, 1) for _ in range(self.n_gate)])
        self.gand = nn.ModuleList([nn.Conv2d(ev_ch, 1, 1) for _ in range(self.n_gate)])
        for mlist in (self.gref, self.gxor, self.gand):
            for m in mlist:
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.refine = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True))
        self.conv_seg = nn.Conv2d(ch, num_classes, 1)

    def forward(self, inputs, ev=None):
        """inputs: list of n_taps maps; ev: list of per-stage 32-ch evidence
        maps at the token grid (deepest first), or None to disable."""
        x = self.reduce(torch.cat(inputs, dim=1))
        for i in range(self.n_gate):
            x = F.interpolate(x, scale_factor=2, mode='bilinear',
                              align_corners=False)
            if ev is not None:
                e = ev[min(i + 1, len(ev) - 1)]
                e_up = F.interpolate(e, size=x.shape[-2:], mode='bilinear',
                                     align_corners=False)
                gate = torch.sigmoid(self.gxor[i](e_up)) * \
                       (1.0 - torch.sigmoid(self.gand[i](e_up)))
                x = x + self.gref[i](x) * gate
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.conv_seg(self.refine(x))


class EviUPerHead(nn.Module):
    """Light per-expert decoder (user 10-11) combining UPerNet's two working
    ingredients — top-down refinement across the four DEPTH taps and PSP
    global context — with the framework's evidence: the 32-ch AND/XOR map
    enters the fusion through a zero-initialised projection, so zeroing the
    evidence input is an exact, same-parameter ablation. ~5M per expert
    (vs 38M for the full UPerExpertHead probe)."""
    def __init__(self, in_ch=1024, ch=192, num_classes=6, ev_ch=32):
        super().__init__()
        self.reduce = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_ch, ch, 1, bias=False), nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True)) for _ in range(4)])
        self.topdown = nn.ModuleList([nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True)) for _ in range(4)])
        self.ppm = nn.ModuleList([nn.Sequential(
            nn.AdaptiveAvgPool2d(s), nn.Conv2d(ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True)) for s in (1, 2, 3, 6)])
        self.ppm_fuse = nn.Sequential(
            nn.Conv2d(ch * 5, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.ev_proj = nn.Conv2d(ev_ch, ch, 1)
        nn.init.zeros_(self.ev_proj.weight)
        nn.init.zeros_(self.ev_proj.bias)
        self.fusion = nn.Sequential(
            nn.Conv2d(ch * 3, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.conv_seg = nn.Conv2d(ch, num_classes, 1)

    def forward(self, inputs, ev=None):
        xs = [r(x) for r, x in zip(self.reduce, inputs)]
        h = self.topdown[3](xs[3])
        for i in (2, 1, 0):
            h = self.topdown[i](xs[i] + h)
        ps = [h] + [F.interpolate(p(h), size=h.shape[-2:], mode='bilinear',
                                  align_corners=False) for p in self.ppm]
        c = self.ppm_fuse(torch.cat(ps, dim=1))
        e = self.ev_proj(ev) if ev is not None else torch.zeros_like(h)
        f = self.fusion(torch.cat([h, c, e], dim=1))
        x = F.interpolate(f, scale_factor=4, mode='bilinear', align_corners=False)
        return self.conv_seg(x)


class UPerExpertHead(nn.Module):
    """Per-expert UPerNet decoder (probe, user 10-11): Feature2Pyramid +
    UPerHead (PSP context + top-down FPN) — the decoder family MFNet uses —
    behind MFNET_HEAD=uper, replacing the LightFPNHead inside the same
    dual-expert + arbitration structure. Interface matches LightFPNHead
    (list of taps; ev accepted and ignored)."""
    def __init__(self, in_ch=1024, ch=256, num_classes=6):
        super().__init__()
        self.pyr = Feature2Pyramid(embed_dim=in_ch)
        self.head = UPerHead(in_ch=in_ch, ch=ch, num_classes=num_classes)

    def forward(self, inputs, ev=None):
        return self.head(self.pyr(list(inputs)))


class LightFPNHead(nn.Module):
    """Minimal convolutional head for MFNET_ARCH=light (~2.3M params).

    Design contract (user 09-24): the framework must be LIGHT and the
    quadratic-term ablation gap must be attributable — so this head has
    ZERO cross-modal capacity (modalities are averaged before it) and no
    attention/heavy fusion: concat depth taps -> 1x1 -> 3x3 -> up4 -> 3x3
    -> cls. All learned cross-modal interaction lives in the SO-LoRA
    bottleneck of the frozen backbone.
    """

    def __init__(self, in_ch=1024, ch=256, num_classes=6, n_taps=4, extra=0,
                 ev_gate=False, ev_ch=32):
        super().__init__()
        self.ev_gate = ev_gate
        if ev_gate:
            self.gref = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.ggate = nn.Conv2d(ev_ch, 1, 1)
            nn.init.zeros_(self.gref.weight)
            nn.init.zeros_(self.ggate.weight)
            if self.ggate.bias is not None:
                nn.init.zeros_(self.ggate.bias)
        self.reduce = nn.Sequential(
            nn.Conv2d(in_ch * (n_taps + extra), ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )
        self.conv_seg = nn.Conv2d(ch, num_classes, 1)

    def forward(self, inputs, ev=None):
        """inputs: list of n_taps same-resolution maps (stride 16).
        ev: optional [B, 32, s, s] AND/XOR evidence at token grid."""
        x = torch.cat(inputs, dim=1)
        x = self.reduce(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear',
                          align_corners=False)
        if self.ev_gate and ev is not None:
            ev_up = F.interpolate(ev, size=x.shape[-2:], mode='bilinear',
                                  align_corners=False)
            x = x + self.gref(x) * torch.sigmoid(self.ggate(ev_up))
        return self.conv_seg(self.refine(x))


class LightFuseHead(nn.Module):
    """Framework-designed decoder (user 10-02, wbf): the head finally reads
    each modality SEPARATELY and fuses WITH the bypass's joint evidence —
    'process first, fuse later' instead of the light head's 'average first,
    process blind'. Params ~7.3M (vs light 2.26M); the MFNet control arm
    uses the same head with a zeroed joint input.
    """
    def __init__(self, in_ch=1024, ch=256, num_classes=6, n_taps=4):
        super().__init__()
        def blk(cin):
            return nn.Sequential(
                nn.Conv2d(cin, ch, 1, bias=False),
                nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
                nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.reduce_0 = blk(in_ch * n_taps)
        self.reduce_1 = blk(in_ch * n_taps)
        self.reduce_J = blk(128 * n_taps)
        self.fuse = nn.Sequential(
            nn.Conv2d(3 * ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.refine = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.conv_seg = nn.Conv2d(ch, num_classes, 1)

    def forward(self, taps0, taps1, jfeats):
        e0 = self.reduce_0(torch.cat(taps0, dim=1))
        e1 = self.reduce_1(torch.cat(taps1, dim=1))
        ej = self.reduce_J(torch.cat(jfeats, dim=1))
        x = self.fuse(torch.cat([e0, e1, ej], dim=1))
        x = F.interpolate(x, scale_factor=4, mode='bilinear',
                          align_corners=False)
        return self.conv_seg(self.refine(x))


class XGLHead(nn.Module):
    """Decoder for Cross-Modal Guided LoRA (user 10-03): per-modality
    branches + interaction evidence stream; the class guidance g from W_g
    is added to the output logits by the model (not this head)."""
    def __init__(self, in_ch=1024, ch=256, num_classes=6, n_taps=4, ev_ch=8):
        super().__init__()
        def blk(cin):
            return nn.Sequential(
                nn.Conv2d(cin, ch, 1, bias=False),
                nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
                nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.reduce_0 = blk(in_ch * n_taps)
        self.reduce_1 = blk(in_ch * n_taps)
        self.reduce_E = blk(ev_ch * 4)
        self.fuse = nn.Sequential(
            nn.Conv2d(3 * ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.refine = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.conv_seg = nn.Conv2d(ch, num_classes, 1)

    def forward(self, taps0, taps1, ev_maps):
        e0 = self.reduce_0(torch.cat(taps0, dim=1))
        e1 = self.reduce_1(torch.cat(taps1, dim=1))
        ee = self.reduce_E(ev_maps)
        x = self.fuse(torch.cat([e0, e1, ee], dim=1))
        x = F.interpolate(x, scale_factor=4, mode='bilinear',
                          align_corners=False)
        return self.conv_seg(self.refine(x))
