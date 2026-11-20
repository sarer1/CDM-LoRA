"""Build the DINOv3 encoder used by CDM-LoRA."""
import os


def build_mm_encoder(encoder_type: str = 'dinov3', size: str = 'large',
                     n_modalities: int = 2, private_r: int = 4, shared_r: int = 2,
                     checkpoint: str = None):
    if encoder_type != 'dinov3':
        raise ValueError("CDM-LoRA uses the DINOv3 encoder")
    if size not in ('small', 'base', 'large'):
        raise ValueError("DINOv3 MM encoder supports small/base/large")
    import glob as _glob
    ckpt = checkpoint or os.environ.get('MFNET_DINO3_CKPT')
    if ckpt is None:
        _stem = {'small': 'vits16', 'base': 'vitb16', 'large': 'vitl16'}[size]
        hits = sorted(_glob.glob(f'weights/dinov3_{_stem}*.pth'))
        if hits:
            ckpt = hits[0]
            print(f"DINOv3 checkpoint auto-discovered: {ckpt}")
    elif ckpt == 'sat':
        ckpt = 'weights/dinov3_vitl16_pretrain_sat493m_eadcf0ff.pth'
    elif ckpt == 'web':
        _stem = {'small': 'vits16', 'base': 'vitb16', 'large': 'vitl16'}[size]
        _h = {'small': '08c60483', 'base': '73cec8be', 'large': '8aa4cbdd'}[size]
        ckpt = f'weights/dinov3_{_stem}_pretrain_lvd1689m_{_h}.pth'
    from MedSAM.models.ImageEncoder.vit.mm_dinov3 import build_dinov3_mm_encoder
    return build_dinov3_mm_encoder(size=size, n_modalities=n_modalities,
                                   private_r=private_r, shared_r=shared_r,
                                   checkpoint=ckpt)
