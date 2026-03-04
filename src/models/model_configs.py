"""
Standard ViT model variant configs from the paper.
Usage: model = build_vit("vit_b16", num_classes=10)
"""

from .vit import VisionTransformer

# (embed_dim, depth, num_heads, patch_size)
_CONFIGS = {
    "vit_ti16": dict(embed_dim=192,  depth=12, num_heads=3,  patch_size=16),
    "vit_s16":  dict(embed_dim=384,  depth=12, num_heads=6,  patch_size=16),
    "vit_s32":  dict(embed_dim=384,  depth=12, num_heads=6,  patch_size=32),
    "vit_b16":  dict(embed_dim=768,  depth=12, num_heads=12, patch_size=16),
    "vit_b32":  dict(embed_dim=768,  depth=12, num_heads=12, patch_size=32),
    "vit_l16":  dict(embed_dim=1024, depth=24, num_heads=16, patch_size=16),
    "vit_l32":  dict(embed_dim=1024, depth=24, num_heads=16, patch_size=32),
    "vit_h14":  dict(embed_dim=1280, depth=32, num_heads=16, patch_size=14),
}


def build_vit(name: str, img_size: int = 224, in_channels: int = 3, num_classes: int = 1000, **kwargs) -> VisionTransformer:
    if name not in _CONFIGS:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_CONFIGS)}")
    cfg = {**_CONFIGS[name], **kwargs}
    return VisionTransformer(
        img_size=img_size,
        patch_size=cfg.pop("patch_size"),
        in_channels=in_channels,
        num_classes=num_classes,
        **cfg,
    )
