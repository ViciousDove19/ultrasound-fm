import torch
import torch.nn as nn

from .patch_embed import PatchEmbed
from .transformer_block import TransformerBlock


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) from "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020).

    Sequence: PatchEmbed → [CLS] prepend → positional embed → L×TransformerBlock → LayerNorm → head
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # +1 for CLS token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, attn_drop, proj_drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Linear classifier head; replace with task-specific head for fine-tuning
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return CLS token embedding and per-block attention weights."""
        B = x.shape[0]
        x = self.patch_embed(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)   # (B, 1 + num_patches, embed_dim)
        x = x + self.pos_embed

        attn_weights = []
        for block in self.blocks:
            x, attn = block(x)
            attn_weights.append(attn)

        x = self.norm(x)
        return x[:, 0], attn_weights  # CLS token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_token, _ = self.forward_features(x)
        return self.head(cls_token)
