"""
3D Vision Transformer for volumetric medical image classification.
Input: (B, 1, D, H, W); output: (B, num_classes).
Volumes are resized to a fixed spatial size inside the model for a consistent patch grid.
"""
import torch
import torch.nn as nn
import math
from typing import Tuple


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class PatchEmbed3D(nn.Module):
    """Split 3D volume into patches and project to embedding dim."""

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 768,
        patch_size: Tuple[int, int, int] = (16, 16, 16),
        spatial_size: Tuple[int, int, int] = (96, 96, 96),
    ):
        super().__init__()
        self.patch_size = patch_size
        self.spatial_size = spatial_size
        self.num_patches = (
            (spatial_size[0] // patch_size[0])
            * (spatial_size[1] // patch_size[1])
            * (spatial_size[2] // patch_size[2])
        )
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W) — will be resized to spatial_size in caller if needed
        x = self.proj(x)
        # (B, embed_dim, D', H', W') -> (B, embed_dim, num_patches)
        x = x.flatten(2)
        # (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=attn_drop, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, dim)
        x = x + self._attn_block(self.norm1(x))
        x = x + self._mlp_block(self.norm2(x))
        return x

    def _attn_block(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        return attn_out

    def _mlp_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class VisionTransformer3D(nn.Module):
    """
    3D ViT for volumetric input (B, 1, D, H, W).
    Pads input to spatial_size internally for a fixed patch grid (no cropping).
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        spatial_size: Tuple[int, int, int] = (96, 96, 96),
        patch_size: Tuple[int, int, int] = (16, 16, 16),
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.spatial_size = spatial_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed3D(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            spatial_size=spatial_size,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_drop=attn_drop,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module)

    @staticmethod
    def _init_module(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _resize_to_fixed(self, x: torch.Tensor) -> torch.Tensor:
        """Pad 3D volume up to self.spatial_size for a fixed patch grid (no cropping)."""
        B, C, D, H, W = x.shape
        target_D, target_H, target_W = self.spatial_size

        # Safety check: we only support volumes not larger than target size
        if D > target_D or H > target_H or W > target_W:
            raise ValueError(
                f"Input volume size {(D, H, W)} exceeds ViT spatial_size {self.spatial_size}. "
                "Please increase vit_spatial_size or resize the data beforehand."
            )

        if (D, H, W) == self.spatial_size:
            return x

        pad_D = target_D - D
        pad_H = target_H - H
        pad_W = target_W - W

        # Pad with zeros on the high ends of each dimension.
        # This is a simple and common choice for medical images where
        # background is near-zero after normalization.
        x = nn.functional.pad(
            x,
            (0, pad_W, 0, pad_H, 0, pad_D),  # (W_low, W_high, H_low, H_high, D_low, D_high)
            mode="constant",
            value=0.0,
        )
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, D, H, W)
        x = self._resize_to_fixed(x)
        x = self.patch_embed(x)
        # (B, num_patches, embed_dim)

        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        cls_out = x[:, 0]
        logits = self.head(cls_out)
        return logits
