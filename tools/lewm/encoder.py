"""Pixel encoders for the LeWM rogue port.

The upstream LeWM repo uses a HuggingFace ViT-tiny (~5M params, 224x224
inputs). The rogue game renders an 8x8 board into a 32x32 RGB tensor, so a
ViT-tiny is wasteful: a few conv layers reach comparable representation
quality at a fraction of the cost.

This module exposes :class:`TinyConvEncoder` (used by ``tools/lewm/jepa.py``)
and :class:`VectorEncoder` (used for parity testing with the existing
``train_world_model.py`` 31-d observation).
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["TinyConvEncoder", "VectorEncoder"]


class TinyConvEncoder(nn.Module):
    """Small CNN encoder for low-resolution game pixels.

    Args:
        in_channels: Channels per frame (3 for RGB, 4 if alpha is used).
        image_size: Spatial size; must be divisible by 8.
        embed_dim: Output embedding dimension.
    """

    def __init__(self, in_channels: int = 3, image_size: int = 32, embed_dim: int = 128):
        super().__init__()
        if image_size % 8 != 0:
            raise ValueError(f"image_size must be divisible by 8, got {image_size}")
        self.in_channels = in_channels
        self.image_size = image_size
        self.embed_dim = embed_dim

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
        )

        final_size = image_size // 8
        self.project = nn.Linear(128 * final_size * final_size, embed_dim)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """Encode a batch of frames.

        Args:
            pixels: ``(B, C, H, W)`` float tensor in ``[0, 1]``.

        Returns:
            ``(B, embed_dim)`` embedding (the analogue of the ViT CLS token).
        """
        if pixels.ndim != 4:
            raise ValueError(f"expected (B, C, H, W), got shape {tuple(pixels.shape)}")
        feats = self.stem(pixels.float())
        return self.project(feats.flatten(1))


class VectorEncoder(nn.Module):
    """Tiny MLP encoder for the existing 31-d hand-engineered observation.

    Useful for sanity-checking the LeWM training loop against the existing
    MLP world model before the Unity pixel pipeline (M2) lands.
    """

    def __init__(self, input_dim: int = 31, embed_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.GELU(),
            nn.Linear(96, embed_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 2 or obs.size(-1) != self.input_dim:
            raise ValueError(f"expected (B, {self.input_dim}), got shape {tuple(obs.shape)}")
        return self.net(obs.float())
