# models/cnn_encoder.py

import torch
from torch import nn


class PatchConvEncoder(nn.Module):
    """
    Patch-based convolutional encoder for Fashion-MNIST.

    The 28x28 input image is divided into 16 non-overlapping 7x7 patches
    using a single convolutional layer with kernel_size=7 and stride=7.
    Each patch is mapped to a 32-dimensional embedding, and all patch
    embeddings are concatenated into a single feature vector.

    Input:  [B, 1, 28, 28]
    Output: [B, 512]  (16 patches * 32 channels)
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 32):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=7,
            stride=7,
            padding=0,
            bias=True,
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, 28, 28]
        feats = self.conv(x)          # [B, 32, 4, 4]
        feats = self.activation(feats)
        feats = feats.view(feats.size(0), -1)  # [B, 32 * 4 * 4 = 512]
        return feats


class CIFARCNNEncoder(nn.Module):
    """
    Patch-based convolutional encoder for CIFAR-10 / CIFAR-100.

    The 32x32 RGB input image is divided into 16 non-overlapping 8x8 patches
    using a single convolutional layer with kernel_size=8 and stride=8.
    Each patch is mapped to a 32-dimensional embedding, and all patch
    embeddings are concatenated into a single 512-dimensional vector.

    Input:  [B, 3, 32, 32]
    Output: [B, 512]  (16 patches * 32 channels)
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 32):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=8,
            stride=8,
            padding=0,
            bias=True,
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, 32, 32]
        feats = self.conv(x)          # [B, 32, 4, 4]
        feats = self.activation(feats)
        feats = feats.view(feats.size(0), -1)  # [B, 32 * 4 * 4 = 512]
        return feats
