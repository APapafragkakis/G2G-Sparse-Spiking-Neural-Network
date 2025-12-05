# cnn_encoder.py

import torch
from torch import nn


class PatchConvEncoder(nn.Module):
    """
    Patch-based convolutional encoder for Fashion-MNIST.

    The 28x28 input image is divided into 16 non-overlapping 7x7 patches
    using a single convolutional layer with kernel_size=7 and stride=7.
    Each patch is mapped to a 32-dimensional embedding, and all patch
    embeddings are concatenated into a single feature vector.

    Output shape: [batch_size, 512] (16 patches * 32 channels).
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 32):
        super().__init__()

        # One conv layer applied over non-overlapping 7x7 patches.
        # For 28x28 images: kernel_size=7, stride=7 => 4x4 spatial positions.
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
        """
        x: [batch_size, 1, 28, 28]
        returns: [batch_size, 512]
        """
        feats = self.conv(x)          # [B, 32, 4, 4]
        feats = self.activation(feats)
        feats = feats.view(feats.size(0), -1)  # [B, 32*4*4 = 512]
        return feats
