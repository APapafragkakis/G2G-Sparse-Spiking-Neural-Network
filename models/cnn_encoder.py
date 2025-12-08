# models/cnn_encoder.py

import torch
from torch import nn

class PatchConvEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 32):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2) 
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)  
        
        self.conv3 = nn.Conv2d(128, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.pool3 = nn.AdaptiveAvgPool2d((4, 4))
        
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = self.activation(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        return x.view(x.size(0), -1)  # [B, 512]

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
