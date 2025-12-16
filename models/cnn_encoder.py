import torch
from torch import nn

class PatchEncoder(nn.Module):
    """
    - Split image into 16 non-overlapping patches (4x4 grid)
    - Apply ONE conv layer to each patch independently using:
        kernel_size = patch_size, stride = patch_size
    - Output: [B, 16 * out_channels]

    Works for:
    - Fashion-MNIST 28x28 -> patch_size=7 -> 4x4 patches
    - CIFAR 32x32 -> patch_size=8 -> 4x4 patches
    """

    def __init__(
        self,
        in_channels: int,
        img_size: int,
        out_channels: int = 32,
        grid_size: int = 4,           # 4x4 = 16 patches
        bias: bool = True,
        activation: bool = True,
    ):
        super().__init__()
        assert img_size % grid_size == 0, (
            f"img_size ({img_size}) must be divisible by grid_size ({grid_size})."
        )
        self.grid_size = grid_size
        self.patch_size = img_size // grid_size
        self.out_channels = out_channels

        # This performs patchify + per-patch embedding in one shot
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
            bias=bias,
        )
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        feats = self.conv(x)          # [B, out_channels, 4, 4]
        feats = self.act(feats)
        feats = feats.flatten(1)      # [B, out_channels * 16]
        return feats
