"""3D Convolutional Pose Encoder for temporal pose conditioning.

Efficiently encodes a sequence of skeleton images [B, T, C, H, W] into
pose embeddings matching the DiT latent dimensions.

Uses an initial 8x spatial downsampling stem followed by residual 3D blocks
to maintain extremely low VRAM footprint (<500MB on 49 frames).
"""

import torch
import torch.nn as nn
from einops import rearrange


class PoseEncoderBlock3D(nn.Module):
    """A single 3D convolutional block with residual connection."""
    def __init__(self, in_channels: int, out_channels: int, stride_spatial: int = 1):
        super().__init__()
        
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3,
            stride=(1, stride_spatial, stride_spatial), padding=1
        )
        self.gn1 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.silu1 = nn.SiLU()
        
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1
        )
        self.gn2 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.silu2 = nn.SiLU()
        
        # Zero-init the last convolution for stable residual start
        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)
            
        # Residual projection
        if in_channels != out_channels or stride_spatial != 1:
            self.residual = nn.Conv3d(
                in_channels, out_channels, kernel_size=1,
                stride=(1, stride_spatial, stride_spatial), padding=0
            )
        else:
            self.residual = nn.Identity()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.silu1(x)
        x = self.conv2(x)
        x = self.gn2(x)
        x = self.silu2(x)
        return x + res


class PoseEncoder3D(nn.Module):
    """Memory-efficient 3D Convolutional Pose Encoder for DiT video models.
    
    Architecture:
    1. Spatial Stem: Conv3D with stride (1, 8, 8) to reduce H, W by 8x immediately
       (e.g., 832x480 -> 104x60), preventing 70GB+ VRAM allocations in 3D convs.
    2. Residual 3D Blocks: Progressively increases channels while preserving temporal depth.
    3. Final projection to match DiT hidden size (e.g., 1280).
    """
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_blocks: int = 4,
        output_channels: int = 1280,
    ):
        super().__init__()
        
        # 1. Stem: spatial 8x downsampling matches latent VAE scale
        self.stem = nn.Sequential(
            nn.Conv3d(
                in_channels, base_channels,
                kernel_size=(1, 8, 8),
                stride=(1, 8, 8),
                padding=0,
            ),
            nn.GroupNorm(min(32, base_channels), base_channels),
            nn.SiLU(),
        )
        
        # 2. Intermediate 3D Conv blocks
        # Channels: base_channels (64) -> 128 -> 256 -> 512
        channels = [base_channels * (2 ** i) for i in range(num_blocks)]
        self.blocks = nn.ModuleList()
        for i in range(num_blocks - 1):
            in_c = channels[i]
            out_c = channels[i + 1]
            stride_s = 2 if i < 2 else 1
            self.blocks.append(PoseEncoderBlock3D(in_c, out_c, stride_spatial=stride_s))
            
        # 3. Final projection to DiT hidden dimension
        self.proj_out = nn.Sequential(
            nn.Conv3d(channels[-1], output_channels, kernel_size=1, stride=1),
            nn.GroupNorm(min(32, output_channels), output_channels),
        )
        # Zero-init output projection for smooth training start
        nn.init.zeros_(self.proj_out[0].weight)
        if self.proj_out[0].bias is not None:
            nn.init.zeros_(self.proj_out[0].bias)
            
        self.output_channels = output_channels

    def forward(self, pose_images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pose_images: Rendered pose skeletons of shape [B, T, C, H, W]
            
        Returns:
            torch.Tensor of shape [B, T, N, output_channels] containing pose embeddings
        """
        # Ensure B, C, T, H, W order for Conv3D
        if pose_images.ndim == 5 and pose_images.shape[2] == 3:
            # [B, T, C, H, W] -> [B, C, T, H, W]
            x = pose_images.permute(0, 2, 1, 3, 4)
        else:
            x = pose_images

        # 1. Stem downsampling (reduces 832x480 -> 104x60)
        x = self.stem(x)
        
        # 2. Residual blocks
        for block in self.blocks:
            x = block(x)
            
        # 3. Projection to hidden size
        x = self.proj_out(x)
        
        # Shape: [B, output_channels, T, H', W']
        # Reshape to DiT token sequence [B, T, H'*W', output_channels]
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(b, t, h * w, c)
        return x
