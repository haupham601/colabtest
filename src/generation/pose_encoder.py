import torch
import torch.nn as nn
from einops import rearrange

class PoseEncoderBlock3D(nn.Module):
    """
    A single 3D convolutional block for the Pose Encoder.
    Applies Conv3D -> GroupNorm -> SiLU twice, with a residual connection.
    """
    def __init__(self, in_channels: int, out_channels: int, stride_spatial: int = 1):
        super().__init__()
        
        # Spatial downsampling is achieved using stride_spatial on H and W, while keeping T stride=1
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, 
            stride=(1, stride_spatial, stride_spatial), padding=1
        )
        # GroupNorm with 32 groups is standard
        self.gn1 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.silu1 = nn.SiLU()
        
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1
        )
        self.gn2 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.silu2 = nn.SiLU()
        
        # Zero-init the last convolution for stable training start
        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)
            
        # Residual connection
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
    """3D Convolutional Pose Encoder for temporal pose conditioning.
    
    Takes a sequence of pose images [B, T, C, H, W] and produces
    pose embeddings that can be added to the DiT's hidden states.
    
    Architecture:
    - 4 blocks of (Conv3D -> GroupNorm -> SiLU -> Conv3D -> GroupNorm -> SiLU)
    - Progressive downsampling in spatial dims, preserving temporal dim
    - Channels: 3 -> 64 -> 128 -> 256 -> output_channels (matches DiT hidden_size)
    - Final projection layer to match DiT hidden dimension
    """
    def __init__(
        self, 
        in_channels: int = 3, 
        base_channels: int = 64, 
        num_blocks: int = 4, 
        output_channels: int = 1280
    ):
        super().__init__()
        
        # Channel progression
        channels = [in_channels] + [base_channels * (2 ** i) for i in range(num_blocks - 1)] + [output_channels]
        
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            in_c = channels[i]
            out_c = channels[i+1]
            
            # Downsample spatially for blocks after the first one
            stride = 2 if i > 0 else 1 
            self.blocks.append(PoseEncoderBlock3D(in_c, out_c, stride))
            
        self.output_channels = output_channels
        
    def forward(self, pose_images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pose_images: Rendered pose skeletons of shape [B, T, C, H, W]
            
        Returns:
            torch.Tensor of shape [B, T, H'*W', output_channels] containing pose embeddings
        """
        # Rearrange to [B, C, T, H, W] for Conv3D operations
        x = rearrange(pose_images, 'b t c h w -> b c t h w')
        
        # Pass through 3D convolutional blocks
        for block in self.blocks:
            x = block(x)
            
        # Current shape: [B, output_channels, T, H', W']
        # Rearrange to match DiT sequential injection format [B, T, N, C]
        x = rearrange(x, 'b c t h w -> b t (h w) c')
        
        return x
