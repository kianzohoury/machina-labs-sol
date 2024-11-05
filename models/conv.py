
from typing import Tuple


import torch
import torch.nn as nn

torch.backends.cudnn.benchmark = True

class ResidualDownBlock(nn.Module):
    """Residual convolutional block for down-sampling + feature extraction."""
    def __init__(self, in_channels: int):
        super(ResidualDownBlock, self).__init__()
        self.in_channels = in_channels 
        out_channels = in_channels * 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(),
            nn.Conv1d(out_channels, out_channels, 3, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(),
        )
        self.proj = nn.Conv1d(in_channels, out_channels, 1)
        self.activation = nn.LeakyReLU()
        self.maxpool = nn.MaxPool1d(2, 2)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        conv_output = self.conv(x)
        skip = self.proj(x)
        residual_output = self.activation(skip + conv_output)
        downsampled_output = self.maxpool(residual_output)
        return conv_output, downsampled_output
        
        
class ResidualUpBlock(nn.Module):
    """Residual convolutional block for up-sampling + signal reconstruction."""
    def __init__(self, in_channels: int):
        super(ResidualUpBlock, self).__init__()
        self.in_channels = in_channels 
        out_channels = in_channels // 2
        self.upsample = nn.Upsample(scale_factor=2,)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels * 2, in_channels, 3, padding="same"),
            nn.BatchNorm1d(in_channels),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels, out_channels, 3, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(),
        )
        self.proj = nn.Conv1d(in_channels, out_channels, 1)
        self.activation = nn.LeakyReLU()
        
        
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        upsampled_output = self.upsample(x)
        # concatenate features
        concatenated_features = torch.cat([upsampled_output, skip], dim=1)
        conv_output = self.conv(concatenated_features)
        residual_output = self.activation(self.proj(upsampled_output) + conv_output)
        return residual_output
    

# convolutional models as baselines    
class DenoiserConv(nn.Module):
    """Convolutional U-Net for point cloud denoising."""
    def __init__(self, d_model: int = 256, **kwargs):
        super(DenoiserConv, self).__init__()
        self.d_model = d_model
        # standard learnable point embeddings to project
        # points from (x, y, z) to higher dimensional space and vice versa
        self.input_projection = nn.Sequential(
            nn.Linear(3, d_model),
            nn.LeakyReLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model)
        )
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 3)
        )
        self.down_conv_1 = ResidualDownBlock(d_model)
        self.down_conv_2 = ResidualDownBlock(d_model * 2)
        self.down_conv_3 = ResidualDownBlock(d_model * 4)
        self.up_conv_1 = ResidualUpBlock(d_model * 8)
        self.up_conv_2 = ResidualUpBlock(d_model * 4)
        self.up_conv_3 = ResidualUpBlock(d_model * 2)
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        point_embeddings = self.input_projection(x).permute(0, 2, 1)
        skip_1, enc_1 = self.down_conv_1(point_embeddings)
        skip_2, enc_2 = self.down_conv_2(enc_1)
        skip_3, enc_3 = self.down_conv_3(enc_2)
        dec_1 = self.up_conv_1(enc_3, skip_3)
        dec_2 = self.up_conv_2(dec_1, skip_2)
        dec_3 = self.up_conv_3(dec_2, skip_1)
        output = self.output_projection(dec_3.permute(0, 2, 1))
        return output
