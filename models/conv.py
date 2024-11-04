
from typing import Tuple


import torch
import torch.nn as nn
import torch.nn.functional as F

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
        offset = self.output_projection(dec_3.permute(0, 2, 1))
        denoised_point_cloud = x + offset
        return denoised_point_cloud


# def knn(x, k):
#     """Compute k-nearest neighbors for each point in the point cloud."""
#     inner = -2 * torch.matmul(x.transpose(2, 1), x)
#     xx = torch.sum(x**2, dim=1, keepdim=True)
#     pairwise_distance = -xx - inner - xx.transpose(2, 1)
#     idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
#     return idx

# def get_graph_feature(x, k=20, idx=None):
#     """Construct graph features for EdgeConv operations."""
#     batch_size, num_dims, num_points = x.size()
#     if idx is None:
#         idx = knn(x, k=k)  # (batch_size, num_points, k)
#     device = x.device

#     idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
#     idx = idx + idx_base
#     idx = idx.view(-1)

#     x = x.transpose(2, 1).contiguous()  # (batch_size * num_points, num_dims)
#     feature = x.view(batch_size * num_points, -1)[idx, :]
#     feature = feature.view(batch_size, num_points, k, num_dims)
#     x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

#     feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
#     return feature  # (batch_size, 2 * num_dims, num_points, k)

# class DGCNNEncoder(nn.Module):
#     """Encoder module using DGCNN architecture."""
#     def __init__(self, k=20, emb_dims=1024):
#         super(DGCNNEncoder, self).__init__()
#         self.k = k

#         self.conv1 = nn.Sequential(
#             nn.Conv2d(6, 64, kernel_size=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(128, 64, kernel_size=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(128, 128, kernel_size=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU()
#         )
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(256, 256, kernel_size=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU()
#         )
#         self.conv5 = nn.Sequential(
#             nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
#             nn.BatchNorm1d(emb_dims),
#             nn.ReLU()
#         )

#     def forward(self, x):
#         batch_size, num_dims, num_points = x.size()

#         x0 = get_graph_feature(x, k=self.k)
#         x = self.conv1(x0)
#         x1 = x.max(dim=-1)[0]

#         x0 = get_graph_feature(x1, k=self.k)
#         x = self.conv2(x0)
#         x2 = x.max(dim=-1)[0]

#         x0 = get_graph_feature(x2, k=self.k)
#         x = self.conv3(x0)
#         x3 = x.max(dim=-1)[0]

#         x0 = get_graph_feature(x3, k=self.k)
#         x = self.conv4(x0)
#         x4 = x.max(dim=-1)[0]

#         x = torch.cat((x1, x2, x3, x4), dim=1)
#         x = self.conv5(x)
#         x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
#         return x

# class DGCNNDecoder(nn.Module):
#     """Decoder module to reconstruct the point cloud."""
#     def __init__(self, emb_dims=1024, num_points=1024):
#         super(DGCNNDecoder, self).__init__()
#         self.num_points = num_points

#         self.fc1 = nn.Linear(emb_dims, 1024)
#         self.fc2 = nn.Linear(1024, 2048)
#         self.fc3 = nn.Linear(2048, num_points * 3)

#         self.bn1 = nn.BatchNorm1d(1024)
#         self.bn2 = nn.BatchNorm1d(2048)

#     def forward(self, x):
#         x = F.relu(self.bn1(self.fc1(x)))
#         x = F.relu(self.bn2(self.fc2(x)))
#         x = self.fc3(x)
#         x = x.view(-1, 3, self.num_points)
#         return x

# class DenoiserConv(nn.Module):
#     """Autoencoder combining the encoder and decoder modules."""
#     def __init__(self, num_points=1024, emb_dims=1024, k=20, **kwargs):
#         super(DenoiserConv, self).__init__()
#         self.encoder = DGCNNEncoder(k=k, emb_dims=emb_dims)
#         self.decoder = DGCNNDecoder(emb_dims=emb_dims, num_points=num_points)

#     def forward(self, x):
#         x = self.encoder(x.permute(0, 2, 1))
#         x = self.decoder(x).permute(0, 2, 1)
#         return x