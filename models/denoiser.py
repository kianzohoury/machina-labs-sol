
import torch
import torch.nn as nn

from typing import List


from .transformer import Encoder, Decoder, DecoderLayer, SelfAttentionBlock, QueryGenerator

class DenoiserTransformer(nn.Module):
    """Denoising transformer using cross-attention."""
    def __init__(
        self,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        d_model: int = 64
    ):
        super(DenoiserTransformer, self).__init__()
        self.num_layers = num_layers 
        self.num_heads = num_heads
        self.dropout = dropout
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
        
        # define encoder
        self.encoder = Encoder(
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            d_model=d_model
        )
        
        # define decoder
        self.decoder = Decoder(
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            d_model=d_model  
        )
        
    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """Generates the estimate complete point cloud given an initial point cloud."""
        # generate higher dimensional point embeddings
        point_embeddings = self.input_projection(point_cloud.clone())
        
        # pass through encoder to get intermediate feature representations
        encoder_features = self.encoder(point_embeddings)
        
        # decode features to generate final point embeddings
        decoded_point_embeddings = self.decoder(
            query_embeddings=encoder_features[-1], encoder_features=encoder_features
        )
        
        # project back onto (x, y, z) to get estimated offset
        output = self.output_projection(decoded_point_embeddings)
        return output
    