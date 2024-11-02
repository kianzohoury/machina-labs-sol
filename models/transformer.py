
from typing import List

import torch
import torch.nn as nn


class QueryGenerator(nn.Module):
    """Generates query point embeddings given the encoded features of the point cloud."""
    def __init__(
        self,
        num_layers: int = 4, 
        d_model: int = 256,
        num_queries: int = 512
    ):
        super(QueryGenerator, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_queries = num_queries

        # define feature projection layer
        self.feature_projection = nn.Linear(num_layers * d_model, d_model * 4)
        
        # define coordinate projection layer: creates num_queries (x, y, z) starting points
        self.coord_projection = nn.Linear(d_model * 4, num_queries * 3)
        
        # define the aggregate projection layer (combines summary features and coordinate projections)
        self.agg_projection = nn.Linear(d_model * 4 + 3, d_model)

    def forward(self, encoder_features: List[torch.Tensor]) -> torch.Tensor:
        """Generates the query points."""
        bsize, seq_len = encoder_features[0].shape[:2]
        
        # concatenate all encoded features along last dimension
        encoder_features = torch.cat(encoder_features, dim=-1)

        # project to a higher dimension
        encoder_features = self.feature_projection(encoder_features)
        
        # max/mean pool to extract a "summary" of the encoded point cloud
        summary_features = encoder_features.mean(dim=1)
        
        # get coordinate projections
        coord_features = self.coord_projection(summary_features)
        
        # reshape
        coord_features = coord_features.view(bsize, self.num_queries, 3)
        
        # concatenate summary features with coordinates
        summary_features = summary_features.unsqueeze(1).expand(bsize, self.num_queries, summary_features.shape[-1])    
        concat_features = torch.cat([summary_features, coord_features], dim=-1)
            
        # aggregate features to create query embeddings
        query_embeddings = self.agg_projection(concat_features)
        return query_embeddings
    

class DecoderLayer(nn.Module):
    """Single decoder layer."""
    def __init__(
        self,
        num_heads: int = 8,
        dropout: float = 0.1,
        d_model: int = 64,
    ) -> None:
        super(DecoderLayer, self).__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.d_model = d_model  
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True
        )
        
        # define the feed forward layers + layer norm
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )
        
        # define layer normalization layers
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        
    def forward(
        self,
        query_embeddings: torch.Tensor,
        encoder_features: torch.Tensor
    ) -> torch.Tensor:
        """Applies cross-attention using the intermediate query embeddings and 
        the corresponding encoder features."""
        cross_attention, _ = self.cross_attention(
            query=query_embeddings, key=encoder_features, value=encoder_features
        )
        skip = self.layer_norm_1(cross_attention + query_embeddings)
        decoded_features = self.layer_norm_2(self.fc(skip) + skip)
        return decoded_features
        
        
class Decoder(nn.Module):
    """Decoder module for generating the missing points using cross-attention."""
    def __init__(
        self,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        d_model: int = 64,
    ) -> None:
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.d_model = d_model
        
        # define self attention layer
        self.self_attention = SelfAttentionBlock(d_model, num_heads, dropout)
        
        # define layer norm 
        self.layer_norm = nn.LayerNorm(d_model)

        # define the cross attention layers
        self.decoder = nn.ModuleList(
            [DecoderLayer(d_model=d_model, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        
    def forward(self, query_embeddings: torch.Tensor, encoder_features: List[torch.Tensor]) -> torch.Tensor:
        """Generates final decoded representations for each predicted missing point."""
        
        # apply self attention to initial query embeddings
        query_embeddings = self.self_attention(query_embeddings)        

        # apply cross attention to drive processing of query embeddings using encoded features
        for i, decoder_layer in enumerate(self.decoder):
            query_embeddings = decoder_layer(
                query_embeddings=query_embeddings, encoder_features=encoder_features[i]
            )
        return query_embeddings


class SelfAttentionBlock(nn.Module):
    """Standard residual self-attention block with feed-forward layers."""
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super(SelfAttentionBlock, self).__init__()
        
        # multi-head attention layer
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True
        )
        
        # layer normalization
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        
        # feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # get self-attention matrix
        attention, _ = self.self_attention(x, x, x)
        x = x + self.dropout(attention)
        x = self.layer_norm_1(x)
        
        # pass through feed-forward layers
        skip = self.feed_forward(x)
        output = self.layer_norm_2(skip + self.dropout(skip))
        return output

class Encoder(nn.Module):
    """Encoder module for extracting local contextual/geometric features"""
    def __init__(
        self,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        d_model: int = 256,
    ) -> None:
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.d_model = d_model
        
        # define individual encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            batch_first=True, 
            dropout=dropout
        )

        # define encoder (enc x num_layers)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers, 
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass returns encoded features 1,...,num_layers."""
        encoder_features = []
        for encoder_layer in self.encoder.layers:
            x = encoder_layer(x)
            encoder_features.append(x)
        return encoder_features
    