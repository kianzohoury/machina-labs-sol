
import torch
import torch.nn as nn


from .transformer import Encoder

class Detector(nn.Module):
    """Simple detection model for identifying synthetic vs. real point clouds."""
    def __init__(
        self,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        d_model: int = 64
    ):
        super(Detector, self).__init__()
        
        # standard learnable point embeddings to project
        # points from (x, y, z) to higher dimensional space
        self.input_projection = nn.Sequential(
            nn.Linear(3, d_model),
            nn.LeakyReLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model)
        )
        
        # define encoder
        self.encoder = Encoder(
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            d_model=d_model
        )
        
        # define classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.LeakyReLU(),
            nn.LayerNorm(d_model * 4),
            nn.Linear(d_model * 4, d_model),
            nn.LeakyReLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )
        
    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """Runs detection model and returns logits."""
        # generate higher dimensional point embeddings
        point_embeddings = self.input_projection(point_cloud)
        
        # pass through encoder and just take last features
        encoder_features = self.encoder(point_embeddings)[-1]
        
        # pool features along sequence dimension
        pooled_features = encoder_features.mean(dim=1)
        
        # pass through classifier
        logits = self.classifier(pooled_features)
        return logits
    