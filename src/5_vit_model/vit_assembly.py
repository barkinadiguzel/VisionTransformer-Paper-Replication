import torch
from torch import nn
from src.input_embedding.patch_embedding import PatchEmbeddingWithClassToken
from src.input_embedding.positional_encoding import PositionalEncoding
from src.encoder.encoder_stack import EncoderStack

class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        embedding_dim=768,
        num_layers=12,
        num_heads=8,
        dim_feedforward=2048,
        num_classes=1000,
        dropout=0.1
    ):
        super().__init__()
        self.patch_embed = PatchEmbeddingWithClassToken(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embedding_dim=embedding_dim
        )
        self.pos_embed = PositionalEncoding(embedding_dim, dropout)
        self.encoder = EncoderStack(
            num_layers=num_layers,
            d_model=embedding_dim,
            n_heads=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.mlp_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_embed(x)
        x = self.encoder(x)
        class_token_output = x[:, 0]  # prepended class token
        out = self.mlp_head(class_token_output)
        return out
