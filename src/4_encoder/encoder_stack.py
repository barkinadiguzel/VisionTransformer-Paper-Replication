import torch
from torch import nn
from src.encoder.encoder_layer import EncoderLayer

class EncoderStack(nn.Module):
    def __init__(self, num_layers=12, d_model=768, n_heads=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model=d_model, n_heads=n_heads, dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
