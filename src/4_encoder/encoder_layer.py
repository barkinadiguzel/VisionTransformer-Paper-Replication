import torch
from torch import nn
from src.attention.scaled_dot_product import ScaledDotProductAttention
from src.attention.multi_head_attention import MultiHeadAttention

class EncoderLayer(nn.Module):
    def __init__(self, d_model=768, n_heads=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model=d_model, h=n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        attn_output, _ = self.mha(x, x, x, mask)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x
