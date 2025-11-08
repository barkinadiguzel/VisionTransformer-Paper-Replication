import torch
from torch import nn

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, num_patches, embedding_dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embedding_dim))

    def forward(self, x):
        x = x + self.pos_embedding
        return x
