import torch
from torch import nn

class PatchEmbeddingWithClassToken(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embedding_dim=768):
        super().__init__()
        self.patch_embedding = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.class_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embedding(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        class_token = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat((class_token, x), dim=1)
        return x
