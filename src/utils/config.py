class ViTConfig:
    # Model architecture
    image_size = 224          # Input image size (224x224)
    patch_size = 16           # Each patch is 16x16
    num_channels = 3          # RGB images
    num_classes = 1000        # e.g., ImageNet

    # Transformer parameters
    dim = 768                 # Embedding dimension
    depth = 12                # Number of encoder layers
    heads = 12                # Number of attention heads
    mlp_dim = 3072            # Feed-forward hidden layer size
    dropout = 0.1
    attention_dropout = 0.1

    # Patch embedding
    patch_embedding_dim = dim  # Linear projection of flattened patches
