import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Splits a 2D image into patches and embeds them into a higher-dimensional space."""

    def __init__(self, in_channels, patch_size, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        # Applies a 2D convolution to create patches from the input image
        self.projection = nn.Conv2d(
            in_channels,  # Number of input channels (e.g., RGB)
            embed_dim,  # Number of output channels (embedding dimension)
            kernel_size=patch_size,  # Size of the patch
            stride=patch_size)  # Stride equal to patch size to ensure non-overlapping patches
        # Flattens the output into a 2D tensor for further processing
        self.flatten = nn.Flatten(2)

    def forward(self, x):
        """
        x: Tensor of shape [B, C, H, W] where B is batch size, C is channels, H is height, W is width
        """
        # Apply convolution to create patches
        x = self.projection(x)
        # Flatten and transpose the output to prepare for transformer input
        x = self.flatten(x).transpose(1, 2)
        return x


class PatchEmbedding3D(nn.Module):
    """Splits a 3D image into patches and embeds them into a higher-dimensional space."""

    def __init__(self, in_channels, patch_size, embed_dim):
        super(PatchEmbedding3D, self).__init__()
        self.patch_size = patch_size
        # Applies a 3D convolution to create patches from the input 3D data
        self.projection = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        # Flattens the output into a 2D tensor for further processing
        self.flatten = nn.Flatten(2)

    def forward(self, x):
        """
        x: Tensor of shape [B, C, D, H, W] where D is depth
        """
        # Apply convolution to create patches
        x = self.projection(x)
        # Flatten and transpose the output to prepare for transformer input
        x = self.flatten(x).transpose(1, 2)
        return x


class PositionalEncoding(nn.Module):
    """Adds positional encoding to the tokens to retain spatial information in 2D data."""

    def __init__(self, num_patches, embed_dim):
        super(PositionalEncoding, self).__init__()
        # Initialize positional encoding parameters
        self.position_encoding = nn.Parameter(torch.randn(1, num_patches, embed_dim))

    def forward(self, x):
        """
        x: Tensor of shape [B, N, embed_dim] where N is the number of patches
        """
        # Add positional encoding to the input tokens
        return x + self.position_encoding


class PositionalEncoding3D(nn.Module):
    """Adds positional encoding to the tokens to retain spatial information in 3D data."""

    def __init__(self, num_patches, embed_dim):
        super(PositionalEncoding3D, self).__init__()
        # Initialize positional encoding parameters
        self.position_encoding = nn.Parameter(torch.randn(1, num_patches, embed_dim))

    def forward(self, x):
        # Add positional encoding to the input tokens
        return x + self.position_encoding


class MultiHeadSelfAttention(nn.Module):
    """Implements the Multi-Head Self-Attention mechanism used in transformers."""

    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # Dimension of each attention head
        self.scale = self.head_dim ** -0.5  # Scaling factor for attention scores
        # Linear layer to compute query, key, and value vectors
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        # Linear layer for output projection
        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        x: Tensor of shape [B, N, embed_dim]
        """
        B, N, E = x.shape  # B: batch size, N: number of tokens, E: embedding dimension
        # Compute Q, K, V matrices
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Compute scaled dot-product attention
        attn = (q @ k.transpose(- 2, -1)) * self.scale
        # Apply softmax to get attention weights
        attn = F.softmax(attn, dim=-1)
        # Compute the output by applying attention weights to the value vectors
        out = (attn @ v).transpose(1, 2).reshape(B, N, E)
        # Project the output back to the original embedding dimension
        return self.projection(out)


class TransformerEncoderLayer(nn.Module):
    """Defines a single layer of the Transformer encoder, consisting of multi-head self-attention and feed-forward networks."""

    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        # Multi-head self-attention layer
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        # Layer normalization applied before the attention and feed-forward networks
        self.ln1 = nn.LayerNorm(embed_dim)
        # Feed-forward network with two linear layers and GELU activation
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim)
        )
        # Layer normalization applied after the feed-forward network
        self.ln2 = nn.LayerNorm(embed_dim)
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: Tensor of shape [B, N, embed_dim]
        """
        # Apply multi-head self-attention and add residual connection
        x = x + self.dropout(self.attn(self.ln1(x)))
        # Apply feed-forward network and add residual connection
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


class MyVIT2D(nn.Module):
    """Defines a Vision Transformer model for 2D image classification."""

    def __init__(self, img_size=(224, 224), patch_size=16, in_channels=3, embed_dim=768, num_heads=8,
                 mlp_dim=2048, depth=12, num_classes=None):
        super(MyVIT2D, self).__init__()
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0  # Ensure the image size is divisible by the patch size

        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size) # Calculate the number of patches
        # Initialize patch embedding and positional encoding layers
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)
        self.pos_encoding = PositionalEncoding(self.num_patches, embed_dim)

        # Create a list of transformer encoder layers
        self.transformer = nn.ModuleList([TransformerEncoderLayer(embed_dim, num_heads, mlp_dim) for _ in range(depth)])
        # Layer normalization applied before the final classification
        self.ln = nn.LayerNorm(embed_dim)
        # Linear layer for classification if num_classes is specified
        self.classifier = nn.Linear(embed_dim, num_classes) if num_classes else None

    def forward(self, x):
        """
        x: Tensor of shape [B, C, H, W]
        """
        # Embed patches and add positional encoding
        x = self.patch_embed(x)
        x = self.pos_encoding(x)

        # Pass through each transformer layer
        for layer in self.transformer:
            x = layer(x)
        # Apply layer normalization
        x = self.ln(x)

        if self.classifier:
            # Average pooling over the sequence of tokens and apply classifier
            x = x.mean(dim=1)
            x = self.classifier(x)

        return x


class MyVIT3D(nn.Module):
    """Defines a Vision Transformer model for 3D data classification."""

    def __init__(self, img_size=(224, 224, 32), patch_size=16, in_channels=3, embed_dim=768, num_heads=8,
                 mlp_dim=2048, depth=12, num_classes=None):
        super(MyVIT3D, self).__init__()
        assert all(size % patch_size == 0 for size in img_size)  # Ensure all dimensions are divisible by patch size

        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size) * (img_size[2] // patch_size)
        # Initialize patch embedding and positional encoding layers for 3D data
        self.patch_embed = PatchEmbedding3D(in_channels, patch_size, embed_dim)
        self.pos_encoding = PositionalEncoding3D(self.num_patches, embed_dim)

        # Create a list of transformer encoder layers
        self.transformer = nn.ModuleList([TransformerEncoderLayer(embed_dim, num_heads , mlp_dim) for _ in range(depth)])
        # Layer normalization applied before the final classification
        self.ln = nn.LayerNorm(embed_dim)
        # Linear layer for classification if num_classes is specified
        self.classifier = nn.Linear(embed_dim, num_classes) if num_classes else None

    def forward(self, x):
        """
        x: Tensor of shape [B, C, D, H, W]
        """
        # Embed patches and add positional encoding
        x = self.patch_embed(x)
        x = self.pos_encoding(x)

        # Pass through each transformer layer
        for layer in self.transformer:
            x = layer(x)

        # Apply layer normalization
        x = self.ln(x)

        if self.classifier:
            # Average pooling over the sequence of tokens and apply classifier
            x = x.mean(dim=1)
            x = self.classifier(x)

        return x


class PatchEmbedding4D(nn.Module):
    def __init__(self, in_channels=3, spatial_patch_size=(16, 16), temporal_patch_size=4, embed_dim=768):
        super(PatchEmbedding4D, self).__init__()
        self.in_channels = in_channels
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.embed_dim = embed_dim

        # 3D convolution
        self.projection = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(self.temporal_patch_size, *self.spatial_patch_size),
            stride=(self.temporal_patch_size, *self.spatial_patch_size)
        )

    def forward(self, x):
        b, c, h, w, d, t = x.shape
        # Reshape to treat temporal dimension as part of the batch dimension
        x = x.view(b * t, c, h, w, d)
        # Apply 3D convolution to each temporal slice
        x = self.projection(x)
        # Flattening the features
        x = x.flatten(2)
        x = x.transpose(1, 2)
        # Reshape back to separate the temporal dimension
        x = x.view(b, t, -1, self.embed_dim)
        return x


class MyVIT4D(nn.Module):
    def __init__(self, in_channels=3, patch_size=(16, 16, 4), embed_dim=768):
        super(MyVIT4D, self).__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Adjust the number of patches based on the output shape from PatchEmbedding4D
        num_patches = ((224 // patch_size[1]) * (224 // patch_size[2])) * (
                    32 // patch_size[0])  # Assuming fixed dimensions of input
        # Initialize patch embedding and positional encoding layers for 4D data
        self.patch_embedding = PatchEmbedding4D(in_channels, patch_size[1:], patch_size[0], embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        # Create a list of transformer encoder layers
        self.transformer = nn.ModuleList([TransformerEncoderLayer(embed_dim, 8, 2048) for _ in range(6)])
        # Linear layer for classification if num_classes is specified
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Embed patches
        x = self.patch_embedding(x)
        b, t, n, _ = x.shape

        # Include the class token
        cls_tokens = self.cls_token.expand(b, t, -1, -1)
        x = torch.cat((cls_tokens, x), dim=2)
        x = x.mean(dim=1)  # Aggregating over the temporal dimension

        # Add position encoding
        x += self.pos_encoding[:, :(n + 1)]
        # Pass through each transformer layer
        for layer in self.transformer:
            x = layer(x)
        # Apply layer normalization
        x = self.ln(x[:, 0])

        return x