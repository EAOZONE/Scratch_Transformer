import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Patch Embedding Modules ----------
class PatchEmbedding2D(nn.Module):
    """Splits a 2D image into patches and embeds them."""

    def __init__(self, in_channels, patch_size, embed_dim):
        super(PatchEmbedding2D, self).__init__()
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten(start_dim=2)  # Flatten spatial dimensions

    def forward(self, x):
        """
        Input: Tensor of shape [B, C, H, W] (batch, channels, height, width)
        Output: Tensor of shape [B, N, embed_dim] (N = number of patches)
        """
        x = self.projection(x)  # Create patches
        x = self.flatten(x)
        x = x.transpose(1, 2)  # Rearrange to [B, N, embed_dim]
        return x

class PatchEmbedding3D(nn.Module):
    """Splits a 3D tensor into patches and embeds them."""

    def __init__(self, in_channels, patch_size, embed_dim):
        super(PatchEmbedding3D, self).__init__()
        self.projection = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten(start_dim=2)  # Flatten spatial and temporal dimensions

    def forward(self, x):
        """
        Input: Tensor of shape [B, C, D, H, W] (batch, channels, depth, height, width)
        Output: Tensor of shape [B, N, embed_dim] (N = number of patches)
        """
        x = self.projection(x)  # Create patches
        x = self.flatten(x)
        x = x.transpose(1, 2)  # Rearrange to [B, N, embed_dim]
        return x


# ---------- Positional Encoding ----------
class PositionalEncoding2D(nn.Module):
    """Adds positional encoding to retain spatial structure."""

    def __init__(self, num_patches, embed_dim):
        super(PositionalEncoding2D, self).__init__()
        self.position_encoding = nn.Parameter(torch.randn(1, num_patches, embed_dim))  # Learnable positional encoding

    def forward(self, x):
        """
        Input: Tensor of shape [B, N, embed_dim]
        Output: Tensor of shape [B, N, embed_dim]
        """
        return x + self.position_encoding

class PositionalEncoding3D(nn.Module):
    """Adds positional encoding to retain spatial and temporal structure."""

    def __init__(self, num_patches, embed_dim):
        super(PositionalEncoding3D, self).__init__()
        self.position_encoding = nn.Parameter(torch.randn(1, num_patches, embed_dim))

    def forward(self, x):
        """
        Input: Tensor of shape [B, N, embed_dim]
        Output: Tensor of shape [B, N, embed_dim]
        """
        return x + self.position_encoding


# ---------- Multi-Head Self-Attention ----------
class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention mechanism."""

    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)  # Query, Key, Value
        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        Input: Tensor of shape [B, N, embed_dim]
        Output: Tensor of shape [B, N, embed_dim]
        """
        B, N, E = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # Split into Q, K, V and rearrange to [3, B, num_heads, N, head_dim]

        attn_weights = (q @ k.transpose(-2, -1)) * self.scale  # Scaled Dot-Product Attention
        attn_weights = F.softmax(attn_weights, dim=-1)  # Attention probabilities
        out = (attn_weights @ v).transpose(1, 2).reshape(B, N, E)  # Combine attention values

        return self.projection(out)  # Project back


# ---------- Transformer Encoder Layer ----------
class TransformerEncoderLayer(nn.Module):
    """A single layer of the Transformer encoder."""

    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Input: Tensor of shape [B, N, embed_dim]
        Output: Tensor of shape [B, N, embed_dim]
        """
        x = x + self.dropout(self.attention(self.norm1(x)))  # Add & Norm
        x = x + self.dropout(self.mlp(self.norm2(x)))  # Add & Norm
        return x


# -------- Vision Transformers ----------
class MyVIT2D(nn.Module):
    """Vision Transformer for 2D images."""

    def __init__(self, img_size, patch_size, in_channels, embed_dim, num_heads, mlp_dim, depth):
        super(MyVIT2D, self).__init__()

        # Calculate number of patches
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        # Layers
        self.patch_embedding = PatchEmbedding2D(in_channels, patch_size, embed_dim)
        self.positional_encoding = PositionalEncoding2D(self.num_patches, embed_dim)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_dim) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Classifier for scalar output (or regression)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x):
        """
        Input: Tensor of shape [B, C, H, W]
        Output: Tensor of shape [B] (Scalar predictions)
        """
        x = self.patch_embedding(x)  # [B, N, embed_dim]
        x = self.positional_encoding(x)  # Add positional information
        for layer in self.encoder_layers:  # Transformer layers
            x = layer(x)
        x = self.norm(x)  # Layer norm
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x).squeeze(-1)  # Regression scalar output

class MyVIT3D(nn.Module):
    """Vision Transformer for 3D data."""

    def __init__(self, img_size, patch_size, in_channels, embed_dim, num_heads, mlp_dim, depth):
        super(MyVIT3D, self).__init__()

        # Calculate number of patches
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (
                    img_size[2] // patch_size[2])

        # Layers
        self.patch_embedding = PatchEmbedding3D(in_channels, patch_size, embed_dim)
        self.positional_encoding = PositionalEncoding3D(self.num_patches, embed_dim)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_dim) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Classifier for scalar output (or regression)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x):
        """
        Input: Tensor of shape [B, C, D, H, W]
        Output: Tensor of shape [B] (Scalar predictions)
        """
        x = self.patch_embedding(x)  # [B, N, embed_dim]
        x = self.positional_encoding(x)  # Add positional information
        for layer in self.encoder_layers:  # Transformer layers
            x = layer(x)
        x = self.norm(x)  # Layer norm
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x).squeeze(-1)  # Regression scalar output