import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Splits the image into patches and embeds them."""

    def __init__(self, in_channels, patch_size, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        # Applies a 2D convolution over an input signal composed of several input planes.
        self.projection = nn.Conv2d(
            in_channels,  # number of features being passed in.
            embed_dim,  # number of kernels being used
            kernel_size=patch_size,  # the size of the kernel.
            stride=patch_size)  # controls the stride for the cross-correlation
        # Flattens a contiguous range of dims into a tensor
        self.flatten = nn.Flatten(2)

    def forward(self, x):
        """
        x: Tensor of shape [B, C, H, W]
        """
        # computes the 2D convolution for the input of x
        x = self.projection(x)
        # flattens and transposes the input
        x = self.flatten(x).transpose(1, 2)
        return x


class PatchEmbedding3D(nn.Module):
    """Splits the 3D volume into patches and embeds them."""

    def __init__(self, in_channels, patch_size, embed_dim):
        super(PatchEmbedding3D, self).__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.flatten = nn.Flatten(2)

    def forward(self, x):
        """
        x: Tensor of shape [B, C, D, H, W]
        """
        x = self.projection(x)
        x = self.flatten(x).transpose(1, 2)
        return x


class PositionalEncoding(nn.Module):
    """Positional Encoding for tokens to retain spatial information."""

    def __init__(self, num_patches, embed_dim):
        super(PositionalEncoding, self).__init__()
        # Define the positional encoding, that will enable the model to effectively process the spatial information
        self.position_encoding = nn.Parameter(torch.randn(1, num_patches, embed_dim))

    def forward(self, x):
        """
        x: Tensor of shape [B, N, embed_dim]
        """
        # Combine the tensor input and position_encoding
        return x + self.position_encoding


class PositionalEncoding3D(nn.Module):
    """Positional Encoding for 3D tokens to retain spatial information."""

    def __init__(self, num_patches, embed_dim):
        super(PositionalEncoding3D, self).__init__()
        self.position_encoding = nn.Parameter(torch.randn(1, num_patches, embed_dim))

    def forward(self, x):
        return x + self.position_encoding


class MultiHeadSelfAttention(nn.Module):
    """"Multi-Head Self-Attention Mechanism."""

    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        # Calculates the query (Q), key (K) and value (V) into three vectors
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        # Creates a linear network for output at end
        self.projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        x: Tensor of shape [B, N, embed_dim]
        """
        B, N, E = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Scale the dot product of the QKV
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # Calculate the softmax of the attention
        attn = F.softmax(attn, dim=-1)
        # Calculate matrix multiplication between the attention and value tensor and transpose and reshape respectively
        out = (attn @ v).transpose(1, 2).reshape(B, N, E)
        # Output is passed through a linear network to get the output
        return self.projection(out)


class TransformerEncoderLayer(nn.Module):
    """Single Transformer Encoder Layer."""

    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        # Gets the MultiHeadSelfAttention calculation using embed_dim and num_heads
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        # Applies Layer Normalization over a mini-batch of inputs.
        self.ln1 = nn.LayerNorm(embed_dim)
        # Using Sequential to create a small model.
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim)
        )
        # Makes another Layer Normalization with the same inputs
        self.ln2 = nn.LayerNorm(embed_dim)
        # Gets dropout to randomly during training zero out some of the input tensor with probability
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: Tensor of shape [B, N, embed_dim]
        """
        # Combines the input with the dropout of the MultiHeadSelfAttention with input of the first layer norm
        x = x + self.dropout(self.attn(self.ln1(x)))
        # Adds the second layer norm input in a sequential model using dropout
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


class MyVIT2D(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, num_heads=8,
                 mlp_dim=2048, depth=12, num_classes=None):
        super(MyVIT2D, self).__init__()
        assert img_size % patch_size == 0

        self.num_patches = (img_size // patch_size) ** 2
        # initializes PatchEmbedding and PositionalEncoding
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)
        self.pos_encoding = PositionalEncoding(self.num_patches, embed_dim)

        # Puts the TransformerEncoderLayer in the PyTorch Neural Network Module List
        self.transformer = nn.ModuleList([TransformerEncoderLayer(embed_dim, num_heads, mlp_dim) for _ in range(depth)])
        # Applies Layer Normalization over a mini-batch of inputs.
        self.ln = nn.LayerNorm(embed_dim)
        # Makes a linear transformation of the embedded data and number of classes
        self.classifier = nn.Linear(embed_dim, num_classes) if num_classes else None

    def forward(self, x):
        """
        x: Tensor of shape [B, C, H, W]
        """
        # Use the tensor input in the patch_embed and pos_encoding
        x = self.patch_embed(x)
        x = self.pos_encoding(x)

        # for every transformer put in the module list the tensor gains another layer
        for layer in self.transformer:
            x = layer(x)
        # uses the layer normalization on the tensor
        x = self.ln(x)

        if self.classifier:
            x = x.mean(dim=1)
            x = self.classifier(x)

        return x


class MyVIT3D(nn.Module):
    def __init__(self, img_size=(224, 224, 32), patch_size=16, in_channels=3, embed_dim=768, num_heads=8,
                 mlp_dim=2048, depth=12, num_classes=None):
        super(MyVIT3D, self).__init__()
        assert all(size % patch_size == 0 for size in img_size)

        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size) * (img_size[2] // patch_size)
        # initializes PatchEmbedding3D and PositionalEncoding3D
        self.patch_embed = PatchEmbedding3D(in_channels, patch_size, embed_dim)
        self.pos_encoding = PositionalEncoding3D(self.num_patches, embed_dim)

        # Puts the TransformerEncoderLayer in the PyTorch Neural Network Module List
        self.transformer = nn.ModuleList([TransformerEncoderLayer(embed_dim, num_heads, mlp_dim) for _ in range(depth)])

        # Applies Layer Normalization over a mini-batch of inputs.
        self.ln = nn.LayerNorm(embed_dim)
        # Makes a linear transformation of the embedded data and number of classes
        self.classifier = nn.Linear(embed_dim, num_classes) if num_classes else None

    def forward(self, x):
        """
        x: Tensor of shape [B, C, H, W, D]
        """
        # Use the tensor input in the patch_embed and pos_encoding
        x = self.patch_embed(x)
        x = self.pos_encoding(x)

        # for every transformer put in the module list the tensor gains another layer
        for layer in self.transformer:
            x = layer(x)

        # uses the layer normalization on the tensor
        x = self.ln(x)

        if self.classifier:
            x = x.mean(dim=1)
            x = self.classifier(x)

        return x


class PatchEmbedding4D(nn.Module):
    def __init__(self, in_channels=3, patch_size=(16, 16, 4, 2), embed_dim=768):
        super(PatchEmbedding4D, self).__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        # Adjust kernel_size and stride to handle new depth*time dimension
        self.projection = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(patch_size[0], patch_size[1], patch_size[2] * patch_size[3]),
            stride=(patch_size[0], patch_size[1], patch_size[2] * patch_size[3])
        )

    def forward(self, x):
        # Reshape input from [B, C, H, W, D, T] to [B, C, H, W, D*T]
        x = x.view(x.size(0), x.size(1), x.size(2), x.size(3), -1)

        # Now perform projection
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class MyVIT4D(nn.Module):
    def __init__(self, img_size=(224, 224, 32, 4), patch_size=(16, 16, 4, 2), in_channels=3, embed_dim=768, num_heads=8,
                 mlp_dim=2048, depth=12, num_classes=None):
        super(MyVIT4D, self).__init__()

        self.patch_embedding = PatchEmbedding4D(in_channels, patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=mlp_dim,
                batch_first=True
            ),
            num_layers=depth
        )

        self.head = nn.Linear(embed_dim, num_classes) if num_classes is not None else nn.Identity()

    def forward(self, x):
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.layer_norm(x)
        x = self.transformer(x)
        cls_token_final = x[:, 0]

        if self.head:
            x = self.head(cls_token_final)

        return x