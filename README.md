# Vision Transformers for 2D, 3D, and 4D Data Classification with PyTorch

This repository contains implementations of Vision Transformers that handle 2D, 3D, and 4D data. The codes are written in PyTorch.

## Understanding the Code

The code primarily consists of two Python files: `main.py` and `my_vit_models.py`.

### main.py

The `main.py` file is the entry point for the program. It imports the necessary libraries and models that we need for our project. Inside this file, we create three separate batches of 2D, 3D, and 4D random data and process them using the corresponding Vision Transformer models (MyVIT2D, MyVIT3D, and MyVIT4D, respectively) imported from `my_vit_models.py`.

The forward pass of each model is wrapped with PyTorch's `autocast('cuda')` context manager, which performs automatic mixed precision training, improving the efficiency of the model.

### my_vit_models.py

In the `my_vit_models.py` file, we define different variants of Vision Transformers for 2D, 3D, and 4D data, respectively named as `MyVIT2D`, `MyVIT3D`, and `MyVIT4D`.

### PatchEmbedding

The `PatchEmbedding` class takes a 2D image as input, splits it into patches, and embeds these patches into a higher-dimensional space using a 2D convolution layer. The output is then flattened and rearranged to prepare it for the transformer layers.

### PatchEmbedding3D

The `PatchEmbedding3D` class works similarly to `PatchEmbedding` but is designed for 3D data. It applies a 3D convolution to split the input 3D data into patches and then embeds these patches.

### PositionalEncoding

`PositionalEncoding` class is used to add positional information to the input tokens. This is essentially a learned embedding for the position of the patches. The purpose of these embeddings is to provide the model with the information about the location of the patches in the original image, which is lost during flattening.

### PositionalEncoding3D

`PositionalEncoding3D` works similarly to `PositionalEncoding` but for 3D data.

### MultiHeadSelfAttention

The `MultiHeadSelfAttention` class implements the multi-head self-attention mechanism used in transformers. This mechanism allows the model to focus on different parts of the input independently (for each head), which can capture various aspects of the information present in the input layer.

### TransformerEncoderLayer

The `TransformerEncoderLayer` class represents a single layer in the Transformer encoder, comprising a multi-head self-attention mechanism and a feed-forward network. These components are made up of linear transformations, layer normalization, and dropout for regularization.

### MyVIT2D

`MyVIT2D` is a Vision Transformer model created specifically for 2D image data. It uses `PatchEmbedding` and `PositionalEncoding` to process the input image, then passes the processed input through a series of transformer layers, applies layer normalization, and if specified, a final linear classifier layer.

### MyVIT3D

`MyVIT3D` follows the same architecture as `MyVIT2D`, but it is designed for 3D data. It uses `PatchEmbedding3D` and `PositionalEncoding3D` to process the input.

### PatchEmbedding4D

`PatchEmbedding4D` is similar to the previous patch embedding classes but handles 4D (spatiotemporal) data. It applies a 3D convolution to each temporal slice and then aggregates these slices.

### MyVIT4D

Finally, `MyVit4D` is a Vision Transformer model that processes 4D (spatiotemporal) data. It uses `PatchEmbedding4D` to process the input, adds positional encoding, and runs a series of transformer layers. Post that, layer normalization and final linear classification layers are applied if specified.
## Usage

You will need a working installation of PyTorch. Then, you can simply run the `main.py` script.

Please note that this is just a basic implementation and might not include practices for efficient and scalable training, data loading etc. For deploying onto a more substantial and complex dataset, additional adjustments may be required.