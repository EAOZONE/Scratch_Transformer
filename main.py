import torch
from my_vit_models import MyVIT2D, MyVIT3D, MyVIT4D
from torch.amp import GradScaler, autocast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Example: Create random 2D data
# Batch size of 8, 3 channels (e.g., RGB), 224x224 images
x = torch.rand(8, 3, 224, 224, device=device)
# 3D data (batch size, channels, height, width, depth)
y = torch.rand(8, 3, 224, 224, 32, device=device)
# 4D data (batch size, channels, height, width, depth, time)
z = torch.rand(8, 3, 224, 224, 32, 4, device=device)

# Initialize GradScaler for mixed precision
scaler = GradScaler('cuda')

# Instantiate the Vision Transformer model
model2d = MyVIT2D()
model2d.to(device)

# Forward pass for 2D model
with autocast('cuda'):
    pred2d = model2d(x)  # Output should have the same shape as input
    # Compute a dummy loss
    loss2d = torch.sum(pred2d)

# Backward pass for 2D model
scaler.scale(loss2d).backward()
print("2D Loss computed:", loss2d.item())

# Instantiate the 3D model
model3d = MyVIT3D()
model3d.to(device)

# Forward pass for 3D model
with autocast('cuda'):
    pred3d = model3d(y)  # Output should have the same shape as input
    # Compute a dummy loss
    loss3d = torch.sum(pred3d)

# Backward pass for 3D model
scaler.scale(loss3d).backward()
print("3D Loss computed:", loss3d.item())

# Instantiate the 4D model
model4d = MyVIT4D()
model4d.to(device)

# Forward pass for 4D model
with autocast('cuda'):
    pred4d = model4d(z)
    loss4d = torch.sum(pred4d)

# Backward pass for 4D model
scaler.scale(loss4d).backward()
print("4D Loss computed:", loss4d.item())
