import os
import random
import torch
from torch import nn, autocast
from torch.utils.data import random_split, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder
from torch.optim import Adam
from torch.cuda.amp import GradScaler
import matplotlib.pyplot as plt
from my_vit_models import MyVIT2D

IMG_SIZE = (900, 600)
PATCH_SIZE = (45, 30)
EMBED_DIM = 768
NUM_HEADS = 8
DEPTH = 6
MLP_DIM = 2048
BATCH_SIZE = 8
EPOCHS = 10
REDUCED_RATIO = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self, data_list, target_list):
        self.data_list = data_list
        self.target_list = target_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = torch.load(self.data_list[idx])
        target = torch.load(self.target_list[idx])
        data = data.permute(0, 3, 1, 2)  # Permute dimensions to [B, C, H, W]
        return data, target

def get_random_files(folder, num_files):
    all_files = [f for f in os.listdir(folder) if f.endswith('.pth')]
    selected_files = random.sample(all_files, num_files)
    return [os.path.join(folder, f) for f in selected_files]




def validate_patch_size(img_size, patch_size):
    """Validate that img_size is divisible by patch_size."""
    if any(img_size[i] % patch_size[i] != 0 for i in range(2)):
        raise ValueError(f"Image dimensions {img_size} must be divisible by patch size {patch_size}.")

def remove_batch_channel(x):
    """Remove extra batch channel."""
    return x.squeeze(0)

def main():
    validate_patch_size(IMG_SIZE, PATCH_SIZE)

    # Set the folder path and number of files to select
    folder_path = 'sample_data/2d'
    num_files = 50

    # Get random files for data and target lists
    data_list = get_random_files(folder_path, num_files)
    target_list = get_random_files(folder_path, num_files)

    dataset = CustomDataset(data_list, target_list)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model
    model = MyVIT2D(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        mlp_dim=MLP_DIM,
        depth=DEPTH,
        in_channels=24
    ).to(DEVICE)

    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    scaler = GradScaler()


    for epoch in range(EPOCHS):
        # Trains model on each batch
        model.train()
        optimizer.zero_grad()
        for i, (data, target) in enumerate(data_loader):
            B, _, H, W, D = data.shape
            data = data.permute(0, 4, 1, 2, 3).reshape(B, H, W, D)
            B, _, H, W, D = target.shape
            target = target.permute(0, 4, 1, 2, 3).reshape(D, H, W, B)
            data, target = data.to(DEVICE), target.to(DEVICE, dtype=torch.float32)

            with autocast(device_type='cuda', dtype=torch.float16):
                output = model(data)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            if (i + 1) % 4 == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        print(f"Epoch {epoch + 1}/{EPOCHS} Loss: {loss.item()}")

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target in data_loader:
            B, _, H, W, D = data.shape
            data = data.permute(0, 4, 1, 2, 3).reshape(B, H, W, D)
            B, _, H, W, D = target.shape
            target = target.permute(0, 4, 1, 2, 3).reshape(D, H, W, B)
            data, target = data.to(DEVICE, dtype=torch.float32), target.to(DEVICE, dtype=torch.float32)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
    print(total_loss / len(data_loader))

if __name__ == "__main__":
    main()