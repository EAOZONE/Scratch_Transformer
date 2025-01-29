import random
import torch
from torch import nn, autocast
from torch.utils.data import random_split, DataLoader, Subset
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
EPOCHS = 100
REDUCED_RATIO = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validate_patch_size(img_size, patch_size):
    """Validate that img_size is divisible by patch_size."""
    if any(img_size[i] % patch_size[i] != 0 for i in range(2)):
        raise ValueError(f"Image dimensions {img_size} must be divisible by patch size {patch_size}.")

def remove_batch_channel(x):
    """Remove extra batch channel."""
    return x.squeeze(0)

def tensor_transform(tensor):
    """Global function to normalize tensors."""
    return (tensor - 0.5) / 0.5


def preprocess_data():
    """Return the global transform function."""
    return tensor_transform

def visualize_image(data, title="Image"):
    """Visualize the data using Matplotlib."""
    data_to_plot = data[:3, :, :]
    data_to_plot = data_to_plot.transpose(1, 2, 0)
    plt.imshow(data_to_plot)
    plt.title(title)
    plt.show()

def file_loader(filepath):
    """file loader function for PyTorch .pth tensor files."""
    data = torch.load(filepath, weights_only=False)
    return data

def main():
    validate_patch_size(IMG_SIZE, PATCH_SIZE)

    # Load the dataset
    dataset = DatasetFolder(
        root='./sample_data/',
        loader=file_loader,
        extensions=('.pth',),
        transform=preprocess_data()
    )

    # Get datasets for training, validation, and testing
    test_size = int(0.2 * len(dataset))
    input_size = 5
    train_size = len(dataset) - test_size - input_size
    train_dataset, test_dataset, sample_input = random_split(dataset, [train_size, test_size, input_size])

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

    all_train_indices = list(range(len(train_dataset)))
    sampled_indices = random.sample(all_train_indices, int(len(train_dataset) * REDUCED_RATIO))
    sampler = torch.utils.data.SubsetRandomSampler(sampled_indices)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)


    for epoch in range(EPOCHS):
        train_loader.sampler.indices = random.sample(all_train_indices, int(len(train_dataset) * REDUCED_RATIO))
        # Trains model on each batch
        model.train()
        optimizer.zero_grad()
        for i, (data, target) in enumerate(train_loader):
            B, _, H, W, D = data.shape
            data = data.permute(0, 4, 1, 2, 3).reshape(B, D, H, W)
            data, target = data.to(DEVICE), target.to(DEVICE, dtype=torch.float32)

            target = target.view(-1, 1)

            with autocast(device_type='cuda', dtype=torch.float16):
                output = model(data)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            if (i + 1) % 4 == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        print(f"Epoch {epoch + 1}/{EPOCHS} Loss: {loss.item()}")


if __name__ == "__main__":
    main()