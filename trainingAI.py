import random
import torch
from torch import nn, autocast
from torch.utils.data import random_split, DataLoader, Subset
from torchvision.datasets import DatasetFolder
from torch.optim import Adam
from torch.cuda.amp import GradScaler
import matplotlib.pyplot as plt
from my_vit_models import MyVIT2D

# ---------- Configuration Variables ----------
IMG_SIZE = (900, 600)  # Dimensions of the input data
PATCH_SIZE = (45, 30)  # Patch size must divide img_size evenly
EMBED_DIM = 768  # Embedding dimension for tokens
NUM_HEADS = 8  # Number of attention heads
DEPTH = 6  # Number of transformer layers
MLP_DIM = 2048  # Feed-forward network dimension in transformer
BATCH_SIZE = 8  # Batch size for training
EPOCHS = 10  # Number of training epochs
REDUCED_RATIO = 0.2  # Percentage of dataset used for training per epoch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- Helper Functions ----------
def validate_patch_size(img_size, patch_size):
    """Validate that img_size is divisible by patch_size."""
    if any(img_size[i] % patch_size[i] != 0 for i in range(2)):
        raise ValueError(f"Image dimensions {img_size} must be divisible by patch size {patch_size}.")


def remove_batch_channel(x):
    """Remove extra batch channel."""
    return x.squeeze(0)  # Assuming the input has an extra channel that needs to be removed


def tensor_transform(tensor):
    """Global function to normalize tensors."""
    return (tensor - 0.5) / 0.5  # Example normalization


def preprocess_data():
    """Return the global transform function."""
    return tensor_transform

def visualize_image(data, title="Image"):
    # Select 3 channels for RGB: for example, use channels 0, 1, and 2
    data_to_plot = data[:3, :, :]  # Shape becomes (3, 900, 600)
    # Transpose to (H, W, C) for Matplotlib
    data_to_plot = data_to_plot.transpose(1, 2, 0)  # Shape now is (900, 600, 3)
    plt.imshow(data_to_plot)
    plt.title(title)
    plt.show()

def visualize_prediction(prediction, title="Prediction"):
    """
    Visualize the prediction using Matplotlib after normalizing to a valid range.
    """

    # Plot the normalized data
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().detach().numpy()  # Convert to NumPy
    prediction = prediction.squeeze()  # Flatten any extra dimensions
    print(prediction.min(), prediction.max())
    prediction_normalized = (prediction - prediction.min()) / (prediction.max() - prediction.min())
    plt.figure(figsize=(4, 4))
    plt.bar(["Prediction"], [prediction_normalized], color='blue')
    plt.title(title)
    plt.ylabel("Value")
    plt.show()

def custom_loader(filepath):
    """Custom loader function for PyTorch .pth tensor files."""
    data = torch.load(filepath)  # Load PyTorch tensor from .pth file
    return data  # Return the loaded tensor

# ---------- Main Code ----------
def main():
    validate_patch_size(IMG_SIZE, PATCH_SIZE)

    # Load the dataset
    dataset = DatasetFolder(
        root='./sample_data/',
        loader=custom_loader,  # Custom loader for .pth tensor files
        extensions=('.pth',),  # Looking for .pth files
        transform=preprocess_data()  # Apply normalization to loaded tensors
    )

    # Split the dataset (same as before)
    test_size = int(0.2 * len(dataset))
    input_size = 5
    train_size = len(dataset) - test_size - input_size
    train_dataset, test_dataset, sample_input = random_split(dataset, [train_size, test_size, input_size])

    # Initialize the 2D Vision Transformer
    model = MyVIT2D(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        mlp_dim=MLP_DIM,
        depth=DEPTH,
        in_channels=24  # For RGB input (update if single-channel input)
    ).to(DEVICE)

    # Define optimizer, criterion, and scaler (unchanged)
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()  # Assuming regression task
    scaler = GradScaler()

    # Training and validation loop (only data shape adjustments if necessary)
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        indices = random.sample(range(len(train_dataset)), int(len(train_dataset) * REDUCED_RATIO))
        epoch_train_dataset = Subset(train_dataset, indices)
        print(f"Total samples in subset: {len(epoch_train_dataset)}")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Expected number of full batches: {len(epoch_train_dataset) // BATCH_SIZE}")
        print(f"Samples in last batch (if any): {len(epoch_train_dataset) % BATCH_SIZE}")
        train_loader = DataLoader(epoch_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                                  pin_memory=True)

        model.train()
        optimizer.zero_grad()
        for i, (data, target) in enumerate(train_loader):
            # Combine depth dimension (D=24) into the channel dimension
            print(data.shape)
            B, _, H, W, D = data.shape  # Data shape: [B, 1, H, W, D]
            data = data.permute(0, 4, 1, 2, 3).reshape(B, D, H, W)  # New shape: [B, D, H, W]
            print(f"Data Shape: {data.shape}")
            # Move data and target to the device
            data, target = data.to(DEVICE).float(), target.to(DEVICE).float()

            # Ensure target shape matches [B, 1]
            target = target.view(-1, 1)

            # Forward pass
            with autocast(device_type='cuda'):
                output = model(data)  # Model expects shape [B, C, H, W]
                loss = criterion(output, target)

            # Backward Pass
            scaler.scale(loss).backward()
            if (i + 1) % 4 == 0:  # Accumulate gradients every 4 batches
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            print(f"Batch {i + 1}, Loss: {loss.item()}")

        # Validation on test set
        model.eval()
        total_loss = 0
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader):
                B, _, H, W, D = data.shape  # Data shape: [B, 1, H, W, D]
                data = data.permute(0, 4, 1, 2, 3).reshape(B, D, H, W)  # New shape: [B, D, H, W]
                print(f"Data Shape: {data.shape}")
                data, target = data.to(DEVICE).float(), target.to(DEVICE).float()
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item()
                print(f"Test Batch {i + 1}, Loss: {loss.item()}")

        print(f"Epoch {epoch + 1} Validation Loss: {total_loss / len(test_loader)}")

    # Sample Prediction
    # Sample Prediction Visualization
    print("Generating Predictions on Sample Input:")
    model.eval()  # Set model to evaluation mode
    sample_loader = DataLoader(sample_input, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    with torch.no_grad():
        for i, (data, target) in enumerate(sample_loader):
            data, target = data.to(DEVICE).float(), target.to(DEVICE).float()

            # Reshape data for the model
            B, _, H, W, D = data.shape  # Data shape: [B, 1, H, W, D]
            data = data.permute(0, 4, 1, 2, 3).reshape(B, D, H, W)  # Reshape to [B, D, H, W]

            # Get model prediction
            prediction = model(data)  # Output should have shape [B, C, H, W] or similar
            print(f"Prediction Shape: {prediction.shape}")

            # Convert to CPU and visualize
            data_np = data[0].cpu().numpy().squeeze()  # Squeeze to get [H, W, D]
            prediction_np = prediction[0].cpu().numpy().squeeze()  # Squeeze for visualization

            # Visualize input data
            visualize_image(data_np, title=f"Sample Input {i} - Data")

            # Visualize prediction
            visualize_prediction(prediction_np, title=f"Sample Input {i} - Prediction")

            # Optionally compare prediction vs target
            target_np = target[0].cpu().numpy().squeeze()
            visualize_prediction(target_np, title=f"Sample Input {i} - Ground Truth")


if __name__ == "__main__":
    main()