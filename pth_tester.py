import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from my_vit_models import MyVIT2D  # Assuming your provided code is in a file named my_vit_models.py

# Load data
data = torch.load('0000.pth')

# Reshape tensor, treat depth as separate variables
# Axis to collapse, assuming the third dimension (index 2) is depth
data = data.mean(dim=2)

# Create an instance of the MyVIT2D model
# Adjust parameters such as embed_dim, num_heads, mlp_dim, and depth as per your requirements
model = MyVIT2D(img_size=data.shape[2], patch_size=12, in_channels=data.shape[1], embed_dim=768, num_heads=8,
                mlp_dim=2048, depth=12, num_classes=10)  # Assuming 10 classes

# Suppose you have labels for your data
labels = torch.randint(0, 10, (data.shape[0],))  # Example labels, replace with your actual labels

# Combining the data and labels
dataset = list(zip(data, labels))
total_size = len(dataset)
train_size = int(total_size * 0.7)
val_size = int(total_size * 0.15)
test_size = total_size - train_size - val_size


# Creating data loaders
train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size],
                                               generator=torch.Generator().manual_seed(
                                                   42))  # Here you'd need to define train_size, val_size, and test_size (they must all add up to the total length of your dataset)
train_loader = DataLoader(train_data, batch_size=16)  # Adjust batch_size as per your requirements
val_loader = DataLoader(val_data, batch_size=16)



import torch.optim as optim

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the device
model = model.to(device)

# Define a Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Optionally, define a Learning Rate scheduler
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Define number of epochs
num_epochs = 100  # Choose what's appropriate for your needs, considering your computational power and overfitting/underfitting balance

# Train the model
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader, 0):
        # Get the inputs and labels (data is a list of [inputs, labels])
        inputs, labels = data[0].to(device), data[1].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass, backward pass, and optimize
        inputs = inputs.float()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Optionally, Step with the scheduler
        # scheduler.step()

        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))

print('Finished Training')
