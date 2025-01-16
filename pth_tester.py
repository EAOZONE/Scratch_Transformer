import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from my_vit_models import MyVIT2D  # Assuming your provided code is in a file named my_vit_models.py

# Load data
data = torch.load('0000.pth')

# Reshape tensor, treat depth as separate variables
# Axis to collapse, assuming the third dimension (index 2) is depth
print(data.shape)
data = data.permute(3,0,1,2)
# Create an instance of the MyVIT2D model
# Adjust parameters such as embed_dim, num_heads, mlp_dim, and depth as per your requirements
model = MyVIT2D(img_size=(900, 600), patch_size=12, in_channels=1, embed_dim=768, num_heads=8,
                mlp_dim=2048, depth=12)  # Assuming 10 classes

# Suppose you have labels for your data
labels = torch.randint(0, 10, (data.shape[0], 768 ))  # Example labels, replace with your actual labels

# Combining the data and labels
dataset = list(zip(data, labels))
total_size = len(dataset)
train_size = int(total_size * 0.5)
val_size = int(total_size * 0.25)
test_size = total_size - train_size - val_size
print(val_size)

# Creating data loaders
train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size],
                                               generator=torch.Generator().manual_seed(
                                                   42))  # Here you'd need to define train_size, val_size, and test_size (they must all add up to the total length of your dataset)
train_loader = DataLoader(train_data, batch_size=1)  # Adjust batch_size as per your requirements
val_loader = DataLoader(val_data, batch_size=1)
test_loader = DataLoader(test_data, batch_size=1)



import torch.optim as optim

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the device
model = model.to(device)

# Define a Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Define number of epochs
num_epochs = 40 # Choose what's appropriate for your needs, considering your computational power and overfitting/underfitting balance

accumulation_steps = 4
for epoch in range(num_epochs):
    optimizer.zero_grad()  # Reset gradients

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        # Forward and backward pass
        inputs = inputs.float()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss = loss / accumulation_steps  # Normalize loss
        loss.backward()

        # Perform optimization step after every `accumulation_steps`
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))

print('Finished Training')


from sklearn.metrics import accuracy_score

# Evaluate on validation set
model.eval()
val_outputs = []
val_labels = []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device).float(), labels.to(device).long()  # Ensure correct dtype
        outputs = model(inputs)

        # Get predicted class indices
        _, predicted = torch.max(outputs, 1)  # Shape: [batch_size]

        # Extend outputs and labels into lists
        val_outputs.extend(predicted.cpu().numpy())  # Predicted class indices
        val_labels.extend(labels.cpu().numpy())  # True class indices

# Ensure outputs and labels are flattened (if not already 1D)
val_outputs = np.array(val_outputs).flatten()
val_labels = np.array(val_labels).flatten()

# Compute accuracy
accuracy = accuracy_score(val_labels, val_outputs)  # Should now work
print(f"Validation Accuracy: {accuracy:.2f}")


# Evaluate on test set
test_outputs = []
test_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device).float(), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        test_outputs.extend(predicted.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

test_accuracy = accuracy_score(test_labels, test_outputs)
print(f"Test Accuracy: {test_accuracy:.2f}")


torch.save(model.state_dict(), "vit_model.pth")

