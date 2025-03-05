import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../../'))
from src.modeling.model import Net
from src.dataset.pytorch_dataset import train_loader, class_names, train_set
import config.config
import tqdm

# Set the best hyperparameters from Ray Tune
best_config = {
    'lr': 0.07213821593952946,
    'momentum': 0.41978204223494453,
    'epochs': 5
}

# Prepare the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.ImageFolder(root=os.path.normpath(config.config.DATASET_DIR), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize the model and optimizer
model = Net(len(class_names))
optimizer = optim.SGD(model.parameters(), lr=best_config['lr'], momentum=best_config['momentum'])

# Loss function
criterion = nn.CrossEntropyLoss()

# Train the model

for epoch in range(best_config['epochs']):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch + 1}/{best_config['epochs']}], Loss: {running_loss / len(train_loader):.4f}")


model.eval()

