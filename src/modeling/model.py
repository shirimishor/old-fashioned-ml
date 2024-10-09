import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, class_num):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  # Input: (3, 224, 224), Output: (16, 224, 224)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)  # Pooling halves the spatial dimensions
        
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # Input: (16, 112, 112), Output: (32, 112, 112)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)  # Input: (32, 56, 56), Output: (64, 56, 56)
        self.bn3 = nn.BatchNorm2d(64)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 28 * 28, 128)  
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, class_num)

        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Apply conv1 -> batchnorm -> relu -> pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Output: (16, 112, 112)
        
        # Apply conv2 -> batchnorm -> relu -> pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Output: (32, 56, 56)
        
        # Apply conv3 -> batchnorm -> relu -> pool
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Output: (64, 28, 28)
        
        # Flatten the tensor for the fully connected layers
        x = torch.flatten(x, 1)  # Flatten: (batch_size, 64*28*28)

        # Fully connected layers with dropout and ReLU activations
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))

        # Output layer
        x = self.fc3(x)
        
        return x