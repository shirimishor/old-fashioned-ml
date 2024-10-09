import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../../'))
import config.config



mean = torch.tensor(config.config.MEAN)
std = torch.tensor(config.config.STD)

# Define the transformation for the dataset (resize, normalize, etc.)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to 224x224
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=mean,  # Normalize to the same stats as ImageNet
                         std=std)
])

dataset = datasets.ImageFolder(root=config.config.DATASET_DIR, transform=transform)
train_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
class_names = dataset.classes

# Create DataLoaders to iterate over the dataset: both train and test
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=True)




