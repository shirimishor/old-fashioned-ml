import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../../'))
import config.config
from src.dataset.full_dataset import merge_datasets

if __name__ == "__main__":       


    # Merging all collected data
    if not os.path.exists(config.config.DATASET_DIR):
        os.makedirs(config.config.DATASET_DIR)

    merge_datasets(config.config.MET_DATA_DIR, config.config.VA_DATA_DIR, config.config.DATASET_DIR)

    # Merging augmented data with original data
    if not os.path.exists(config.config.DATA_WITH_AUG_DIR):
        os.makedirs(config.config.DATA_WITH_AUG_DIR)

    merge_datasets(config.config.AUG_DIR, config.config.DATASET_DIR, config.config.DATA_WITH_AUG_DIR)

# Creating dataset: transmorming data and creating DataLoader objects
mean = torch.tensor(config.config.MEAN)
std = torch.tensor(config.config.STD)

# Define the transformation for the dataset (resize, normalize, etc.)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=mean,  
                        std=std)
])

dataset = datasets.ImageFolder(root=os.path.normpath(config.config.DATA_WITH_AUG_DIR), transform=transform)
train_set, test_set = torch.utils.data.random_split(dataset, [0.8, 0.2])
class_names = dataset.classes





