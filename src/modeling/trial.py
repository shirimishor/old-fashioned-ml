import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../../'))
from src.modeling.model import Net
from src.dataset.pytorch_dataset import train_set, dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_tune(config):
    print(f"train_tune started with config: {config}")
    net = Net(config["l1"], config["l2"])
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=config["momentum"])
    for epoch in range(10):
        running_loss = 0.0
        # Use tqdm to wrap the DataLoader for progress tracking
        for i, data in enumerate(train_loader):
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            print(loss.item())
            # accumulate the running loss
            running_loss += loss.item()
            print(f"running loss: {running_loss}")
        
        avg_loss = running_loss / len(train_loader)
        print(f'[{epoch + 1}, {i + 1}] loss: {avg_loss}')


search_space = {
    "lr":  0.039842,
    "momentum": 0.476454,
    "batch_size": 8,
    "l1": 16,
    "l2": 64
}

