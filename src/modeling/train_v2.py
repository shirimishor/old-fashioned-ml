import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm # progress bar
from model import Net
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../../'))
from src.modeling.model import Net
from src.dataset.pytorch_dataset import train_set
from torch.utils.data import DataLoader

# Set the best hyperparameters from Ray Tune
best_config ={
  "batch_size": 32,
  "l1": 256,
  "l2": 64,
  "lr": 0.05793603995620943,
  "momentum": 0.22478537696556533
}

def train(config):
    print(f"training started with config: {config}")
    net = Net(config["l1"], config["l2"])
    print(net.fc3)
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=config["momentum"])
    for epoch in range(100):
        running_loss = 0.0
        # Use tqdm to wrap the DataLoader for progress tracking
        for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Progress")):
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # accumulate the running loss
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f'[{epoch + 1}, {i + 1}] loss: {avg_loss}')
    print("finished training!")
    torch.save(net.state_dict(), "models/nn5.pth")

if __name__ == "__main__":
    train(best_config)
    
