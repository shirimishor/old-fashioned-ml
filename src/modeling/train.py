import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm # progress bar
from collections import Counter
from model import Net
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../../'))
from src.dataset.pytorch_dataset import class_names, train_loader



def loss_weights(loader):
    class_counts = Counter()

    for _, labels in loader:
        class_counts.update(labels.tolist())

    # Get the total number of samples
    total_samples = sum(class_counts.values())

    # Compute class weights
    class_weights = {class_id: total_samples / count for class_id, count in class_counts.items()}
    class_weights_tensor = torch.tensor([class_weights[i] for i in range(len(class_counts))])
    return class_weights_tensor


net = Net(len(class_names))


criterion = nn.CrossEntropyLoss(weight=loss_weights(train_loader))
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


if __name__ == "__main__":
    print("starting training...")
    for epoch in range(5):  # loop over the dataset multiple times
        running_loss = 0.0
        # Use tqdm to wrap the DataLoader for progress tracking
        for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Progress")):
            # get the inputs; data is a list of [inputs, labels]
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

        print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 28:.3f}')

    print('Finished Training')

    # Saving the model as a dictionary
    torch.save(net.state_dict(), "models/nn1.pth")

