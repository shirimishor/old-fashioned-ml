import torch
from model import Net
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../../'))
from src.dataset.pytorch_dataset import class_names, test_set
from torch.utils.data import DataLoader


if __name__ == "__main__":
    print("Loading model...")
    model = Net(256, 64)
    print(model.fc3)
    model.load_state_dict(torch.load(r"C:\old-fashioned-ml\models\nn5.pth"))
    model.eval()
    print("Model loaded")
    test_loader = DataLoader(test_set, batch_size=32, shuffle=True)


    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct // total} %')

    # prepare to count predictions for each class
correct_pred = {classname: 0 for classname in class_names}
total_pred = {classname: 0 for classname in class_names}

# again no gradients needed
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[class_names[label]] += 1
            total_pred[class_names[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    