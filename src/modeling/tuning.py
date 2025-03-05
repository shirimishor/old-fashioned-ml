from ray.train import RunConfig
from ray import tune
from ray.tune import Tuner, TuneConfig
import ray
from ray.air import session  
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../../'))
from src.modeling.model import Net
from src.dataset.pytorch_dataset import train_set
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader



def train_tune(config):
    print(f"train_tune started with config: {config}")
    net = Net(config["l1"], config["l2"])
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=config["momentum"])
    
    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            labels = labels.long()  # Ensure correct dtype

            # Debugging: Check label values
            if labels.max() >= 14 or labels.min() < 0:
                print(f"🚨 Error: Labels out of bounds! {labels}")

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        session.report({"loss": avg_loss})

    

# Define search space
search_space = {
    "lr": tune.uniform(0.0001, 0.1),
    "momentum": tune.uniform(0.1, 0.9),
    "batch_size": tune.choice([16, 24, 32]),
    "l1": tune.choice([2 ** i for i in range(3, 9)]),
    "l2": tune.choice([2 ** i for i in range(3, 9)])
}

ray.shutdown()

ray.init(
    num_cpus=os.cpu_count(),
    num_gpus=0
)

# Set a simple storage path
storage_path = os.path.abspath("./ray_results")

# Initialize the Tuner
tuner = Tuner(
    train_tune,
    param_space=search_space,
    tune_config=TuneConfig(
        metric="loss",
        mode="min",
        num_samples=5
    ),
    run_config=RunConfig(
        verbose=1,
        storage_path=storage_path,  # Simplified storage path
        callbacks=[]
    ),
)

# Run the Tuner
results = tuner.fit()

print("Results:", results)