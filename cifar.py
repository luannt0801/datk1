from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import time
from GoogleNet import GoogLeNet
DATA_ROOT = "./dataset"

# Hyper-parameters 
n_epochs = 3
batch_size = 4
learning_rate = 0.00001
momentum = 0.9
NUM_CLIENTS = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data() -> (
    Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict]
):
    """Load CIFAR-10 (training and test set)."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = CIFAR10(DATA_ROOT, train=True, download=True, transform=transform_train)
    #trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testset = CIFAR10(DATA_ROOT, train=False, download=True, transform=transform_test)
    #testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    #num_examples = {"trainset": len(trainset), "testset": len(testset)}

    # Split training set into 10 partitions to simulate the individual dataset
    partition_size = len(trainset) // NUM_CLIENTS
    lengths = [partition_size] * NUM_CLIENTS
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=batch_size))
    testloader = DataLoader(testset, batch_size=batch_size)
    num_examples = {"trainset": partition_size, "testset": len(testset)}
    return trainloaders, valloaders, testloader, num_examples
    #return trainloader, testloader, num_examples


def train(
    model: GoogLeNet,
    trainloader: torch.utils.data.DataLoader,
    n_epochs: int
) -> None:
    """Train the network."""
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # Train the network
    model.to(device)
    model.train()
    for epoch in range(1, n_epochs + 1):  # loop over the dataset multiple times
        time_start = time.time()
        for i, (images, labels) in enumerate (tqdm(trainloader)):
        # origin shape: [4, 3, 32, 32] = 4, 3, 1024
        # input_layer: 3 input channels, 6 output channels, 5 kernel size
            images = images.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        time_stop = time.time()
        print(f"Epoch {epoch} training time: {time_stop - time_start} seconds\n")
            
def test(
    model: GoogLeNet,
    testloader: torch.utils.data.DataLoader
) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    # Define loss and metrics
    criterion = nn.CrossEntropyLoss()
    correct, loss = 0, 0.0

    # Evaluate the network
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate (tqdm(testloader)):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)  # pylint: disable=no-member
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Centralized PyTorch training")
    print("Load data")
    trainloader, _ ,testloader, _ = load_data()
    model = GoogLeNet().to(device)

    print("Start training")
    train(model= model, trainloader=trainloader[0], n_epochs=n_epochs)
    
    print('Finished Training')
    PATH = './GoogleNet.pth'
    torch.save(model.state_dict(), PATH)


    print("Evaluate model")
    loss, accuracy = test(model= model, testloader=testloader)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

if __name__ == "__main__":
    main()