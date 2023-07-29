"""Flower client example using PyTorch for CIFAR-10 image classification."""
import os
import sys
import timeit
from collections import OrderedDict
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torchvision
import cifar
from cifar import GoogLeNet

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# pylint: enable=no-member


# Flower Client
class CifarClient(fl.client.NumPyClient):
    """Flower client implementing CIFAR-10 image classification using
    PyTorch."""

    def __init__(
        self,
        model: GoogLeNet,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        num_examples: Dict
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        self.model.train()
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        self.model.train()
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        cifar.train(self.model, self.trainloader, n_epochs=1)
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = cifar.test(self.model, self.testloader)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}


def main() -> None:
    """Load data, start CifarClient."""

    # Load data
    trainloader,_ ,testloader, num_examples = cifar.load_data()

    # Load model
    model = GoogLeNet().to(DEVICE).train()

    # Start client
    client = CifarClient(model, trainloader[0], testloader, num_examples)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


    #print('Finished Training')
    #PATH = './GoogleNet_flwr.pth'
    #torch.save(model.state_dict(), PATH)


if __name__ == "__main__":
    main()
    