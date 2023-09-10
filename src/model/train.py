import time

import torch
from tqdm import tqdm
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def train(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device
) -> float:
    """
    Train the given model using the provided data loader.

    Args:
        model (nn.Module): The neural network model to train.
        loader (DataLoader): The data loader for training data.
        criterion (nn.Module): The loss function.
        optimizer (Optimizer): The optimizer.
        device (torch.device): The device (CPU or GPU) on which to perform training.

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    training_loss = 0.0

    for batch_num, (inputs, labels) in enumerate(tqdm(loader)):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        training_loss += loss.item()

    training_loss /= len(loader)
    print('Training Loss:', training_loss)

    return training_loss
