import torch
from torch import nn
from torch.utils.data import DataLoader


def test(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device
) -> tuple[float, float]:
    """
    Test the given model using the provided data loader.

    Args:
        model (nn.Module): The neural network model to test.
        loader (DataLoader): The data loader for test data.
        criterion (nn.Module): The loss function.
        device (torch.device): The device (CPU or GPU) on which to perform testing.

    Returns:
        tuple: The evaluated loss and accuracy
    """

    with torch.no_grad():
        model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            total_samples += labels.size(0)

            # Predicts.
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get predicted class with the highest prob in axis=1.
            correct_predictions += (predicted == labels).sum().item()

            # Calc loss.
            loss = criterion(outputs, labels)
            total_loss += loss.item()

        # Calc metrics.
        average_loss = total_loss / len(loader)
        accuracy = (correct_predictions / total_samples) * 100.0

        print(f'Testing Loss: {average_loss}')
        print(f'Testing Accuracy: {accuracy:.2f} %')

        return average_loss, accuracy
