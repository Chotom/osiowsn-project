import torch
from torchvision import models


def get_model(device, class_num: int) -> models.MobileNetV2:
    """
    Prepare the model. Change last layer to predict given number of classes.
    Cast model to the device.

    Args:
        device (torch.device): The device (CPU or GPU) on which to perform testing.
        class_num (int): Number of classes to predict

    Returns:
        MobileNetV2 Model.
    """
    model = models.mobilenet_v2(weights='DEFAULT')

    num_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_features, class_num)

    model = model.to(device)

    return model
