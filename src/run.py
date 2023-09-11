import time

import torch
from torch.nn.utils import prune
from torch.utils.data import DataLoader
from torchvision import transforms

from src.visualize.plot_dataset_images import plot_example_images
from src.data.read_cifar100 import load_cifar100_data
from src.data.transform_cifar100 import CifarAnimals
from src.model.model import get_model
from src.model.test import test
from src.model.train import train
from src.visualize.plot_train_metrics import plot_metrics

# Consts
PLOT_MODE = False
EPOCHS = 7
NOISE_LABELS = 0.2
DATA_DIR = 'data/raw'
WATER_ANIMALS = sorted(['flatfish', 'lobster', 'dolphin', 'seal', 'shark'])
LAND_ANIMALS = sorted(['lizard', 'bear', 'elephant', 'chimpanzee', 'squirrel'])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRUNING_AMOUNT = 0.4
FINE_TUNING_EPOCHS = 1
print(f'Uses: {DEVICE} for torch.')

transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor()
])

# ---------------------------------------------------------------------------------------------------------------------

# Prepare original dataset.
org_train, org_test = load_cifar100_data(DATA_DIR, transform=transform)
land_animals_idx = [idx for idx, name in enumerate(org_train.classes) if name in LAND_ANIMALS]
water_animals_idx = [idx for idx, name in enumerate(org_train.classes) if name in WATER_ANIMALS]

if PLOT_MODE:
    plot_example_images(org_test, water_animals_idx, WATER_ANIMALS)
    plot_example_images(org_test, land_animals_idx, LAND_ANIMALS)

# ---------------------------------------------------------------------------------------------------------------------

# Prepare data for the new purpose. Creates custom dataset with noised labels in training set.
transformed_train = CifarAnimals(
    org_train,
    noise_prob=NOISE_LABELS,
    water_animal_indices=water_animals_idx,
    land_animal_indices=land_animals_idx
)

transformed_test = CifarAnimals(
    org_test,
    noise_prob=0.0,
    water_animal_indices=water_animals_idx,
    land_animal_indices=land_animals_idx
)

if PLOT_MODE:
    plot_example_images(transformed_test, [0], ['Water animal'])
    plot_example_images(transformed_test, [1], ['Land animal'])

# ---------------------------------------------------------------------------------------------------------------------

# Prepare the baseline model using pretrained mobilenet with default weights.
# Info about the model: https://pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v2.html.
train_loader = DataLoader(transformed_train, batch_size=64, shuffle=True)
test_loader = DataLoader(transformed_test, batch_size=64, shuffle=False)

model = get_model(DEVICE, 2)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train and evaluate the baseline model.
train_losses = []
test_losses = []
test_accuracies = []
for i in range(EPOCHS):
    print(f'[{i + 1}/{EPOCHS}]')
    train_loss = train(model, train_loader, criterion, optimizer, DEVICE)
    train_losses.append(train_loss)

    test_loss, test_acc = test(model, test_loader, criterion, DEVICE)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    time.sleep(0.5)

if PLOT_MODE:
    plot_metrics(train_losses, test_losses, test_accuracies)

# ---------------------------------------------------------------------------------------------------------------------

# Model sparsification for inference

parameters_to_prune = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
        parameters_to_prune.append((module, 'weight'))

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=PRUNING_AMOUNT,
)

print(f'After pruning')
test(model, test_loader, criterion, DEVICE)

sparse_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f'Fine tuning after pruning')

train_losses = []
test_losses = []
test_accuracies = []
for i in range(FINE_TUNING_EPOCHS):
    print(f'[{i + 1}/{FINE_TUNING_EPOCHS}]')
    train_loss = train(model, train_loader, criterion, sparse_optimizer, DEVICE)
    train_losses.append(train_loss)

    test_loss, test_acc = test(model, test_loader, criterion, DEVICE)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    time.sleep(0.5)

if PLOT_MODE:
    plot_metrics(train_losses, test_losses, test_accuracies)