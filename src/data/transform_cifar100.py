import torch.utils.data
import torchvision
import os
from torchvision import datasets, transforms
from torch.utils.data import Subset, Dataset
import numpy as np


class CifarAnimals(Dataset):
    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            noise_prob: float,
            water_animal_indices: list[int],
            land_animal_indices: list[int]
    ):
        """
        Args:
            dataset (torch.utils.data.Dataset): The original dataset to be filtered and potentially noisy-labeled.
            noise_prob (float): The probability of introducing label noise (0.0 to 1.0).
            water_animal_indices (list of int): Indices representing water animal classes.
            land_animal_indices (list of int): Indices representing land animal classes.
        """

        self.water_animal_indices = water_animal_indices
        self.land_animal_indices = land_animal_indices
        self.noise_prob = noise_prob

        self.dataset = [i for i in dataset if i[1] in water_animal_indices + land_animal_indices]
        self.noisy_labels = self.compute_noisy_labels()

    def compute_noisy_labels(self):
        noisy_labels = []

        for _, label in self.dataset:
            # Randomly flip the label with the specified noise probability
            if np.random.rand() < self.noise_prob:
                if label in self.water_animal_indices:
                    # Assign a random land animal class index
                    new_label = np.random.choice(self.land_animal_indices)
                else:
                    # Assign a random water animal class index
                    new_label = np.random.choice(self.water_animal_indices)
                noisy_labels.append(new_label)
            else:
                # Save original label
                noisy_labels.append(label)

        return noisy_labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, _ = self.dataset[index]
        noisy_label = self.noisy_labels[index]

        animal_class = 0 if noisy_label in self.water_animal_indices else 1

        return data, animal_class
