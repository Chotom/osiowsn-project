import os
from torchvision import datasets, transforms


def load_cifar100_data(data_dir: str, transform=None) -> tuple:
    if not os.path.exists(os.path.join(data_dir, 'cifar-100-python')):
        raise FileNotFoundError(f"The CIFAR-100 dataset is not found in '{data_dir}'. Please download it first.")

    if transform is None:
        transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=False, transform=transform)
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=False, transform=transform)

    return train_dataset, test_dataset


# Example: Print the number of samples in the train and test set
if __name__ == "__main__":
    data_directory = "data/raw"
    train_data, test_data = load_cifar100_data(data_directory)

    print(f"Number of training samples: {len(train_data)}")
    print(f"Number of testing samples: {len(test_data)}")
