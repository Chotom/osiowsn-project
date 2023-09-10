import os
import torchvision
from torchvision import datasets


def download_cifar100_dataset(data_dir):
    os.makedirs(data_dir, exist_ok=True)

    # Download and save the CIFAR-100 dataset
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
    datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)

    print(f"CIFAR-100 dataset downloaded and saved to '{data_dir}' directory.")


if __name__ == "__main__":
    data_directory = "data/raw"
    download_cifar100_dataset(data_directory)
