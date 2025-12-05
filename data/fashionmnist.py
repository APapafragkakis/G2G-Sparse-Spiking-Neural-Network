# data/fashionmnist.py

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_fashion_loaders(batch_size, root: str = "./data"):
    transform = transforms.ToTensor()

    train_set = datasets.FashionMNIST(
        root=root,
        train=True,
        download=True,
        transform=transform,
    )

    test_set = datasets.FashionMNIST(
        root=root,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader
