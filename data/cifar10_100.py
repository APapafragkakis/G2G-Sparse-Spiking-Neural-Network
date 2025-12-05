# data/cifar10_100.py
# CIFAR-10 & CIFAR-100 loaders in the same style as FashionMNIST

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Standard normalization used in many CIFAR baselines
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def get_cifar10_loaders(batch_size, root: str = "./data"):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )

    train_set = datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=transform,
    )

    test_set = datasets.CIFAR10(
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


def get_cifar100_loaders(batch_size, root: str = "./data"):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )

    train_set = datasets.CIFAR100(
        root=root,
        train=True,
        download=True,
        transform=transform,
    )

    test_set = datasets.CIFAR100(
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
