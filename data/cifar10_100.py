# data/cifar10_100.py

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def get_cifar10_loaders(batch_size, root: str = "./data"):
    # Training: WITH augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),      # ← ΣΗΜΑΝΤΙΚΟ
        transforms.RandomHorizontalFlip(),         # ← ΣΗΜΑΝΤΙΚΟ
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    
    # Testing: WITHOUT augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_set = datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=transform_train,  # ← Διαφορετικό transform
    )

    test_set = datasets.CIFAR10(
        root=root,
        train=False,
        download=True,
        transform=transform_test,  # ← Διαφορετικό transform
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,      # ← Πρόσθεσε για speed
        pin_memory=True,    # ← Πρόσθεσε αν έχεις GPU
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, test_loader


def get_cifar100_loaders(batch_size, root: str = "./data"):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_set = datasets.CIFAR100(
        root=root,
        train=True,
        download=True,
        transform=transform_train,
    )

    test_set = datasets.CIFAR100(
        root=root,
        train=False,
        download=True,
        transform=transform_test,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, test_loader