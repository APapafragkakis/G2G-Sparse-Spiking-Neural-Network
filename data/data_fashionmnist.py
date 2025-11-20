from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_fashion_loaders(batch_size, root="./data"):
    # Convert images to tensors in the range [0, 1]
    transform = transforms.ToTensor()

    # Training split of Fashion-MNIST
    train_set = datasets.FashionMNIST(
        root=root,
        train=True,
        download=True,
        transform=transform
    )

    # Test split of Fashion-MNIST
    test_set = datasets.FashionMNIST(
        root=root,
        train=False,
        download=True,
        transform=transform
    )

    # DataLoader for training 
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )

    # DataLoader for testing 
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader
