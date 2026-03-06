from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def get_data():

    # Apply augmentations and randomness to training data to improve generalization
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
            transforms.RandomErasing(p=0.3),
        ]
    )

    # Normalize testing data
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )

    # Load training data
    train = CIFAR10(
        root="./data",
        train=True,
        download=False,
        transform=train_transform,
    )

    # Load testing data
    test = CIFAR10(
        root="./data",
        train=False,
        download=False,
        transform=test_transform,
    )


    # Create batched data loaders for training loop
    train_loader = DataLoader(
        train,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, test_loader
