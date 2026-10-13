import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_data_loader(data_dir, batch_size=32, train=True):
    """CIFAR10 loader (as in the module notebooks), resized to 64x64 for the CNN."""
    transform = transforms.Compose(
        [transforms.Resize((64, 64)), transforms.ToTensor()]
    )
    dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=train, download=True, transform=transform
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return loader


def get_transform(image_size=64):
    # Same preprocessing as get_data_loader, used by the API
    return transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    )
