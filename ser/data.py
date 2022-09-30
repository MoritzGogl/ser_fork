from torchvision import datasets
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets


def loading_data(batch_size, DATA_DIR, ts):
        # dataloaders
    training_dataloader = DataLoader(
        datasets.MNIST(root="../data", download=True, train=True, transform=ts),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )

    validation_dataloaader = DataLoader(
        datasets.MNIST(root=DATA_DIR, download=True, train=False, transform=ts),
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
    )

    return ([training_dataloader, validation_dataloaader])