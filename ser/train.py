import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ser.model import Net

def run_training(name, epochs, batch_size, learning_rate, DATA_DIR, model, device):

    print(f"Running experiment {name}")

    
    #batch_size = 1000
    #learning_rate = 0.01

    # save the parameters!

    # setup params
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # torch transforms
    ts = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # dataloaders
    training_dataloader = DataLoader(
        datasets.MNIST(root="../data", download=True, train=True, transform=ts),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )

    validation_dataloader = DataLoader(
        datasets.MNIST(root=DATA_DIR, download=True, train=False, transform=ts),
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
    )

    # train
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(training_dataloader):
            images, labels = images.to(device), labels.to(device)
            model.train()
            optimizer.zero_grad()
            output = model(images)
            loss = F.nll_loss(output, labels)
            loss.backward()
            optimizer.step()
            print(
                f"Train Epoch: {epoch} | Batch: {i}/{len(training_dataloader)} "
                f"| Loss: {loss.item():.4f}"
            )
            # validate
            val_loss = 0
            correct = 0
            with torch.no_grad():
                for images, labels in validation_dataloader:
                    images, labels = images.to(device), labels.to(device)
                    model.eval()
                    output = model(images)
                    val_loss += F.nll_loss(output, labels, reduction="sum").item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(labels.view_as(pred)).sum().item()
                val_loss /= len(validation_dataloader.dataset)
                val_acc = correct / len(validation_dataloader.dataset)

                print(
                    f"Val Epoch: {epoch} | Avg Loss: {val_loss:.4f} | Accuracy: {val_acc}"
                )
