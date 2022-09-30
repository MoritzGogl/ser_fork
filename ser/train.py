import torch
from torch import optim
import torch.nn.functional as F
from ser.data import loading_data
from ser.transforms import loading_transforms
import json
import time
import os

from dataclasses import dataclass

@dataclass
class DataClass_Hyperparams:
    best_val_acc: float
    name: str
    epochs: int
    best_epoch: int
    learning_rate: float
    batch_size: int 


def run_training(name, epochs, batch_size, learning_rate, DATA_DIR, model, device, PROJECT_ROOT):

    print(f"Running experiment {name}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    ts=loading_transforms()

    training_dataloader = loading_data(batch_size, DATA_DIR, ts)[0]
    validation_dataloader = loading_data(batch_size, DATA_DIR, ts)[1]

    timestr = time.strftime("%Y%m%d-%H%M%S")
    path=str(PROJECT_ROOT) + "/outputs/"+timestr
    os.mkdir(path)

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

            best_val_acc=0

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

                if val_acc > best_val_acc:
                    best_val_acc=val_acc
                    best_epoch=epoch
                    print("New best accuracy!")

                    torch.save(model.state_dict(), path+'/model_weights.pth')

                    dictionary = { 
                    "name": name, 
                    "epochs": epochs, 
                    "batch size": batch_size,
                    "learning rate": learning_rate,
                    "best epoch": best_epoch,
                    "best accuracy": best_val_acc
                    } 
     

                    with open(path + '/params.json', "w") as outfile:
                         json.dump(dictionary, outfile)
  

                    Params = DataClass_Hyperparams(best_val_acc, name, epochs, best_epoch, learning_rate, batch_size)
                    print(Params.best_val_acc, Params.name, Params.epochs, Params.best_epoch, Params.learning_rate, Params.batch_size )

    


