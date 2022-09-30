from pathlib import Path
#import torch
#from torch import optim
#import torch.nn as nn
#import torch.nn.functional as F
#from torch.utils.data import DataLoader
#from torchvision import datasets, transforms

import typer

from train import run_training

main = typer.Typer()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@main.command()
def train(    
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),

    epochs: int = typer.Option(
        ..., "-e", "--epochs", help="Number of epochs"
    ),

    batch_size: int = typer.Option(
        ..., "-b", "--batch_size", help="Batch size"
    ),

    learning_rate: float = typer.Option(
        ..., "-r", "--learning_rate", help="Learning rate"
    ),
    ):
    run_training(name, epochs, batch_size, learning_rate, DATA_DIR)



@main.command()
def infer():
    print("This is where the inference code will go")
