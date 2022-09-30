from pathlib import Path
import typer
import torch
from ser.train import run_training
from ser.model import Net
import git
from git import Repo
main = typer.Typer()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


repo = Repo(PROJECT_ROOT)
print(repo.index.diff(None))

assert repo.index.diff(None) == []
#repo = git.Repo(search_parent_directories=True)
#sha = repo.head.object.hexsha
#print(sha)

print("===================================")

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Net().to(device)

    run_training(name, epochs, batch_size, learning_rate, DATA_DIR, model, device, PROJECT_ROOT)
    


@main.command()
def infer():
    print("This is where the inference code will go")
