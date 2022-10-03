from datetime import datetime
from pathlib import Path

import torch

from ser.train import train as run_train
from ser.data import train_dataloader, val_dataloader, test_dataloader
from ser.params import Params, save_params
from ser.transforms import transforms, normalize
import json


def inference(run_path, label):


    # TODO load the parameters from the run_path so we can print them out!
    params_path=str(run_path) + "/params.json"

    with open(params_path, 'r') as f:
        data_dict = json.load(f)
     
    print(data_dict)

    # select image to run inference for
    dataloader = test_dataloader(1, transforms(normalize))
    images, labels = next(iter(dataloader))
    while labels[0].item() != label:
        images, labels = next(iter(dataloader))

    # load the model
    model = torch.load(run_path / "model.pt")

    # run inference
    model.eval()
    output = model(images)
    pred = output.argmax(dim=1, keepdim=True)[0].item()
    certainty = max(list(torch.exp(output)[0]))
    pixels = images[0][0]
    print(generate_ascii_art(pixels))
    print(f"This is a {pred}")

    print("Certainty:", certainty)

def generate_ascii_art(pixels):
    ascii_art = []
    for row in pixels:
        line = []
        for pixel in row:
            line.append(pixel_to_char(pixel))
        ascii_art.append("".join(line))
    return "\n".join(ascii_art)


def pixel_to_char(pixel):
    if pixel > 0.99:
        return "O"
    elif pixel > 0.9:
        return "o"
    elif pixel > 0:
        return "."
    else:
        return " "