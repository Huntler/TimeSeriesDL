import math
from multiprocessing import freeze_support
import os
import numpy as np
from tqdm import tqdm
import torch
from data.dataset import Dataset
from model.base_model import BaseModel
from utils.config import config
from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize
from torch.utils.data import DataLoader
import argparse

from utils.plotter import plot_curve


# TODO: add dropout to LSTM
# TODO: add multiple LSTM layers (currently only on available)


future_steps = 1
config_dict = None


def train():
    # define parameters (depending on device, lower the precision to save memory)
    device = config_dict["device"]
    precision = torch.float16 if device == "cuda" else torch.float32

    freeze_support()

    future_steps = config_dict["dataset_args"]["future_steps"]

    # load the data, normalize them and convert them to tensor
    dataset = Dataset(**config_dict["dataset_args"])
    split_sizes = [int(math.ceil(len(dataset) * 0.8)), int(math.floor(len(dataset) * 0.2))]
    
    trainset, valset = torch.utils.data.random_split(dataset, split_sizes)
    trainloader = DataLoader(trainset, **config_dict["dataloader_args"])
    valloader = DataLoader(valset, **config_dict["dataloader_args"])

    # create model
    model_name = config_dict["model_name"]
    model: BaseModel = config.get_model(model_name)(
        input_size=dataset.sample_size, sequence_length=config_dict["dataset_args"]["sequence_length"], 
        future_steps=future_steps, **config_dict["model_args"])

    # create the model and train it, if epochs > 0
    epochs = config_dict["train_epochs"]
    if epochs == 0:
        return

    # define log path in config and move the current hyperparameters to
    # this driectory
    config_dict["evaluation"] = model.log_path
    config.store_args(f"{model.log_path}/config.yml", config_dict)

    # train the model
    print(f"Starting training of model: {model.log_path}")
    model.learn(train=trainloader, validate=valloader,
                epochs=config_dict["train_epochs"])
    model.save_to_default()


def load():
    device = config_dict["device"]
    precision = torch.float16 if config_dict["device"] == "cuda" else torch.float32

    future_steps = config_dict["dataset_args"]["future_steps"]

    # load the data, normalize them and convert them to tensor
    dataset = Dataset(**config_dict["dataset_args"])
    dataloader = DataLoader(dataset)

    # load the model given the path
    path = []
    root_folder = config_dict["evaluation"]
    for file in os.listdir(root_folder):
        if ".torch" in file:
            path.append(file)
    if len(path) == 0:
        print("No model to evaluate.")
    path = f"{root_folder}/{max(path)}"

    # create the model instance based on an already trained model
    _config_dict = config_dict
    _config_dict["model_args"]["log"] = False
    model_class = config.get_model(name=config_dict["model_name"])
    model: BaseModel = model_class(
        input_size=dataset.sample_size, sequence_length=config_dict["dataset_args"]["sequence_length"], 
        future_steps=future_steps, precision=precision, **_config_dict["model_args"])
    model.load(path)

    # do the prediction
    actual_data = []
    pred_data = []
    index = 0
    for X, y in tqdm(dataloader):
        pred_data += model.predict(X)
        actual_data += list(y.ravel().numpy())
        index += 1

    pred_data = np.array([[i, d] for i, d in enumerate(pred_data)])
    actual_data = np.array([[i, d] for i, d in enumerate(actual_data)])
    plot_curve(data=[actual_data, pred_data], data_name=["actual", "prediction"],
               title=f"Look ahead: {future_steps}", save_path=f"{root_folder}/look_ahead_{future_steps}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument("--config", dest="config", help="")
    parser.add_argument("--load", dest="load", help="")
    args = parser.parse_args()

    if args.config:
        config_dict = config.get_args(args.config)
        future_steps = 1
        train()
        load()

    if args.load:
        config_dict = config.get_args(args.load)
        load()
