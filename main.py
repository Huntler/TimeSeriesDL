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
# FIXME: fix GRU model in case n_samples % batch_size != 0


future_steps = 1
config_dict = None


def train():
    # define parameters (depending on device, lower the precision to save memory)
    device = config_dict["device"]
    precision = torch.float16 if device == "cuda" else torch.float32

    freeze_support()

    # load the data, normalize them and convert them to tensor
    dataset = Dataset(**config_dict["dataset_args"])
    split_sizes = [int(len(dataset) * 0.8), int(len(dataset) * 0.2)]
    trainset, valset = torch.utils.data.random_split(dataset, split_sizes)
    trainloader = DataLoader(trainset, **config_dict["dataloader_args"])
    valloader = DataLoader(valset, **config_dict["dataloader_args"])

    # create model
    model_name = config_dict["model_name"]
    model: BaseModel = config.get_model(model_name)(
        input_size=dataset.sample_size, sequence_length=config_dict["dataset_args"]["sequence_length"], 
        **config_dict["model_args"])

    # create the model and train it, if epochs > 0
    epochs = config_dict["train_epochs"]
    if epochs == 0:
        return

    # define log path in config and move the current hyperparameters to
    # this driectory
    config_dict["evaluation"] = model.log_path
    config.store_args(f"{model.log_path}/config.yml", config_dict)

    # train the model
    model.learn(train=trainloader, validate=valloader,
                epochs=config_dict["train_epochs"])
    model.save_to_default()


def load():
    device = config_dict["device"]
    precision = torch.float16 if config_dict["device"] == "cuda" else torch.float32

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
        precision=precision, **_config_dict["model_args"])
    model.load(path)

    # do the prediction
    actual_data = []
    pred_data = []
    index = 0
    for X, y in tqdm(dataloader):
        if len(pred_data) <= index:
            _y = model.predict(X, future_steps=future_steps)
            pred_data += _y
        actual_data += [y.ravel().numpy()[0]]
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
    parser.add_argument("--steps", dest="steps", type=int, default=1, help="")
    args = parser.parse_args()

    if args.config:
        config_dict = config.get_args(args.config)
        future_steps = 1
        train()
        load()

    if args.load:
        config_dict = config.get_args(args.load)
        future_steps = args.steps
        load()
