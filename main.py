from multiprocessing import freeze_support
import os
from tqdm import tqdm
import torch
from data.dataset import Dataset
from model.base_model import BaseModel
from utils.config import config
from torchvision import transforms  
from torchvision.transforms import ToTensor, Normalize
from torch.utils.data import DataLoader
import argparse


# TODO: add dropout to LSTM
# TODO: add multiple LSTM layers (currently only on available)
# TODO: add visualization methods to plot our dataset (either using matplotlib or tensorboard logging)
# FIXME: fix GRU model in case n_samples % batch_size != 0


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
    model: BaseModel = config.get_model(model_name)(input_size=dataset.sample_size, **config_dict["model_args"])

    # create the model and train it, if epochs > 0
    epochs = config_dict["train_epochs"]
    if epochs == 0:
        return

    # define log path in config and move the current hyperparameters to 
    # this driectory
    config_dict["evaluation"] = model.log_path
    config.store_args(f"{model.log_path}/config.yml", config_dict)

    # train the model
    model.learn(train=trainloader, validate=valloader, epochs=config_dict["train_epochs"])
    model.save_to_default()


def load():
    device = config_dict["device"]
    precision = torch.float16 if config_dict["device"] == "cuda" else torch.float32

    # load the data, normalize them and convert them to tensor
    dataset = Dataset(**config_dict["dataset_args"])
    dataloader = DataLoader(dataset, **config_dict["dataloader_args"])
    
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
    model: BaseModel = model_class(input_size=dataset.sample_size, precision=precision, **_config_dict["model_args"])
    model.load(path)

    # do the prediction
    for X, y in tqdm(dataloader):
        y = model.predict(X)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument("--config", dest="config", help="")
    args = parser.parse_args()

    config_dict = config.get_args(args.config)
    train()
    # load()