import math
from multiprocessing import freeze_support
import os
import random
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
import copy
import optuna

from utils.plotter import plot_curve
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


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

    # Load test data
    test_config_dict = copy.deepcopy(config_dict)
    test_config_dict["dataset_args"]["d_type"]="test"
    test_dataset = Dataset(**test_config_dict["dataset_args"])
    testloader = DataLoader(test_dataset, **test_config_dict["dataloader_args"])

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
    model.learn(train=trainloader, validate=valloader, test=testloader,
                epochs=config_dict["train_epochs"])
    model.save_to_default()

    return model

def load():
    device = config_dict["device"]
    precision = torch.float16 if config_dict["device"] == "cuda" else torch.float32

    future_steps = config_dict["dataset_args"]["future_steps"]
    config_dict["dataset_args"]["future_steps"] = 1

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
        future_steps=1, precision=precision, **_config_dict["model_args"])
    model.load(path)

    # do the prediction in a recursive fashion 
    actual_data = []
    pred_data = []
    index = 0
    losses = []
    for X, y in tqdm(dataloader):
        seq_len = X.size(1)
        if index == 0:
            pass
        elif index < seq_len:
            sub_seq = torch.tensor(pred_data)
            sub_seq = torch.unsqueeze(sub_seq, 0)
            sub_seq = torch.unsqueeze(sub_seq, -1)
            X[:, :index] = sub_seq
        else:
            sub_seq = torch.tensor(pred_data[-seq_len:])
            sub_seq = torch.unsqueeze(sub_seq, 0)
            sub_seq = torch.unsqueeze(sub_seq, -1)
            X = sub_seq

        predicted = model.predict(X)
        actual = list(y.ravel().numpy())

        pred_data += predicted
        actual_data += actual

        loss = abs(predicted[0] - actual[0])
        losses.append(loss)

        index += 1

        if index == future_steps:
            index = 0

    mae_mean = np.mean(losses)
    print("MAE Loss mean for test data: " + str(mae_mean))

    pred_data = np.array([[i, d] for i, d in enumerate(pred_data)])
    actual_data = np.array([[i, d] for i, d in enumerate(actual_data)])
    d_type = config_dict["dataset_args"]["d_type"]
    plot_curve(data=[actual_data, pred_data], data_name=["actual", "prediction"],
               title=f"Look ahead: {future_steps}", save_path=f"{root_folder}/look_ahead_{future_steps}_{d_type}.png")




l = list(range(1, 200))
new_l = [item for item in l if item % 5 == 0]
trial_seq_lens = []

def objective(trial):
    x = new_l[trial.number]

    config_dict["dataset_args"]["sequence_length"] = x
    trial_seq_lens.append(x)
    print("Training model with sequence length: " + str(x))
    model = train()

    (acc, var, mse, rmse, mae) = model.test_stats
    return acc

def fine_tune_seq_len():
    study = optuna.create_study(direction=optuna.study.StudyDirection.MAXIMIZE)
    study.optimize(objective, n_trials=40)

    best_seq_len = trial_seq_lens[study.best_trial.number]
    print("Best performing seqence length is " + str(best_seq_len))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument("--config", dest="config", help="")
    parser.add_argument("--load", dest="load", help="")
    parser.add_argument("--fine_tune", dest="fine_tune", help="")

    args = parser.parse_args()

    if args.fine_tune:
        config_dict = config.get_args(args.fine_tune)
        fine_tune_seq_len()

    if args.config:
        config_dict = config.get_args(args.config)
        future_steps = 1
        train()
        config_dict["dataset_args"]["d_type"] = "test"
        config_dict["dataset_args"]["future_steps"] = 200
        load()

    if args.load:
        config_dict = config.get_args(args.load)
        load()
