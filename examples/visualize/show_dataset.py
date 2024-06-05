"""This example shows how to visualize a dataset"""
import argparse
import os
from typing import Dict

import numpy as np
import torch
from scipy.io import savemat
from tqdm import trange
from TimeSeriesDL.data import Dataset
from TimeSeriesDL.debug import VisualizeDataset
from TimeSeriesDL.model.base_model import BaseModel
from TimeSeriesDL.utils import config


def predicted_dataset(train_args: Dict, dataset: Dataset) -> Dataset:
    """Loads a trained model from the train_config and iterates over the
    provided dataset to create a new dataset with the models predictions

    Args:
        train_args (Dict): The config args of a trained model.
        dataset (Dataset): The dataset

    Returns:
        Dataset: _description_
    """
    train_args["model"]["log"] = False
    model: BaseModel = config.get_model(train_args["model_name"])(**train_args["model"])
    model.load(train_args["model_path"])

    # create storage of prediction
    window_len = train_args["dataset"]["sequence_length"]
    f_len = train_args["dataset"]["future_steps"]
    full_sequence = np.zeros(dataset.shape)
    full_sequence[0:window_len, :] = dataset.slice(0, window_len)

    # predict based on sliding window
    print("Predicting...")
    for i in trange(0, dataset.sample_size - window_len, f_len):
        window = full_sequence[i:i + window_len]
        window = torch.tensor(window, device=train_args["device"], dtype=torch.float32)
        window = torch.unsqueeze(window, 0)
        sample = model.predict(window)
        full_sequence[i + window_len:i + window_len + f_len] = sample.detach().cpu().numpy()

    # remove the channel and prepare to save the predicted data
    full_sequence = np.squeeze(full_sequence, 1)
    full_sequence = dataset.scale_back(full_sequence)
    full_sequence = np.swapaxes(full_sequence, 0, 1)

    # save prediction using the label names from the original dataset
    export = {}
    for i, label_name in enumerate(dataset.label_names):
        export[label_name] = list(full_sequence[i, :])
    savemat("prediction.mat", export)

    # load saved matrix as a dataset and delete the temporary file
    dataset = Dataset(custom_path="prediction.mat")
    os.remove("prediction.mat")

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--compare", default=False, action="store_true")
    parser.add_argument("--features", type=int, nargs="+")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--output", type=str)

    # define args
    args = parser.parse_args()
    train_args = config.get_args(args.config)

    # load the dataset and put it into the visualizer
    data = Dataset(**train_args["dataset"])
    vis = VisualizeDataset(data, name="Train")

    # load the second dataset to compare to the already loaded one if the argument is provided
    if args.compare:
        data2 = predicted_dataset(train_args, data)
        overlay = VisualizeDataset(data2, name="Predicted", overlay_mode=True)
        vis.set_overlay(overlay)

    # visualize the dataset(s)
    features = args.get("features", list(range(len(data.label_names))))
    vis.set_feature(features)

    # test save last configuration but of all samples
    vis.visualize(start=args.start, end=args.end, save=args.get("output", None))
