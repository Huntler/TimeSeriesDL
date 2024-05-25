"""Encodes a dataset and stores it."""
from typing import Dict
from tqdm import tqdm
from scipy.io import savemat
import torch
import numpy as np
from TimeSeriesDL.data import Dataset
from TimeSeriesDL.model.conv_ae_model import ConvAE


def encode_dataset(
    train_args: Dict, export_path: str = "./examples/train_encoded.mat"
) -> None:
    """Loads a trained ConvAE using the config dictionary.

    Args:
        train_args (Dict): The config of a trained ConvAE.
    """
    # check if the model to load is trained and type AE
    if train_args["model_name"] != "ConvAE":
        print("To encode a dataset, an AE model is required.")
        exit(1)

    if not train_args["model_path"]:
        print("The AE model needs to be trained first.")
        exit(1)

    # load the AE and prevent logging
    train_args["model"]["log"] = False
    ae = ConvAE(**train_args["model"])
    ae.load(path=train_args["model_path"])

    # load the dataset which should be encoded, make sure to disable AE mode
    train_args["dataset"]["ae_mode"] = False
    data = Dataset(**train_args["dataset"])

    encoded = []
    for i in tqdm(range(0, data.sample_size, train_args["dataset"]["sequence_length"])):
        # get the data as tensor
        x, _ = data[i]
        x = torch.unsqueeze(torch.tensor(x), 0)

        # encode the data and unwrap the batch
        x = ae.encode(x)
        x = torch.transpose(x, 2, 1)
        x = list(x.cpu().detach().numpy()[0, :, :])
        encoded += x

    # save the encoded dataset
    encoded = np.array(encoded)
    encoded = np.swapaxes(encoded, 0, 1)
    savemat(export_path, {"train": list(encoded)})
