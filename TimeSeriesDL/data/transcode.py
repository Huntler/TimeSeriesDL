"""Encodes a dataset and stores it."""
from typing import Dict, Tuple
from tqdm import tqdm
from scipy.io import savemat
import torch
import numpy as np
from TimeSeriesDL.data import Dataset
from TimeSeriesDL.model.base_model import BaseModel
from TimeSeriesDL.utils import config


def check_model_type(train_args: Dict) -> bool:
    """Checks if the model described in the provided dictionary is trained and type AE.

    Args:
        train_args (Dict): The dictionary.

    Returns:
        bool: True if trained and auto-encoder, else False.
    """
    # check if the model to load is trained and type AE
    if "AE" not in train_args["model_name"]:
        print("To encode a dataset, an AE model is required.")
        return False

    if not train_args["model_path"]:
        print("The AE model needs to be trained first.")
        return False

    return True


def load(train_args: Dict, decode: bool = False) -> Tuple[Dataset, BaseModel]:
    """Loads the dataset and AE using the dictionary.

    Args:
        train_args (Dict): The dictionary describing ConvAE and Dataset.
        decode (bool): Sets the dataset sequence length to the AE expected decode size.

    Returns:
        Tuple[Dataset, ConvAE]: Loaded Dataset and ConvAE model.
    """
    # check if the model to load is trained and type AE
    if not check_model_type(train_args):
        exit(1)

    # load the AE and prevent logging
    train_args["model"]["log"] = False
    model_name = train_args["model_name"]
    ae = config.get_model(model_name)(**train_args["model"])
    ae.load(path=train_args["model_path"])

    # load the dataset which should be encoded, make sure to disable AE mode
    train_args["dataset"]["ae_mode"] = False

    if decode:
        train_args["dataset"]["sequence_length"] = ae.latent_length

    data = Dataset(**train_args["dataset"])

    return data, ae


def encode_dataset(
    train_args: Dict, export_path: str = "./examples/train_encoded.mat"
) -> np.array:
    """Loads a trained ConvAE using the config dictionary. Then encodes the
    dataset specified in the dictionary and stores it to 'export_path'.

    Args:
        train_args (Dict): The config of a trained ConvAE.
        export_path (str): The path of the encoded dataset.
    """
    data, ae = load(train_args)

    encoded = []
    for i in tqdm(range(0, data.sample_size, train_args["dataset"]["sequence_length"])):
        # get the data as tensor
        x, _ = data[i]
        x = torch.unsqueeze(torch.tensor(x, dtype=ae.precision), 0)

        # encode the data and unwrap the batch
        x = ae.encode(x)
        x = torch.transpose(x, 2, 1)
        x = list(x.cpu().detach().numpy()[0, :, :])
        encoded += x

    # save the encoded dataset
    encoded = np.array(encoded)
    encoded = np.swapaxes(encoded, 0, 1)

    export = {}
    for i in range(encoded.shape[0]):
        export[f"encoded_feature_{i}"] = list(encoded[i, :])
    savemat(export_path, export)

    return encoded


def decode_dataset(train_args: Dict, export_path: str = "./examples/train_decoded.mat") -> None:
    """Loads a trained ConvAE using the config dictionary. Then decodes the
    dataset specified in the dictionary and stores it to 'export_path'.

    Args:
        train_args (Dict): The config of a trained ConvAE.
        export_path (str): The path of the decoded dataset.
    """
    data, ae = load(train_args, decode=True)

    decoded = []
    for i in tqdm(range(0, data.sample_size, ae.latent_length)):
        # get the data as tensor, apply 0-padding as sequence might be to small
        x = np.zeros((1, ae.latent_length, data.shape[-1]))
        d, _ = data[i]
        x[0, :d.shape[0], :] = d
        x = torch.tensor(x, dtype=ae.precision)
        x = torch.transpose(x, 2, 1)

        # encode the data and unwrap the batch
        x = ae.decode(x)
        x = list(x.cpu().detach().numpy()[0, :, :])
        decoded += x

    # save the encoded dataset
    decoded = np.array(decoded)
    decoded = np.swapaxes(decoded, 0, 1)

    export = {}
    for i in range(decoded.shape[0]):
        export[f"decoded_{i}"] = list(decoded[i, :])
    savemat(export_path, export)

    return decoded
