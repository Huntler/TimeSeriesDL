"""Encodes a dataset and stores it."""
from typing import Callable, Dict, List, Tuple
from tqdm import tqdm
import torch
import numpy as np
from TimeSeriesDL.data import Dataset
from TimeSeriesDL.model.auto_encoder import AutoEncoder
from TimeSeriesDL.utils import config


def check_model_type(ae: AutoEncoder) -> bool:
    """Checks if the model described in the provided model is type AE.

    Args:
        ae (AutoEncoder): The model.

    Returns:
        bool: True if auto-encoder, else False.
    """
    # check if the model to load is trained and type AE
    if isinstance(ae, AutoEncoder):
        return True

    print("To encode a dataset, an AE model is required.")
    return False


def load(train_args: Dict, decode: bool = False) -> Tuple[Dataset, AutoEncoder]:
    """Loads the dataset and AE using the dictionary.

    Args:
        train_args (Dict): The dictionary describing ConvAE and Dataset.
        decode (bool): Sets the dataset sequence length to the AE expected decode size.

    Returns:
        Tuple[Dataset, ConvAE]: Loaded Dataset and ConvAE model.
    """
    if not train_args["model_path"]:
        print("The AE model needs to be trained first.")
        exit(1)

    # load the AE
    model_name = train_args["model_name"]
    ae = config.get_model(model_name)(**train_args["model"])
    ae.load(path=train_args["model_path"])

    # check if the model to load is trained and type AE
    if not check_model_type(ae):
        exit(1)

    # load the dataset which should be encoded, make sure to disable AE mode
    train_args["dataset"]["ae_mode"] = False

    if decode:
        train_args["dataset"]["sequence_length"] = ae.latent_length

    data = Dataset(**train_args["dataset"])
    return data, ae


def encode_dataset(data: Dataset, model: AutoEncoder) -> Dataset:
    """Loads a trained ConvAE using the config dictionary. Then encodes the
    dataset specified in the dictionary and stores it to 'export_path'.

    Args:
        data (Dataset): The dataset to encode.
        model (AutoEncoder): The trained ConvAE model to encode with.
    
    Returns:
        Dataset: The encoded dataset.
    """
    sequence_length, _, _ = data.sample_shape()

    encoded = []
    for i in tqdm(range(0, len(data), sequence_length)):
        # get the data as tensor
        x, _ = data[i]
        x = torch.unsqueeze(torch.tensor(x, dtype=model.precision), 0)

        # encode the data and unwrap the batch
        x = model.encode(x, as_array=True)
        x = np.swapaxes(x[:, :, 0, :], 2, 1)
        x = list(x[0, :, :])
        encoded += x

    # save the encoded dataset
    encoded = np.array(encoded)
    encoded = np.expand_dims(encoded, 1)

    labels = [f"enc_{i}" for i in encoded.shape[-1]]

    dataset = Dataset()
    dataset.overwrite_content(encoded, labels)

    return dataset


def decode_dataset(
        data: Dataset,
        model: AutoEncoder,
        scaler: Callable,
        labels: List[str] = None) -> Dataset:
    """Loads a trained ConvAE using the config dictionary. Then decodes the
    dataset specified in the dictionary and stores it to 'export_path'.

    Args:
        data (Dataset): The dataset to decode.
        model (AutoEncoder): The trained ConvAE model to decode with.
        scaler (Callable): The scaleback function of the original dataset.
        labels (List[str]): Original labels for each feature. Defaults to None.
    
    Returns:
        Dataset: The encoded dataset.
    """
    data.set_sequence(model.latent_length)

    decoded = []
    for i in tqdm(range(0, len(data), model.latent_length)):
        # get the data as tensor, apply 0-padding as sequence might be to small
        x = np.zeros((1, model.latent_length, 1, data.shape[-1]))
        d, _ = data[i]
        x[0, :, :d.shape[0], :] = d
        x = np.swapaxes(x, 1, 3)
        x = torch.tensor(x, dtype=model.precision)

        # encode the data and unwrap the batch
        x = model.decode(x)
        x = scaler(list(x.cpu().detach().numpy()[0, :, :, :]))
        decoded += list(x)

    # save the encoded dataset
    decoded = np.array(decoded)

    if not labels:
        labels = [f"dec_{i}" for i in range(decoded.shape[0])]

    dataset = Dataset()
    dataset.overwrite_content(decoded, labels)

    return dataset
