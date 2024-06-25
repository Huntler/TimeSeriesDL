"""Encodes a dataset and stores it."""
from typing import Callable, Dict, List, Tuple
from tqdm import trange
import torch
import numpy as np
from TimeSeriesDL.data import Dataset
from TimeSeriesDL.model.auto_encoder import AutoEncoder
from TimeSeriesDL.utils import model_register


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
    ae = model_register.get_model(model_name)(**train_args["model"])
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


def encode_dataset(data: Dataset, model: AutoEncoder, verbose: bool = False) -> Dataset:
    """Loads a trained ConvAE using the config dictionary. Then encodes the
    dataset specified in the dictionary and stores it to 'export_path'.

    Args:
        data (Dataset): The dataset to encode.
        model (AutoEncoder): The trained ConvAE model to encode with.
        verbose (bool): Show progress bar. Defaults to False.
    
    Returns:
        Dataset: The encoded dataset.
    """
    assert data.num_matrices == 1, "Single-matrix dataset expected."
    sequence_length, _, _ = data.sample_shape()
    future_steps, _, _ = data.sample_shape(True)

    encoded = []
    loop = trange if verbose else range
    for i in loop(0, len(data), sequence_length):
        # get the data as tensor
        x = data.slice(i, i + sequence_length)
        x = torch.unsqueeze(torch.tensor(x, dtype=model.precision), 0)

        # encode the data and unwrap the batch
        x = model.encode(x, as_array=True)
        encoded += list(x[0])

    # save the encoded dataset
    encoded = np.array(encoded)

    labels = [f"enc_{i}" for i in range(encoded.shape[-1])]
    dataset = Dataset(sequence_length=sequence_length, future_steps=future_steps)
    dataset.overwrite_content(encoded, labels)

    return dataset


def decode_dataset(
        data: Dataset,
        model: AutoEncoder,
        scaler: Callable = None,
        labels: List[str] = None,
        verbose: bool = False) -> Dataset:
    """Loads a trained ConvAE using the config dictionary. Then decodes the
    dataset specified in the dictionary and stores it to 'export_path'.

    Args:
        data (Dataset): The dataset to decode.
        model (AutoEncoder): The trained ConvAE model to decode with.
        scaler (Callable): The scaleback function of the original dataset. Defaults to None.
        labels (List[str]): Original labels for each feature. Defaults to None.
        verbose (bool): Show progress bar. Defaults to False.

    
    Returns:
        Dataset: The encoded dataset.
    """
    assert data.num_matrices == 1, "Single-matrix dataset expected."
    data.set_sequence(model.latent_length)

    decoded = []
    loop = trange if verbose else range
    for i in loop(0, len(data), model.latent_length):
        x = data.slice(i, i + model.latent_length)

        # add batch and reshape the input as it was after encoding
        x = torch.unsqueeze(torch.tensor(x, dtype=model.precision), 0)
        x: torch.tensor = torch.swapaxes(x, 1, 3)
        x: torch.tensor = torch.swapaxes(x, 1, 2)

        # encode the data and unwrap the batch
        x = model.decode(x, as_array=True)[0, :, :, :]
        if scaler:
            x = scaler(x)
        decoded += list(x)

    # save the encoded dataset
    decoded = np.array(decoded)

    if not labels:
        labels = [f"dec_{i}" for i in range(decoded.shape[0])]

    dataset = Dataset(sequence_length=model.latent_length)
    dataset.overwrite_content(decoded, labels)

    return dataset
