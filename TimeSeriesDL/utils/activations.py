"""This module registers activation functions from the torch library to be 
used with the config manager."""

from typing import Callable
import torch

_str_to_activation = {"relu": torch.relu, "sigmoid": torch.sigmoid, "tanh": torch.tanh}


def get_activation_from_string(activation_str: str) -> Callable:
    """Returns an activation function corresponding to the provided string. Available
    activation functions are:
    - relu
    - sigmoid
    - tanh

    Args:
        activation_str (str): The name of the activation function.

    Returns:
        Callable: The activation function from the torch library.
    """
    activation_str = activation_str.strip().lower()
    func = _str_to_activation.get(activation_str, None)

    if not func:
        print(
            f"No activation function candidate found for {activation_str}. Fallback to ReLU."
        )
        return _str_to_activation["relu"]

    return func
