"""This module contains various loss functions"""
from torch import nn


_loss_register = {
    "MSELoss": nn.MSELoss(),
    "CELoss": nn.CrossEntropyLoss(),
    "L1Loss": nn.L1Loss(),
    "BCELoss": nn.BCELoss(),
    "BCEWithLogitsLoss": nn.BCEWithLogitsLoss(),
    "KLDivLoss": nn.KLDivLoss()
}

def get_loss_by_name(name: str) -> nn.Module:
    """Get loss function by name.

    Args:
        name (str): Name of the loss function to be retrieved

    Raises:
        ValueError: If the loss is unknown
        
    Returns:
        nn.Module: Loss function
    """
    if name not in _loss_register:
        raise ValueError(f"Unknown loss {name}")

    return _loss_register[name]
