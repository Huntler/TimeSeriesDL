"""This module contains various loss functions"""
from typing import Tuple
import torch
from torch import nn


def rmse_loss_fn(yhat, y):
    """Calculates the RMSE loss sqrt(mean(y_hat - y)**2)

    Args:
        yhat (torch.tensor): Predicted value.
        y (torch.tensor): Actual value.

    Returns:
        torch.tensor: Loss.
    """
    return torch.sqrt(torch.mean((yhat - y) ** 2))
