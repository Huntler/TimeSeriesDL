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

def reproductionLoss(x: Tuple[torch.tensor, torch.tensor, torch.tensor],
                     x_hat: torch.tensor) -> torch.tensor:
    """Calculates the Reproduction Loss of a variational auto-encoder (VAE).

    Args:
        x (Tuple[torch.tensor, torch.tensor, torch.tensor]): x, mean, var.
        x_hat (torch.tensor): x_hat.

    Returns:
        torch.tensor: The calculated loss.
    """
    x, mean, log_var = x
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + kld
