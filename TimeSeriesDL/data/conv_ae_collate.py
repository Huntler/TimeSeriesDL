"""This module replaces the collate function of a DataLoader to encode/decode the data."""
from typing import Callable, List, Tuple
import numpy as np
import torch
from TimeSeriesDL.model.auto_encoder.conv_ae_model import ConvAE


class AutoEncoderCollate:
    """The wrapper class defines a custom collate function to be used in a DataLoader.

    Args:
        Dataset (Dataset): The base dataset class.
    """

    def __init__(self, ae: ConvAE, device: str = "cpu") -> None:
        self._device = device
        self._ae = ae

        self._ae.use_device(self._device)

    def collate_fn(self) -> Callable[..., Tuple[torch.tensor, torch.tensor]]:
        """Custom collate function designed to work with a DataLoader. Uses
        the object's ConvAE to encode a batch of samples before returning it.
        The function swaps feature and sequence in the dimensions.

        Returns:
            Callable[..., Tuple[torch.tensor, torch.tensor]]: The collate function.
        """
        def _collate_fn(data: List[Tuple[torch.tensor, torch.tensor]]):
            x, y = zip(*data)

            x = torch.tensor(np.array(x), device=self._device)
            y = torch.tensor(np.array(y), device=self._device)

            x = self._ae.encode(x)
            y = self._ae.encode(y)

            x = torch.swapaxes(x, 1, 2)
            y = torch.swapaxes(y, 1, 2)

            return x, y
        return _collate_fn
