"""This module defines the interface for an auto-encoder."""
import numpy as np
import torch
from TimeSeriesDL.model.base_model import BaseModel


class AutoEncoder(BaseModel):
    """The auto-encoder interface.

    Args:
        BaseModel (BaseModel): Inherits from the base model.

    Raises:
        NotImplementedError: Latent space length.
        NotImplementedError: Encoder.
        NotImplementedError: Decoder.
    """
    @property
    def precision(self) -> torch.dtype:
        """The precision required by the model.

        Returns:
            torch.dtype: The model's precision.
        """
        return self._precision

    @property
    def latent_length(self) -> int:
        """Sample length of the latent space.

        Returns:
            int: Length of the latent space.
        """
        raise NotImplementedError

    def encode(self, x: torch.tensor, as_array: bool = False) -> torch.tensor:
        """Encodes the input.

        Args:
            x (torch.tensor): The input.
            as_array (bool): Retuns the encoded value as np.array. Defaults to False.

        Returns:
            torch.tensor: The encoded output as tensor if as_array is set to False.
        """
        raise NotImplementedError

    def decode(self, x: torch.tensor, as_array: bool = False) -> torch.tensor:
        """Decodes the input, should be the same as before encoding the data.

        Args:
            x (torch.tensor): The input.
            as_array (bool): Retuns the encoded value as np.array. Defaults to False.

        Returns:
            torch.tensor: The decoded data.
        """
        raise NotImplementedError
