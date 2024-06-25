"""This module contains a dummy auto encoder class, useful to test other modules."""
import torch
import numpy as np
from TimeSeriesDL.model.auto_encoder import AutoEncoder

class DummyAutoEncoder(AutoEncoder):
    """The dummy auto encoder mimics the behaviour of a real auto encoder,
    required to test the transcode functions.
    """
    def __init__(self, sequence: int) -> None:
        super().__init__()
        self._sequence = sequence

    def encode(self, x: torch.tensor, as_array: bool = False) -> np.array:
        """Always returns the input as numpy array."""
        if as_array:
            return x.cpu().detach().numpy()

        x: torch.tensor = torch.swapaxes(x, 1, 3)
        x: torch.tensor = torch.swapaxes(x, 2, 1)
        return x

    def decode(self, x: torch.tensor, as_array: bool = False) -> np.array:
        """Swaps the axis of the input as a real auto encoder would do."""
        x: torch.tensor = torch.swapaxes(x, 2, 1)
        x: torch.tensor = torch.swapaxes(x, 1, 3)

        if as_array:
            return x.cpu().detach().numpy()
        return x

    @property
    def precision(self):
        """Returns the precision."""
        return torch.float32

    @property
    def latent_length(self):
        """Returns an abitrary latent length."""
        return self._sequence

    def forward(self, batch: torch.tensor) -> torch.tensor:
        return batch
