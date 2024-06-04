"""This module contains a simple CNN/LSTM model."""
from typing import Tuple

import torch
from torch.optim.lr_scheduler import ExponentialLR
from TimeSeriesDL.utils.activations import get_activation_from_string
from TimeSeriesDL.utils.config import config
from TimeSeriesDL.model.base_model import BaseModel


class ConvLSTM(BaseModel):
    """The simple model consists of a conv2d layer followed by a LSTM layer and
    two dense layers.

    Args:
        BaseModel (BaseModel): The base model.
    """

    def __init__(
        self,
        features: int = 1,
        sequence_length: int = 1,
        latent_size: int = 1,
        hidden_dim: int = 64,
        kernel_size: int = 15,
        stride: int = 1,
        padding: int = 0,
        lstm_layers: int = 1,
        out_act: str = "sigmoid",
        lr: float = 1e-3,
        lr_decay: float = 9e-1,
        adam_betas: Tuple[float, float] = (9e-1, 999e-3),
        tag: str = "",
        log: bool = True,
        precision: torch.dtype = torch.float32,
    ) -> None:
        super().__init__("ConvLSTM", tag, log)

        # define sequence parameters
        self._features = features
        self._sequence_length = sequence_length
        self._latent_size = latent_size
        self._last_activation = get_activation_from_string(out_act)

        # CNN hyperparameters
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding

        # LSTM hyperparameters
        self._hidden_dim = hidden_dim
        self._lstm_layers = lstm_layers
        self._precision = precision

        # lstm1, linear, cnn are all layers in the network
        # create the layers and initilize them based on our hyperparameters
        self._conv_1 = torch.nn.Conv2d(
            1,
            self._latent_size,
            (self._features, self._kernel_size),
            self._stride,
            self._padding,
            dtype=self._precision,
        )

        self._lstm = torch.nn.LSTM(
            self._latent_size,
            self._hidden_dim,
            self._lstm_layers,
            batch_first=True,
            dtype=self._precision,
        )

        self._linear_1 = torch.nn.Linear(self._hidden_dim, 64, dtype=self._precision)
        self._linear_2 = torch.nn.Linear(64, self._features, dtype=self._precision)

        self._loss_suite.add_loss_fn("MSE", torch.nn.MSELoss(), main=True)
        self._loss_suite.add_loss_fn("L1", torch.nn.L1Loss())

        self._optim = torch.optim.AdamW(self.parameters(), lr=lr, betas=adam_betas)
        self._scheduler = ExponentialLR(self._optim, gamma=lr_decay)

    @property
    def precision(self) -> torch.dtype:
        """The models precision.

        Returns:
            torch.dtype: The models precision.
        """
        return self._precision

    def load(self, path) -> None:
        """Loads the model's parameter given a path"""
        self.load_state_dict(torch.load(path))
        self.eval()

    def _reduction_network(self, x: torch.tensor) -> torch.tensor:
        # reduce the LSTM's output by using a few dense layers
        x = self._linear_1(x)
        x = torch.relu(x)

        x = self._linear_2(x)
        x = self._last_activation(x)
        return x

    def forward(self, x: torch.tensor):
        # CNN forward pass batch, channels, features, samples
        x: torch.tensor = torch.swapaxes(x, 1, 3) # batch, feature, channel, sample
        x: torch.tensor = torch.swapaxes(x, 2, 1) # batch, channel, feature, sample
        x = self._conv_1(x)
        x = torch.relu(x)

        # change output to batch, samples, channel, features
        x: torch.tensor = torch.swapaxes(x, 2, 1) # batch, feature, channel, sample
        x: torch.tensor = torch.swapaxes(x, 1, 3) # batch, sample, channel, feature

        # unwrap single value, possible as channel needs to be 1
        x = torch.squeeze(x, 3)

        # LSTM forward pass
        x, _ = self._lstm(x)
        x = torch.relu(x)

        # the last values are our predection ahead
        x = x[:, -1:, :]

        # finalize the prediction
        x = self._reduction_network(x)

        # add the single channel again to ensure correct loss calculation
        return torch.unsqueeze(x, 2)


config.register_model("ConvLSTM", ConvLSTM)
