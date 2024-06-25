"""This module contains a simple CNN/LSTM model."""

import torch
from TimeSeriesDL.utils import get_activation_from_string
from TimeSeriesDL.utils import model_register
from TimeSeriesDL.model.base_model import BaseModel


class ConvLSTM(BaseModel):
    """The simple model consists of a conv2d layer followed by a LSTM layer and
    two dense layers.

    Args:
        BaseModel (BaseModel): The base model.
    """

    def __init__(
        self,
        in_features: int = 1,
        sequence_length: int = 1,
        future_steps: int = 1,
        latent_features: int = 1,
        hidden_dim: int = 64,
        kernel: int = 3,
        padding: int = 0,
        lstm_layers: int = 1,
        dropout: float = 0.1,
        out_act: str = "sigmoid",
        loss: str = "MSELoss",
        optimizer: str = "Adam",
        lr: float = 1e-3
    ) -> None:
        super().__init__(loss, optimizer, lr)

        # define sequence parameters
        self._in_features = in_features
        self._sequence_length = sequence_length
        self._latent_features = latent_features
        self._last_activation = get_activation_from_string(out_act)

        # CNN hyperparameters
        self._kernel = (self._in_features, kernel)
        self._padding = padding
        latent_size = future_steps if future_steps > 1 else sequence_length
        self._stride = min((sequence_length - kernel) // (latent_size - 1), kernel // 2)
        self._stride = max(self._stride, 1)
        print("Kernel", self._kernel, "with stride", self._stride)

        # LSTM hyperparameters
        self._hidden_dim = hidden_dim
        self._lstm_layers = lstm_layers
        self._future_steps = future_steps

        # lstm1, linear, cnn are all layers in the network
        # create the layers and initilize them based on our hyperparameters
        self._conv = torch.nn.Conv2d(
            1,
            self._latent_features,
            self._kernel,
            self._stride,
            self._padding
        )

        self._lstm = torch.nn.LSTM(
            self._latent_features,
            self._hidden_dim,
            self._lstm_layers,
            batch_first=True,
            dropout=dropout if self._lstm_layers > 1 else 0
        )

        self._regressor = torch.nn.Sequential(
            torch.nn.Linear(
                self._hidden_dim, self._hidden_dim // 2
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
                self._hidden_dim // 2, self._latent_features
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(
            self._latent_features, self._in_features
            )
        )

    def _reduction_network(self, x: torch.tensor) -> torch.tensor:
        # reduce the LSTM's output by using a few dense layers
        x = self._linear_1(x)
        x = torch.relu(x)

        x = self._linear_2(x)
        x = self._last_activation(x)
        return x

    def forward(self, batch: torch.tensor):
        # CNN forward pass batch, channels, features, samples
        x = torch.unsqueeze(batch, dim=-2)
        x: torch.tensor = torch.swapaxes(x, 1, 3)  # batch, feature, channel, sample
        x: torch.tensor = torch.swapaxes(x, 2, 1)  # batch, channel, feature, sample
        x = self._conv(x)
        x = torch.relu(x)

        # change output to batch, samples, channel, features
        x: torch.tensor = torch.swapaxes(x, 2, 1)  # batch, feature, channel, sample
        x: torch.tensor = torch.swapaxes(x, 1, 3)  # batch, sample, channel, feature

        # unwrap single value, possible as channel needs to be 1
        x = torch.squeeze(x, 3)

        # LSTM forward pass
        _, (hidden, _) = self._lstm(x)
        x = torch.relu(hidden[-1])

        # reduce the LSTM's output to match it's input
        x = self._regressor(x)
        x = self._last_activation(x)

        x = torch.unsqueeze(x, 1)
        return x


model_register.register_model("ConvLSTM", ConvLSTM)
