"""This module contains a straightfowrad LSTM model."""
import torch
from TimeSeriesDL.model.base_model import BaseModel
from TimeSeriesDL.utils.activations import get_activation_from_string
from TimeSeriesDL.utils.config import config


class LSTM(BaseModel):
    def __init__(
        self,
        features: int = 1,
        future_steps: int = 1,
        lstm_layers: int = 1,
        hidden_dim: int = 64,
        last_activation: str = "relu",
        tag: str = "",
        precision: torch.dtype = torch.float32
    ) -> None:
        # initialize components using the parent class
        super().__init__("LSTM", tag)

        # LSTM hyperparameters
        self._hidden_dim = hidden_dim
        self._lstm_layers = lstm_layers

        # data parameter
        self._features = features
        self._future_steps = future_steps

        self._output_activation = get_activation_from_string(last_activation)
        self._precision = precision

        # define the layers
        self._lstm = torch.nn.LSTM(
            self._features,
            self._hidden_dim,
            self._lstm_layers,
            batch_first=True,
            dtype=self._precision,
        )
        self._linear_1 = torch.nn.Linear(
            self._hidden_dim, 64, dtype=self._precision)

        self._linear_2 = torch.nn.Linear(
            64, self._features, dtype=self._precision)

    def forward(self, x):
        # LSTM forward pass
        x, _ = self._lstm(x)

        # the last values are our predection ahead
        x = x[:, -self._future_steps:, :]

        # reduce the LSTM's output by using a few dense layers
        x = torch.relu(x)
        x = self._linear_1(x)
        x = torch.relu(x)
        x = self._linear_2(x)

        return self._output_activation(x)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path))
        self.eval()


config.register_model("LSTM", LSTM)
