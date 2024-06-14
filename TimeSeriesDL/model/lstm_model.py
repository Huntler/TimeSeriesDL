"""This module contains a straightfowrad LSTM model."""
import torch
from TimeSeriesDL.model.base_model import BaseModel
from TimeSeriesDL.utils import get_activation_from_string
from TimeSeriesDL.utils import model_register


class LSTM(BaseModel):
    """Straightforward LSTM model predicting one timestep ahead.

    Args:
        BaseModel (BaseModel): The base class.
    """
    def __init__(
        self,
        features: int = 1,
        lstm_layers: int = 1,
        dropout: float = 0.1,
        hidden_dim: int = 64,
        last_activation: str = "sigmoid",
        loss: str = "MSELoss",
        optimizer: str = "Adam",
        lr: float = 1e-3
    ) -> None:
        # initialize components using the parent class
        super().__init__(loss, optimizer, lr)

        # LSTM hyperparameters
        self._hidden_dim = hidden_dim
        self._lstm_layers = lstm_layers

        # data parameter
        self._features = features

        self._output_activation = get_activation_from_string(last_activation)

        # define the layers
        self._lstm = torch.nn.LSTM(
            self._features,
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
                self._hidden_dim // 2, self._features
            )
        )

    def forward(self, batch: torch.tensor):
        # LSTM forward pass
        _, (hidden, _) = self._lstm(batch)
        x = torch.relu(hidden[-1])

        # reduce the LSTM's output by using a few dense layers
        x = self._regressor(x)

        return self._output_activation(x)


model_register.register_model("LSTM", LSTM)
