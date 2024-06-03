"""This module contains a straightfowrad LSTM model."""

from datetime import datetime
from typing import Tuple
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from TimeSeriesDL.model.base_model import BaseModel
from TimeSeriesDL.utils.activations import get_activation_from_string
from TimeSeriesDL.utils.config import config


class LSTM(BaseModel):
    def __init__(
        self,
        features: int = 1,
        lstm_layers: int = 1,
        hidden_dim: int = 64,
        last_activation: str = "relu",
        lr: float = 1e-3,
        lr_decay: float = 9e-1,
        adam_betas: Tuple[float, float] = (9e-1, 999e-3),
        log: bool = True,
        tag: str = "",
        precision: torch.dtype = torch.float32
    ) -> None:
        # if logging enalbed, then create a tensorboard writer, otherwise prevent the
        # parent class to create a standard writer
        if log:
            now = datetime.now()
            self._tb_sub = now.strftime("%d%m%Y_%H%M%S")
            self._tb_path = f"runs/{tag}/LSTM/{self._tb_sub}"
            self._writer = SummaryWriter(self._tb_path)
        else:
            self._writer = False

        # initialize components using the parent class
        super(LSTM, self).__init__(self._writer)

        # LSTM hyperparameters
        self._hidden_dim = hidden_dim
        self._lstm_layers = lstm_layers

        # data parameter
        self._features = features

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
        self._linear_1 = torch.nn.Linear(self._hidden_dim, 64, dtype=self._precision)

        self._linear_2 = torch.nn.Linear(64, self._features, dtype=self._precision)

        # define optimizer, loss function and variable learning rate
        self._loss_suite.add_loss_fn("MSE", torch.nn.MSELoss())
        self._loss_suite.add_loss_fn("L1", torch.nn.L1Loss())

        self._optim = torch.optim.AdamW(self.parameters(), lr=lr, betas=adam_betas)
        self._scheduler = ExponentialLR(self._optim, gamma=lr_decay)

    def forward(self, x, future_steps: int = 1):
        # LSTM forward pass
        x, _ = self._lstm(x)

        # the last values are our predection ahead
        x = x[:, -future_steps:, :]

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