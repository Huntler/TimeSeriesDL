"""This module contains a simple CNN/LSTM model."""

from argparse import ArgumentError
from datetime import datetime
from typing import Tuple

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from TimeSeriesDL.utils.config import config
from TimeSeriesDL.model.base_model import BaseModel


class ConvLSTM(BaseModel):
    """The simple model consists of a conv1d layer followed by a LSTM layer and
    two dense layers.

    Args:
        BaseModel (BaseModel): The base model.
    """

    def __init__(
        self,
        input_size: int,
        latent_size: int = None,
        hidden_dim: int = 64,
        xavier_init: bool = False,
        out_act: str = "relu",
        lr: float = 1e-3,
        lr_decay: float = 9e-1,
        adam_betas: Tuple[float, float] = (9e-1, 999e-3),
        kernel_size: int = 15,
        stride: int = 1,
        padding: int = 0,
        channels: int = 1,
        lstm_layers: int = 1,
        future_steps: int = 1,
        tag: str = "",
        log: bool = True,
        precision: torch.dtype = torch.float32,
    ) -> None:
        # if logging enabled, then create a tensorboard writer, otherwise prevent the
        # parent class to create a standard writer
        if log:
            now = datetime.now()
            self._tb_sub = now.strftime("%d%m%Y_%H%M%S")
            self._tb_path = f"runs/{tag}/SimpleModel/{self._tb_sub}"
            self._writer = SummaryWriter(self._tb_path)
        else:
            self._writer = False

        # initialize components using the parent class
        super(ConvLSTM, self).__init__(self._writer)

        # define sequence parameters
        self._future_steps = future_steps
        self._input_size = input_size
        self._latent_size = input_size if not latent_size else latent_size

        # CNN hyperparameters
        self._kernel_size = kernel_size
        self._channels = channels
        self._stride = stride
        self._padding = padding

        # LSTM hyperparameters
        self._hidden_dim = hidden_dim
        self._lstm_layers = lstm_layers

        # everything else
        self._xavier = xavier_init
        self._output_activation = out_act
        self._precision = precision

        # lstm1, linear, cnn are all layers in the network
        # create the layers and initilize them based on our hyperparameters
        self._conv_1 = torch.nn.Conv1d(
            in_channels=self._input_size,
            out_channels=self._latent_size,
            kernel_size=self._kernel_size,
            stride=self._stride,
            padding=self._padding,
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

        self._linear_2 = torch.nn.Linear(64, self._input_size, dtype=self._precision)

        if self._xavier:
            self._linear_1.weight = torch.nn.init.xavier_normal_(self._linear_1.weight)
            self._linear_2.weight = torch.nn.init.xavier_normal_(self._linear_2.weight)
        else:
            self._linear_1.weight = torch.nn.init.zeros_(self._linear_1.weight)
            self._linear_2.weight = torch.nn.init.zeros_(self._linear_2.weight)

        self._loss_suite.add_loss_fn("MSE", torch.nn.MSELoss())
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
        x = torch.relu(x)
        x = self._linear_1(x)
        x = torch.relu(x)
        x = self._linear_2(x)

        # output from the last layer
        if self._output_activation == "relu":
            x = torch.relu(x)
        elif self._output_activation == "sigmoid":
            x = torch.sigmoid(x)
        elif self._output_activation == "tanh":
            x = torch.tanh(x)
        else:
            raise ArgumentError(
                argument=None,
                message="Wrong output actiavtion specified "
                + "(relu | sigmoid | tanh).",
            )

        return x

    def forward(self, x: torch.tensor, future_steps: int = 1):
        # CNN forward pass
        x: torch.tensor = torch.transpose(x, 2, 1)
        x = self._conv_1(x)
        x: torch.tensor = torch.transpose(x, 2, 1)
        x = torch.relu(x)

        # LSTM forward pass
        x, _ = self._lstm(x)

        # the last values are our predection ahead
        x = x[:, -future_steps:, :]

        # finalize the prediction
        x = self._reduction_network(x)
        return x


config.register_model("ConvLSTM", ConvLSTM)
