"""This module contains a simple CNN/LSTM model."""
from argparse import ArgumentError
from datetime import datetime
from typing import Tuple

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from TimeSeriesDL.model.base_model import BaseModel


class SimpleModel(BaseModel):
    """The simple model consists of a Conv Layer

    Args:
        BaseModel (_type_): _description_
    """

    def __init__(self, input_size: int, hidden_dim: int = 64, xavier_init: bool = False,
                 out_act: str = "relu", lr: float = 1e-3, lr_decay: float = 9e-1,
                 adam_betas: Tuple[float, float] = (9e-1, 999e-3), kernel_size: int = 15,
                 stride: int = 1, padding: int = 0, channels: int = 1,
                 sequence_length: int = 1, future_steps: int = 1, log: bool = True,
                 precision: torch.dtype = torch.float32) -> None:
        # if logging enalbed, then create a tensorboard writer, otherwise prevent the
        # parent class to create a standard writer
        if log:
            now = datetime.now()
            self._tb_sub = now.strftime("%d%m%Y_%H%M%S")
            self._tb_path = f"runs/SimpleModel/{self._tb_sub}"
            self._writer = SummaryWriter(self._tb_path)
        else:
            self._writer = False

        # initialize components using the parent class
        super(SimpleModel, self).__init__(self._writer)

        # define sequence parameters
        self._sequence_length = sequence_length
        self._future_steps = future_steps
        self._input_size = input_size

        # CNN hyperparameters
        self._kernel_size = kernel_size
        self._channels = channels
        self._stride = stride
        self._padding = padding

        # LSTM hyperparameters
        self._hidden_dim = hidden_dim

        # everything else
        self._xavier = xavier_init
        self._output_activation = out_act
        self._precision = precision

        # lstm1, linear, cnn are all layers in the network
        # create the layers and initilize them based on our hyperparameters
        self._conv_1 = torch.nn.Conv1d(
            in_channels=self._input_size,
            out_channels=self._input_size,
            kernel_size=self._kernel_size,
            stride=self._stride,
            padding=self._padding,
            dtype=self._precision
        )

        self._lstm_1 = torch.nn.LSTMCell(
            self._input_size,
            self._hidden_dim,
            dtype=self._precision
        )
        self._linear_1 = torch.nn.Linear(
            self._hidden_dim,
            64,
            dtype=self._precision
        )

        self._linear_2 = torch.nn.Linear(
            64,
            1,
            dtype=self._precision
        )

        if self._xavier:
            self._linear_1.weight = torch.nn.init.xavier_normal_(
                self._linear_1.weight)
            self._linear_2.weight = torch.nn.init.xavier_normal_(
                self._linear_2.weight)
        else:
            self._linear_1.weight = torch.nn.init.zeros_(
                self._linear_1.weight)
            self._linear_2.weight = torch.nn.init.zeros_(
                self._linear_2.weight)

        self._loss_fn = torch.nn.MSELoss()
        self._optim = torch.optim.AdamW(
            self.parameters(), lr=lr, betas=adam_betas)
        self._scheduler = ExponentialLR(self._optim, gamma=lr_decay)

    @property
    def precision(self) -> torch.dtype:
        """The models precision.

        Returns:
            torch.dtype: The models precision.
        """
        return self._precision

    def _init_hidden_states(self, batch_size: int, n_samples: int) -> Tuple[torch.tensor]:
        """This method is used to initialize the hidden cell states of a LSTM layer.
        Args:
            batch_size (int): The batch size of our input.
            n_samples (int): The amount of samples per batch.
        Returns:
            Tuple[torch.tensor]: The cell states as tuple (hidden, cell)
        """
        h_t = torch.zeros(batch_size, n_samples,
                          self._hidden_dim, dtype=self._precision)
        c_t = torch.zeros(batch_size, n_samples,
                          self._hidden_dim, dtype=self._precision)

        return h_t, c_t

    def load(self, path) -> None:
        """Loads the model's parameter given a path
        """
        self.load_state_dict(torch.load(path))
        self.eval()

    def _network(self, x: torch.tensor, h: torch.tensor, c: torch.tensor):
        """This method contains the model's network architecture.
        Args:
            x (torch.tensor): Data x
            h (torch.tensor): Hidden states
            c (torch.tensor): Cell states
        Raises:
            ArgumentError: Possible activation functions are 'relu', 'sigmoid' and 'tanh'.
        Returns:
            torch.tensor: The output of all hidden states and the current hidden and cell states.
        """
        batch_size, n_samples, _ = h.shape
        _, _, dim = x.shape

        # initial hidden and cell states
        output_batch = torch.empty(batch_size, n_samples, dim)
        for i, batch in enumerate(x):
            hidden = (h[i], c[i])
            out, hidden = self._lstm_1(batch, hidden)

            # reduce the LSTM's output by using a few dense layers
            x = torch.relu(out)
            x = self._linear_1(x)
            x = torch.relu(x)
            x = self._linear_2(x)

            # output from the last layer
            if self._output_activation == "relu":
                output = torch.relu(x)
            elif self._output_activation == "sigmoid":
                output = torch.sigmoid(x)
            elif self._output_activation == "tanh":
                output = torch.tanh(x)
            else:
                raise ArgumentError(argument=None,
                                    message="Wrong output actiavtion specified " +
                                    "(relu | sigmoid | tanh).")

            output_batch[i, :, :] = output

        return output_batch, (h, c)

    def forward(self, x, future_steps: int = 1):
        # conv1d forward pass
        x: torch.tensor = torch.transpose(x, 2, 1)
        x = self._conv_1(x)
        x: torch.tensor = torch.transpose(x, 2, 1)
        x = torch.relu(x)

        # LSTM preparation
        # reset the hidden states for each sequence we train
        batch_size, n_samples, _ = x.shape
        h, c = self._init_hidden_states(batch_size, n_samples)

        # LSTM forward pass
        output_batch, (h, c) = self._network(x, h, c)

        # look several steps ahead
        for _ in range(future_steps):
            output_batch, (h, c) = self._network(output_batch, h, c)

        # the last values are our predection ahead
        output_batch = output_batch[:, -future_steps:, :]
        return output_batch
