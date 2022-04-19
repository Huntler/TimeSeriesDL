from argparse import ArgumentError
from datetime import datetime
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from .base_model import BaseModel
from torch.optim.lr_scheduler import ExponentialLR


class LstmModel(BaseModel):
    def __init__(self, input_size: int, hidden_dim: int = 64, xavier_init: bool = False, out_act: str = "relu",
                 lr: float = 1e-3, lr_decay: float = 9e-1, adam_betas: List[float] = [9e-1, 999e-3],
                 sequence_length: int = 1, future_steps: int = 1,
                 log: bool = True, precision: torch.dtype = torch.float32) -> None:
        # if logging enalbed, then create a tensorboard writer, otherwise prevent the
        # parent class to create a standard writer
        if log:
            now = datetime.now()
            self.__tb_sub = now.strftime("%d%m%Y_%H%M%S")
            self._tb_path = f"runs/LSTM_Model/{self.__tb_sub}"
            self._writer = SummaryWriter(self._tb_path)
        else:
            self._writer = False

        # initialize components using the parent class
        super(LstmModel, self).__init__()

        self.__input_size = input_size
        self.__sequence_length = sequence_length
        self.__future_steps = future_steps
        self.__xavier = xavier_init
        self.__output_activation = out_act
        self.__n_layers = 0
        self.__hidden_dim = hidden_dim
        self.__precision = precision

        # lstm1, linear are all layers in the network
        self.__lstm_1 = torch.nn.LSTMCell(
            self.__input_size,
            hidden_size=self.__hidden_dim,
            dtype=self.__precision
        )

        # create the dense layers and initilize them based on our hyperparameters
        self.__linear_1 = torch.nn.Linear(
            self.__hidden_dim,
            64,
            dtype=self.__precision
        )

        self.__linear_2 = torch.nn.Linear(
            64,
            1,
            dtype=self.__precision
        )

        if self.__xavier:
            self.__linear_1.weight = torch.nn.init.xavier_normal_(
                self.__linear_1.weight)
            self.__linear_2.weight = torch.nn.init.xavier_normal_(
                self.__linear_2.weight)
        else:
            self.__linear_1.weight = torch.nn.init.zeros_(
                self.__linear_1.weight)
            self.__linear_2.weight = torch.nn.init.zeros_(
                self.__linear_2.weight)

        # define loss function, optimizer and scheduler for the learning rate
        self._loss_fn = torch.nn.L1Loss()
        self._optim = torch.optim.AdamW(self.parameters(), lr=lr, betas=adam_betas)
        self._scheduler = ExponentialLR(self._optim, gamma=lr_decay)

    @property
    def precision(self) -> torch.dtype:
        return self.__precision

    def __init_hidden_states(self, batch_size: int) -> Tuple[torch.tensor]:
        """This method is used to initialize the hidden cell states of a LSTM layer.

        Args:
            batch_size (int): The batch size of our input.
            n_samples (int): The amount of samples per batch.

        Returns:
            Tuple[torch.tensor]: The cell states as tuple (hidden, cell)
        """
        h_t = torch.zeros(batch_size,
                          self.__hidden_dim, dtype=self.__precision)
        c_t = torch.zeros(batch_size,
                          self.__hidden_dim, dtype=self.__precision)

        return h_t, c_t

    def load(self, path) -> None:
        """Loads the model's parameter given a path
        """
        self.load_state_dict(torch.load(path))
        self.eval()

    def network(self, X, h, c):
        """This method contains the model's network architecture.

        Args:
            X (_type_): Data X
            h (_type_): Hidden states
            c (_type_): Cell states

        Raises:
            ArgumentError: _description_

        Returns:
            _type_: The output of all hidden states and the current hidden and cell states.
        """
        # iterate over each time step and predict the output using the LSTM
        for input_t in X.split(1, dim=1):
            h, c = self.__lstm_1(input_t[:, 0], (h, c))

        x = h
        # x = torch.relu(h)

        # pass the normalized output of the LSTM into the Dense layers
        x = self.__linear_1(x)
        x = torch.relu(x)

        x = self.__linear_2(x)
        if self.__output_activation == "relu":
                output = torch.relu(x)
        elif self.__output_activation == "sigmoid":
                output = torch.sigmoid(x)
        elif self.__output_activation == "tanh":
                output = torch.tanh(x)
        elif self.__output_activation == "linear":
            output = x
        else:
            raise ArgumentError(
                "Wrong output actiavtion specified (relu | sigmoid | tanh).")

        return output, (h, c)

    def forward(self, X):
        # based on the official PyTorch documentation: 
        # https://github.com/pytorch/examples/blob/main/time_sequence_prediction/train.py
        batch_size, n_samples, dim = X.shape

        # reset the hidden states for each sequence we train
        h, c = self.__init_hidden_states(batch_size)

        # fit the hidden states to the given batch X
        output_batch, (h, c) = self.network(X, h, c)
        output_batch = torch.unsqueeze(output_batch, -1) 

        # look several steps ahead
        outputs = torch.empty(batch_size, self.__future_steps, dim)
        for i in range(self.__future_steps):
            output_batch, (h, c) = self.network(output_batch, h, c)
            outputs[:, i, :] = output_batch
            output_batch = torch.unsqueeze(output_batch, -1) 

        return outputs
