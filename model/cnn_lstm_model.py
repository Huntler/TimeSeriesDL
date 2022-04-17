from argparse import ArgumentError
from datetime import datetime
from math import ceil
from typing import List, Tuple
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from .base_model import BaseModel


class CnnLstmModel(BaseModel):
    def __init__(self, input_size: int, hidden_dim: int = 64, xavier_init: bool = False, out_act: str = "relu",
                 lr: float = 1e-3, lr_decay: float = 9e-1, adam_betas: List[float] = [9e-1, 999e-3],
                 kernel_size: int = 15, stride: int = 1, padding: int = 0, channels: int = 1,
                 sequence_length: int = 1,
                 log: bool = True, precision: torch.dtype = torch.float32) -> None:
        # if logging enalbed, then create a tensorboard writer, otherwise prevent the
        # parent class to create a standard writer
        if log:
            now = datetime.now()
            self.__tb_sub = now.strftime("%d%m%Y_%H%M%S")
            self._tb_path = f"runs/CNN_LSTM_Model/{self.__tb_sub}"
            self._writer = SummaryWriter(self._tb_path)
        else:
            self._writer = False

        # initialize components using the parent class
        super(CnnLstmModel, self).__init__()

        self.__sequence_length = sequence_length
        self.__input_size = input_size

        # CNN hyperparameters
        self.__kernel_size = kernel_size
        self.__stride = stride
        self.__padding = padding
        self.__channels = channels

        # LSTM hyperparameters
        self.__hidden_dim = hidden_dim

        # everything else
        self.__xavier = xavier_init
        self.__output_activation = out_act
        self.__precision = precision

        # lstm1, linear, cnn are all layers in the network
        self.__conv_1 = torch.nn.Conv1d(
            in_channels=self.__input_size, 
            out_channels=self.__channels, 
            kernel_size=self.__kernel_size, 
            stride=self.__stride, 
            padding=self.__padding, 
            dtype=self.__precision
        )

        # size: [(W−K+2P)/S]+1
        size = int(((self.__sequence_length - self.__kernel_size + 2 * self.__padding) / self.__stride) + 1)
        self.__batch_norm_0 = torch.nn.BatchNorm1d(size, track_running_stats=False)
        self.__cnn_activation = torch.nn.LeakyReLU()

        self.__lstm_1 = torch.nn.LSTMCell(
            self.__channels,
            self.__hidden_dim,
            dtype=self.__precision
        )
        self.__batch_norm_1 = torch.nn.BatchNorm1d(self.__hidden_dim, track_running_stats=False)

        # create the dense layers and initilize them based on our hyperparameters
        self.__linear_1 = torch.nn.Linear(
            self.__hidden_dim,
            64,
            dtype=self.__precision
        )
        self.__batch_norm_2 = torch.nn.BatchNorm1d(64, track_running_stats=False)

        self.__linear_2 = torch.nn.Linear(
            64,
            self.__channels,
            dtype=self.__precision
        )
        self.__batch_norm_3 = torch.nn.BatchNorm1d(self.__channels, track_running_stats=False)
        
        self.__linear_3 = torch.nn.Linear(
            self.__channels,
            1,
            dtype=self.__precision
        )

        if self.__xavier:
            self.__linear_1.weight = torch.nn.init.xavier_normal_(
                self.__linear_1.weight)
            self.__linear_2.weight = torch.nn.init.xavier_normal_(
                self.__linear_2.weight)
            self.__linear_3.weight = torch.nn.init.xavier_normal_(
                self.__linear_3.weight)
        else:
            self.__linear_1.weight = torch.nn.init.zeros_(self.__linear_1.weight)
            self.__linear_2.weight = torch.nn.init.zeros_(self.__linear_2.weight)
            self.__linear_3.weight = torch.nn.init.zeros_(self.__linear_3.weight)

        self._loss_fn = torch.nn.MSELoss()
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
        h_t = torch.zeros(batch_size, self.__hidden_dim, dtype=self.__precision)
        c_t = torch.zeros(batch_size, self.__hidden_dim, dtype=self.__precision)

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
            h, c = self.__lstm_1(input_t[:, 0, :], (h, c))
            
        x = self.__batch_norm_1(h) if h.size(0) > 1 else h
        x = torch.relu(x)

        # reduce the LSTM's output by using a few dense layers
        x = self.__linear_1(x)
        x = self.__batch_norm_2(x) if x.size(0) > 1 else x
        x = torch.relu(x)

        x = self.__linear_2(x)
        x = self.__batch_norm_3(x) if x.size(0) > 1 else x
        output = torch.relu(x)
        
        return output, (h, c)

    def forward(self, X, future_steps: int = 1):
        # CNN forward pass
        x: torch.tensor = torch.transpose(X, 2, 1)
        x = self.__conv_1(x)
        x: torch.tensor = torch.transpose(x, 2, 1)
        x = self.__batch_norm_0(x)
        x = self.__cnn_activation(x)

        # LSTM preparation
        # reset the hidden states for each sequence we train
        batch_size, n_samples, dim = x.shape
        h, c = self.__init_hidden_states(batch_size)

        # LSTM forward pass
        output_batch, (h, c) = self.network(x, h, c)
        output_batch = torch.unsqueeze(output_batch, 1) 

        # look several steps ahead
        outputs = []
        for i in range(future_steps):
            output_batch, (h, c) = self.network(output_batch, h, c)
            output_batch = torch.unsqueeze(output_batch, 1) 
            outputs += output_batch 
        
        # reshape the ouput again to match the input's shape
        outputs = torch.cat(outputs, dim=0)
        outputs = torch.unsqueeze(outputs, 1)

        # reduce the amount of features (occur if CNN has multiple channels)
        x = self.__linear_3(outputs)

        # output from the last layer
        if self.__output_activation == "relu":
                output = torch.relu(x)
        elif self.__output_activation == "sigmoid":
                output = torch.sigmoid(x)
        elif self.__output_activation =="tanh":
                output = torch.tanh(x)
        else:
                raise ArgumentError(
                    "Wrong output actiavtion specified (relu | sigmoid | tanh).")

        return output
