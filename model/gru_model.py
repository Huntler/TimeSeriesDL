from argparse import ArgumentError
from datetime import datetime
from typing import List
import torch
from torch.utils.tensorboard import SummaryWriter
from .base_model import BaseModel
from torch.optim.lr_scheduler import ExponentialLR


class GruModel(BaseModel):
    def __init__(self, input_size: int, out_act: str = "relu", xavier_init: bool = False, n_layers: int = 2,
                 hidden_dim: int = 64, dropout: float = 0.1, log: bool = True,
                 lr: float = 1e-3, lr_decay: float = 9e-1, betas: List[float] = [9e-1, 999e-3],
                 precision: torch.dtype = torch.float16) -> None:
        # if logging enabled, then create a tensorboard writer, otherwise prevent the
        # parent class to create a standard writer
        if log:
            now = datetime.now()
            self.__tb_sub = now.strftime("%d%m%Y_%H%M%S")
            self._tb_path = f"runs/GRU_Model/{self.__tb_sub}"
            self._writer = SummaryWriter(self._tb_path)
        else:
            self._writer = False

        super(GruModel, self).__init__()

        # set up hyperparameters
        self.__input_size = input_size
        self.__output_activation = out_act
        self.__xavier = xavier_init
        self.__n_layers = n_layers
        self.__hidden_dim = hidden_dim
        self.__dropout = dropout
        self.__precision = precision

        # set up the model's layers
        self.__gru = torch.nn.GRU(
            self.__input_size,
            self.__hidden_dim,
            self.__n_layers,
            batch_first=True,
            dropout=self.__dropout)

        self.__h = None

        # create the dense layers and initilize them based on our hyperparameters
        self.__linear_1 = torch.nn.Linear(self.__hidden_dim, 32)
        self.__linear_2 = torch.nn.Linear(32, 1)

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

        # output from the last layer
        match self.__output_activation:
            case "relu":
                self.__activation = lambda x: torch.relu(x)
            case "sigmoid":
                self.__activation = lambda x: torch.sigmoid(x)
            case "tanh":
                self.__activation = lambda x: torch.tanh(x)
            case _:
                raise ArgumentError(
                    "Wrong output actiavtion specified (relu | sigmoid | tanh).")

        # define loss function, optimizer and scheduler for the learning rate
        self._loss_fn = torch.nn.MSELoss()
        self._optim = torch.optim.AdamW(self.parameters(), lr=lr, betas=betas)
        self._scheduler = ExponentialLR(self._optim, gamma=lr_decay)

    def __init_hidden_states(self, batch_size: int) -> torch.tensor:
        """Initializes the hidden cell state of our GRU layer. Either zero or 
        Xavier normal distribution.

        Args:
            batch_size (int): The batch size of the input.

        Returns:
            torch.tensor: The hidden cell state.
        """
        hidden = torch.empty(self.__n_layers, batch_size,
                             self.__hidden_dim, device=self._device)
        if self.__xavier:
            hidden = torch.nn.init.xavier_normal_(hidden)
        else:
            hidden = torch.nn.init.zeros_(hidden)
        return hidden

    def load(self, path) -> None:
        """Loads the model's parameter given a path
        """
        self.load_state_dict(torch.load(path))
        self.eval()

    def forward(self, X):
        batch_size, n_samples, dim = X.shape
        output = torch.empty(batch_size, 1, 1, dtype=torch.float32)

        # initilize the hidden cell state based on the batch's input size
        if self.__h is None:
            self.__h = self.__init_hidden_states(batch_size)

        # run for all samples t of the provided sequence window
        for t in range(0, n_samples):
            # pass batch of sample at time step t into GRU
            input = X[:, t].unsqueeze(1)
            x, self.__h = self.__gru(input, self.__h.data)

            # afterwards feed through a dense network
            x = self.__activation(x)
            x = self.__linear_1(x)
            x = self.__activation(x)
            x = self.__linear_2(x)

            # and output the data using one activation function
            match self.__output_activation:
                case "relu":
                    x = torch.relu(x)
                case "sigmoid":
                    x = torch.sigmoid(x)
                case _:
                    raise ArgumentError(
                        "Wrong output actiavtion specified (relu | sigmoid).")
                # FIXME: maybe tanh?

        output[:, 0, :] = x[:, 0, :]
        return output
