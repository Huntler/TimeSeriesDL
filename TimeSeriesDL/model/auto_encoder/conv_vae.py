"""This module contains a variational auto-encoder based on CNN."""

from datetime import datetime
from typing import Any, Tuple
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from TimeSeriesDL.model.auto_encoder.base import AutoEncoder
from TimeSeriesDL.utils.activations import get_activation_from_string
from TimeSeriesDL.utils.config import config
from TimeSeriesDL.loss import RMSELoss


class ConvVAE(AutoEncoder):
    """This model uses a variational auto-encoder (VAE) based on Convolutional
    layers to auto-encode time-series.

    Args:
        BaseModel (BaseModel): The base model class.
    """

    def __init__(
        self,
        features: int = 1,
        sequence_length: int = 1,
        extracted_features: int = 1,
        latent_size: int = 1,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        last_activation: str = "sigmoid",
        lr: float = 1e-3,
        lr_decay: float = 9e-1,
        adam_betas: Tuple[float, float] = (9e-1, 999e-3),
        tag: str = "",
        log: bool = True,
        precision: torch.dtype = torch.float32,
    ) -> None:
        # if logging enalbed, then create a tensorboard writer, otherwise prevent the
        # parent class to create a standard writer
        if log:
            now = datetime.now()
            self._tb_sub = now.strftime("%d%m%Y_%H%M%S")
            self._tb_path = f"runs/{tag}/VAE/{self._tb_sub}"
            self._writer = SummaryWriter(self._tb_path)
        else:
            self._writer = False

        super().__init__(self._writer)

        # data parameter
        self._features = features
        self._extracted_features = extracted_features
        self._sequence_length = sequence_length

        # cnn parameter
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._precision = precision

        self._latent_space = latent_size
        self._last_activation = get_activation_from_string(last_activation)

        # check if the latent space will be bigger than the output after the first conv1d layer
        ef_length = int(
            (sequence_length - kernel_size + 2 * padding) / stride) + 1
        ls_length = int((ef_length - kernel_size + 2 * padding) / stride) + 1
        if ef_length < ls_length:
            print(
                "Warning: Output after first encoder layer is smaller than latent "
                + f"space. {ef_length} < {ls_length}"
            )

        self._enc_1_len = ef_length
        self._enc_2_len = ls_length

        # setup the encoder based on CNN
        self._encoder_1 = nn.Conv1d(
            self._features,
            self._extracted_features,
            self._kernel_size,
            self._stride,
            self._padding,
            dtype=self._precision,
        )

        self._encoder_2 = nn.Conv1d(
            self._extracted_features,
            self._latent_space,
            self._kernel_size,
            self._stride,
            self._padding,
            dtype=self._precision,
        )

        # setup decoder
        self._decoder_1 = nn.ConvTranspose1d(
            self._latent_space,
            self._extracted_features,
            self._kernel_size,
            self._stride,
            self._padding,
            dtype=self._precision,
        )

        self._decoder_2 = nn.ConvTranspose1d(
            self._extracted_features,
            self._features,
            self._kernel_size,
            self._stride,
            self._padding,
            dtype=self._precision,
        )

        # setup latent space distribution
        self._mean_layer = nn.Linear(
            self._enc_2_len * self._latent_space, self._enc_2_len * self._latent_space, dtype=self._precision
        )

        self._var_layer = nn.Linear(
            self._enc_2_len * self._latent_space, self._enc_2_len * self._latent_space, dtype=self._precision
        )

        # setup loss suite
        self._loss_suite.add_loss_fn("L1", torch.nn.L1Loss())
        self._loss_suite.add_loss_fn("RMSE", RMSELoss)
        self._loss_suite.add_loss_fn("BCE", torch.nn.BCELoss(), main=True)

        # setup optimizer
        self._optim = torch.optim.AdamW(
            self.parameters(), lr=lr, betas=adam_betas)
        self._scheduler = ExponentialLR(self._optim, gamma=lr_decay)

    @property
    def latent_length(self) -> int:
        return self._enc_2_len

    def reparameterization(self, mean: torch.tensor, var: torch.tensor) -> torch.tensor:
        """Samples from the latent representation, which is a random distribution in this case.

        Args:
            mean (torch.tensor): The mean of the encoded date.
            var (torch.tensor): The log_variance of the encoded data.

        Returns:
            torch.tensor: The sampled encoded data.
        """
        epsilon = torch.randn_like(var).to(self._device)
        z = mean + var * epsilon
        return z

    def encode(self, x: torch.tensor, variance: bool = False) -> Any:
        # change input to batch, features, samples
        x: torch.tensor = torch.swapaxes(x, 2, 1)

        x = self._encoder_1.forward(x)
        x = torch.relu(x)

        x = self._encoder_2.forward(x)
        x = torch.relu(x)

        batch, features, samples = x.shape
        x = x.view(batch, features * samples)
        mean = self._mean_layer(x).view(batch, features, samples)

        if variance:
            log_var = self._var_layer(x).view(batch, features, samples)
            return mean, log_var

        return mean

    def decode(self, x: torch.tensor) -> torch.tensor:
        batch, _, _ = x.shape

        x = self._decoder_1.forward(
            x, [batch, self._extracted_features, self._enc_1_len]
        )
        x = torch.relu(x)

        x = self._decoder_2.forward(
            x, [batch, self._features, self._sequence_length])

        # change output to batch, samples, features
        x: torch.tensor = torch.swapaxes(x, 2, 1)
        return self._last_activation(x)

    def freeze(self, unfreeze: bool = False) -> None:
        """Freezes or unfreezes the parameter of this AE to enable or disable parameter tuning.

        Args:
            unfreeze (bool, optional): Unfreezes the parameter if set to True. Defaults to False.
        """
        layers = [
            self._encoder_1.parameters(),
            self._encoder_2.parameters(),
            self._mean_layer.parameters(),
            self._var_layer.parameters(),
            self._decoder_1.parameters(),
            self._decoder_2.parameters(),
        ]

        for layer in layers:
            for param in layer:
                param.requires_grad = not unfreeze

    def forward(self, x: torch.tensor):
        # encode the data to mean/log_var of latent space
        mean, log_var = self.encode(x, variance=True)

        # takes exponential function (log var -> var)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))

        # decode data back
        return self.decode(z)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path))
        self.eval()


config.register_model("ConvVAE", ConvVAE)
