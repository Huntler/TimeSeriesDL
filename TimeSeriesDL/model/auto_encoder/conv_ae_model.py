"""This module contains a basic auto-encoder based on CNN."""
from typing import Tuple
import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from TimeSeriesDL.model.auto_encoder import AutoEncoder
from TimeSeriesDL.utils.activations import get_activation_from_string
from TimeSeriesDL.utils.config import config


class ConvAE(AutoEncoder):
    """This model uses CNN to auto-encode time-series.

    Args:
        BaseModel (BaseModel): The base model class.
    """

    def __init__(
        self,
        features: int = 1,
        sequence_length: int = 1,
        channels: int = 1,
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
        save_every: int = 0,
        precision: torch.dtype = torch.float32,
    ) -> None:
        super().__init__("ConvAE", save_every, tag, log)

        # data parameter
        self._features = features
        self._extracted_features = extracted_features
        self._sequence_length = sequence_length

        # cnn parameter
        self._channels = channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._precision = precision

        self._latent_space = latent_size
        self._last_activation = get_activation_from_string(last_activation)

        # check if the latent space will be bigger than the output after the first conv1d layer
        ef_length = int(
            (sequence_length - kernel_size + 2 * padding) / stride) + 1
        ls_length = int((ef_length - kernel_size//2 + 2 * padding) / stride) + 1
        if ef_length < ls_length:
            print(
                "Warning: Output after first encoder layer is smaller than latent "
                + f"space. {ef_length} < {ls_length}"
            )

        self._enc_1_len = ef_length
        self._enc_2_len = ls_length

        # setup the encoder based on CNN
        self._encoder_1 = nn.Conv2d(
            self._channels,
            self._extracted_features,
            (self._features, self._kernel_size),
            self._stride,
            self._padding,
            dtype=self._precision,
        )

        self._encoder_2 = nn.Conv2d(
            self._extracted_features,
            self._latent_space,
            (self._channels, self._kernel_size//2),
            self._stride,
            self._padding,
            dtype=self._precision,
        )

        # setup decoder
        self._decoder_1 = nn.ConvTranspose2d(
            self._latent_space,
            self._extracted_features,
            (self._channels, self._kernel_size//2),
            self._stride,
            self._padding,
            dtype=self._precision,
        )

        self._decoder_2 = nn.ConvTranspose2d(
            self._extracted_features,
            self._channels,
            (self._features, self._kernel_size),
            self._stride,
            self._padding,
            dtype=self._precision,
        )

        self._loss_suite.add_loss_fn("BCE", torch.nn.BCELoss())
        self._loss_suite.add_loss_fn("MSE", torch.nn.MSELoss(), main=True)
        self._loss_suite.add_loss_fn("L1", torch.nn.L1Loss())

        self._optim = torch.optim.AdamW(self.parameters(), lr=lr, betas=adam_betas)
        self._scheduler = ExponentialLR(self._optim, gamma=lr_decay)

    @property
    def latent_length(self) -> int:
        return self._enc_2_len

    @property
    def precision(self) -> torch.dtype:
        return self._precision

    def encode(self, x: torch.tensor, as_array: bool = False) -> torch.tensor:
        # change input to batch, channels, features, samples
        x: torch.tensor = torch.swapaxes(x, 1, 3) # batch, feature, channel, sample
        x: torch.tensor = torch.swapaxes(x, 2, 1) # batch, channel, feature sample

        x = self._encoder_1.forward(x)
        x = torch.relu(x)

        x = self._encoder_2.forward(x)
        x = torch.relu(x)

        if as_array:
            return x.cpu().detach().numpy()
        return x

    def decode(self, x: torch.tensor, as_array: bool = False) -> torch.tensor:
        batch, _, _, _ = x.shape

        output_size = [batch, self._extracted_features, self._channels, self._enc_1_len]
        x = self._decoder_1.forward(x, output_size)
        x = torch.relu(x)

        output_size = [batch, self._channels, self._features, self._sequence_length]
        x = self._decoder_2.forward(x, output_size)
        x = self._last_activation(x)

        # change output to batch, samples, channel, features
        x: torch.tensor = torch.swapaxes(x, 2, 1) # batch, feature, channel, sample
        x: torch.tensor = torch.swapaxes(x, 1, 3) # batch, sample, channel, feature

        if as_array:
            return x.cpu().detach().numpy()
        return x

    def freeze(self, unfreeze: bool = False) -> None:
        """Freezes or unfreezes the parameter of this AE to enable or disable parameter tuning.

        Args:
            unfreeze (bool, optional): Unfreezes the parameter if set to True. Defaults to False.
        """
        layers = [
            self._encoder_1.parameters(),
            self._encoder_2.parameters(),
            self._decoder_1.parameters(),
            self._decoder_2.parameters()]

        for layer in layers:
            for param in layer:
                param.requires_grad = not unfreeze

    def forward(self, x: torch.tensor):
        x = self.encode(x)
        return self.decode(x)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path))
        self.eval()


config.register_model("ConvAE", ConvAE)
