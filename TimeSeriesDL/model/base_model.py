"""This module contains the base model."""

from typing import List
import os

import torch
from torch import nn


class BaseModel(nn.Module):
    """BaseModel class of any neural network predicting time series.

    Args:
        nn (nn.Module): Torch nn module.
    """

    def __init__(self, name: str, tag: str = "") -> None:
        """Initializes the TensorBoard logger and checks for available GPU(s).

        Args:
            writer (SummaryWriter, optional): TensorBoard writer. Defaults to a generic path.
        """
        super().__init__()

        # enable tensorboard
        self._init_log_path(name, tag)

        # check for gpu
        self._device = "cpu"
        if torch.cuda.is_available():
            self._device_name = torch.cuda.get_device_name(0)
            print(f"GPU acceleration available on {self._device_name}")

        self._precision = torch.float32

    def _init_log_path(self, name: str, tag: str) -> None:
        """Initializes the Log path writer.

        Args:
            name (str): The name of the model folder.
            tag (str): The tag name of a primary folder.
        """
        path = f"runs/{tag}/{name}"

        # check if log path exists, if so add "_<increment>" to the path string
        for i in range(100):
            if os.path.exists(path):
                path += f"_{i}"
            else:
                break

        self._tb_path = path + "/"

    @property
    def log_path(self) -> str:
        """Returns the log path of TensorBoard.

        Returns:
            str: Path as string.
        """
        return self._tb_path

    @property
    def device(self) -> str:
        """Returns the device, the model uses.

        Returns:
        str: The device as string.
        """
        return self._device

    def use_device(self, device: str) -> None:
        """Sets the current device to run the model on. e.g. 'cpu', 'cuda', 'mps'.

        Args:
            device (str): The device to use.
        """
        self._device = "cpu"
        if device == "cuda":
            if not torch.cuda.is_available():
                print("No CUDA support on your system. Fallback to CPU.")
            else:
                self._device = "cuda"

        if device == "mps":
            if torch.backends.mps.is_available():
                if not torch.backends.mps.is_built():
                    print(
                        "MPS not available because the current PyTorch install was not "
                        "built with MPS enabled."
                    )
                else:
                    self._device = "mps"
            else:
                print("MPS not available.")

        print(f"Using {self._device} backend.")
        self.to(self._device)

    def save_to_default(self, post_fix: str | int = None) -> None:
        """This method saves the current model state to the tensorboard
        directory.

        Args:
            post_fix (str | int): Adds a postfix to the model path. Defaults to None.‚
        """
        params = self.state_dict()
        post_fix = f"_{post_fix}" if post_fix is not None else ""
        if not os.path.exists(f"{self._tb_path}/models/"):
            os.makedirs(f"{self._tb_path}/models/")

        torch.save(params, f"{self._tb_path}/models/model{post_fix}.torch")

    def load(self, path: str) -> None:
        """Loads a model with its parameters into this object.

        Args:
            path (str): The model's path.

        Raises:
            NotImplementedError: The Base model has not implemented this.
        """
        raise NotImplementedError()

    def forward(self, x):
        """
        This method performs the forward call on the neural network
        architecture.

        Args:
            x (Any): The input passed to the defined neural network. The shape is
            (batch_size, sequence_length, values)
            future_steps (int, optional): The amount of steps predicted.

        Raises:
            NotImplementedError: The Base model has not implementation
                                 for this.
        """
        raise NotImplementedError

    def predict(self, x: torch.tensor, as_array: bool = False) -> List:
        """This method only predicts future steps based on the given curve described by
        the datapoints X.

        Args:
            x (torch.tensor): The datapoints.
            as_array (bool, optional): Returns the prediction as np.array. Defaults to False.

        Returns:
            List: The prediction.
        """
        x = x.to(self._device)
        with torch.no_grad():
            out = self(x)
            if as_array:
                out = out.cpu().numpy()

        return out
