"""Module contains the LossSuite to calulate and log losses."""
from typing import Callable, List, Tuple
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch


class LossMeasurementSuite:
    """The LossMeasurementSuite is used by the BaseModel-class to calculate the prediction's
    loss of a model during training. To use this suite, add a loss function in the model's
    init and tag it as main function.
    """
    def __init__(self, writer: SummaryWriter, tag: str = "Train") -> None:
        self._writer = writer
        self._tag = tag
        self._main = None
        self._register = {}

        self._epoch_data = {}

    @property
    def tag(self) -> str:
        """The TensorBoard tag (e.g. 'Train')

        Returns:
            str: The current tag used.
        """
        return self._tag

    def set_tag(self, tag: str) -> None:
        """Sets a tag which is shown in the TensorBoard.

        Args:
            tag (str): The tag to use.
        """
        self._tag = tag

    def add_loss_fn(self, name: str, loss_fn: Callable, main: bool = False) -> None:
        """Adds a loss function to the suite. If the loss should be used, set the parameter
        'main' to `True`. All other loss functions do not use a gradient.

        Args:
            name (str): The name of the loss function.
            loss_fn (Callable): The loss function, accepting two inputs: y, y_hat.
            main (bool, optional): Main loss function to optimize the model. Defaults to False.
        """
        if name in self._register.keys():
            print(f"Loss function for name '{name}' already registered.")
            return

        self._register[name] = loss_fn

        if main:
            self._main = name

    def remove_loss_fn(self, name: str) -> None:
        """Removes a loss function from the suite.

        Args:
            name (str): The loss function's name to be removed.
        """
        if name in self._register.keys():
            del self._register[name]

    def calulate(self,
                 y: torch.tensor,
                 y_hat: torch.tensor,
                 log_pos: int = None
                 ) -> Tuple[torch.tensor, List[Tuple]]:
        """Calculates the loss for each registered loss function.

        Args:
            y (torch.tensor): Predicted value.
            y_hat (torch.tensor): Expected value.
            log_pos (int, optional): Logging position of the TensorBoard. Defaults to None.

        Returns:
            Tuple[torch.tensor, List[Tuple]]: main loss to optimize, other loss values (name, value)
        """
        if not self._main:
            print("Main loss function not defined in LossMeasurementSuite.")
            exit(1)

        result = []
        for name, fn in self._register.items():
            with torch.no_grad():
                loss = fn(y, y_hat)
                result.append((name, loss.item()))

            if log_pos:
                self._writer.add_scalar(f"{self._tag}/{name}", loss, log_pos)

        main_loss = self._register[self._main](y, y_hat)
        return main_loss, result

    def add_epoch_data(self, data: List[Tuple]) -> None:
        """Adds data to an internal bucket which is requried to log epoch data. To do
        so call 'log_epoch'.

        Args:
            data (List[Tuple]): The data gathered from 'calculate'.
        """
        for name, loss in data:
            losses = self._epoch_data.get(name, [])
            losses.append(loss)
            self._epoch_data[name] = losses

    def log_epoch(self, epoch: int) -> None:
        """Logs the internally stored epoch_data and clears the bucket.

        Args:
            epoch (int): The current epoch.
        """
        # log the gathered epoch data per loss function
        # summarize the loss data as mean over the epoch
        for name, losses in self._epoch_data.items():
            loss = np.mean(losses)
            self._writer.add_scalar(f"Epoch/{self._tag}/{name}", loss, epoch)

        # clear the recorded epoch data
        self._epoch_data = {}
