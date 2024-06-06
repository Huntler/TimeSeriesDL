"""Module contains the LossSuite to calulate and log losses."""
from typing import Callable, List, Tuple
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import _LRScheduler
from torch import optim
from torch.optim.optimizer import ParamsT
from torch.optim.lr_scheduler import ExponentialLR
import torch

from TimeSeriesDL.loss import get_loss_by_name
from TimeSeriesDL.loss.optimizer import get_optimizer_by_name


class LossMeasurementSuite:
    """The LossMeasurementSuite is used by the BaseModel-class to calculate the prediction's
    loss of a model during training. To use this suite, add a loss function in the model's
    init and tag it as main function.
    """
    def __init__(self, optimizer: str, main: str = None, log: List[str] = [],
                 lr: float = 1e-3, lr_decay: float = None) -> None:
        self._writer = None
        self._tag = "Train"
        self._main = None

        self._lr = lr
        self._lr_decay = lr_decay
        self._optim = get_optimizer_by_name(optimizer)
        self._scheduler: _LRScheduler = None

        self._register = {}
        self._epoch_data = {}

        # initialize provided parameters
        if main:
            self.add_loss_fn(main, get_loss_by_name(main), True)

        for name in log:
            self.add_loss_fn(name, get_loss_by_name(name))

    def init_optimizer(self, parameters: ParamsT) -> None:
        """Initialize the optimizer and scheduler.

        Args:
            parameters (ParamsT): The model's parameter to optimize.
        """
        self._optim = self._optim(parameters, lr=self._lr)

        # the scheduler only works with the Adam optimizer
        if self._lr_decay is not None and isinstance(self._optim, optim.Adam):
            self._scheduler = ExponentialLR(self._optim, gamma=self._lr_decay)

    def set_writer(self, writer: SummaryWriter) -> None:
        """Sets the writer used by this suite.

        Args:
            writer (SummaryWriter): The writer used by this suite.
        """
        self._writer = writer

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

    def __call__(self,
                 y: torch.tensor,
                 y_hat: torch.tensor,
                 log_pos: int = None
                 ) -> List[Tuple]:
        """Calculates the loss for each registered loss function. Then optimizes the model using
        the main loss function.

        Args:
            y (torch.tensor): Predicted value.
            y_hat (torch.tensor): Expected value.
            log_pos (int, optional): Logging position of the TensorBoard. Defaults to None.

        Returns:
            List[Tuple]: Loss values (name, value)
        """
        assert self._main, "Main loss function not defined in LossMeasurementSuite."
        assert self._writer, "TensorBoard writer not defined."

        result = self.calculate(y, y_hat, log_pos)

        # calculuate the loss which is to be optimized
        main_loss = self._register[self._main](y, y_hat)

        # run backpropagation on the main_loss
        self._optim.zero_grad()
        main_loss.backward()
        self._optim.step()

        return result

    def calculate(self,
                 y: torch.tensor,
                 y_hat: torch.tensor,
                 log_pos: int = None) -> List[Tuple]:
        """Calculates the loss for each registered loss function without optimizing the model.

        Args:
            y (torch.tensor): Predicted value.
            y_hat (torch.tensor): Expected value.
            log_pos (int, optional): Logging position of the TensorBoard. Defaults to None.
        
        Returns:
            List[Tuple]: Loss values (name, value).
        """
        assert self._writer, "TensorBoard writer not defined."

        result = []
        for name, fn in self._register.items():
            with torch.no_grad():
                loss = fn(y, y_hat)
                result.append((name, loss.item()))

            if log_pos:
                self._writer.add_scalar(f"{self._tag}/{name}", loss, log_pos)

        return result

    def scheduler_step(self, epoch: int) -> None:
        """Performs a step of the scheduler and logs to the TensorBoard.

        Args:
            epoch (int): The current epoch.
        """
        if self._scheduler:
            self._scheduler.step()
            lr = self._scheduler.get_last_lr()[0]
            self._writer.add_scalar(f"Epoch/{self._tag}/learning_rate", lr, epoch)

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
        assert self._writer, "TensorBoard writer not defined."

        # log the gathered epoch data per loss function
        # summarize the loss data as mean over the epoch
        for name, losses in self._epoch_data.items():
            loss = np.mean(losses)
            self._writer.add_scalar(f"Epoch/{self._tag}/{name}", loss, epoch)

        # clear the recorded epoch data
        self._epoch_data = {}
