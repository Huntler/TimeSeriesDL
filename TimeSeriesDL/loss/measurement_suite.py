from typing import Callable, List, Tuple
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch


class LossMeasurementSuite:
    def __init__(self, writer: SummaryWriter, tag: str = "Train") -> None:
        self._writer = writer
        self._tag = tag
        self._main = None
        self._register = {}

        self._epoch_data = {}

    @property
    def tag(self) -> str:
        return self._tag

    def set_tag(self, tag: str) -> None:
        self._tag = tag

    def add_loss_fn(self, name: str, loss_fn: Callable, main: bool = False) -> None:
        if name in self._register.keys():
            print(f"Loss function for name '{name}' already registered.")
            return

        self._register[name] = loss_fn

        if main:
            self._main = name

    def remove_loss_fn(self, name: str) -> None:
        if name in self._register.keys():
            del self._register[name]

    def calulate(self,
                 y: torch.tensor,
                 y_hat: torch.tensor,
                 log_pos: int = None
                 ) -> Tuple[torch.tensor, List[Tuple]]:
        main_loss = None
        if not self._main:
            print("Main loss function not defined in LossMeasurementSuite.")
            exit(1)

        result = []
        for name, fn in self._register.items():
            loss = fn(y, y_hat)
            result.append((name, loss.item()))

            if name == self._main:
                main_loss = loss

            if log_pos:
                self._writer.add_scalar(f"{self._tag}/{name}", loss, log_pos)

        return main_loss, result

    def add_epoch_data(self, data: List[Tuple]) -> None:
        for name, loss in data:
            losses = self._epoch_data.get(name, [])
            losses.append(loss)
            self._epoch_data[name] = losses

    def log_epoch(self, epoch: int) -> None:
        # log the gathered epoch data per loss function
        # summarize the loss data as mean over the epoch
        for name, losses in self._epoch_data.items():
            loss = np.mean(losses)
            self._writer.add_scalar(f"Epoch/{self._tag}/{name}", loss, epoch)

        # clear the recorded epoch data
        self._epoch_data = {}
