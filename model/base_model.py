from datetime import datetime
from typing import List

import numpy as np
import torch
from torch import nn
from torch.nn import Module
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import contextlib

@contextlib.contextmanager
def no_autocast():
    yield None


class BaseModel(nn.Module):
    def __init__(self) -> None:
        super(BaseModel, self).__init__()

        # enable tensorboard
        if self._writer is None:
            self.__tb_sub = datetime.now().strftime("%m-%d-%Y_%H%M%S")
            self._tb_path = f"runs/{self.__tb_sub}"
            self._writer = SummaryWriter(self._tb_path)
        self.__sample_position = 0

        # check for gpu
        self._device = "cpu"
        if torch.cuda.is_available():
            self._device_name = torch.cuda.get_device_name(0)
            print(f"GPU acceleration available on {self._device_name}")

        # define object which where defined by children of this class
        self._scheduler: _LRScheduler = None
        self._optim: Optimizer = None
        self._loss_fn: Module = None

    @property
    def log_path(self) -> str:
        return self._tb_path

    def use_device(self, device: str) -> None:
        self._device = device
        self.to(self._device)

    def save_to_default(self) -> None:
        """This method saves the current model state to the tensorboard 
        directory.
        """
        model_tag = datetime.now().strftime("%H%M%S")
        params = self.state_dict()
        torch.save(params, f"{self._tb_path}/model_{model_tag}.torch")

    def load(self, path) -> None:
        raise NotImplementedError()

    def forward(self, X, future_steps: int = 1):
        """
        This method performs the forward call on the neural network 
        architecture.

        Args:
            X (Any): The input passed to the defined neural network.
            future_steps (int, optional): The amount of steps predicted.

        Raises:
            NotImplementedError: The Base model has not implementation 
                                 for this.
        """
        raise NotImplementedError

    def learn(self, train, validate=None, epochs: int = 1):
        # set the model into training mode
        self.train()

        # run for n epochs specified
        for e in tqdm(range(epochs)):
            ep_losses = []

            # run for each batch in training set
            for X, y in train:
                losses = []

                # run for a given amount of epochs
                with torch.cuda.amp.autocast() if self._device == "cuda" else no_autocast():
                    # perform the presiction and measure the loss between the prediction
                    # and the expected output
                    pred_y = self(X)

                    # calculate the gradient using backpropagation of the loss
                    loss = self._loss_fn(pred_y, y)

                    # reset the gradient and run backpropagation
                    self._optim.zero_grad()
                    loss.backward()
                    self._optim.step()
                    # print(y.detach().numpy(), pred_y.ravel().detach().numpy(), loss.item())

                    losses.append(loss.item())

                # log for the statistics
                losses = np.mean(losses, axis=0)
                ep_losses.append(losses)
                self._writer.add_scalar(
                    "Train/loss", loss, self.__sample_position)
                self.__sample_position += X.size(0)

            # if there is an adaptive learning rate (scheduler) available
            if self._scheduler:
                self._scheduler.step()
                lr = self._scheduler.get_last_lr()[0]
                self._writer.add_scalar("Train/learning_rate", lr, e)

            # log for the statistics
            ep_losses = np.mean(ep_losses)
            self._writer.add_scalar("Train/epoch_loss", ep_losses, e)

            # runn a validation of the current model state
            if validate:
                # set the model to eval mode, run validation and set to train mode again
                self.eval()
                accuracy = self.validate(validate, e)
                self.train()

        self.eval()
        self._writer.flush()

    def validate(self, dataloader, log_step: int = -1) -> float:
        """Method validates model's accuracy based on the given data. In validation, the model
        only looks one step ahead.

        Args:
            dataloader (_type_): The dataloader which contains value, not used for training.
            log_step (int, optional): The step of the logger, can be disabled by setting to -1.

        Returns:
            float: The model's accuracy.
        """
        accuracies = []
        # predict all y's of the validation set and append the model's accuracy 
        # to the list
        for X, y in dataloader:
            _y = self.predict(X, as_list=False)
            loss = self._loss_fn(_y, y)
            accuracies.append(1 - loss.item())

        # calculate some statistics based on the data collected
        accuracy = np.mean(np.array(accuracies))
        variance = np.mean(np.var(np.array(accuracies)))

        # log to the tensorboard if wanted
        if log_step != -1:
            self._writer.add_scalar("Val/accuracy_mean", accuracy, log_step)
            self._writer.add_scalar("Val/accuracy_var", variance, log_step)

        return accuracy

    def predict(self, X, future_steps: int = 1, as_list: bool = True) -> List:
        """This method only predicts future steps based on the given curve described by the datapoints X.

        Args:
            X (_type_): The datapoints.
            future_steps (int, optional): The amount of steps to look into future. Defaults to 1.

        Returns:
            List: The prediction.
        """
        with torch.no_grad():
            out = self(X, future_steps=future_steps)
            if as_list:
                out = list(out.ravel().numpy())

        return out
