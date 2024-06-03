"""This module contains the base model."""

from datetime import datetime
from typing import List

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from TimeSeriesDL.loss import LossMeasurementSuite


class BaseModel(nn.Module):
    """BaseModel class of any neural network predicting time series.

    Args:
        nn (nn.Module): Torch nn module.
    """

    def __init__(self, writer: SummaryWriter = None) -> None:
        """Initializes the TensorBoard logger and checks for available GPU(s).

        Args:
            writer (SummaryWriter, optional): TensorBoard writer. Defaults to a generic path.
        """
        super(BaseModel, self).__init__()

        # enable tensorboard
        self._writer = writer
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

        self._precision = torch.float32

        # define object which where defined by children of this class
        self._loss_suite = LossMeasurementSuite(self._writer)
        self._scheduler: _LRScheduler = None
        self._optim: Optimizer = None

        self.test_stats = None

    @property
    def log_path(self) -> str:
        """Returns the log path of TensorBoard.

        Returns:
            str: Path as string.
        """
        return self._tb_path

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

    def save_to_default(self) -> None:
        """This method saves the current model state to the tensorboard
        directory.
        """
        params = self.state_dict()
        torch.save(params, f"{self._tb_path}/model.torch")

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

    def learn(self, train, validate=None, test=None, epochs: int = 1):
        """Trains the model on a dataset. Valdiation- and Testdatasets can be set as well.

        Args:
            train (Dataset): Dataset to train on.
            validate (Dataset, optional): Validate model on this dataset. Defaults to None.
            test (Dataset, optional): Test model on this dataset. Defaults to None.
            epochs (int, optional): Run training for given amount of epochs. Defaults to 1.
        """
        # set the model into training mode
        self.train()

        # check if the dataset is wrapped within a Dataloader
        if not isinstance(train, DataLoader):
            print("Please provide the dataset wrapped in a torch DataLoader.")
            exit(1)

        # run for n epochs specified
        pbar = tqdm(
            total=epochs * len(train) * train.batch_size,
            desc=f"Epochs {epochs}",
            leave=True,
        )
        for e in range(epochs):
            # run for each batch in training set
            for x, y_hat in train:
                x = x.to(self._device)
                y_hat = y_hat.to(self._device)

                # perform the presiction and measure the loss between the prediction
                # and the expected output
                y = self(x)

                # calculate the gradient using backpropagation of the loss
                loss, losses = self._loss_suite.calulate(y, y_hat, self.__sample_position)

                # reset the gradient and run backpropagation
                self._optim.zero_grad()
                loss.backward()
                self._optim.step()

                # log for the statistics
                self._loss_suite.add_epoch_data(losses)

                self.__sample_position += x.size(0)
                pbar.update(train.batch_size)

            # if there is an adaptive learning rate (scheduler) available
            if self._scheduler:
                self._scheduler.step()
                lr = self._scheduler.get_last_lr()[0]
                self._writer.add_scalar("Epoch/Train/learning_rate", lr, e)

            # log for the statistics
            self._loss_suite.log_epoch(e)

            # runn a validation of the current model state
            if validate:
                # set the model to eval mode, run validation and set to train mode again
                self.eval()
                _ = self.validate(validate, e)
                self.train()

            if test:
                self.eval()
                _ = self.validate(test, e, "Test")
                self.train()

        self.eval()
        self._writer.flush()

    def validate(self, loader: DataLoader, epoch: int, tag: str = "Validate") -> float:
        """This method validates/tests the model on a different dataset and logs 
        losses/accuracies to the tensorboard.

        Args:
            loader (DataLoader): The dataset to validate/test on.
            epoch (int): The epoch in which the model is validated/tested.
            tag (str, optional): The tag describing which values are logged. 
            Defaults to "Validate".

        Returns:
            float: The models accuracy on the given dataset.
        """
        accuracies = []
        self._loss_suite.set_tag(tag)

        # predict all y's of the validation set and append the model's accuracy
        # to the list
        for x, y_hat in loader:
            y = self.predict(x, as_list=False)
            y_hat = y_hat.to(self._device)

            loss, data = self._loss_suite.calulate(y, y_hat)
            self._loss_suite.add_epoch_data(data)

            accuracies.append(1 - loss.item())

        # calculate some statistics based on the data collected
        accuracy = np.mean(np.array(accuracies))
        variance = np.mean(np.var(np.array(accuracies)))

        self._loss_suite.log_epoch(epoch)

        # log to the tensorboard if wanted
        self._writer.add_scalar(f"{tag}/accuracy_mean", accuracy, epoch)
        self._writer.add_scalar(f"{tag}/accuracy_var", variance, epoch)

        return accuracy

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
