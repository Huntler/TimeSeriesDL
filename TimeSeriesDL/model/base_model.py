"""This module contains the base model."""
from datetime import datetime
from typing import List

import numpy as np
import torch
from torch import nn
from torch.nn import Module
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def rmse_loss_fn(yhat, y):
    """Calculates the RMSE loss sqrt(mean(y_hat - y)**2)

    Args:
        yhat (torch.tensor): Predicted value.
        y (torch.tensor): Actual value.

    Returns:
        torch.tensor: Loss.
    """
    return torch.sqrt(torch.mean((yhat-y)**2))


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
        self._scheduler: _LRScheduler = None
        self._optim: Optimizer = None
        self._loss_fn: Module = torch.nn.MSELoss()

        self._l1_loss = nn.L1Loss()
        self._rmse_loss = rmse_loss_fn
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
                    print("MPS not available because the current PyTorch install was not "
                        "built with MPS enabled.")
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
        model_tag = datetime.now().strftime("%H%M%S")
        params = self.state_dict()
        torch.save(params, f"{self._tb_path}/model_{model_tag}.torch")

    def load(self, path: str) -> None:
        """Loads a model with its parameters into this object.

        Args:
            path (str): The model's path.

        Raises:
            NotImplementedError: The Base model has not implemented this.
        """
        raise NotImplementedError()

    def forward(self, x, future_steps: int = 1):
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

    def learn(self, train, validate=None, test=None, epochs: int = 1, verbose: bool = False):
        """Trains the model on a dataset. Valdiation- and Testdatasets can be set as well.

        Args:
            train (Dataset): Dataset to train on.
            validate (Dataset, optional): Validate model on this dataset. Defaults to None.
            test (Dataset, optional): Test model on this dataset. Defaults to None.
            epochs (int, optional): Run training for given amount of epochs. Defaults to 1.
            verbose (bool, optional): Show progress on CLI. Defaults to False.
        """
        # set the model into training mode
        self.train()

        # check if the dataset is wrapped within a Dataloader
        if not isinstance(train, DataLoader):
            print("Please provide the dataset wrapped in a torch DataLoader")
            exit(1)

        # run for n epochs specified
        for e in tqdm(range(epochs)):
            train_iterator = tqdm(train) if verbose else train
            mse_ep_losses = []
            rmse_ep_losses = []
            mae_ep_losses = []

            # run for each batch in training set
            for x, y in train_iterator:
                mse_losses = []
                rmse_losses = []
                mae_losses = []

                x = x.to(self._device)
                y = y.to(self._device)

                # perform the presiction and measure the loss between the prediction
                # and the expected output
                pred_y = self(x)

                # calculate the gradient using backpropagation of the loss
                loss = self._loss_fn(pred_y, y)

                # calculate rmse and mae losses as well
                rmse_loss = self._rmse_loss(pred_y, y)
                mae_loss = self._l1_loss(y, pred_y)

                # reset the gradient and run backpropagation
                self._optim.zero_grad()
                loss.backward()
                self._optim.step()
                # print(y.detach().numpy(), pred_y.ravel().detach().numpy(), loss.item())

                mse_losses.append(loss.item())
                rmse_losses.append(rmse_loss.item())
                mae_losses.append(mae_loss.item())

                # log for the statistics
                mse_losses = np.mean(mse_losses, axis=0)
                mse_ep_losses.append(mse_losses)
                self._writer.add_scalar(
                    "Train/loss", loss, self.__sample_position)

                rmse_losses = np.mean(rmse_losses, axis=0)
                rmse_ep_losses.append(rmse_losses)
                self._writer.add_scalar(
                    "Train/rmse_loss", rmse_loss, self.__sample_position)

                mae_losses = np.mean(mae_losses, axis=0)
                mae_ep_losses.append(mae_losses)
                self._writer.add_scalar(
                    "Train/mae_loss", mae_loss, self.__sample_position)
                self.__sample_position += x.size(0)

            # if there is an adaptive learning rate (scheduler) available
            if self._scheduler:
                self._scheduler.step()
                lr = self._scheduler.get_last_lr()[0]
                self._writer.add_scalar("Train/learning_rate", lr, e)

            # log for the statistics
            mse_ep_losses = np.mean(mse_ep_losses)
            self._writer.add_scalar("Train/epoch_loss", mse_ep_losses, e)

            rmse_ep_losses = np.mean(rmse_ep_losses)
            self._writer.add_scalar("Train/rmse_epoch_loss", rmse_ep_losses, e)

            mae_ep_losses = np.mean(mae_ep_losses)
            self._writer.add_scalar("Train/mae_epoch_loss", mae_ep_losses, e)

            # runn a validation of the current model state
            if validate:
                # set the model to eval mode, run validation and set to train mode again
                self.eval()
                _ = self.validate(validate, e)
                self.train()

            if test:
                self.eval()
                _ = self.test(test, e)
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
        mse_losses = []
        rmse_losses = []
        mae_losses = []

        # predict all y's of the validation set and append the model's accuracy
        # to the list
        for x, y in dataloader:
            _y = self.predict(x, as_list=False)

            y = y.to(self._device)
            loss = self._loss_fn(_y, y)
            rmse_loss = self._rmse_loss(_y, y)
            mae_loss = self._l1_loss(_y, y)

            mse_losses.append(loss.item())
            rmse_losses.append(rmse_loss.item())
            mae_losses.append(mae_loss.item())

            accuracies.append(1 - loss.item())

        # calculate some statistics based on the data collected
        accuracy = np.mean(np.array(accuracies))
        variance = np.mean(np.var(np.array(accuracies)))

        mse_loss = np.mean(np.array(mse_losses))
        rmse_loss = np.mean(np.array(rmse_losses))
        mae_loss = np.mean(np.array(mae_losses))

        # log to the tensorboard if wanted
        if log_step != -1:
            self._writer.add_scalar("Val/accuracy_mean", accuracy, log_step)
            self._writer.add_scalar("Val/accuracy_var", variance, log_step)

            self._writer.add_scalar("Val/mse_loss", mse_loss, log_step)
            self._writer.add_scalar("Val/rmse_loss", rmse_loss, log_step)
            self._writer.add_scalar("Val/mae_loss", mae_loss, log_step)

        return accuracy

    def test(self, dataloader, log_step: int = -1) -> float:
        """Method validates model's accuracy based on the given data. In validation, the model
        only looks one step ahead.

        Args:
            dataloader (_type_): The dataloader which contains value, not used for training.
            log_step (int, optional): The step of the logger, can be disabled by setting to -1.

        Returns:
            float: The model's accuracy.
        """
        accuracies = []
        mse_losses = []
        rmse_losses = []
        mae_losses = []

        # predict all y's of the validation set and append the model's accuracy
        # to the list
        for x, y in dataloader:
            _y = self.predict(x, as_list=False)

            y = y.to(self._device)
            loss = self._loss_fn(_y, y)
            rmse_loss = self._rmse_loss(_y, y)
            mae_loss = self._l1_loss(_y, y)

            mse_losses.append(loss.item())
            rmse_losses.append(rmse_loss.item())
            mae_losses.append(mae_loss.item())

            accuracies.append(1 - loss.item())

        # calculate some statistics based on the data collected
        accuracy = np.mean(np.array(accuracies))
        variance = np.mean(np.var(np.array(accuracies)))

        mse_loss = np.mean(np.array(mse_losses))
        rmse_loss = np.mean(np.array(rmse_losses))
        mae_loss = np.mean(np.array(mae_losses))

        self.test_stats = (
            accuracy, variance, mse_loss, rmse_loss, mae_loss
        )

        # log to the tensorboard if wanted
        if log_step != -1:
            self._writer.add_scalar("Test/accuracy_mean", accuracy, log_step)
            self._writer.add_scalar("Test/accuracy_var", variance, log_step)

            self._writer.add_scalar("Test/mse_loss", mse_loss, log_step)
            self._writer.add_scalar("Test/rmse_loss", rmse_loss, log_step)
            self._writer.add_scalar("Test/mae_loss", mae_loss, log_step)

        return accuracy

    def predict(self, x: torch.tensor, as_list: bool = True) -> List:
        """This method only predicts future steps based on the given curve described by 
        the datapoints X.

        Args:
            x (torch.tensor): The datapoints.
            future_steps (int, optional): The amount of steps to look into future. Defaults to 1.

        Returns:
            List: The prediction.
        """
        x = x.to(self._device)
        with torch.no_grad():
            out = self(x)
            if as_list:
                out = list(out.ravel().cpu().numpy())

        return out
