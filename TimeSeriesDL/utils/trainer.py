"""This module contains the trainer class used to train and validate a model"""
from typing import Dict
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from TimeSeriesDL.loss.measurement_suite import LossMeasurementSuite
from TimeSeriesDL.model.base_model import BaseModel


class ModelTrainer:
    """This class is used to train and validate a model"""
    def __init__(self, epochs: int = 1, save_every: int = 0) -> None:
        self._epochs = epochs
        self._save_every = save_every
        self._writer = None

        self._trainloader = None
        self._testloader = None
        self._loss_suite = None

    def set_dataset(self, dataloader: DataLoader) -> None:
        """Sets a train dataset.

        Args:
            dataloader (DataLoader): The dataloader containing the dataset.
        """
        self._trainloader = dataloader

    def set_testset(self, dataloader: DataLoader) -> None:
        """Sets a test dataset.

        Args:
            dataloader (DataLoader): The dataloader containing the dataset.
        """
        self._testloader = dataloader

    def set_loss_suite(self, loss_suite: LossMeasurementSuite) -> None:
        """Sets a loss suite, which handles TensorBoard logging and parameter
        optimization.

        Args:
            loss_suite (LossMeasurementSuite): The configured loss suite.
        """
        self._loss_suite = loss_suite

    def train(self, model: BaseModel):
        """Trains the given model on the train dataset in this object. Make sure
        to have it set by calling 'set_dataset()' before. Also, a loss suite is 
        required which handles parameter optimization and logging. Set it by calling 
        'set_loss_suite()'.

        Args:
            model (BaseModel): The model to optimize.
        """
        assert self._trainloader, "No training data provided."
        assert self._loss_suite, "No loss suite provided."
        writer = SummaryWriter(model.log_path)

        # prepare the training
        self._loss_suite.init_optimizer(model.parameters())
        self._loss_suite.set_tag("Train")
        self._loss_suite.set_writer(writer)
        model.train()

        # run for n epochs specified
        pos = 0
        pbar = tqdm(
            total=self._epochs * len(self._trainloader) * self._trainloader.batch_size,
            desc=f"Epochs {self._epochs}",
            leave=True,
        )

        for e in range(self._epochs):
            # run for each batch in training set
            for x, y_hat in self._trainloader:
                # move the batch to the correct device
                x = x.to(model.device)
                y_hat = y_hat.to(model.device)

                # do a prediction on the model
                y = model(x)

                # compute the loss
                losses = self._loss_suite(y, y_hat, pos)

                # log for the statistics
                self._loss_suite.add_epoch_data(losses)

                pos += x.size(0)
                pbar.update(self._trainloader.batch_size)

            # if there is an adaptive learning rate (scheduler) available
            self._loss_suite.scheduler_step(e)

            # log for the statistics
            self._loss_suite.log_epoch(e)

            # save the model after every n epochs
            if self._save_every > 0 and (e + 1) % self._save_every == 0:
                model.save_to_default(e + 1)

        model.eval()

    def test(self, model: BaseModel) -> Dict:
        """This method tests the model on test dataset and logs
        losses/accuracies to the tensorboard.

        Args:
            model (BaseModel): The model that will be tested

        Returns:
            Dict: The models accuracy on the given dataset.
        """
        assert self._testloader, "No test data provided."
        assert self._loss_suite, "No loss suite provided."

        accuracies = {}
        self._loss_suite.set_tag("Test")

        # predict all y's of the validation set and append the model's accuracy
        # to the list
        pos = 0
        for x, y_hat in tqdm(self._testloader):
            x = x.to(model.device)
            y_hat = y_hat.to(model.device)

            y = model.predict(x, as_array=False)

            losses = self._loss_suite.calculate(y, y_hat, pos)
            for name, loss in losses:
                loss_list = accuracies.get(name, [])
                loss_list.append(1 - loss)
                accuracies[name] = loss_list

            pos += x.size(0)

        # calculate some statistics based on the data collected
        result = {}
        for name, accuracy in accuracies.items():
            result[name] = {
                "mean": np.mean(np.array(accuracy)),
                "std": np.mean(np.std(np.array(accuracy))),
                "var": np.mean(np.var(np.array(accuracy)))
            }

        return result
