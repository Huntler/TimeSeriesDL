"""This module contains the base model."""

from typing import List

import lightning as L
import torch

from TimeSeriesDL.utils import get_loss_by_name, get_optimizer_by_name


class BaseModel(L.LightningModule):
    """BaseModel class of any neural network predicting time series.

    Args:
        (lightning.LightningModule): LightningModule.
    """

    def __init__(self, loss: str = "MSELoss", optimizer: str = "Adam", lr: float = 1e-3) -> None:
        """Initializes the TensorBoard logger and checks for available GPU(s).
        """
        super().__init__()

        self._loss_name = loss
        self._loss = get_loss_by_name(loss)
        self._optim = optimizer
        self._lr = lr

        self._dropout = torch.nn.Dropout()
        self._mc_iteration = 10

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def training_step(self, batch: torch.Tensor, **kwargs) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self._loss(y_hat, y)
        self.log(f"train/{self._loss_name}", loss)
        return loss

    def test_step(self, batch: torch.Tensor, **kwargs) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self._loss(y_hat, y)
        self.log(f"test/{self._loss_name}", loss)
        return loss

    def validation_step(self, batch: torch.Tensor, **kwargs) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self._loss(y_hat, y)
        self.log(f"validate/{self._loss_name}", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return get_optimizer_by_name(self._optim)(self.parameters(), lr=self._lr)

    def predict_step(self, batch: torch.Tensor, **kwargs) -> torch.Tensor:
        # enable Monte Carlo Dropout
        self._dropout.train()

        # take average of MC-iterations
        pred = [self._dropout(self(batch)).unsqueeze(0) for _ in range(self._mc_iteration)]
        pred = torch.vstack(pred).mean(dim=0)
        return pred

    def predict(self, x: torch.tensor, as_array: bool = False) -> List:
        """This method only predicts future steps based on the given curve described by
        the datapoints X.

        Args:
            x (torch.tensor): The datapoints.
            as_array (bool, optional): Returns the prediction as np.array. Defaults to False.

        Returns:
            List: The prediction.
        """
        with torch.no_grad():
            out = self.predict_step(x)
            if as_array:
                out = out.cpu().numpy()

        return out
