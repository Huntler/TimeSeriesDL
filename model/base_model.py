from datetime import datetime
from typing import List
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


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

    def learn(self, X, y, epochs: int = 1):
        # measure history
        losses = []

        # run for a given amount of epochs
        with torch.cuda.amp.autocast(enabled=(self._device == "cuda")):
            for e in range(0, epochs):
                # perform the presiction and measure the loss between the prediction
                # and the expected output
                pred_y = self(X)

                # calculate the gradient using backpropagation of the loss
                loss = self._loss_fn(pred_y, y)

                self._optim.zero_grad
                loss.backward()
                self._optim.step()

                losses.append(loss.item())

        # log for the statistics
        losses = np.mean(losses, axis=0)
        self._writer.add_scalar("Train/loss", loss, self.__sample_position)
        self.__sample_position += X.size(0)

        self.eval()
        self._writer.flush()

    def predict(self, X, future_steps: int = 1) -> List:
        """This method only predicts future steps based on the given curve described by the datapoints X.

        Args:
            X (_type_): The datapoints.
            future_steps (int, optional): The amount of steps to look into future. Defaults to 1.

        Returns:
            List: The prediction.
        """
        with torch.no_grad():
            out = self(X)

        return out
