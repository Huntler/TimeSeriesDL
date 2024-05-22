
"""This module contains the Dataset loader."""
from typing import List, Tuple
from sklearn.preprocessing import MinMaxScaler
import scipy.io
import torch
import numpy as np


class Dataset(torch.utils.data.Dataset):
    """The dataset class loads a dataset from a scipy-matrix and normalize all values according 
    to boundaries defined. On very large dataset, this class can run in system out of memory issues.
    The matrix to load should have the following shape: (n_features, n_samples)

    Args:
        torch (torch.utils.data.Dataset): Based on torch's dataset class.
    """
    def __init__(self, d_type: str = "train", normalize: bool = True, bounds: Tuple[int] = (0, 1),
                 future_steps: int = 1, sequence_length: int = 32, precision: np.dtype = np.float32,
                 custom_path: str = None):
        super(Dataset, self).__init__()

        self._precision = precision
        self._seq = sequence_length
        self._f_seq = future_steps

        # load the dataset specified
        self._d_type = d_type
        self._file = custom_path if custom_path else f"./data/{self._d_type}.mat"
        self._mat = self.load_data()

        # normalize the dataset between values of o to 1
        self._scaler = None
        if normalize:
            self._scaler = MinMaxScaler(feature_range=bounds)
            self._scaler = self._scaler.fit(self._mat)
            self._mat = self. _scaler.transform(self._mat)

        self._mat = self._mat[:, 0]

    def load_data(self) -> np.array:
        """Loads the dataset from the path self._file which is generated as './data/{d_type}.mat'.

        Returns:
            np.array: The dataset.
        """
        mat: np.array = scipy.io.loadmat(self._file).get(f"{self._d_type}")
        mat = np.swapaxes(mat, 0, 1)
        return mat.astype(self._precision)

    @property
    def sample_size(self) -> int:
        """Returns the sample size of the dataset.

        Returns:
            int: The dataset's sample size.
        """
        return self._mat.shape[0]

    def scale_back(self, data: List):
        """Scales the given data back using the inverse scaler.

        Args:
            data (List): Data.

        Returns:
            np.array: Data scaled to input range.
        """
        data = np.array(data, dtype=self._precision)
        return self._scaler.inverse_transform(data)

    def __len__(self):
        return max(1, len(self._mat) - self._f_seq - self._seq)

    def __getitem__(self, index):
        x = self._mat[index:self._seq + index]
        y = self._mat[self._seq + index:self._seq + index + self._f_seq]
        return x, y
