"""This module contains the Dataset loader."""

import re
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

    def __init__(
        self,
        d_type: str = "train",
        normalize: bool = True,
        bounds: Tuple[int] = (0, 1),
        future_steps: int = 1,
        sequence_length: int = 32,
        precision: np.dtype = np.float32,
        custom_path: str = None,
        ae_mode: bool = False
    ):
        super(Dataset, self).__init__()

        self._precision = precision
        self._seq = sequence_length
        self._f_seq = future_steps
        self._ae_mode = ae_mode

        # load the dataset specified
        self._d_type = d_type
        self._file = custom_path if custom_path else f"./data/{self._d_type}.mat"
        self._mat, self._labels = self.load_data()
        print("Dataset shape:", self._mat.shape)

        # normalize the dataset between values of o to 1
        self._scaler = None
        if normalize:
            self._scaler = MinMaxScaler(feature_range=bounds)
            self._scaler = self._scaler.fit(self._mat)
            self._mat = self._scaler.transform(self._mat)

    @property
    def label_names(self) -> List[str]:
        """Returns the list of label names corresponding to the data loaded.

        Returns:
            List[str]: The list of label names.
        """
        return self._labels

    @property
    def d_type(self) -> str:
        """Returns the d_type (=name) of the dataset.

        Returns:
            str: The d_type of the dataset.
        """
        return self._d_type

    def load_data(self) -> Tuple[np.array, List[str]]:
        """Loads the dataset from the path self._file which is generated as './data/{d_type}.mat'.

        Returns:
            Tuple[np.array, List[str]]: The dataset and labels.
        """
        labels = []
        mat = []
        for label, data in scipy.io.loadmat(self._file).items():
            # skip entries which are not labels
            if re.search("__\\w*__", label):
                continue

            labels.append(label)
            mat.append(data[0, :])

        mat: np.array = np.array(mat)
        mat = np.swapaxes(mat, 0, 1)
        return mat.astype(self._precision), labels

    @property
    def shape(self) -> Tuple[int]:
        """Returns the shape of the dataset.

        Returns:
            Tuple[int]: The dataset's shape as tuple of ints.
        """
        return self._mat.shape

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
        if self._scaler:
            data = np.array(data, dtype=self._precision)
            return self._scaler.inverse_transform(data)

        print("Warning, no scaler defined.")
        return data

    def slice(self, start: int, end: int, index: int | np.ndarray = None) -> np.array:
        """Slices the dataset.

        Args:
            start (int): Start index of slice.
            end (int): End index of slice.
            index (int | np.array): The index where to slice.

        Returns:
            np.array: The sliced array.
        """
        if isinstance(index, np.ndarray) or isinstance(index, int):
            return self._mat[start:end, index]
        return self._mat[start:end, :]

    def __len__(self):
        return max(1, len(self._mat) - self._f_seq - self._seq)

    def __getitem__(self, index):
        x = self._mat[index: self._seq + index]

        # the auto encoder requires input = output
        if self._ae_mode:
            return x, x

        y = self._mat[self._seq + index: self._seq + index + self._f_seq]
        return x, y
