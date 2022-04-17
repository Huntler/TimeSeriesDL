from typing import List, Tuple
from sklearn.preprocessing import MinMaxScaler
import scipy.io
import torch
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, d_type: str = "train", normalize: bool = True, bounds: Tuple[int] = (0, 1),
                future_steps: int= 1, sequence_length: int = 1, precision: np.dtype = np.float32):
        super(Dataset, self).__init__()

        self._precision = precision
        self._seq = sequence_length
        self._f_seq = future_steps

        # load the dataset specified
        self._file = f"./data/{d_type}.mat"
        self._mat = scipy.io.loadmat(self._file).get(f"X{d_type}")
        self._mat = self._mat.astype(self._precision)

        # normalize the dataset between values of o to 1
        if normalize:
            _scaler = MinMaxScaler(feature_range=bounds)
            _scaler.fit(self._mat)
            self._mat = _scaler.transform(self._mat)

    @property
    def sample_size(self) -> int:
        return self._mat.shape[1]

    def __getitem__(self, index):
        # get the label (or label sequence)
        y = self._mat[index * self._f_seq:(index + 1) * self._f_seq, np.newaxis]
        y = y.astype(self._precision)

        x_start = index * self._f_seq
        x_end = x_start - self._seq
        X = np.zeros((self._seq, 1), dtype=self._precision)

        if x_end == -self._seq:
            X[self._seq - x_start:] = self._mat[:x_start]
            return X, y

        if x_end < 0:
            X[self._seq - x_start:] = self._mat[:x_start]
            return X, y

        X = self._mat[x_end:x_start, :]
        return X, y

    def __len__(self):
        return len(self._mat) / self._f_seq
