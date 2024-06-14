"""This module contains the Dataset loader."""

import re
from typing import List, Tuple
from sklearn.preprocessing import MinMaxScaler
import scipy.io
from scipy.io import savemat
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
        normalize: bool = True,
        bounds: Tuple[int] = (0, 1),
        future_steps: int = 1,
        sequence_length: int = 32,
        precision: np.dtype = np.float32,
        path: str | List[str] = None,
        ae_mode: bool = False
    ):
        super().__init__()

        self._precision = precision
        self._seq = sequence_length
        self._f_seq = future_steps
        self._ae_mode = ae_mode

        # load the dataset specified, if a path is provided,
        # otherwise create an empty dataset
        if path is None:
            return

        self._file = path if isinstance(path, list) else [path]
        self._mat = None
        self._labels = None
        self._load_data()
        self._shape = self.shape

        # normalize the dataset between values of o to 1
        self._scaler = None
        if normalize:
            for i, mat in enumerate(self._mat):
                self._scaler = MinMaxScaler(feature_range=bounds)
                self._scaler = self._scaler.fit(mat[:, 0, :])
                self._mat[i][:, 0, :] = self._scaler.transform(mat[:, 0, :])

        # pre-compute which index corresponds to which matrix
        self._precompute_indices()

        assert len(self._shape) == 3, f"Expect dataset dimensions to be 3, got {len(self._shape)}"
        assert self._shape[1] == 1, f"Expect dataset channel dimension to be 1, got {self._shape[1]}"

    def _precompute_indices(self) -> None:
        """Method pre-computes the indices for eventual sub-matrices.
        """
        self._max_indices = []
        prev_size = 0
        for m in self._mat:
            l = len(m) - self._seq - self._f_seq + 1
            self._max_indices.append(prev_size + l)
            prev_size += l
        self._max_indices = np.array(self._max_indices)

    @property
    def label_names(self) -> List[str]:
        """Returns the list of label names corresponding to the data loaded.

        Returns:
            List[str]: The list of label names.
        """
        return self._labels

    @property
    def num_matrices(self) -> int:
        """Property describes the number of matrices load in this dataset. Useful, as not
        all operations can be done on a list of matrices and require a single-matrix
        dataset to work.

        Returns:
            int: The number of matrices in this dataset.
        """
        if not self._mat:
            return 0

        return len(self._mat)

    def overwrite_content(self, mat: np.array, labels: List[str]) -> None:
        """Overwrites the matrix of this dataset. Make sure that the shape of the new
        matrix is compatible, especially the matrix needs to have 3 dimensions: Samples, 
        Channles, Features. Runtime is O(1).

        Args:
            mat (np.array): The new matrix.
        """
        self._mat = [mat]
        self._labels = labels
        self._shape = self.shape
        self._precompute_indices()

        assert len(self._shape) == 3, f"Expect dataset dimensions to be 3, got {len(self._shape)}"
        assert self._shape[1] == 1, f"Expect dataset channel dimension to be 1, got {self._shape[1]}"

    def _load_data(self) -> None:
        self._mat = []
        self._labels = []

        # Runtime is O(n_files * n_labels).
        for path in self._file:
            labels = []
            mat = []
            for label, data in sorted(scipy.io.loadmat(path).items()):
                # skip entries which are not labels
                if re.search("__\\w*__", label):
                    continue

                labels.append(label)
                mat.append(data[0, :])

            # swap axes to have feature, samples
            mat: np.array = np.array(mat, dtype=self._precision)
            mat = np.swapaxes(mat, 0, 1)

            self._mat.append(mat)
            self._labels = labels

        # add a dimension to have one channel per feature/sample
        if len(self._mat[0].shape) != 3:
            self._mat = [np.expand_dims(m, 1) for m in self._mat]

    def set_sequence(self, length: int) -> None:
        """Sets the sequence length of the samples output. Runtime is O(1).

        Args:
            length (int): The length to output.
        """
        self._seq = length

    @property
    def shape(self) -> Tuple[int]:
        """Returns the shape of the dataset. Runtime is O(n_matrices).

        Returns:
            Tuple[int]: The dataset's shape as tuple of ints.
        """

        # calculate the matrix shape
        shape = list(self._mat[0].shape)
        shape[0] -= self._seq + self._f_seq - 1
        for i in range(1, len(self._mat)):
            shape[0] += (self._mat[i].shape[0] - self._seq - self._f_seq + 1)
            assert self._mat[0].shape[1:] == self._mat[i].shape[1:], \
                "The shape of the dataset is not consistent"
        return tuple(shape)

    def sample_shape(self, label: bool = False) -> Tuple[int, int, int]:
        """Returns the shape of one sample. Runtime is O(1).

        Args:
            label (bool): Return the shape of a label

        Returns:
            Tuple[int, int, int]: The shape of one sample as tuple of ints.
        """
        _, channels,features = self._mat[0].shape
        if not label:
            return self._seq, channels, features
        return self._f_seq, channels, features

    def scale_back(self, data: List):
        """Scales the given data back using the inverse scaler. Runtime is O(1).

        Args:
            data (List): Data.

        Returns:
            np.array: Data scaled to input range.
        """
        input_shape = data.shape
        assert len(self._mat) == 1, "Revert scaling is only supported when one dataset is loaded"
        if len(input_shape) == 3:
            assert data.shape[1] == 1, f"Can not scale back on multi-channel data. {data.shape} got {data.shape[1]}"
            data = data[:, 0, :]

        if self._scaler:
            data = np.array(data, dtype=self._precision)
            data = self._scaler.inverse_transform(data)
            if len(input_shape) == 3:
                data = np.expand_dims(data, 1)
            return data

        print("Warning, no scaler defined.")
        return data

    def slice(self, start: int, end: int, index: int | np.ndarray = None) -> np.array:
        """Slices the dataset along the samples axis. Runtime is O(n_matrices).

        Args:
            start (int): Start index of slice.
            end (int): End index of slice.
            index (int | np.array): The index where to slice.

        Returns:
            np.array: The sliced array.
        """
        assert len(self._mat) == 1, "Single-matrix dataset supported only"
        assert start < end, "Start index must be smaller than end index."
        assert start >= 0, "Start index must be >= 0"
        assert end != -1, "End must be well defined."

        if isinstance(index, np.ndarray) or isinstance(index, int):
            return self._mat[0][start:end, :, index]
        return self._mat[0][start:end, :, :]

    def save(self, path: str) -> None:
        """Saves the dataset to the provided path as scipy matrix. Runtime is
        O(n_labels).
        
        Args:
            path (str): Location of the exported matrix.
        """
        assert len(self._mat) == 0, "Single-matrix dataset supported only"

        export = {}
        for i, label in enumerate(self.label_names):
            export[label] = self._mat[:, :, i]
        savemat(path, export)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        assert 0 <= index < len(self), f"Invalid index {index}"
        mat_index = np.argmax(self._max_indices - (index + 1) >= 0)
        if mat_index > 0:
            index -= self._max_indices[mat_index - 1]

        x = self._mat[mat_index][index: self._seq + index, :, :]

        # the auto encoder requires input = output
        if self._ae_mode:
            return x, x

        y = self._mat[mat_index][self._seq + index: self._seq + index + self._f_seq, :, :]
        return x, y
