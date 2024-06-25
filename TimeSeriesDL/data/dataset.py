"""This module contains the Dataset loader."""

from typing import List, Tuple
import torch
import numpy as np

from TimeSeriesDL.data.tensor_normalizer import TensorNormalizer
from TimeSeriesDL.model.base_model import BaseModel


class Dataset(torch.utils.data.Dataset):
    """The dataset class loads a dataset from a scipy-matrix and normalize all values according
    to boundaries defined. On very large dataset, this class can run in system out of memory issues.
    The matrix to load should have the following shape: (n_features, n_samples)

    Args:
        torch (torch.utils.data.Dataset): Based on torch's dataset class.
    """

    def __init__(
        self,
        future_steps: int = 1,
        sequence_length: int = 32,
        scaler: str = None,
        precision: np.dtype = np.float32,
        path: str | List[str] = None
    ):
        super().__init__()

        self._precision = precision
        self._seq = sequence_length
        self._f_seq = future_steps
        self._scaler = scaler
        assert scaler in [None, "normalize", "standardize"]

        # load the dataset specified, if a path is provided,
        # otherwise create an empty dataset
        if path is None:
            return

        self._file = path if isinstance(path, list) else [path]
        self._mat = None
        self._labels = None
        self._load_data()
        self._shape = self.shape

        # pre-compute which index corresponds to which matrix
        self._precompute_indices()

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

    def _load_data(self) -> None:
        # clear container
        self._mat = []
        self._labels = None

        for file in self._file:
            # get the labels first
            with open(file, "r") as f:
                labels = f.readlines()[0][2:-1].split(",")
                if not self._labels:
                    self._labels = labels
                assert self._labels == labels, f"Expected labels to be equal, got {self._labels} and {labels}"

            # load the data and store them to the matrix
            data = np.loadtxt(file, delimiter=",", dtype=self._precision)
            self._mat.append(data)

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
        _,features = self._mat[0].shape
        if not label:
            return self._seq, features
        return self._f_seq, features

    def scale_back(self, data: List):
        """Scales the given data back using the inverse scaler. Runtime is O(1).

        Args:
            data (List): Data.

        Returns:
            np.array: Data scaled to input range.
        """
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
            return self._mat[0][start:end, index]
        return self._mat[0][start:end, :]

    def get(self, index: int) -> Tuple[np.array, np.array, TensorNormalizer]:
        """Returns the data and scaler at a given index.
        Args:
            index (int): Index of the data to return.
        Returns:
            Tuple[np.array, np.array]: The data and the scaler at that index.
        """
        assert 0 <= index < len(self), f"Invalid index {index}"
        mat_index = np.argmax(self._max_indices - (index + 1) >= 0)
        if mat_index > 0:
            index -= self._max_indices[mat_index - 1]

        # define sequence
        enc_input = self._mat[mat_index][index: self._seq + index, :]
        dec_input = self._mat[mat_index][self._seq + index - 1: self._seq + index + self._f_seq - 1, :]
        dec_output = self._mat[mat_index][self._seq + index: self._seq + index + self._f_seq, :]

        # scale sequences
        scaler = None
        if self._scaler:
            standardize = self._scaler == "standardize"
            scaler, enc_input = TensorNormalizer(standardize).fit_transform(enc_input)
            dec_input = scaler.transform(dec_input)
            dec_output = scaler.transform(dec_output)

        return enc_input, dec_input, dec_output, scaler


    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        x, _, y, _ = self.get(index)
        return x, y
