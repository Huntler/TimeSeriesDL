"""This module manages multiple datasets and loads them in the required training stages."""
import math
import os
import copy
from typing import List, Dict
import lightning as L
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import  DataLoader

from TimeSeriesDL.data.dataset import Dataset


class TSDataModule(L.LightningDataModule):
    def __init__(self, files: str | List[str], data_kwargs: Dict, loader_kwargs: Dict) -> None:
        super().__init__()

        self._data_kwargs = data_kwargs
        self._loader_kwargs = loader_kwargs

        # check the provided file paths
        if isinstance(files, str):
            assert os.path.exists(files), "Can not examine provided path."
            self._files = [files]

        else:
            for f in files:
                assert os.path.exists(f), f"Can not examine provided path {f}."
            self._files = files

        self._train = None
        self._val = None
        self._test = None

    def _free_memory(self) -> None:
        self._train = None
        self._val = None
        self._test = None

    def setup(self, stage: str) -> None:
        assert stage in ["fit", "test"], "Unsopported stage, supported are [fit, test]."
        self._free_memory()

        # define the splits: 0.7, 0.2, 0.1
        test_split = 0
        if len(self._files) >= 2:
            test_split = math.ceil(0.2 * len(self._files))

        val_split = 0
        if len(self._files) >= 3:
            val_split = math.ceil(0.1 * len(self._files))

        train_split = len(self._files) - (test_split + val_split)

        # load the datasets
        if stage == "fit":
            f_train = self._files[0:train_split]
            self._train = Dataset(path=f_train, **self._data_kwargs)

            f_val = self._files[train_split:train_split + val_split]
            self._val = Dataset(path=f_val, **self._data_kwargs)

        if stage == "test":
            f_test = self._files[train_split + val_split:]
            self._test  = Dataset(path=f_test, **self._data_kwargs)

    def train_dataloader(self) -> CombinedLoader:
        return DataLoader(self._train, **self._loader_kwargs)

    def val_dataloader(self) -> CombinedLoader:
        # disable shuffle
        kwargs = copy.deepcopy(self._loader_kwargs)
        kwargs["shuffle"] = False

        return DataLoader(self._val, **kwargs)

    def test_dataloader(self) -> CombinedLoader:
        # disable shuffle
        kwargs = copy.deepcopy(self._loader_kwargs)
        kwargs["shuffle"] = False

        return DataLoader(self._test, **kwargs)
