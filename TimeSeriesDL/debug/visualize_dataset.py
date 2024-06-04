"""This module visualizes a dataset's data."""
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from TimeSeriesDL.data.dataset import Dataset



def _transform_feature(feature: int | List[int], max_f: int) -> np.array:
    # make sure the feature is type of list
    if isinstance(feature, int):
        feature = [feature]

    # verify provided features to be compatible with dataset
    for f in feature:
        assert f >= 0, f"Expect feature index {f} to be greater equal than 0."
        assert f < max_f, (
            f"Expect feature index {f} to be smaller than "
            + "number of features in the dataset."
        )

    return np.array(feature)


class VisualizeDataset:
    """This class visualizes a dataset, with the ability to overwrite label names, select features
    to visualizes, and possibility to overlay a second dataset.
    """
    def __init__(self, dataset: Dataset, name: str = None, overlay_mode: bool = False) -> None:
        """Initializes a visualization object to show insights into a dataset.

        Args:
            dataset (Dataset): The dataset to be visualized. Requires the base functionality of the
            TimeSeriesDL's dataset class.
            name (str, optional): The name of the dataset. Defaults to the dataset file name.
            overlay_mode (bool, optional): This object should be used as an overlay. Defaults to False.
        """
        self._dataset = dataset
        self._name = name if name else self._dataset.d_type
        assert len(self._dataset.shape) == 3, "Expected dataset to have two dimensions."

        self._overlay: VisualizeDataset = None
        self._overlay_mode = overlay_mode

        self._feature = None
        self._label = None

    @property
    def dataset(self) -> Dataset:
        """Returns the dataset obtained by this object.

        Returns:
            Dataset: The dataset.
        """
        return self._dataset

    @property
    def name(self) -> str:
        """Returns the name to be shown on the plot.

        Returns:
            str: The name of the plot visualized.
        """
        return self._name

    @property
    def is_overlay(self) -> bool:
        """Is this object an overlay.

        Returns:
            bool: Returns True if the object should be used as an overlay.
        """
        return self._overlay_mode

    def set_overlay(self, overlay: "VisualizeDataset") -> None:
        """Sets an overlay to this object, which gets plotted in the same 
        graph as self. Setting an overlay to an overly is not permitted. An overlay requires
        the 'overlay_mode' parameter during initialization.

        Args:
            overlay (VisualizeDataset): The object to overlay.
        """
        assert not self._overlay_mode, "Can not set an overlay to an overlay."
        assert overlay.is_overlay, "Expected overlay to be an overlay, got normal object."
        self._overlay = overlay

    def set_feature(self, feature: int | List[int], label: str | List[str] = None) -> None:
        """Selects one or multiple features to be shown on the graph.

        Args:
            feature (int | List[int]): The features to be visualized.
            label (str | List[str], optional): Overwrites the datasets label names. 
            Defaults to None.
        """
        assert not self._overlay_mode, "The overlay uses the same features as the main plot."
        feature = _transform_feature(feature, self._dataset.shape[-1])

        if isinstance(label, str):
            label = [label]

        if not label:
            label = [self._dataset.label_names[i] for i in feature]

        assert len(feature) == len(label), "Expected a label name for each feature, or 'None'."
        self._feature = feature
        self._label = label

    def visualize(self, start: int = 0, end: int = -1, save: str = None) -> None:
        """Visualizes the selected features of the dataset and the overlay if set.

        Args:
            start (int, optional): Start index of plot. Defaults to 0.
            end (int, optional): End index of plot. Defaults to -1.
            save (str, optional): Save path. Defaults to None and prevents saving.
        """
        assert self._feature is not None, \
            "No feature to visualize defined. Did you call 'set_feature()'?"

        if end == -1:
            end = self._dataset.shape[0]

        # if this is an overlay, then apply the features and labels set
        # otherwise start the matplotlib figure
        _overlay_data = None
        if self._overlay:
            _overlay_data = self._overlay.dataset.slice(start, end, self._feature)
        else:
            plt.figure(1)

        # slice the dataset as required to view start/end/selected features
        data = self._dataset.slice(start, end, self._feature)

        plt_index = len(self._feature) * 100 + 10
        for i in self._feature:
            plt_index += 1
            plt.subplot(plt_index)

            # visualize the data
            for c in range(data.shape[1]):
                x = np.linspace(start, end, data.shape[0])
                plt.plot(x, data[:, c, i], label=self.name)
                plt.ylabel(self._label[i])

                # add the overlay
                if self._overlay:
                    x = np.linspace(start, end, _overlay_data.shape[0])
                    plt.plot(x, _overlay_data[:, c, i], label=self._overlay.name)

            plt.legend(loc="upper left")
        if save:
            plt.savefig(save)
        plt.show()
