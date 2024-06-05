"""This module visualizes a dataset's data."""
import math
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from TimeSeriesDL.data.dataset import Dataset



def _transform_feature(feature: int | List[int], max_f: int) -> np.array:
    """This function transforms a feature or a list of features into a NumPy array,
    ensuring that the input is valid and compatible with the dataset. It also ensures
    that each feature index is within the bounds of the total number of features in
    the dataset.

    Args:
        feature (int | List[int]): an integer or a list of integers representing the
        feature(s) to be transformed.
        max_f (int): an integer representing the maximum index of the features in the
        dataset.

    Returns:
        np.array: A NumPy array containing the input feature(s).
    """
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
            overlay_mode (bool, optional): This object should be used as an overlay.
            Defaults to False.
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
        """Sets an overlay to this visualized dataset, which will be plotted in the
        same graph. This method can only be called on a normal object (i.e., not an
        overlay itself). If you try to set an overlay to another overlay, this method 
        will raise an assertion error. 
        
        To use this method, the object being passed as the `overlay` parameter must 
        be an instance of `VisualizeDataset` and must have been initialized with the 
        `overlay_mode` parameter.

        Args:
            overlay (VisualizeDataset): The object to overlay on top of this visualized dataset.
        """
        assert not self._overlay_mode, "Can not set an overlay to an overlay."
        assert isinstance(overlay, VisualizeDataset), "Expected overlay to be an instance of VisualizeDataset"
        assert getattr(overlay, 'is_overlay', False), "Expected overlay to have 'is_overlay' set to True"
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

    def _get_grid(self, max_h, num_features) -> Tuple[int, int]:
        sqrt = math.sqrt(num_features)
        if sqrt * sqrt == num_features and num_features <= max_h:
            return int(sqrt), int(sqrt)
        elif num_features <= max_h:
            return num_features, 1

        h, w = max_h, math.ceil(num_features / max_h)
        if (h * w - num_features) % w == 0:
            return h - (h * w - num_features) // 2, w
        return h, w

    def visualize(self, start: int = 0, end: int = -1, size: int = 3, save: str = None) -> None:
        """Visualizes the selected features of the dataset and the overlay if set.

        Args:
            start (int, optional): Start index of plot. Defaults to 0.
            end (int, optional): End index of plot. Defaults to -1.
            size (int, optional): Size of the plots. Defaults to 3.
            save (str, optional): Save path. Defaults to None and prevents saving.
        """
        assert self._feature is not None, \
            "No feature to visualize defined. Did you call 'set_feature()'?"
        assert not self._overlay_mode, "The overlay should not call visualize."

        if end == -1:
            end = self._dataset.shape[0]

        # if this is an overlay, then apply the features and labels set
        # otherwise start the matplotlib figure
        _overlay_data = None
        if self._overlay:
            _overlay_data = self._overlay.dataset.slice(start, end, self._feature)

        # slice the dataset as required to view start/end/selected features
        data = self._dataset.slice(start, end, self._feature)

        # setup graph and layout
        grid = self._get_grid(4, len(self._feature))
        figsize = (grid[1] * size * 2, grid[0] * size)
        fig, axs = plt.subplots(grid[0], grid[1], figsize=figsize, sharex=True, squeeze=False)
        fig.tight_layout(pad=2.0)

        # iterate over the graph's layout
        i = 0
        for plt_x in range(grid[0]):
            for plt_y in range(grid[1]):
                # visualize the data
                for c in range(data.shape[1]):
                    x = np.linspace(start, end, data.shape[0])
                    line, = axs[plt_x, plt_y].plot(x, data[:, c, i])
                    axs[plt_x, plt_y].set_ylabel(self._label[i])
                    if i == 0:
                        line.set_label(self.name)

                    # add the overlay
                    if self._overlay:
                        x = np.linspace(start, end, _overlay_data.shape[0])
                        line, = axs[plt_x, plt_y].plot(x, _overlay_data[:, c, i])
                        if i == 0:
                            line.set_label(self._overlay.name)
                i += 1
                if i == len(self._feature):
                    break
            if i == len(self._feature):
                break

            fig.legend()
        if save:
            fig.savefig(save)
        plt.show()
