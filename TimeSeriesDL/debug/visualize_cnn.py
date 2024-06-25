"""This module analyzes a model's first layer."""
import numpy as np
import matplotlib.pyplot as plt


class VisualizeConv:
    """
    A class for visualizing convolutional neural network (CNN) weights.

    Attributes:
        _model (BaseModel): The CNN model to be visualized.

    Methods:
        visualize(use_mean=True, save=None)
    """

    def __init__(self, model) -> None:
        self._model = model

        # get the first layer
        first_layer_name = next(iter(self._model.state_dict()))
        self._first_layer = self._model.state_dict()[first_layer_name]

        # Check if the first layer is a Conv2d layer
        dim_len = len(self._first_layer.shape)
        assert (
            dim_len == 4
        ), f"The first layer of the model must be a Conv2d layer but got dimension {dim_len}"

    def visualize(self, save: str = None) -> None:
        """
        Visualizes the weights of the first convolutional layer in the CNN.

        This method displays the weights of the first convolutional layer as an image,
        showing how they are distributed over features and kernels. Additionally, a boxplot
        is displayed to show the spread of kernel values over each feature.

        Args:
            save (str): Optional path to save the plot. If not provided, the plot will be
            shown immediately.
                Defaults to None.

        Returns:
            None: The method does not return any value. Instead, it displays the plot and
            saves it if a file path is provided.
        """
        conv_weights = self._first_layer.numpy()
        conv_weights = conv_weights.transpose(1, 3, 2, 0)

        # calculate the mean over all extracted features
        conv_weights = np.sum(np.abs(conv_weights), axis=3)

        # Display the weights using imshow
        fig, ax = plt.subplots(2, 1, figsize=(10, 5))
        fig.tight_layout(pad=2.0)

        ax[1].imshow(conv_weights[0].T, cmap="Reds", origin='lower')
        ax[1].set_xlabel("Kernel")
        ax[1].set_ylabel("Feature")

        # display a boxplot showing how the kernel values spread
        # over a feature
        ax[0].set_title("First Layer Analysis")
        pos = [_ for _ in range(conv_weights.shape[-1])]
        bp = ax[0].boxplot(conv_weights[0], vert=True, positions=pos)
        ax[0].set_ylabel('Kernel Values')
        ax[0].set_xlabel('Feature')
        plt.setp(bp['boxes'], color='red')

        if save:
            plt.savefig(save)
        plt.show()
