from argparse import ArgumentError
from typing import List
from matplotlib import pyplot as plt
import numpy as np


def plot_curve(data: List or np.array, data_name: List or np.array = None,
               title: str = "Unknown", n_samples: int = -1,
               save_path: str = None) -> None:
    fig, axs = plt.subplots(1)
    fig.suptitle(title)

    # acutal plot method
    def closure(data: np.array, n_samples: int) -> None:
        samples = n_samples if n_samples != -1 else len(data)
        axs.plot(data[:samples, 0], data[:samples, 1])

    # plot one or multiple data curves
    if isinstance(data, List):
        if data_name is not None:
            if not isinstance(data_name, List):
                raise ArgumentError("If multiple data curves are given, then provided a list of names too.")
            if len(data_name) != len(data):
                raise ArgumentError("Each data curve needs a corresponding name.")

        for _data in data:
            closure(_data, n_samples)

        axs.legend(data_name)
    else:
        closure(data, n_samples)
        axs.legend(data_name)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
