"""This module contains various test functions."""
from typing import Tuple
import numpy as np


def test_1(size: int) -> Tuple[np.array, np.array]:
    """(1 / x) + math.sin(x**x) / (x - 4)

    Args:
        size (int): Number of samples to generate.

    Returns:
        Tuple: x and y values.
    """
    x = np.linspace(0.5, 3.5, num=size)
    return x, (1 / x) + np.sin(np.power(x, x)) / (x - 4)
