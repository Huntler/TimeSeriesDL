"""This module contains various test functions."""
from typing import Tuple
import numpy as np

def f(x: float) -> float:
    """(1 / x) + math.sin(x**x) / (x - 4)

    Args:
        x (float): x.

    Returns:
        Tuple: f(x).
    """
    return (1 / x) + np.sin(np.power(x, x)) / (x - 4)

def test_1(size: int) -> Tuple[np.array, np.array]:
    """f(x)

    Args:
        size (int): Number of samples to generate.

    Returns:
        Tuple: x and y values.
    """
    x = np.linspace(0.5, 3.5, num=size)
    return x, f(x)

def test_2(size: int) -> Tuple[np.array, np.array]:
    """f(x * 0.98 + 0.2)

    Args:
        size (int): Number of samples to generate.

    Returns:
        Tuple: x and y values.
    """
    x = np.linspace(0.5, 3.5, num=size)
    return x, f(x * 0.98 + 0.2)
