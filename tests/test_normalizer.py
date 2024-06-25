"""This module tests the TimeSeriesDL.data collate function."""
import unittest
import torch
import numpy as np

from TimeSeriesDL.data import TensorNormalizer


class TestNormalizer(unittest.TestCase):
    def test_standardize_norm(self):
        # Standardize
        original = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        scaler, scaled = TensorNormalizer(standardize=True).fit_transform(original)
        unscaled = scaler.inverse_transform(scaled)
        self.assertTrue((original == unscaled).all())

    def test_normalize_norm(self):
        # Normalize
        original = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        scaler, scaled = TensorNormalizer(standardize=False).fit_transform(original)
        unscaled = scaler.inverse_transform(scaled)
        self.assertTrue((original == unscaled).all())
