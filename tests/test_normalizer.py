"""This module tests the TimeSeriesDL.data collate function."""
import unittest
import torch

from TimeSeriesDL.data import TensorNormalizer


class TestNormalizer(unittest.TestCase):
    def test_standardize_norm(self):
        # Standardize
        original = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float)
        scaler, scaled = TensorNormalizer(standardize=True).fit_transform(original)
        unscaled = scaler.inverse_transform(scaled)
        self.assertTrue(torch.equal(original, unscaled))

    def test_normalize_norm(self):
        # Normalize
        original = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float)
        scaler, scaled = TensorNormalizer(standardize=False).fit_transform(original)
        unscaled = scaler.inverse_transform(scaled)
        self.assertTrue(torch.equal(original, unscaled))
