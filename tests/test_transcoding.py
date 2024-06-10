"""This module tests the TimeSeriesDL.data transcoding."""
import unittest
import random
import torch
import numpy as np
from TimeSeriesDL.data import Dataset, encode_dataset, decode_dataset


paths = [f"examples/train_{i + 1}.mat" for i in range(4)]
sequence = random.randint(1, 50)
future = random.randint(1, 50)
length = 100_000 - sequence - future + 1


class DummyAutoEncoder:
    """The dummy auto encoder mimics the behaviour of a real auto encoder,
    required to test the transcode functions.
    """
    def encode(self, x: torch.tensor, as_array: bool = False) -> np.array:
        """Always returns the input as numpy array."""
        return x.cpu().detach().numpy()

    def decode(self, x: torch.tensor, as_array: bool = False) -> np.array:
        """Swaps the axis of the input as a real auto encoder would do."""
        x: torch.tensor = torch.swapaxes(x, 2, 1)
        x: torch.tensor = torch.swapaxes(x, 1, 3)
        return x.cpu().detach().numpy()

    @property
    def precision(self):
        """Returns the precision."""
        return torch.float32

    @property
    def latent_length(self):
        """Returns an abitrary latent length."""
        return sequence


class TestTranscode(unittest.TestCase):
    def test_failure(self):
        """Tests if the encode_dataset function throws an error when trying to encode
        a multi-matrix dataset.
        """
        # Create a dataset from all files in the list of paths
        data = Dataset(sequence_length=sequence, future_steps=future, path=paths)
        model = DummyAutoEncoder()

        encode_fn = lambda: encode_dataset(data, model)
        self.assertRaises(AssertionError, encode_fn)

    def test_encoder(self):
        """Tests if the encoder outputs the input when using a dumm auto encoder.
        """
        # Create a dataset from all files in the list of paths
        data = Dataset(sequence_length=sequence, future_steps=future, path=paths[0])
        model = DummyAutoEncoder()

        encoded = encode_dataset(data, model)
        self.assertTrue((data[0][0] == encoded[0][0]).all())

    def test_decoder(self):
        """Tests if the decoder outputs the input when using a dummy auto encoder.
        """
        # Create a dataset from all files in the list of paths
        data = Dataset(normalize=False, sequence_length=sequence, future_steps=future, path=paths[0])
        model = DummyAutoEncoder()

        decoded = decode_dataset(data, model)
        self.assertTrue((data[0][0] == decoded[0][0]).all())

    def test_decoder_scaling(self):
        """Test equals test_decoder, but tests the dataset.scale_back function when decoding.
        """
        # Create a dataset from all files in the list of paths
        data = Dataset(normalize=True, sequence_length=sequence, future_steps=future, path=paths[0])
        model = DummyAutoEncoder()

        decoded = decode_dataset(data, model, scaler=data.scale_back)
        input_scaled = data.scale_back(data[0][0])
        self.assertTrue((input_scaled == decoded[0][0]).all())



if __name__ == "__main__":
    unittest.main()
