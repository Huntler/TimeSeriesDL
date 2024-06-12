"""This module tests the TimeSeriesDL.data transcoding."""
import unittest
import random
from TimeSeriesDL.data import Dataset, encode_dataset, decode_dataset
from TimeSeriesDL.model import DummyAutoEncoder


paths = [f"examples/train_{i + 1}.mat" for i in range(4)]
sequence = random.randint(1, 50)
future = random.randint(1, 50)
length = 100_000 - sequence - future + 1
model = DummyAutoEncoder(sequence=sequence)


class TestTranscode(unittest.TestCase):
    def test_failure(self):
        """Tests if the encode_dataset function throws an error when trying to encode
        a multi-matrix dataset.
        """
        # Create a dataset from all files in the list of paths
        data = Dataset(sequence_length=sequence, future_steps=future, path=paths)

        encode_fn = lambda: encode_dataset(data, model)
        self.assertRaises(AssertionError, encode_fn)

    def test_encoder(self):
        """Tests if the encoder outputs the input when using a dumm auto encoder.
        """
        # Create a dataset from all files in the list of paths
        data = Dataset(sequence_length=sequence, future_steps=future, path=paths[0])

        encoded = encode_dataset(data, model)
        self.assertTrue((data[0][0] == encoded[0][0]).all())

    def test_decoder(self):
        """Tests if the decoder outputs the input when using a dummy auto encoder.
        """
        # Create a dataset from all files in the list of paths
        data = Dataset(normalize=False, sequence_length=sequence, future_steps=future, path=paths[0])

        decoded = decode_dataset(data, model)
        self.assertTrue((data[0][0] == decoded[0][0]).all())

    def test_decoder_scaling(self):
        """Test equals test_decoder, but tests the dataset.scale_back function when decoding.
        """
        # Create a dataset from all files in the list of paths
        data = Dataset(normalize=True, sequence_length=sequence, future_steps=future, path=paths[0])

        decoded = decode_dataset(data, model, scaler=data.scale_back)
        input_scaled = data.scale_back(data[0][0])
        self.assertTrue((input_scaled == decoded[0][0]).all())



if __name__ == "__main__":
    unittest.main()
