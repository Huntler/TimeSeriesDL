"""This module tests the TimeSeriesDL.data collate function."""
import unittest
import random
import torch
from TimeSeriesDL.data import Dataset, AutoEncoderCollate
from TimeSeriesDL.model.auto_encoder import DummyAutoEncoder

paths = [f"examples/data/train_{i + 1}.mat" for i in range(4)]
sequence = random.randint(1, 50)
future = random.randint(1, 50)
length = 100_000 - sequence - future + 1
model = DummyAutoEncoder(sequence=sequence)

class TestCollae(unittest.TestCase):
    def test_function_sample(self):
        """
        Tests if the collate function works on a single sample.
        """
        data = Dataset(sequence_length=sequence, future_steps=future, path=paths[0])
        collate = AutoEncoderCollate(model)
        fn = collate.collate_fn()

        # convert to tensor
        x, y = data[0]
        x = torch.tensor(x)
        y = torch.tensor(y)

        # test the call
        result = fn([(x, y)])

        # add batch, encode, swap features and channels
        x = torch.unsqueeze(x, 0)
        x = model.encode(x)
        x = torch.swapaxes(x, 1, 2)

        self.assertEqual(result[0].shape, x.shape)
        self.assertTrue(result[0].all() == x.all())

    def test_function_batch(self):
        """
        Tests if the collate function works on a batch of data.
        """
        data = Dataset(sequence_length=sequence, future_steps=future, path=paths[0])
        _, channels, features = data.sample_shape()
        collate = AutoEncoderCollate(model)
        fn = collate.collate_fn()

        batch = []
        to_encode = torch.empty((100, sequence, channels, features))
        for i in range(100):
            # convert to tensor
            x, y = data[0]
            x = torch.tensor(x)
            y = torch.tensor(y)

            batch.append((x, y))
            to_encode[i] = x

        # test the call
        result = fn(batch)

        # add batch, encode, swap features and channels
        x = model.encode(to_encode)
        x = torch.swapaxes(x, 1, 2)

        self.assertEqual(result[0].shape, x.shape)
        self.assertTrue(result[0].all() == x.all())
