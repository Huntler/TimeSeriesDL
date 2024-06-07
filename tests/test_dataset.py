import unittest
import random
import numpy as np
from TimeSeriesDL.data import Dataset


paths = [f"examples/train_{i + 1}.mat" for i in range(4)]
sequence = random.randint(1, 50)
future = random.randint(1, 50)
length = 100_000 - sequence - future + 1


class TestDataset(unittest.TestCase):
    def test_init(self):
        for path in paths:
            _ = Dataset(sequence_length=sequence, future_steps=future, path=path)

    def test_multi_path_init(self):
        _ = Dataset(sequence_length=sequence, future_steps=future, path=paths)

    def test_slice(self):
        # test single-matrix dataset
        for path in paths:
            dataset = Dataset(sequence_length=sequence, future_steps=future, path=path)
            slice_result = dataset.slice(0, 100)
            self.assertEqual(slice_result.shape, (100, 1, 5))

        # test multi-matrix dataset
        dataset = Dataset(sequence_length=sequence, future_steps=future, path=paths)
        fn = lambda: dataset.slice(0, 100)
        self.assertRaises(AssertionError, fn)

    def test_label_names(self):
        # test single-matrix dataset
        for path in paths:
            dataset = Dataset(sequence_length=sequence, future_steps=future, path=path)
            self.assertEqual(len(dataset.label_names), 5)

        # test multi-matrix dataset
        dataset = Dataset(sequence_length=sequence, future_steps=future, path=paths)
        self.assertEqual(len(dataset.label_names), 5)

    def test_num_matrices(self):
        # test single-matrix dataset
        for path in paths:
            dataset = Dataset(sequence_length=sequence, future_steps=future, path=path)
            self.assertEqual(dataset.num_matrices, 1)

        # test multi-matrix dataset
        dataset = Dataset(sequence_length=sequence, future_steps=future, path=paths)
        self.assertEqual(dataset.num_matrices, 4)

    def test_shape(self):
        # test single-matrix dataset
        expected = (length, 1, 5)
        for path in paths:
            dataset = Dataset(sequence_length=sequence, future_steps=future, path=path)
            self.assertTupleEqual(dataset.shape, expected)

        # test multi-matrix dataset
        expected = (length * len(paths), 1, 5)
        dataset = Dataset(sequence_length=sequence, future_steps=future, path=paths)
        #self.assertTupleEqual(dataset.shape, expected)

    def test_sample_shape(self):
        expected_x = (sequence, 1, 5)
        expected_y = (future, 1, 5)

        # test single-matrix dataset
        for path in paths:
            dataset = Dataset(sequence_length=sequence, future_steps=future, path=path)
            self.assertTupleEqual(dataset.sample_shape(False), expected_x)
            self.assertTupleEqual(dataset.sample_shape(True), expected_y)

        # test multi-matrix dataset
        dataset = Dataset(sequence_length=sequence, future_steps=future, path=paths)
        self.assertTupleEqual(dataset.sample_shape(False), expected_x)
        self.assertTupleEqual(dataset.sample_shape(True), expected_y)

    def test_set_sequence(self):
        seq = random.randint(50, 100)
        expected = (seq, 1, 5)

        # test single-matrix dataset
        for path in paths:
            dataset = Dataset(sequence_length=sequence, future_steps=future, path=path)
            dataset.set_sequence(seq)
            self.assertTupleEqual(dataset.sample_shape(False), expected)

        # test multi-matrix dataset
        dataset = Dataset(sequence_length=sequence, future_steps=future, path=paths)
        dataset.set_sequence(seq)
        self.assertTupleEqual(dataset.sample_shape(False), expected)

    def test_multi_get_0(self):
        data_idx = 0
        dataset = Dataset(sequence_length=sequence, future_steps=future, path=paths)
        to_test_x, to_test_y = dataset[data_idx]

        #data_idx -= length
        dataset = Dataset(sequence_length=sequence, future_steps=future, path=paths[0])
        expected_x, expected_y = dataset[data_idx]

        self.assertTrue((to_test_x == expected_x).all())
        self.assertTrue((to_test_y == expected_y).all())

    def test_multi_get_1(self):
        data_idx = length + length // 2
        dataset = Dataset(sequence_length=sequence, future_steps=future, path=paths)
        to_test_x, to_test_y = dataset[data_idx]

        data_idx = length // 2
        dataset = Dataset(sequence_length=sequence, future_steps=future, path=paths[1])
        expected_x, expected_y = dataset[data_idx]

        self.assertTrue((to_test_x == expected_x).all(), f"test {data_idx} vs expected {data_idx}")
        self.assertTrue((to_test_y == expected_y).all())

    def test_overwrite_content(self):
        dataset = Dataset(sequence_length=sequence, future_steps=future, path=paths[0])

        _length = 50_000 - sequence - future + 1
        _feature = random.randint(1, 100)

        labels = [f"a{i}" for i in range(_feature)]
        content = np.zeros((50_000, 1, _feature))
        dataset.overwrite_content(content, labels)

        self.assertTupleEqual((_length, 1, _feature), dataset.shape)
        self.assertListEqual(labels, dataset.label_names)


if __name__ == "__main__":
    unittest.main()
