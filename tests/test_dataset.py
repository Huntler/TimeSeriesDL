"""This module tests the TimeSeriesDL.data.Dataset class."""
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
        """
        Tests the initialization of a single-matrix or multi-matrix dataset.
        """
        for path in paths:
            # Create a dataset from each file in the list of paths
            _ = Dataset(sequence_length=sequence, future_steps=future, path=path)

    def test_multi_path_init(self):
        """
        Tests the initialization of a multi-matrix dataset.
        """
        # Create a dataset from all files in the list of paths
        _ = Dataset(sequence_length=sequence, future_steps=future, path=paths)

    def test_slice(self):
        """
        Tests slicing a single-matrix or multi-matrix dataset.
        """
        # Test single-matrix dataset
        # Slice the dataset from index 0 to 100 and check that the shape is correct
        for path in paths:
            dataset = Dataset(sequence_length=sequence, future_steps=future, path=path)
            slice_result = dataset.slice(0, 100) 
            self.assertEqual(slice_result.shape, (100, 1, 5))  

        # Test multi-matrix dataset
        # The slice method should throw an assertion error when used on a multi-matrix dataset
        dataset = Dataset(sequence_length=sequence, future_steps=future, path=paths)
        fn = lambda: dataset.slice(0, 100)
        self.assertRaises(AssertionError, fn)

    def test_label_names(self):
        """
        Tests getting the label names from a single-matrix or multi-matrix dataset.
        """
        # Test single-matrix dataset
        # Check that the number of label names is correct matches the number of features
        for path in paths:
            dataset = Dataset(sequence_length=sequence, future_steps=future, path=path)
            self.assertEqual(len(dataset.label_names), 5)

        # Test multi-matrix dataset
        # Check that the number of label names is correct matches the number of features
        dataset = Dataset(sequence_length=sequence, future_steps=future, path=paths)
        self.assertEqual(len(dataset.label_names), 5)

    def test_num_matrices(self):
        """
        Tests getting the number of matrices from a single-matrix or multi-matrix dataset.
        """
        # Test single-matrix dataset
        # A single-matrix dataset has num_matrices == 1
        for path in paths:
            dataset = Dataset(sequence_length=sequence, future_steps=future, path=path)
            self.assertEqual(dataset.num_matrices, 1)

        # Test multi-matrix dataset
        dataset = Dataset(sequence_length=sequence, future_steps=future, path=paths)
        self.assertEqual(dataset.num_matrices, 4)

    def test_shape(self):
        """
        Tests getting the shape from a single-matrix or multi-matrix dataset.
        """
        # Test single-matrix dataset
        expected = (length, 1, 5)
        for path in paths:
            dataset = Dataset(sequence_length=sequence, future_steps=future, path=path)
            self.assertTupleEqual(dataset.shape, expected)

        # Test multi-matrix dataset
        expected = (length * len(paths), 1, 5)
        dataset = Dataset(sequence_length=sequence, future_steps=future, path=paths)
        self.assertTupleEqual(dataset.shape, expected)

    def test_sample_shape(self):
        """
        Tests getting the sample shape from a single-matrix or multi-matrix dataset.
        """
        expected_x = (sequence, 1, 5)
        expected_y = (future, 1, 5)

        # Test single-matrix dataset
        # Check that the sample shape matches the expected values, the parameter True/False
        # returns the shape of a sample/label
        for path in paths:
            dataset = Dataset(sequence_length=sequence, future_steps=future, path=path)
            self.assertTupleEqual(dataset.sample_shape(False), expected_x)
            self.assertTupleEqual(dataset.sample_shape(True), expected_y)

        # Test multi-matrix dataset
        dataset = Dataset(sequence_length=sequence, future_steps=future, path=paths)
        self.assertTupleEqual(dataset.sample_shape(False), expected_x)
        self.assertTupleEqual(dataset.sample_shape(True), expected_y)

    def test_set_sequence(self):
        """
        Tests setting a new sequence length to the dataset.
        """
        # define the new sequence to set to know what to expect
        seq = random.randint(50, 100)
        expected = (seq, 1, 5)

        # test single-matrix dataset
        # .sample_shape() uses the dataset._seq attribute, as all other methods
        # check if its output matches the expected tuple
        for path in paths:
            dataset = Dataset(sequence_length=sequence, future_steps=future, path=path)
            dataset.set_sequence(seq)
            self.assertTupleEqual(dataset.sample_shape(False), expected)

        # test multi-matrix dataset
        dataset = Dataset(sequence_length=sequence, future_steps=future, path=paths)
        dataset.set_sequence(seq)
        self.assertTupleEqual(dataset.sample_shape(False), expected)

    def test_multi_get_0(self):
        """
        Test if the getter at index=0 works as intended on a multi-matrix dataset.
        """
        # get the first value of the multi-matrix dataset
        # as those matrices are loaded as arranged in 'paths', the output should
        # match a single-matrix output at the same index
        data_idx = 0
        dataset = Dataset(sequence_length=sequence, future_steps=future, path=paths)
        to_test_x, to_test_y = dataset[data_idx]

        # get the first value of the single-matrix dataset
        dataset = Dataset(sequence_length=sequence, future_steps=future, path=paths[0])
        expected_x, expected_y = dataset[data_idx]

        self.assertTrue((to_test_x == expected_x).all())
        self.assertTrue((to_test_y == expected_y).all())

    def test_multi_get_1(self):
        """
        Test if the getter at index>single-matrix.length works as intended on a multi-matrix 
        dataset.
        """
        data_idx = length + length // 2
        dataset = Dataset(sequence_length=sequence, future_steps=future, path=paths)
        to_test_x, to_test_y = dataset[data_idx]

        data_idx = length // 2
        dataset = Dataset(sequence_length=sequence, future_steps=future, path=paths[1])
        expected_x, expected_y = dataset[data_idx]

        self.assertTrue((to_test_x == expected_x).all(), f"test {data_idx} vs expected {data_idx}")
        self.assertTrue((to_test_y == expected_y).all())

    def test_multi_get_transition(self):
        """
        Test if the transition between matrices results in consistent returns.
        """
        data_idx = length - sequence - future - 5
        dataset = Dataset(sequence_length=sequence, future_steps=future, path=paths)
        expected_x, expected_y = dataset[data_idx]
        for i in range(1, (sequence + future) * 2):
            to_test_x, to_test_y = dataset[data_idx + i]

            msg = f"{length} ({sequence}, {future}): {data_idx + i - 1} vs {data_idx + i}"
            self.assertTupleEqual(to_test_x.shape, expected_x.shape, msg)
            self.assertTupleEqual(to_test_y.shape, expected_y.shape, msg)

            expected_x, expected_y = to_test_x, to_test_y

    def test_overwrite_content(self):
        """
        Tests if overwriting the content of a dataset changes the shape as well.
        """
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
