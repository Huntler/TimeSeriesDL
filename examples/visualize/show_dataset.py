"""This example shows how to visualize a dataset"""
from TimeSeriesDL.data.dataset import Dataset
from TimeSeriesDL.debug.visualize_dataset import VisualizeDataset


_dataset = Dataset(custom_path="examples/train.mat")
vis = VisualizeDataset(_dataset, name="Train")

# test one feature, overwrite label name
vis.set_feature(0, "Test")
vis.visualize()

# test two features, use label names of dataset
# only show samples 60_000 to 80_000
vis.set_feature([0, 1])
vis.visualize(start=60_000, end=80_000)

# test save last configuration but of all samples
vis.visualize(save="examples/train_mat.png")

# test overlaying another (here the same) dataset
second = VisualizeDataset(_dataset, name="Predicted", overlay_mode=True)
vis.set_overlay(second)
vis.visualize()
