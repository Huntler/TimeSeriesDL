"""This example shows how to visualize a dataset"""
import argparse

from TimeSeriesDL.data import Dataset
from TimeSeriesDL.debug import VisualizeDataset
from TimeSeriesDL.model.base_model import BaseModel
from TimeSeriesDL.utils import config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--compare-to", type=str, default=None)
    parser.add_argument("--features", type=int, nargs="+")
    parser.add_argument("--unscale", action="store_true", default=False)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--output", type=str, default=None)

    # define args
    args = parser.parse_args()
    train_args = config.get_args(args.config)
    if args.compare_to:
        train_args["dataset"]["custom_path"] = args.compare_to

    # load the dataset and put it into the visualizer
    data = Dataset(**train_args["dataset"])
    vis = VisualizeDataset(data, name="Input", scale_back=args.unscale)

    # load the second dataset to compare to the already loaded one if the argument is provided
    if args.compare_to:
        train_args["model"]["log"] = False
        model: BaseModel = config.get_model(train_args["model_name"])(**train_args["model"])
        model.load(train_args["model_path"])

        vis.generate_overlay(model)

    # visualize the dataset(s)
    features = args.features if args.features else list(range(len(data.label_names)))
    vis.set_feature(features)

    # test save last configuration but of all samples
    vis.visualize(start=args.start, end=args.end, save=args.output)
