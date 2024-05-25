"""Example usage of the any model."""

import argparse
from TimeSeriesDL.data import encode_dataset
from TimeSeriesDL.utils import config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", help="Configuration file to load", required=True
    )
    args = parser.parse_args()

    train_args = config.get_args(args.config)
    encode_dataset(train_args)
