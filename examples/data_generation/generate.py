"""This module generates a dataset using functions from the function's module."""
import argparse
from scipy.io import savemat
import matplotlib.pyplot as plt
import functions

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--samples", help="Number of samples to generate")
parser.add_argument("-p", "--path", help="Path to store the dataset (default: examples/train.mat)")
parser.add_argument("-v", "--visualize", help="Visualize function", action="store_true")
args = parser.parse_args()

# check important arguments
if not args.samples:
    print("Provide a number of samples to generate.")
    exit(1)

path = args.path if args.path else"examples/train.mat"

# generate the dataset
x, y = functions.test_1(size=int(args.samples))

# store the dataset
savemat(path, {"train": y})

# show the dataset
if args.visualize:
    fig, ax = plt.subplots()
    ax.scatter(x, y, c="tab:blue", label="test_1", alpha=0.3, edgecolors='none')

    ax.legend()
    ax.grid(True)

    plt.show()
