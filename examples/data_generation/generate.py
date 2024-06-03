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
samples = int(args.samples)
x, y1 = functions.test_1(size=samples)
x, y2 = functions.test_2(size=samples)

# store the dataset
savemat(path, {"train_1": y1, "train_2": y2})

# show the dataset
if args.visualize:
    fig, ax = plt.subplots()
    ax.scatter(x, y1, c="tab:blue", label="test_1", alpha=0.3, edgecolors='none')
    ax.scatter(x, y2, c="tab:red", label="test_2", alpha=0.3, edgecolors='none')

    ax.legend()
    ax.grid(True)

    plt.show()
