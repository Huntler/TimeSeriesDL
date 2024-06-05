"""This module generates a dataset using functions from the function's module."""
import argparse
import numpy as np
from scipy.io import savemat
import matplotlib.pyplot as plt
import functions

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--samples", help="Number of samples to generate")
parser.add_argument("-p", "--path", help="Path to store the dataset (default: examples/train.mat)")
parser.add_argument("--visualize", help="Visualize function", action="store_true")
parser.add_argument("-v", "--version", help="The dataset version to generate.")
args = parser.parse_args()

# check important arguments
if not args.samples:
    print("Provide a number of samples to generate.")
    exit(1)

path = args.path if args.path else"examples/train_1.mat"
samples = int(args.samples)
version = int(args.version) if args.version else 0
print(f"Generating version {version}")

d = {}
if version == 0:
    # generate the dataset
    x, y1 = functions.test_1(size=samples)
    x, y2 = functions.test_2(size=samples)

    # store the dataset
    d = {"train_1": y1, "train_2": y2}
    savemat(path, d)

elif version == 1:
    # Generate a sequence of x values
    x = np.linspace(0, 10, samples)

    # Calculate y values for each function
    y1 = (x**2 + 2*x + 1) * np.exp(-x*0.5)
    y2 = np.exp(y1)
    y3 = np.sin(y1*2)
    y4 = np.sin(np.exp(x/2))
    y5 = np.log(x + 1)

    d = {"train_1": y1, "train_2": y2, "train_3": y3, "train_4": y4, "train_5": y5}
    savemat(path, d)

elif version == 2:
    # same as version 1 but shifted by x=5
    # Generate a sequence of x values
    x = np.linspace(1, 1, samples)

    # Calculate y values for each function
    y1 = (x**2 + 2*x + 1) * np.exp(-x*0.5)
    y2 = np.exp(y1)
    y3 = np.sin(y1*2)
    y4 = np.sin(np.exp(x/2))
    y5 = np.log(x + 1)

    d = {"train_1": y1, "train_2": y2, "train_3": y3, "train_4": y4, "train_5": y5}
    savemat(path, d)

elif version == 3:
    # same as version 1 but shifted by x=1 and y*0.2
    # Generate a sequence of x values
    x = np.linspace(0, 10, samples)

    # Calculate y values for each function
    factor = 0.8
    y1 = (x**2 + 2*x + 1) * np.exp(-x*0.5) * factor
    y2 = np.exp(y1) * factor
    y3 = np.sin(y1*2) * factor
    y4 = np.sin(np.exp(x/2)) * factor
    y5 = np.log(x + 1) * factor

    d = {"train_1": y1, "train_2": y2, "train_3": y3, "train_4": y4, "train_5": y5}
    savemat(path, d)

# show the dataset
if args.visualize:
    # Create a figure for plotting
    plt.figure(figsize=(14, 8))

    # Plot each function by iterating over the dictionary
    for i, (key, value) in enumerate(d.items(), start=1):
        plt.subplot(3, 2, i)  # Create a subplot in a 3x2 grid
        plt.plot(x, value, label=key)
        plt.legend()

    # Adjust the layout and display the plot
    plt.tight_layout()
    plt.show()
