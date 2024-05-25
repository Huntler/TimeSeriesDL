"""Example usage of the any model."""
import argparse
import torch
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.io import savemat
from TimeSeriesDL.data import Dataset
from TimeSeriesDL.model.ae_model import AE
from TimeSeriesDL.utils import config

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="Configuration file to load", required=True)
args = parser.parse_args()

train_args = config.get_args(args.config)

# check if the model to load is trained and type AE
if train_args["model_name"] != "AE":
    print("To encode a dataset, an AE model is required.")
    exit(1)

if not train_args["model_path"]:
    print("The AE model needs to be trained first.")
    exit(1)

# load the AE and prevent logging
train_args["model"]["log"] = False
ae = AE(**train_args["model"])
ae.load(path=train_args["model_path"])

# load the dataset which should be encoded, make sure to disable AE mode
train_args["dataset"]["ae_mode"] = False
data = Dataset(**train_args["dataset"])
dataloader = DataLoader(data)

encoded = []
for x, _ in tqdm(dataloader):
    # encode the data and unwrap the batch
    x = ae.encode(x)
    x = torch.transpose(x, 2, 1)
    x = list(x.cpu().detach().numpy()[0, :, :])
    encoded.append(x)

# save the encoded dataset
savemat("./examples/train_encoded.mat", {"train": encoded})
