"""Example usage of the any model."""
from torch.utils.data import DataLoader
from TimeSeriesDL.model import BaseModel
from TimeSeriesDL.data import Dataset
from TimeSeriesDL.utils import config
from TimeSeriesDL.debug.visualize_dataset import VisualizeDataset

# load training arguments (equals example/simple_model.py)
train_args = config.get_args("./examples/simple/config.yaml")

# create a dataset loader which loads a matplotlib matrix from ./train.mat
data = Dataset(**train_args["dataset"])
dataloader = DataLoader(data, **train_args["dataloader"])

# create a model based on what is defined in the config
# to do so, a model needs to be registered using config.register_model()
model_name = train_args["model_name"]
model: BaseModel = config.get_model(model_name)(**train_args["model"])
model.use_device(train_args["device"])

# train the model on the dataset for 5 epochs and log the progress in a CLI
# to review the model's training performance, open TensorBoard in a browser
model.learn(train=dataloader, epochs=train_args["train_epochs"])

# save the model to its default location 'runs/{time_stamp}/model_SimpleModel.torch'
model.save_to_default()

# also, store a modified copy of the training arguments containing the model path
# this makes comparisons between multiple experiments easier<
train_args["model_path"] = model.log_path + "/model.torch"
config.store_args(f"{model.log_path}/config.yml", train_args)

import torch
import numpy as np
from scipy.io import savemat
from tqdm import trange

# create storage of prediction
window_len = train_args["dataset"]["sequence_length"]
f_len = train_args["dataset"]["future_steps"]
full_sequence = np.zeros(data.shape)
full_sequence[0:window_len, :] = data.slice(0, window_len)

# predict based on sliding window
for i in trange(0, data.sample_size - window_len, f_len):
    window = full_sequence[i:i + window_len]
    window = torch.tensor(window, device=train_args["device"], dtype=torch.float32)
    window = torch.unsqueeze(window, 0)
    sample = model.predict(window)
    full_sequence[i + window_len:i + window_len + f_len] = sample.detach().cpu().numpy()

# remove the channel and prepare to save the predicted data
full_sequence = np.squeeze(full_sequence, 1)
full_sequence = data.scale_back(full_sequence)
full_sequence = np.swapaxes(full_sequence, 0, 1)

# save prediction using the label names from the original dataset
export = {}
for i, label_name in enumerate(data.label_names):
    export[f"pred_feature_{i}"] = list(full_sequence[i, :])
savemat("examples/simple/prediction.mat", export)

# load dataset without normalization and compare against prediction
data = Dataset(normalize=False, **train_args["dataset"])
predicted = Dataset(normalize=False, custom_path="examples/simple/prediction.mat")
input_vis = VisualizeDataset(data, name="Input")
output_vis = VisualizeDataset(predicted, name="Predicted", overlay_mode=True)

input_vis.set_feature([0, 1])
input_vis.set_overlay(output_vis)
input_vis.visualize(save="examples/simple/comparioson.png")
