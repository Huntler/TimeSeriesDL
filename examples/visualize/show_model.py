"""This example shows how to visualize a model"""
import argparse
from TimeSeriesDL.debug import VisualizeConv
from TimeSeriesDL.model import BaseModel, ConvLSTM
from TimeSeriesDL.utils import model_register

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Config file of a trained model.")
args = parser.parse_args()

# create or load a model, disable logging to avoid creating a
# run folder and tensorboard
args = model_register.get_args(args.config)
args["model"]["log"] = False
model: BaseModel = ConvLSTM(**args["model"])
model.load(args["model_path"])

# visualize the model
save = args["model_path"].replace("model.torch", "analysis.png")
vis = VisualizeConv(model)
vis.visualize(save)
