"""This example shows how to visualize a model"""
from TimeSeriesDL.debug import VisualizeConv
from TimeSeriesDL.model import BaseModel, ConvLSTM

# create or load a model, disable logging to avoid creating a
# run folder and tensorboard
model: BaseModel = ConvLSTM(log=False)

# visualize the model
vis = VisualizeConv(model)
vis.visualize()
