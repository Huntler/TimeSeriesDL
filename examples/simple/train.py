"""Example usage of the any model."""
from torch.utils.data import DataLoader
from TimeSeriesDL.debug.visualize_cnn import VisualizeConv
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

# also, store a modified copy of the training arguments containing the model path
# this makes comparisons between multiple experiments easier
train_args["model_path"] = model.log_path + "/models/model.torch"
config.store_args(f"{model.log_path}/config.yml", train_args)

# train the model on the dataset for 5 epochs and log the progress in a CLI
# to review the model's training performance, open TensorBoard in a browser
epochs = train_args["train_epochs"]
model.learn(train=dataloader, epochs=epochs)

# save the model to its default location 'runs/{time_stamp}/model_SimpleModel.torch'
model.save_to_default()

# load dataset without normalization and compare against prediction
input_vis = VisualizeDataset(data, name="Input")
input_vis.generate_overlay(model)

input_vis.set_feature(list(range(len(data.label_names))))
input_vis.visualize(save=f"{model.log_path}/predict_on_train.png")

# visualize the model
model.use_device("cpu")
vis = VisualizeConv(model)
vis.visualize(f"{model.log_path}/analysis.png")
