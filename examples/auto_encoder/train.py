"""Example usage of the any model."""
from torch.utils.data import DataLoader
from TimeSeriesDL.model import BaseModel
from TimeSeriesDL.data import Dataset
from TimeSeriesDL.utils import config

# load training arguments (equals example/simple_model.py)
train_args = config.get_args("./examples/auto_encoder/config.yaml")

# create a dataset loader which loads a matplotlib matrix from ./train.mat
data = Dataset(**train_args["dataset"])
dataloader = DataLoader(data, **train_args["dataloader"])

# create a model based on what is defined in the config
# to do so, a model needs to be registered using config.register_model()
model_name = train_args["model_name"]
simple_model: BaseModel = config.get_model(model_name)(**train_args["model"])
simple_model.use_device(train_args["device"])

# train the model on the dataset for 5 epochs and log the progress in a CLI
# to review the model's training performance, open TensorBoard in a browser
simple_model.learn(train=dataloader, epochs=train_args["train_epochs"])

# save the model to its default location 'runs/{time_stamp}/model_SimpleModel.torch'
simple_model.save_to_default()

# also, store a modified copy of the training arguments containing the model path
# this makes comparisons between multiple experiments easier<
train_args["model_path"] = simple_model.log_path + "/model.torch"
config.store_args(f"{simple_model.log_path}/config.yml", train_args)