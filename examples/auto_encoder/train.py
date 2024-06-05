"""Example usage of the any model."""
from torch.utils.data import DataLoader
from TimeSeriesDL.debug.visualize_dataset import VisualizeDataset
from TimeSeriesDL.model import BaseModel
from TimeSeriesDL.data import Dataset, encode_dataset, decode_dataset
from TimeSeriesDL.utils import config

# load training arguments (equals example/simple_model.py)
train_args = config.get_args("./examples/auto_encoder/config.yaml")

# create a dataset loader which loads a matplotlib matrix from ./train.mat
data = Dataset(**train_args["dataset"])
dataloader = DataLoader(data, **train_args["dataloader"])

# create a model based on what is defined in the config
# to do so, a model needs to be registered using config.register_model()
model_name = train_args["model_name"]
ae_model: BaseModel = config.get_model(model_name)(**train_args["model"])
ae_model.use_device(train_args["device"])

# train the model on the dataset for 5 epochs and log the progress in a CLI
# to review the model's training performance, open TensorBoard in a browser
ae_model.learn(train=dataloader, epochs=train_args["train_epochs"])

# save the model to its default location 'runs/{time_stamp}/model_SimpleModel.torch'
ae_model.save_to_default()

# also, store a modified copy of the training arguments containing the model path
# this makes comparisons between multiple experiments easier<
train_args["model_path"] = ae_model.log_path + "/model.torch"
config.store_args(f"{ae_model.log_path}/config.yml", train_args)

# use the trained AE to encode -> decode a dataset...
encode_dataset(train_args, export_path="examples/auto_encoder/train_encoded.mat")
train_args["dataset"]["custom_path"] = "examples/auto_encoder/train_encoded.mat"

decode_dataset(train_args, data.scale_back, labels=data.label_names,
               export_path="examples/auto_encoder/train_decoded.mat")
decoded = Dataset(custom_path="examples/auto_encoder/train_decoded.mat")

# ...and visualize the decoded dataset against the input
input_vis = VisualizeDataset(data, name="Input")
output_vis = VisualizeDataset(decoded, name="Predicted", overlay_mode=True)

input_vis.set_feature([0, 1])
input_vis.set_overlay(output_vis)
input_vis.visualize()
input_vis.visualize(save="examples/auto_encoder/train_transcoded_comparison.png")
