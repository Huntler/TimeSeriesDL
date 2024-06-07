"""Example usage of the any model."""
import json
from torch.utils.data import DataLoader
from TimeSeriesDL.debug.visualize_cnn import VisualizeConv
from TimeSeriesDL.loss.measurement_suite import LossMeasurementSuite
from TimeSeriesDL.model import BaseModel
from TimeSeriesDL.data import Dataset
from TimeSeriesDL.utils import config, ModelTrainer
from TimeSeriesDL.debug.visualize_dataset import VisualizeDataset

# load training arguments (equals example/simple_model.py)
train_args = config.get_args("./examples/simple/config.yaml")

# create the loss suite handling loss calculation for the
# backpropagation and logging
loss_suite = LossMeasurementSuite(**train_args["loss_suite"])

# create a dataset loader which loads one or multiple scipy matrices
data = Dataset(**train_args["dataset"])
dataloader = DataLoader(data, **train_args["dataloader"])

# create a testset, used to optimize parameters of the model
test = Dataset(**train_args["testset"])
testloader = DataLoader(test, batch_size=train_args["dataloader"]["batch_size"])

trainer = ModelTrainer(**train_args["trainer"])
trainer.set_dataset(dataloader)
trainer.set_testset(testloader)
trainer.set_loss_suite(loss_suite)

# create a model based on what is defined in the config
# to do so, a model needs to be registered using config.register_model()
model_name = train_args["model_name"]
model: BaseModel = config.get_model(model_name)(**train_args["model"])
model.use_device(train_args["device"])

# also, store a modified copy of the training arguments containing the model path
# this makes comparisons between multiple experiments easier
train_args["model_path"] = model.log_path + "/models/model.torch"
config.store_args(f"{model.log_path}/config.yml", train_args)

# train the model using the cvonfigured trainer
trainer.train(model)

# test the model and store the accuracies
result = trainer.test(model)
with open(model.log_path + "/test_results.json", "w", encoding="UTF-8") as f:
    json.dump(result, f, indent=-1)

# save the model to its default location 'runs/{time_stamp}/model_SimpleModel.torch'
model.save_to_default()

# visualize the train data
if data.num_matrices == 1:
    test_vis = VisualizeDataset(data, name="Input")
    test_vis.generate_overlay(model)

    test_vis.set_feature(list(range(len(data.label_names))))
    test_vis.visualize(save=f"{model.log_path}/predict_on_train.png")

# visualize the test data
if test.num_matrices == 1:
    test_vis = VisualizeDataset(test, name="Input")
    test_vis.generate_overlay(model)

    test_vis.set_feature(list(range(len(data.label_names))))
    test_vis.visualize(save=f"{model.log_path}/predict_on_test.png")

# visualize the model
model.use_device("cpu")
vis = VisualizeConv(model)
vis.visualize(f"{model.log_path}/analysis.png")
