"""Example usage of the any model."""
from torch.utils.data import DataLoader
from TimeSeriesDL.model import BaseModel
from TimeSeriesDL.data import Dataset, encode_dataset, decode_dataset, AutoEncoderCollate
from TimeSeriesDL.utils import config, ModelTrainer
from TimeSeriesDL.loss.measurement_suite import LossMeasurementSuite
from TimeSeriesDL.debug.visualize_dataset import VisualizeDataset
from TimeSeriesDL.debug.visualize_cnn import VisualizeConv


def train(path: str, collate = None) -> BaseModel:
    """Trains based on a given config.

    Args:
        path (str): The path to a train config.
        collate_fn (Callable): The coolate function used by the dataloader.

    Returns:
        BaseModel: The trained model.
    """
    # load training arguments (equals example/simple_model.py)
    train_args = config.get_args(path)

    # create the loss suite handling loss calculation for the
    # backpropagation and logging
    loss_suite = LossMeasurementSuite(**train_args["loss_suite"])

    # create a dataset loader which loads a matplotlib matrix from ./train.mat
    _data = Dataset(**train_args["dataset"])
    dataloader = DataLoader(_data, **train_args["dataloader"])
    if collate:
        dataloader = DataLoader(_data, collate_fn=collate, **train_args["dataloader"])

    # set up the trainer
    trainer = ModelTrainer(**train_args["trainer"])
    trainer.set_dataset(dataloader)
    trainer.set_loss_suite(loss_suite)

    # create a model based on what is defined in the config
    # to do so, a model needs to be registered using config.register_model()
    model_name = train_args["model_name"]
    model: BaseModel = config.get_model(model_name)(**train_args["model"])
    model.use_device(train_args["device"])

    # also, store a modified copy of the training arguments containing the model path
    # this makes comparisons between multiple experiments easier<
    train_args["model_path"] = model.log_path + "/models/model.torch"
    config.store_args(f"{model.log_path}/config.yml", train_args)

    # train the model using the cvonfigured trainer
    trainer.train(model)

    # save the model to its default location 'runs/{time_stamp}/model_SimpleModel.torch'
    model.save_to_default()
    return model, _data, train_args


if __name__ == "__main__":
    # train the auto encoder and encode the dataset
    print("Train the ConvAE")
    ae, data, ae_args = train("./examples/lstm_and_ae/ae_config.yaml")

    print("\nEncode dataset")
    export = "runs/" + ae_args["model"]["tag"]
    encode_dataset(train_args=config.get_args(ae.log_path + "/config.yml"),
                   export_path=export + "/encoded.mat")

    # train the lstm on the encoded dataset
    print("\nTrain LSTM")
    collate_fn = AutoEncoderCollate(ae, device=ae.device).collate_fn()
    lstm, encoded, lstm_args = train("./examples/lstm_and_ae/lstm_config.yaml", collate_fn)

    print("\nApply LSTM")
    encoded.apply(lstm)
    data.save(export + "/prediction.mat")

    print("\nDecode prediction")
    ae_args["dataset"]["custom_path"] = export + "/prediction.mat"
    decode_dataset(ae_args, data.scale_back, export_path=export + "/decoded.mat")

    ae_args["dataset"]["custom_path"] = export + "/decoded.mat"
    ae_args["dataset"]["ae_mode"] = False
    pred = Dataset(**ae_args["dataset"])

    # visualize the test data
    visualization = VisualizeDataset(data, name="Input")
    overlay = VisualizeDataset(pred, name="Prediction", overlay_mode=True)

    visualization.set_overlay(overlay)
    visualization.set_feature(list(range(len(data.label_names))))
    visualization.visualize(save=f"{export}/predict_on_test.png")

    # visualize first layer of AE
    ae.use_device("cpu")
    vis = VisualizeConv(ae)
    vis.visualize(f"{ae.log_path}/analysis.png")
