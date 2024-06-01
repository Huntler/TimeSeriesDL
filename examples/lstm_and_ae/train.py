"""Example usage of the any model."""
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from TimeSeriesDL.model import BaseModel
from TimeSeriesDL.data import Dataset, encode_dataset
from TimeSeriesDL.model import ConvAE, LSTM
from TimeSeriesDL.utils import config

def train(path: str) -> BaseModel:
    """Trains based on a given config.

    Args:
        path (str): The path to a train config.

    Returns:
        BaseModel: The trained model.
    """
    # load training arguments (equals example/simple_model.py)
    train_args = config.get_args(path)

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
    model.learn(train=dataloader, epochs=train_args["train_epochs"], verbose=True)

    # save the model to its default location 'runs/{time_stamp}/model_SimpleModel.torch'
    model.save_to_default()

    # also, store a modified copy of the training arguments containing the model path
    # this makes comparisons between multiple experiments easier<
    train_args["model_path"] = model.log_path + "/model.torch"
    config.store_args(f"{model.log_path}/config.yml", train_args)
    return model

if __name__ == "__main__":
    # train the auto encoder and encode the dataset
    print("Train the ConvAE")
    ae: ConvAE = train("./examples/lstm_and_ae/ae_config.yaml")
    print(f"Latent space shape is {ae.latent_space_shape}")

    print("\nEncode dataset")
    encoded: np.array = encode_dataset(config.get_args(ae.log_path + "/config.yml"))

    # train the lstm on the encoded dataset, then decode: ae.decode(lstm.predict(x))
    print("\nTrain LSTM")
    lstm: LSTM = train("./examples/lstm_and_ae/lstm_config.yaml")

    # use trained lstm to predict on a dataset
    print("\nPredict using LSTM/Decoder")
    sequence, latent_features = ae.latent_space_shape
    print(encoded.shape)
    for i in range(0, encoded.shape[1], sequence):
        x = torch.tensor([encoded[:, i:i + sequence + 1]])
        x = torch.swapaxes(x, 1, 2)
        x = lstm.predict(x, as_array=True)

        if i + sequence + 1 < encoded.shape[1]:
            encoded[:, i + sequence + 1] = x
    print(encoded.shape)

    decoded = []
    for i in range(0, encoded.shape[1], sequence):
        x = torch.tensor([encoded[:, i:i + sequence]])
        if x.shape[2] == sequence:
            x = ae.decode(x)
            decoded += list(x.cpu().detach().numpy()[0])

    decoded = np.array(decoded)
    print(decoded.shape)

    fig, ax = plt.subplots()
    x = np.linspace(0.5, 3.5, len(decoded[:, 0]))
    ax.scatter(x, decoded[:, 0], c="tab:blue", label="test_1", alpha=0.3, edgecolors='none')
    ax.scatter(x, decoded[:, 1], c="tab:red", label="test_2", alpha=0.3, edgecolors='none')

    ax.legend()
    ax.grid(True)

    plt.show()
