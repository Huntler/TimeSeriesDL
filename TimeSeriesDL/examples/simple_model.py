"""Example usage of the SimpleModel."""
from torch.utils.data import DataLoader
from TimeSeriesDL.model import SimpleModel
from TimeSeriesDL.data import Dataset

# create a dataset loader which loads a matplotlib matrix from ./data/train.mat
# for this example, a custom path is set
data = Dataset(d_type="train", sequence_length=128, custom_path="./train.mat")
dataloader = DataLoader(data, batch_size=16)

# create a SimpleModel based on CNN/LSTM architecture, which predicts the next
# value of a sequence based on the last 128 values
# the example dataset only contains one value type, hence the input_size is 1
simple_model = SimpleModel(input_size=1)

# train the model on the dataset for 5 epochs and log the progress in a CLI
# to review the model's training performance, open TensorBoard in a browser
simple_model.learn(train=dataloader, epochs=5, verbose=True)

# save the model to its default location 'runs/{time_stamp}/model_SimpleModel.torch'
simple_model.save_to_default()

# create a new instance to load a trained model
new_instance = SimpleModel(input_size=128)
new_instance.load("runs/5-21-2024_203057/model_SimpleModel.torch")

# predict on a trained model using the corresponding method and ask the output
# to be a list instead of a tensor
new_instance.predict(data[0], as_list=True)
