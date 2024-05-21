"""Example usage of the SimpleModel."""
from TimeSeriesDL.model import SimpleModel
from TimeSeriesDL.data import Dataset

# create a dataset loader which loads a matplotlib matrix from ./data/train.mat
data = Dataset(d_type="train")

# create a SimpleModel based on CNN/LSTM architecture, which predicts the next
# value of a sequence based on the last 128 values
simple_model = SimpleModel(input_size=128)

# train the model on the dataset for 5 epochs and log the progress in a CLI
# to review the model's training performance, open TensorBoard in a browser
simple_model.learn(train=data, epochs=5, verbose=True)

# save the model to its default location 'runs/{time_stamp}/model_SimpleModel.torch'
simple_model.save_to_default()

# create a new instance to load a trained model
new_instance = SimpleModel(input_size=128)
new_instance.load("runs/5-21-2024_203057/model_SimpleModel.torch")

# predict on a trained model using the corresponding method and ask the output
# to be a list instead of a tensor
new_instance.predict(data[0], as_list=True)
