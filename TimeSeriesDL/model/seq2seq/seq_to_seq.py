import math
import torch
import torch.nn.functional as F

from TimeSeriesDL.model.base_model import BaseModel
from TimeSeriesDL.utils.activations import get_activation_from_string
from TimeSeriesDL.utils.register import model_register

from .encoder import Encoder
from .decoder_attention import DecoderWithAttention
from .decoder_vanilla import DecoderVanilla

class Seq2Seq(BaseModel):
    def __init__(self,
                 in_features: int = 1,
                 hidden_dim: int = 64,
                 gru_layers: int = 1,
                 dropout: float = 0.1,
                 probabilistic: bool = True,
                 attention: bool = True,
                 teacher_force_decay: float = None,
                 out_act: str = "sigmoid",
                 loss: str = "GaussianNLLLoss",
                 optimizer: str = "Adam",
                 lr: float = 1e-3):
        super().__init__(loss, optimizer, lr)
        self.encoder = Encoder(in_features, hidden_dim, gru_layers, dropout)
        
        dec_params = {
            "dec_feature_size": in_features,
            "dec_target_size": in_features,
            "dropout": dropout,
            "dist_size": 2 if probabilistic else 1,
            "probabilistic": probabilistic,
            "num_gru_layers": gru_layers,
            "hidden_dim": hidden_dim
        }
        self.decoder = DecoderWithAttention(**dec_params) if attention else DecoderVanilla(**dec_params)
        self.decoder.output_sampler(Seq2Seq.sample_from_output)

        self._last_activation = get_activation_from_string(out_act)

        self._epoch = 0
        self._teacher_force_decay = teacher_force_decay
        self._teacher_force = 1
        self.probabilistic = probabilistic
    
    @staticmethod
    def compute_smape(prediction, target):
        return torch.mean(torch.abs(prediction - target) / ((torch.abs(target) + torch.abs(prediction)) / 2. + 1e-8)) * 100.
    
    @staticmethod
    def get_dist_params(output):
        mu = output[:, :, :, 0]
        # softplus to constrain to positive
        sigma = F.softplus(output[:, :, :, 1])
        return mu.to(output.device), sigma.to(output.device)
    
    @staticmethod
    def sample_from_output(output):
        # in - output: (batch size, dec seq len, num targets, num dist params)
        # out - output: (batch size, dec seq len, num targets)
        if output.shape[-1] > 1:  # probabilistic can be assumed
            mu, sigma = Seq2Seq.get_dist_params(output)
            return torch.normal(mu, sigma)
        # No sample just reshape if pointwise
        return output.squeeze(-1)

    def _inverse_sigmoid_decay(decay):
        # inverse sigmoid decay from https://arxiv.org/pdf/1506.03099.pdf
        def compute(indx):
            return decay / (decay + math.exp(indx / decay))
        return compute

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self._epoch += 1
        self._teacher_force = self._inverse_sigmoid_decay(self._teacher_force_decay)(self._epoch)
        self.log("epoch/teacher_force", self._teacher_force)
    
    def forward(self, batch: torch.tensor):
        # enc_inputs: (batch size, input seq length, num enc features)
        # dec_inputs: (batch size, output seq length, num dec features)
        enc_inputs, dec_inputs = batch
        
        # enc_outputs: (batch size, input seq len, hidden size)
        # hidden: (num gru layers, batch size, hidden dim), ie the last hidden state
        enc_outputs, hidden = self.encoder(enc_inputs)
        
        # outputs: (batch size, output seq len, num targets, num dist params)
        outputs = self.decoder(dec_inputs, hidden, enc_outputs, self._teacher_force)
        
        return outputs

    def _compute_loss(self, prediction, target):
        # prediction: (batch size, dec seq len, num targets, num dist params)
        # target: (batch size, dec seq len, num targets)
        if self.probabilistic:
            mu, sigma = Seq2Seq.get_dist_params(prediction)
            var = sigma ** 2
            loss = self._loss(mu, target, var)
        else:
            loss = self._loss(prediction.squeeze(-1), target)
        return loss
    
    def training_step(self, batch: torch.Tensor, **kwargs) -> torch.Tensor:
        # unpack the batch and hand the input to the model
        x, y = batch
        y_hat = self(x)

        # calculate the loss of the model's prediction
        loss = self._compute_loss(y_hat, y)
        self.log(f"train/{self._loss_name}", loss)

        # forward the loss to lighning's optimizer
        return loss

    def test_step(self, batch: torch.Tensor, **kwargs) -> torch.Tensor:
        # unpack the batch and hand the input to the model
        x, y = batch
        y_hat = self(x)

        # calculate the loss of the model's prediction
        loss = self._compute_loss(y_hat, y)
        self.log(f"test/{self._loss_name}", loss)
        return loss

    def validation_step(self, batch: torch.Tensor, **kwargs) -> torch.Tensor:
        # unpack the batch and hand the input to the model
        x, y = batch
        y_hat = self(x)

        # calculate the loss of the model's prediction
        loss = self._compute_loss(y_hat, y)
        self.log(f"validate/{self._loss_name}", loss)
        return loss

model_register.register_model("Seq2Seq", Seq2Seq)
