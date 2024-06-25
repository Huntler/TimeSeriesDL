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
                 teacher_force: float = None,
                 out_act: str = "sigmoid",
                 loss: str = "GaussianNLLLoss",
                 optimizer: str = "Adam",
                 lr: float = 1e-3):
        super().__init__(loss, optimizer, lr)
        self.encoder = Encoder(in_features, hidden_dim, gru_layers, dropout)
        self.decoder = DecoderWithAttention() if attention else DecoderVanilla()

        self._last_activation = get_activation_from_string(out_act)

        self._teacher_force = teacher_force
        self.probabilistic = probabilistic
    
    @staticmethod
    def compute_smape(prediction, target):
        return torch.mean(torch.abs(prediction - target) / ((torch.abs(target) + torch.abs(prediction)) / 2. + 1e-8)) * 100.
    
    @staticmethod
    def get_dist_params(output):
        mu = output[:, :, :, 0]
        # softplus to constrain to positive
        sigma = F.softplus(output[:, :, :, 1])
        return mu, sigma
    
    @staticmethod
    def sample_from_output(output):
        # in - output: (batch size, dec seq len, num targets, num dist params)
        # out - output: (batch size, dec seq len, num targets)
        if output.shape[-1] > 1:  # probabilistic can be assumed
            mu, sigma = Seq2Seq.get_dist_params(output)
            return torch.normal(mu, sigma)
        # No sample just reshape if pointwise
        return output.squeeze(-1)
    
    def forward(self, enc_inputs, dec_inputs):
        # enc_inputs: (batch size, input seq length, num enc features)
        # dec_inputs: (batch size, output seq length, num dec features)
        
        # enc_outputs: (batch size, input seq len, hidden size)
        # hidden: (num gru layers, batch size, hidden dim), ie the last hidden state
        enc_outputs, hidden = self.encoder(enc_inputs)
        
        # outputs: (batch size, output seq len, num targets, num dist params)
        outputs = self.decoder(dec_inputs, hidden, enc_outputs, self._teacher_force)
        
        return outputs

    def compute_loss(self, prediction, target, override_func=None):
        # prediction: (batch size, dec seq len, num targets, num dist params)
        # target: (batch size, dec seq len, num targets)
        if self.probabilistic:
            mu, sigma = Seq2Seq.get_dist_params(prediction)
            var = sigma ** 2
            loss = self.loss_func(mu, target, var)
        else:
            loss = self.loss_func(prediction.squeeze(-1), target)
        return loss if self.training else loss.item()
    
    def optimize(self, prediction, target):
        # prediction & target: (batch size, seq len, output dim)
        self.opt.zero_grad()
        loss = self.compute_loss(prediction, target)
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.opt.step()
        return loss.item()

model_register.register_model("Seq2Seq", Seq2Seq)
