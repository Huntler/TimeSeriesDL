import torch
from torch import nn
from torch.nn import functional as F

from .decoder_base import DecoderBase


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # the hidden size for the output of attn (and input of v) can actually be any number  
        # Also, using two layers allows for a non-linear act func inbetween
        self.attn = nn.Linear(2 * hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, decoder_hidden_final_layer, encoder_outputs):
        # decoder_hidden_final_layer: (batch size, hidden size)
        # encoder_outputs: (batch size, input seq len, hidden size)
        
        # Repeat decoder hidden state input seq len times
        hidden = decoder_hidden_final_layer.unsqueeze(1).repeat(1, encoder_outputs.shape[1], 1)
        
        # Compare decoder hidden state with each encoder output using a learnable tanh layer
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        
        # Then compress into single values for each comparison (energy)
        attention = self.v(energy).squeeze(2)
        
        # Then softmax so the weightings add up to 1
        weightings = F.softmax(attention, dim=1)
                
        # weightings: (batch size, input seq len)
        return weightings


class DecoderWithAttention(DecoderBase):
    def __init__(self, dec_feature_size, dec_target_size, hidden_size, 
                 num_gru_layers, target_indices, dropout, dist_size,
                 probabilistic):
        super().__init__(dec_target_size, target_indices, dist_size, probabilistic)
        self.attention_model = Attention(hidden_size, num_gru_layers)
        # GRU takes previous timestep target and weighted sum of encoder hidden states
        self.gru = nn.GRU(dec_feature_size + hidden_size, hidden_size, num_gru_layers, batch_first=True, dropout=dropout)
        # Output layer takes decoder hidden state output, weighted sum and decoder input
        # Feeding decoder input into the output layer essentially acts as a skip connection
        self.out = nn.Linear(hidden_size + hidden_size + dec_feature_size, dec_target_size * dist_size)

    def run_single_recurrent_step(self, inputs, hidden, enc_outputs):
        # inputs: (batch size, 1, num dec features)
        # hidden: (num gru layers, batch size, hidden size)
        # enc_outputs: (batch size, input seq len, hidden size)
        
        # Get attention weightings
        # weightings: (batch size, input seq len)
        weightings = self.attention_model(hidden[-1], enc_outputs)
        
        # Then compute weighted sum
        # weighted_sum: (batch size, 1, hidden size)
        weighted_sum = torch.bmm(weightings.unsqueeze(1), enc_outputs)
                
        # Then input into GRU
        # gru inputs: (batch size, 1, num dec features + hidden size)
        # output: (batch size, 1, hidden size)
        output, hidden = self.gru(torch.cat((inputs, weighted_sum), dim=2), hidden)
            
        # Get prediction
        # out input: (batch size, 1, hidden size + hidden size + num targets)
        output = self.out(torch.cat((output, weighted_sum, inputs), dim=2))
        output = output.reshape(output.shape[0], output.shape[1], self.target_size, self.dist_size)
        
        # output: (batch size, 1, num targets, num dist params)
        # hidden: (num gru layers, batch size, hidden size)
        return output, hidden
