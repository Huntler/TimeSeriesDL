import random
import torch
from torch import nn
import lightning as L

# Decoder superclass whose forward is called by Seq2Seq but other methods filled out by subclasses
class DecoderBase(L.LightningModule):
    def __init__(self, dec_target_size, dist_size, probabilistic):
        super().__init__()
        self.target_size = dec_target_size
        self.dist_size = dist_size
        self.probabilistic = probabilistic
        self._sample_from_output = lambda x: x
    
    def output_sampler(self, func) -> None:
        self._sample_from_output = func
    
    # Have to run one step at a time unlike with the encoder since sometimes not teacher forcing
    def run_single_recurrent_step(self, inputs, hidden, enc_outputs):
        raise NotImplementedError()
    
    def forward(self, inputs, hidden, enc_outputs, teacher_force_prob=None):
        # inputs: (batch size, output seq length, num dec features)
        # hidden: (num gru layers, batch size, hidden dim), ie the last hidden state
        # enc_outputs: (batch size, input seq len, hidden size)
        
        batch_size, dec_output_seq_length, _ = inputs.shape
        
        # Store decoder outputs
        # outputs: (batch size, output seq len, num targets, num dist params)
        outputs = torch.zeros(batch_size, dec_output_seq_length, self.target_size, self.dist_size, dtype=torch.float)

        # curr_input: (batch size, 1, num dec features)
        curr_input = inputs[:, 0:1, :]
        
        for t in range(dec_output_seq_length):
            # dec_output: (batch size, 1, num targets, num dist params)
            # hidden: (num gru layers, batch size, hidden size)
            dec_output, hidden = self.run_single_recurrent_step(curr_input, hidden, enc_outputs)
            # Save prediction
            outputs[:, t:t+1, :, :] = dec_output
            # dec_output: (batch size, 1, num targets)
            dec_output = self._sample_from_output(dec_output)
            
            # If teacher forcing, use target from this timestep as next input o.w. use prediction
            teacher_force = random.random() < teacher_force_prob if teacher_force_prob is not None else False
            
            curr_input = inputs[:, t:t+1, :].clone()
            if not teacher_force:
                curr_input[:, :, :] = dec_output
        return outputs.to(inputs.device)
