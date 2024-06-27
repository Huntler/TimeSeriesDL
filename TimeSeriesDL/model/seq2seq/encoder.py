from torch import nn

class Encoder(nn.Module):
    def __init__(self, enc_feature_size, hidden_size, num_gru_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(enc_feature_size, hidden_size, num_gru_layers, batch_first=True, dropout=dropout)
        
    def forward(self, inputs):
        # inputs: (batch size, input seq len, num enc features)
        output, hidden = self.gru(inputs)
            
        # output: (batch size, input seq len, hidden size)
        # hidden: (num gru layers, batch size, hidden size)
        return output, hidden
