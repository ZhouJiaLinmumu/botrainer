import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class PADRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super(PADRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, x_lengths):
        x_pack = pack_padded_sequence(input, x_lengths, batch_first=True)
        output, (h_n, c_n) = self.lstm(x_pack)
        h_n = h_n.permute(1, 0, 2)
        output = self.fc(h_n)
        output = output.squeeze(1)
        return output