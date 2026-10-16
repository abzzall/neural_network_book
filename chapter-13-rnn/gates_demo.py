"""One LSTM and GRU step, reconstructed from their documented gate equations."""

import torch
from torch import nn


def check_gates():
    torch.manual_seed(42)
    x = torch.randn(2, 3, dtype=torch.double)
    h = torch.randn(2, 4, dtype=torch.double)
    c = torch.randn(2, 4, dtype=torch.double)
    lstm = nn.LSTMCell(3, 4).double()
    gates = x @ lstm.weight_ih.T + lstm.bias_ih + h @ lstm.weight_hh.T + lstm.bias_hh
    i, f, g, o = gates.chunk(4, dim=-1)
    cell = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
    hidden = torch.sigmoid(o) * torch.tanh(cell)
    actual_h, actual_c = lstm(x, (h, c))
    lstm_error = max((hidden - actual_h).abs().max(), (cell - actual_c).abs().max()).item()
    gru = nn.GRUCell(3, 4).double()
    xr, xz, xn = (x @ gru.weight_ih.T + gru.bias_ih).chunk(3, dim=-1)
    hr, hz, hn = (h @ gru.weight_hh.T + gru.bias_hh).chunk(3, dim=-1)
    reset, update = torch.sigmoid(xr + hr), torch.sigmoid(xz + hz)
    candidate = torch.tanh(xn + reset * hn)
    expected = (1 - update) * candidate + update * h
    gru_error = (expected - gru(x, h)).abs().max().item()
    return lstm_error, gru_error


if __name__ == '__main__':
    lstm_error, gru_error = check_gates()
    print(f'LSTM gate error below 1e-12: {lstm_error < 1e-12}')
    print(f'GRU gate error below 1e-12: {gru_error < 1e-12}')
