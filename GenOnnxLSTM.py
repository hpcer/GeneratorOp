import torch
import torch.nn as nn
import numpy as np

# L sequence lenth
# D 2 if bidirectional= True otherwise 1
# N batch size
# H_in input_size
# H_cell hidden_size
# H_out proj_size if proj_size > 0 otherwise hidden_size

L = 1024

D = 1

N = 1

H_in = 256

H_cell = 64 # OCR

num_layers=1 #default

input = torch.randn(N, L, H_in)

h0 = torch.randn(D * num_layers, N, H_cell)

c0 = torch.randn(D * num_layers, N, H_cell)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.LSTM = nn.LSTM(input_size = H_in, hidden_size = H_cell, num_layers=num_layers, batch_first=True, bidirectional = False)

    def forward(self, in0, h):
        x = self.LSTM(in0, h)
        return x

model = Model()

model.eval()

torch.onnx.export(
    model, (input, (h0, c0)), 'lstm.onnx',
    input_names=['input', 'h0', 'c0'],
    output_names=['output', 'hn', 'cn'],
    dynamic_axes={'input': {0: 'sequence'}, 'output': {0: 'sequence'}}
    )