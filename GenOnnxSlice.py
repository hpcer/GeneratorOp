import torch
import torch.nn as nn
import numpy as np

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, src):
        out = src[:, :, :, 0:64]
        return out

model = Model()

model.eval()

x = torch.randn(1, 4, 128, 128)

torch.onnx.export(model,
    x,
    'slice.onnx',
    input_names = ["input"],
    output_names = ["output"],
    dynamic_axes={'input':{0:'batch'}, 'output':{0:'batch'}}
)


