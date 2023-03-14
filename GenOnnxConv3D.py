import torch
import torch.nn as nn
import numpy as np

in_channels = 32
in_h = 128
in_w = 128
depth = 32

out_channels = 16

kernel_size_0 = 3
kernel_size_1 = 3
kernel_size_2 = 3

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.Conv3D = nn.Conv3d(in_channels, out_channels, (kernel_size_0, kernel_size_1, kernel_size_2), stride = 1, padding = 0, dilation = 1, groups = 1, bias = True)

    def forward(self, src):
        x = self.Conv3D(src)
        return x

model = Model()

model.eval()

x = torch.rand(1, in_channels, depth, in_h, in_w)


torch.onnx.export(model,
    x,
    'conv3d.onnx',
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={'input':{0:'batch'}, 'output':{0:'batch'}}  # 动态轴
)

