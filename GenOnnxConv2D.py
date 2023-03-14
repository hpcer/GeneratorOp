import torch
import torch.nn as nn
import numpy as np

in_channels = 16
in_h = 486
in_w = 864

out_channels = 16

kernel_size = (3, 3)

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.Conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, bias = True)
    def forward(self, x):
        x = self.Conv2d(x)
        return x

model=Model()
model.eval() 

x=torch.randn((1, in_channels, in_h, in_w))

torch.onnx.export(model, # 搭建的网络
    x, # 输入张量
    'Conv2D.onnx', # 输出模型名称
    input_names=["input"], # 输入命名
    output_names=["output"], # 输出命名
    dynamic_axes={'input':{0:'batch'}, 'output':{0:'batch'}}  # 动态轴
)