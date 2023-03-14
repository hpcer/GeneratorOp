import torch
import torch.nn as nn
import numpy as np

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.softmax = nn.Softmax(dim=2)
    def forward(self, x):
        x = self.softmax(x)
        return x

model=Model()
model.eval() 

x=torch.randn((4, 32, 64, 64))

torch.onnx.export(model, # 搭建的网络
    x, # 输入张量
    'Softmax.onnx', # 输出模型名称
    input_names=["input"], # 输入命名
    output_names=["output"], # 输出命名
    dynamic_axes={'input':{0:'batch'}, 'output':{0:'batch'}}  # 动态轴
)