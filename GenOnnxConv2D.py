import torch
import torch.nn as nn
import numpy as np
import argparse

from GenRandom import GeneratorRandom


class Model(nn.Module):
    def __init__(self, args):
        super(Model,self).__init__()
        in_channels = args['IC']
        out_channels = args['OC']
        kernel_size = (args['kH'], args['kW'])
        stride = (args['sH'], args['sW'])
        padding = 'same' if (args['padType'] == 0) else 'valid'
        self.Conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias = True if (args['hasBias'] == 1) else False)
    def forward(self, x):
        x = self.Conv2d(x)
        return x

def CreateConv2D(args):
    model=Model(args)
    
    x = torch.from_numpy(GeneratorRandom(args))
    y = model(x)

    # model.eval() 

    torch.onnx.export(model, # 搭建的网络
        x, # 输入张量
        'Conv2D.onnx', # 输出模型名称
        input_names=["input"], # 输入命名
        output_names=["output"], # 输出命名
        dynamic_axes={'input':{0:'batch'}, 'output':{0:'batch'}}  # 动态轴
    )

    np.save("Conv2D_Input.npy", x.numpy())
    np.save("Conv2D_Output.npy", y.detach().numpy())

parser = argparse.ArgumentParser()
parser.add_argument('data', nargs='+', type=int, help='input data')
args = parser.parse_args()

conv2DArgsDic={'IC': 0, 'iH': 0, 'iW': 0, 'OC': 0, 'kH': 0, 'kW': 0, 'sH': 0, 'sW': 0, 'hasBias': 0, 'batch': 1, 'dH': 1, 'dW': 1, 'group': 1, 'padType': 0}

conv2DArgsList = list(conv2DArgsDic.keys())[:len(args.data)]

tmpConv2DArgs = dict(zip(conv2DArgsList, args.data))

conv2DArgsDic = {**conv2DArgsDic, **tmpConv2DArgs}

CreateConv2D(conv2DArgsDic)