import torch
import torch.nn as nn
import argparse

from GenRandom import GeneratorRandom
import numpy as np

class Depth2Space(torch.nn.Module):
    def __init__(self, block_size):
        super(Depth2Space, self).__init__()
        self.block_size = block_size
        self.PixelShuffle = nn.PixelShuffle(block_size)
    def forward(self, x):
        x = self.PixelShuffle(x)
        return x

def CreateD2S(args):
    # example usage
    input_tensor = torch.from_numpy(GeneratorRandom(args))
    # input_tensor = torch.randn(args['batch'], args['IC'], args['iH'], args['iW']) # input tensor with shape (N, C, H, W)
    block_size = args['BlockSize'] # block size for spatial shuffling
    depth2space = Depth2Space(block_size)
    output_tensor = depth2space(input_tensor)

    # export as onnx model
    dummy_input = torch.randn(args['batch'], args['IC'], args['iH'], args['iW'])
    torch.onnx.export(depth2space, dummy_input, "Depth2Space.onnx")
    np.save("Depth2Space_Input.npy", input_tensor.numpy())
    np.save("Depth2Space_Output.npy", output_tensor.numpy())


parser = argparse.ArgumentParser()
parser.add_argument('data', nargs='+', type=int, help='input data')
args = parser.parse_args()

# kind == 0: CRD --> torch PixelShuffle
# kind == 1: DCR --> not support

d2sArgsDic={'IC': 0, 'iH': 0, 'iW': 0, 'BlockSize': 2, 'D2SKind': 0, 'batch': 1}

d2sArgsList = list(d2sArgsDic.keys())[:len(args.data)]

tmpD2SArgs = dict(zip(d2sArgsList, args.data))

d2sArgsDic = {**d2sArgsDic, **tmpD2SArgs}

if d2sArgsDic['D2SKind'] != 0:
    print('Only support pixel shuffle, CRD(kind == 0), not support DCR(kind == 1)')
else:
    CreateD2S(d2sArgsDic)