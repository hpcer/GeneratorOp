import torch
import torch.nn as nn
import argparse

from GenRandom import GeneratorRandom
import numpy as np

class Space2Depth(torch.nn.Module):
    def __init__(self, block_size):
        super(Space2Depth, self).__init__()
        self.block_size = block_size
    def forward(self, x):
        n, c, h, w = x.size()
        unfolded_x = x.view(n, c, h // self.block_size, self.block_size, w // self.block_size, self.block_size)
        transposed_x = unfolded_x.permute(0, 1, 3, 5, 2, 4)
        return transposed_x.reshape(n, c * (self.block_size ** 2), h // self.block_size, w // self.block_size)

def CreateS2D(args):
    # example usage
    input_tensor = torch.from_numpy(GeneratorRandom(args))
    print(input_tensor.size())
    # input_tensor = torch.randn(args['batch'], args['IC'], args['iH'], args['iW']) # input tensor with shape (N, C, H, W)
    block_size = args['BlockSize'] # block size for spatial shuffling
    space2depth = Space2Depth(block_size)
    output_tensor = space2depth(input_tensor)

    # export as onnx model
    dummy_input = torch.randn(args['batch'], args['IC'], args['iH'], args['iW'])
    torch.onnx.export(space2depth, dummy_input, "Space2Depth.onnx")

    np.save("Space2Depth_Input.npy", input_tensor.numpy())
    np.save("Space2Depth_Output.npy", output_tensor.numpy())


parser = argparse.ArgumentParser()
parser.add_argument('data', nargs='+', type=int, help='IC iH iW BlockSize batch')
args = parser.parse_args()

# kind == 0: CRD --> torch PixelShuffle
# kind == 1: DCR --> not support

s2dArgsDic={'IC': 0, 'iH': 0, 'iW': 0, 'BlockSize': 2, 'batch': 1}

s2dArgsList = list(s2dArgsDic.keys())[:len(args.data)]

tmpS2DArgs = dict(zip(s2dArgsList, args.data))

s2dArgsDic = {**s2dArgsDic, **tmpS2DArgs}

print(s2dArgsDic)

CreateS2D(s2dArgsDic)
