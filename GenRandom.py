import numpy as np
import argparse

def GeneratorRandom(args):
    a = np.random.rand(args['batch'], args['IC'], args['iH'], args['iW']).astype(np.float32)
    return a

# argsDic = {"IC": 0, "iH": 0, "iW": 0, "batch": 1}