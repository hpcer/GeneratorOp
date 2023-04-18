import tensorflow as tf
from tensorflow import keras

import numpy as np

import os
import argparse

def ConvertToTFLite(model, output_file):
    filename = os.path.abspath(output_file)
    out_dir  = os.path.dirname(filename)

    if not os.path.isdir(out_dir):
        raise RuntimeError('output %s directory does not exist' % output_dir)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    open(filename, "wb").write(tflite_model)
    return filename

def CreateConv2D(args):
    model = tf.keras.models.Sequential(
        [
            keras.layers.InputLayer(input_shape=(args['iH'], args['iW'], args['IC']), batch_size = args['batch'],
                dtype = 'float32', name="data"),

            # Keras not has group.
            keras.layers.Conv2D(
                filters = args['OC'], kernel_size=(args['kH'], args['kW']),
                strides=(args['sH'], args['sW']), padding = ('same' if args['padType'] == 0 else 'valid'),
                data_format="channels_last", name="conv1", use_bias = (True if args['hasBias'] == 1 else False)),
        ],
        name = "Conv2D"
    )

    in_shape  = model.get_layer('conv1').input_shape
    out_shape = model.get_layer('conv1').output_shape

    input  = tf.random.uniform(in_shape)
    output = tf.random.uniform(out_shape)

    model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
    model.fit(input, output)

    print('Conv2D Params =====>>>>>>>>')
    print('input shape: ', in_shape)
    print('kernel size: ', args['kH'], args['kW'])
    print('stride size: ', args['sH'], args['sW'])
    print('padding type: ', 'same' if args['padType'] == 0 else 'valid')
    print('output shape: ', out_shape)
    print('---------------------------')

    print(model.layers)

    model.summary()

    # save tensorflow model
    # if save_model:
    #     filename = SaveModel(model, model_path)
    #     print("Save model to %s" % filename)

    filename = ConvertToTFLite(model, './Conv2D.tflite')
    print("Generator tflite to %s" % filename)


parser = argparse.ArgumentParser()
parser.add_argument('data', nargs='+', type=int, help='input data')
args = parser.parse_args()

# padType 0 is same, 1 is valid

conv2DArgsDic={'IC': 0, 'iH': 0, 'iW': 0, 'OC': 0, 'kH': 0, 'kW': 0, 'sH': 0, 'sW': 0, 'hasBias': 0, 'batch': 1, 'dH': 1, 'dW': 1, 'group': 1, 'padType': 0}

conv2DArgsList = list(conv2DArgsDic.keys())[:len(args.data)]

tmpConv2DArgs = dict(zip(conv2DArgsList, args.data))

conv2DArgsDic = {**conv2DArgsDic, **tmpConv2DArgs}

CreateConv2D(conv2DArgsDic)