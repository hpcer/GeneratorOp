import tensorflow as tf
from tensorflow import keras

import numpy as np

import os
import argparse

def SaveModel(model, output_file):
    filename = os.path.abspath(output_file)
    out_dir  = os.path.dirname(filename)

    if not os.path.isdir(out_dir):
        raise RuntimeError('output %s directory does not exist' % output_dir)

    model.save(filename)
    return filename

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

def CreateDeConv2D(N, inC, outC, inH, inW, kernelH, kernelW,
    strideH, strideW, padding, save_model, cvt_tflite,
    model_path, tflite_path):
    model = tf.keras.models.Sequential(
        [
            keras.layers.InputLayer(input_shape=(inH, inW, inC), batch_size = N,
                dtype = 'float32', name="data"),

            keras.layers.Conv2DTranspose(
                filters = outC, kernel_size=(kernelH, kernelW),
                strides=(strideH, strideW), padding=padding,
                output_padding=0, data_format = "channels_last", name="deconv1",
                use_bias = False),
                # use_bias = true, convert to dlc(for snpe) will failed
                # not support group
                # padding: 'same' or 'valid'
        ],
        name = 'DeConv2D'
    )

    in_shape  = model.get_layer('deconv1').input_shape
    out_shape = model.get_layer('deconv1').output_shape
    # input = np.random.rand(1, inH, inW, inC)
    # output = np.random.rand(outShape)
    input  = tf.random.uniform(in_shape)
    output = tf.random.uniform(out_shape)

    model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
    model.fit(input, output)

    print('DeConv2D Params =====>>>>>>>>')
    print('input shape: ', in_shape)
    print('kernel size: ', kernelH, kernelW)
    print('stride size: ', strideH, strideW)
    print('padding type: ', padding)
    print('output shape: ', out_shape)
    print('---------------------------')

    print(model.layers)

    model.summary()

    if save_model:
        filename = SaveModel(model, model_path)
        print("Save model to %s" % filename)

    if cvt_tflite:
        filename = ConvertToTFLite(model, tflite_path)
        print("Generator tflite to %s" % filename)

def CreateSoftmax(N, inC, inH, inW, axis, save_model, cvt_tflite, model_path, tflite_path):
    model = tf.keras.models.Sequential(
        [
            keras.layers.InputLayer(input_shape=(inH, inW, inC), batch_size = N,
                dtype = 'float32', name="data"),

            keras.layers.Softmax(axis, name="softmax1"),
                # use_bias = true, convert to dlc(for snpe) will failed
                # not support group
                # padding: 'same' or 'valid'
        ],
        name = 'Softmax'
    )

    in_shape  = model.get_layer('softmax1').input_shape
    out_shape = model.get_layer('softmax1').output_shape
    # input = np.random.rand(1, inH, inW, inC)
    # output = np.random.rand(outShape)
    input  = tf.random.uniform(in_shape)
    output = tf.random.uniform(out_shape)

    model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
    model.fit(input, output)

    print('Softmax Params =====>>>>>>>>')
    print('input shape: ', in_shape)
    print('axis: ', axis)
    print('output shape: ', out_shape)
    print('---------------------------')

    print(model.layers)

    model.summary()

    if save_model:
        filename = SaveModel(model, model_path)
        print("Save model to %s" % filename)

    if cvt_tflite:
        filename = ConvertToTFLite(model, tflite_path)
        print("Generator tflite to %s" % filename)


def CreateConv2D(N, inC, outC, inH, inW, kernelH, kernelW, strideH, strideW, padding,
    save_model, cvt_tflite, model_path, tflite_path):
    model = tf.keras.models.Sequential(
        [
            keras.layers.InputLayer(input_shape=(inH, inW, inC), batch_size = N,
                dtype = 'float32', name="data"),

            keras.layers.Conv2D(
                filters = outC, kernel_size=(kernelH, kernelW),
                strides=(strideH, strideW), padding = padding,
                data_format="channels_last", name="conv1", use_bias = False),
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
    print('kernel size: ', kernelH, kernelW)
    print('stride size: ', strideH, strideW)
    print('padding type: ', padding)
    print('output shape: ', out_shape)
    print('---------------------------')

    print(model.layers)

    model.summary()

    if save_model:
        filename = SaveModel(model, model_path)
        print("Save model to %s" % filename)

    if cvt_tflite:
        filename = ConvertToTFLite(model, tflite_path)
        print("Generator tflite to %s" % filename)


def main():
    parser = argparse.ArgumentParser(description="Create model file of tensoflow or tflite.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-mt', '--model_type',
                        help='Generator model type.',
                        default='DeConv2D[support DeConv2D/Softmax/Conv2D]')

    parser.add_argument('-sm', '--save_model',
                        help='Is save tensorflow model.',
                        type=bool,
                        default=False)

    parser.add_argument('-smf', '--save_model_file',
                        help='Model file dir to save tensoflow model.',
                        default='./model')

    parser.add_argument('-st', '--save_tflite',
                        help='Is save tflite model.',
                        type=bool,
                        default=True)

    parser.add_argument('-stf', '--save_tflite_file',
                        help='TFLite model file dir to save file.',
                        default='./model.tflite')

    parser.add_argument('-in', '--input_batches',
                        help='input batches.',
                        type=int,
                        default=1)

    parser.add_argument('-ic', '--input_channel',
                        help='input channels.',
                        type=int,
                        default=1)

    parser.add_argument('-ih', '--input_height',
                        help='input height.',
                        type=int,
                        default=32)

    parser.add_argument('-iw', '--input_width',
                        help='input width.',
                        type=int,
                        default=32)

    parser.add_argument('-oc', '--output_channel',
                        help='output channel.',
                        type=int,
                        default=1)

    parser.add_argument('-kh', '--kernel_height',
                        help='kernel height.',
                        type=int,
                        default=3)

    parser.add_argument('-kw', '--kernel_width',
                        help='kernel width.',
                        type=int,
                        default=3)

    parser.add_argument('-sh', '--stride_height',
                        help='stride height.',
                        type=int,
                        default=2)

    parser.add_argument('-sw', '--stride_width',
                        help='stride width.',
                        type=int,
                        default=2)

    parser.add_argument('-pt', '--pad_type',
                        help='DeConv2D padding type[same/valid].',
                        default='same')

    parser.add_argument('-axis', '--axis',
                        help="axis",
                        type=int,
                        default=-1)

    args = parser.parse_args()

    print(tf.version.VERSION)

    if args.model_type == 'DeConv2D':
        CreateDeConv2D(args.input_batches, args.input_channel, args.output_channel,
            args.input_height, args.input_width, args.kernel_height, args.kernel_width,
            args.stride_height, args.stride_width, args.pad_type, args.save_model,
            args.save_tflite, args.save_model_file, args.save_tflite_file)

    if args.model_type == 'Conv2D':
        CreateConv2D(args.input_batches, args.input_channel, args.output_channel,
            args.input_height, args.input_width, args.kernel_height, args.kernel_width,
            args.stride_height, args.stride_width, args.pad_type, args.save_model,
            args.save_tflite, args.save_model_file, args.save_tflite_file)

    if args.model_type == 'Softmax':
        CreateSoftmax(args.input_batches, args.input_channel, args.input_height, args.input_width,
            args.axis, args.save_model, args.save_tflite, args.save_model_file, args.save_tflite_file)

if __name__ == '__main__':
    main()




# model = keras.Sequential()
# model.add(keras.Input(shape=(64, 224, 224), batch_size = 1, dtype = 'float32'))
# model.add(keras.layers.Conv2DTranspose(
#         32, 3, (2, 2), padding='same',
#         output_padding=0, data_format = "channels_first") )

# def grouped_conv2d_transpose(inputs, filters, kernel_size, strides, groups):
#     """Performs grouped transposed convolution.

#     Args:
#         inputs: A `Tensor` of shape `[batch_size, h, w, c]`.
#         filters: The number of convolutional filters.
#         kernel_size: The spatial size of the convolutional kernel.
#         strides: The convolutional stride.
#         groups: The number of groups to use in the grouped convolution step.
#             The input channel count needs to be evenly divisible by `groups`.
#     Returns:
#         A `Tensor` of shape `[batch_size, new_h, new_w, filters]`.
#     """
#     splits = tf.split(inputs, groups, axis=-1)
#     convolved_splits = [
#         tf.keras.layers.Conv2DTranspose(
#             filters // groups,
#             kernel_size,
#             strides
#         )(split) for split in splits
#     ]
#     return tf.concat(convolved_splits, -1)