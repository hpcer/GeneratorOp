import tensorflow as tf
from tensorflow import keras

import numpy as np

import os
import argparse

batches = 1

input = (4, 16, 16, 8) # D H W C

outC = 4

kernel = (3, 3, 3) # D H W
stride = (1, 1, 1) # D H W

pad = (1, 1, 1) # D H W

model = tf.keras.models.Sequential(
    [
        keras.layers.InputLayer(input_shape=input, batch_size = batches,
        dtype = 'float32', name= "in_data"),

        keras.layers.Conv3D(filters=outC, kernel_size=kernel, strides=stride, padding='same',
            groups = 1, use_bias = False, data_format="channels_last", name="conv3d"),
    ],
    name = "Conv3D"
)

in_shape = model.get_layer("conv3d").input_shape
out_shape = model.get_layer("conv3d").output_shape

input = tf.random.normal(in_shape)
output = tf.random.normal(out_shape)

model.compile(optimizer="Adam", loss="mse", metrics=["mae"])

model.fit(input, output)

print('Conv3D Params ======>>>>>')
print('input shape: ', in_shape)
print('kernel shape: ', kernel)
print('stride shape: ', stride)
print('output shape: ', out_shape)
print('-----------------------------------')

print(model.layers)

model.summary()

converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
open('./conv3d.tflite', "wb").write(tflite_model)