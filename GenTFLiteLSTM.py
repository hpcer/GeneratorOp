import tensorflow as tf
from tensorflow import keras

import numpy as np

import os
import argparse

batches = 5

input = (1, 128, 10) # T D

model = tf.keras.models.Sequential(
    [
        keras.layers.InputLayer(input_shape=input,
        dtype = 'float32', name= "in_data"),

        tf.keras.layers.LSTM(4),
    ],
    name = "LSTM"
)

in_shape = model.get_layer("lstm").input_shape
output_shape = model.get_layer("lstm").output_shape

print('in shape: ', in_shape)

input = tf.random.normal(in_shape)
output = tf.random.normal(output_shape)

model.compile(optimizer="Adam", loss="mse", metrics=["mae"])

model.fit(input, output)

print('Conv3D Params ======>>>>>')
print('input shape: ', in_shape)
print('input shape: ', output_shape)
print('-----------------------------------')

print(model.layers)

model.summary()

converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
open('./lstm.tflite', "wb").write(tflite_model)