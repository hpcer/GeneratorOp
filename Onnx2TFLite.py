from onnx_tf.backend import prepare
import onnx

#onnx to tf
onnx_model_path = 'lstm.onnx'
tf_model_path = 'model_tf'

onnx_model = onnx.load(onnx_model_path)
tf_rep = prepare(onnx_model)
tf_rep.export_graph(tf_model_path)


#tf to tflite
import tensorflow as tf

saved_model_dir = 'model_tf'
tflite_model_path = 'model.tflite'

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

# Save the model
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)