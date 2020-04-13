import tensorflow as tf

graph_def_file = "F:/new_img/sign/retrained_graph.pb"
input_arrays = ["Mul"]
output_arrays = ["final_result"]

converter = tf.lite.TFLiteConverter.from_frozen_graph(
  graph_def_file, input_arrays, output_arrays)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)


# import tensorflow as tf
# saved_model_dir = "F:/new_img/sign/"
# converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
# tflite_model = converter.convert()
# open("converted_model.tflite", "wb").write(tflite_model)