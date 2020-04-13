from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import time
import numpy as np
import cv2


cap = cv2.VideoCapture(0)

from PIL import Image
import tflite_runtime.interpreter as tflite
from tflite_runtime.interpreter import Interpreter


def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}


def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
  """Returns a sorted array of classification results."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      default='C:/Users/Kaiwalya/Desktop/sign_language/py_api/model.tflite')
  parser.add_argument(
      '--labels',
      default='C:/Users/Kaiwalya/Desktop/sign_language/py_api/dict.txt')
  args = parser.parse_args()

  labels = load_labels(args.labels)

  interpreter = Interpreter(args.model)
  interpreter.allocate_tensors()
  _, height, width, _ = interpreter.get_input_details()[0]['shape']
  while(1):
    ret, frame = cap.read()
    dim = (224, 224)
    image = cv2.resize(frame,dim)
    results = classify_image(interpreter, image)
    label_id, prob = results[0]
    cv2.imshow('frame',frame)
    print(labels[label_id],prob)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


if __name__ == '__main__':
  main()
