

# TFLite for audio classification model

import tensorflow as tf
from PIL import Image
import numpy as np
import librosa


model = tf.lite.Interpreter(model_path="model.tflite")
classes = [  "Music" ,  "2pop" ,  ]

waveform , sr = librosa.load('./sample.wav' , sr=16000)

if waveform.shape[0] % 16000 != 0:
	waveform = np.concatenate([waveform, np.zeros(16000)])

input_details = model.get_input_details()
output_details = model.get_output_details()

model.resize_tensor_input(input_details[0]['index'], (1, len(waveform)))
model.allocate_tensors()

model.set_tensor(input_details[0]['index'], waveform[None].astype('float32'))
model.invoke()

class_scores = model.get_tensor(output_details[0]['index'])

print("")
print("class_scores", class_scores)
print("Class : ", classes[class_scores.argmax()])