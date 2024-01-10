
# TF for image segmentation model
import tensorflow
import numpy
import numpy as np
model = tensorflow.saved_model.load( './' )
classes = [  "Music" ,  "2pop" ,  ]
import librosa
waveform , sr = librosa.load('./sample.wav' , sr=16000)
if waveform.shape[0] % 16000 != 0:
	waveform = np.concatenate([waveform, np.zeros(16000)])
inp = tensorflow.constant( np.array([waveform]) , dtype='float32'  )
class_scores = model( inp )[0].numpy()
print("")
print("class_scores" , class_scores)
print("Class : " , classes[  class_scores.argmax()])