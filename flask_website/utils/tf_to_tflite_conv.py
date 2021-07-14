import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model

import cv2
from os.path import join, dirname
dirname = dirname(__file__)

with tf.device('/cpu:0'):
	# load model
	model_path = '../model/hello_model.h5'
	model = tf.keras.models.load_model(join(dirname, model_path), compile=False)

	#convert the model
	converter = tf.lite.TFLiteConverter.from_keras_model(model)
	tflite_model = converter.convert()
	# Save the model.
	with open(join(dirname, '../model/model.tflite'), 'wb') as f:
		f.write(tflite_model)
