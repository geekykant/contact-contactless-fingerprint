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
	# model definition
	# def build_model():
	# 	x1 = layers.Input(shape = (160, 160, 1))
	# 	x2 = layers.Input(shape = (160, 160, 1))
	#
	# 	# share weights both inputs
	# 	inputs = layers.Input(shape = (160, 160, 1))
	# 	feature = layers.Conv2D(32, kernel_size = 3, activation = 'relu')(inputs)
	# 	feature = layers.MaxPooling2D(pool_size = 2)(feature)
	# 	feature = layers.Conv2D(64, kernel_size = 3, activation = 'relu')(feature)
	# 	feature = layers.MaxPooling2D(pool_size = 2)(feature)
	# 	feature = layers.Conv2D(128, kernel_size = 3, activation = 'relu')(feature)
	# 	feature = layers.MaxPooling2D(pool_size = 2)(feature)
	# 	feature_model = Model(inputs = inputs, outputs = feature)
	#
	# 	# two feature models that sharing weights
	# 	x1_net = feature_model(x1)
	# 	x2_net = feature_model(x2)
	#
	# 	# subtract features
	# 	net = layers.Subtract()([x1_net, x2_net])
	# 	net = layers.Conv2D(128, kernel_size = 3, activation = 'relu')(net)
	# 	net = layers.MaxPooling2D(pool_size = 2)(net)
	# 	net = layers.Flatten()(net)
	# 	net = layers.Dense(512, activation = 'relu')(net)
	# 	net = layers.Dense(1, activation = 'sigmoid')(net)
	# 	model = Model(inputs = [x1, x2], outputs = net)
	#
	# 	# compile
	# 	model.compile(loss = 'binary_crossentropy', optimizer = optimizers.Adam(lr=1e-4), metrics = ['acc'])
	# 	return (model, feature_model)
	#
	# (model, feature_model) = build_model()

	# load model
	model_path = '../model/build4/10_70_fp160.h5'
	model = tf.keras.models.load_model(join(dirname, model_path), compile=False)

	#convert the model
	converter = tf.lite.TFLiteConverter.from_keras_model(model)
	tflite_model = converter.convert()
	# Save the model.
	with open('new_model.tflite', 'wb') as f:
		f.write(tflite_model)
